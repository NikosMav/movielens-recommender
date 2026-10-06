"""Stage-4 orchestration: validation candidate choice, LambdaRank, test refit.

Retrievers used to train the ranker are fit on fit-train. Labels are the
validation window. Before test scoring, retrievers and features are rebuilt on
full train. The ranker weights are not updated on test labels. See
:data:`REFIT_SEMANTICS`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.config import RunConfig
from movielens_recommender.data import load_users
from movielens_recommender.evaluate import evaluate_recommender, format_metrics
from movielens_recommender.ranker.candidates import (
    CANDIDATE_SET_NAMES,
    RANKER_CANDIDATE_SETS,
    PrecomputedRecommender,
    choose_candidate_set,
    materialize_candidate_set,
    retrieval_recall,
    score_rank_maps,
)
from movielens_recommender.ranker.explain import recreate_command, save_ranker
from movielens_recommender.ranker.features import (
    FeatureContext,
    assert_context_uses_only,
    assert_group_affinity_uses_only,
    build_feature_context,
    build_feature_matrix,
    categorical_feature_names,
    drop_retriever_score_rank,
    relevance_pairs,
)
from movielens_recommender.ranker.gate import gate_against_s3
from movielens_recommender.ranker.train import (
    RankerMatrices,
    assemble_matrices,
    feature_importance_gain,
    train_fixed_rounds,
    train_with_early_stop_then_refit,
)
from movielens_recommender.scale import release_memory
from movielens_recommender.split import (
    SplitConfig,
    SplitResult,
    apply_cold_start_policy,
    chronological_tail_holdout,
)

REFIT_SEMANTICS = (
    "Before scoring test, both retrievers are retrained on full train "
    "(fit-train union validation) and every feature is rebuilt from that "
    "matrix and those retrievers. The ranker is trained only on "
    "validation-window labels with features from the train part (fit-train). "
    "A held-out slice of validation users is used only to early-stop and "
    "record best_iteration. The refit ranker is then trained on all "
    "validation-window users for num_boost_round equal to best_iteration, "
    "still with fit-train features, and with no validation callback. "
    "Test-window labels never enter ranker training, early stopping, or "
    "candidate-set selection. If the ranker is retrained on a later window, "
    "num_boost_round is fixed to that best_iteration (primary seed): no new "
    "early stopping and no new candidate-set choice. On the global-time "
    "cutoff, the later window is the pre-cutoff train. Its chronological tail "
    "(val_fraction) supplies labels; features and retrievers for that "
    "training come from the head only. Before scoring the post-cutoff test, "
    "retrievers and features are rebuilt on the full pre-cutoff train. "
    "Post-cutoff labels are never used."
)


def _release_two_tower_pack(model: Any) -> None:
    """Drop a two-tower feature pack after its top-k lists are stored.

    A second pack (fit-train, while the full-train pack is still live) does
    not fit in the ml-32M commit limit (ADR-0011).
    """
    model._features = None
    model._seen = {}
    model._item_vectors = None
    model._model = None


def _sample_user_ids(config: RunConfig, frame: pd.DataFrame) -> list[int] | None:
    """Users to score when an evaluation sample is configured.

    ``None`` keeps the all-user path used by ml-1m and ml-latest-small.
    """
    if int(config.eval.user_sample_size) <= 0:
        return None
    return [int(uid) for uid in frame["user_id"].unique()]


def _round(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def _eval_ks(config: RunConfig) -> list[int]:
    return sorted(set(int(k) for k in list(config.eval.ks) + list(config.eval.retrieval_ks)))


def _seed_summary(per_seed: list[dict[str, Any]], keys: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(row["metrics"][key]) for row in per_seed if key in row["metrics"]]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": _round(float(arr.mean())),
            "std": _round(float(arr.std(ddof=0))),
            "min": _round(float(arr.min())),
            "max": _round(float(arr.max())),
            "values": [_round(float(v)) for v in vals],
        }
    return out


def _retriever_hit(hit: tuple[float, int] | None) -> dict[str, float | int] | None:
    if hit is None:
        return None
    return {"score": float(hit[0]), "rank": int(hit[1])}


def _provenance(
    items: Sequence[int],
    item_item: Mapping[int, tuple[float, int]],
    two_tower: Mapping[int, tuple[float, int]],
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        iid = int(item)
        rows.append(
            {
                "item_item": _retriever_hit(item_item.get(iid)),
                "two_tower": _retriever_hit(two_tower.get(iid)),
            }
        )
    return rows


def _blocks_for_users(
    ctx: FeatureContext,
    lists: Mapping[int, Sequence[int]],
    ii_maps: Mapping[int, Mapping[int, tuple[float, int]]],
    tt_maps: Mapping[int, Mapping[int, tuple[float, int]]],
    positives: set[tuple[int, int]],
    *,
    users: set[int] | None = None,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    blocks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for uid in sorted(lists):
        if users is not None and int(uid) not in users:
            continue
        items = [int(i) for i in lists[uid]]
        if len(items) < 2:
            continue
        ii = ii_maps.get(int(uid), {})
        tt = tt_maps.get(int(uid), {})
        matrix = build_feature_matrix(ctx, int(uid), items, ii, tt)
        labels = np.asarray(
            [1.0 if (int(uid), item) in positives else 0.0 for item in items],
            dtype=np.float64,
        )
        blocks.append((int(uid), matrix, labels))
    return blocks


def _with_columns(matrices: RankerMatrices, names: list[str]) -> RankerMatrices:
    if names == matrices.feature_names:
        return matrices
    index = [matrices.feature_names.index(name) for name in names]
    kept_cat = [name for name in matrices.categorical_features if name in names]
    return RankerMatrices(
        x=matrices.x[:, index],
        y=matrices.y,
        group=matrices.group,
        user_ids=matrices.user_ids,
        feature_names=list(names),
        categorical_features=kept_cat,
    )


def _rerank(
    booster: Any,
    ctx: FeatureContext,
    lists: Mapping[int, Sequence[int]],
    ii_maps: Mapping[int, Mapping[int, tuple[float, int]]],
    tt_maps: Mapping[int, Mapping[int, tuple[float, int]]],
    feature_names: Sequence[str],
) -> dict[int, list[int]]:
    """Score every candidate with ``booster`` and sort descending (stable)."""
    wanted = list(feature_names)
    blocks: list[np.ndarray] = []
    owners: list[tuple[int, list[int]]] = []
    ranked: dict[int, list[int]] = {}
    for uid in sorted(lists):
        items = [int(i) for i in lists[uid]]
        if not items:
            ranked[int(uid)] = []
            continue
        matrix = build_feature_matrix(
            ctx,
            int(uid),
            items,
            ii_maps.get(int(uid), {}),
            tt_maps.get(int(uid), {}),
        )
        if wanted != ctx.names:
            matrix = matrix[:, [ctx.names.index(name) for name in wanted]]
        blocks.append(matrix)
        owners.append((int(uid), items))
    if not blocks:
        return ranked
    scores = np.asarray(booster.predict(np.vstack(blocks)), dtype=np.float64)
    offset = 0
    for uid, items in owners:
        n = len(items)
        order = np.argsort(-scores[offset : offset + n], kind="mergesort")
        ranked[uid] = [items[int(i)] for i in order]
        offset += n
    return ranked


def _evaluate_lists(
    ranked: Mapping[int, Sequence[int]],
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: RunConfig,
    split: SplitResult,
    include_segments: bool,
) -> dict[str, Any]:
    model = PrecomputedRecommender(ranked)
    metrics = evaluate_recommender(
        model.recommend,
        train,
        test,
        relevance_threshold=config.eval.relevance_threshold,
        ks=tuple(_eval_ks(config)),
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
        split=split,
        include_segments=include_segments,
    )
    return format_metrics(metrics)


def _ranker_kwargs(config: RunConfig) -> dict[str, Any]:
    rank = config.models.ranker
    return {
        "learning_rate": rank.learning_rate,
        "num_leaves": rank.num_leaves,
        "min_data_in_leaf": rank.min_data_in_leaf,
        "feature_fraction": rank.feature_fraction,
        "bagging_fraction": rank.bagging_fraction,
    }


def _fit_item_item(
    train: pd.DataFrame, hyperparams: Mapping[str, Any]
) -> ItemItemCosineRecommender:
    return ItemItemCosineRecommender(
        min_common=int(hyperparams.get("min_common", 1)),
        k_neighbors=int(hyperparams.get("k_neighbors", 0)),
        shrinkage=float(hyperparams.get("shrinkage", 0.0)),
    ).fit(train)


def _fit_two_tower(
    train: pd.DataFrame,
    *,
    config: RunConfig,
    movies: pd.DataFrame,
    hyperparams: Mapping[str, Any],
    n_epochs: int,
    seed: int,
):
    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    return fit_two_tower_recommender(
        train,
        dataset=config.dataset,
        data_dir=config.data_dir,
        movies=movies,
        hyperparams=dict(hyperparams),
        seed=seed,
        relevance_threshold=config.eval.relevance_threshold,
        n_epochs=int(n_epochs),
        val_split=None,
        show_progress=True,
    )


def _relevant_users(
    train: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    relevance_threshold: float,
) -> dict[int, set[int]]:
    """Warm relevant items in ``labels`` against the ``train`` catalog."""
    view = SplitResult(
        train=train,
        test=labels,
        config=SplitConfig(relevance_threshold=relevance_threshold),
        n_users_kept=int(train["user_id"].nunique()),
        n_users_dropped=0,
    )
    _seen, relevant, _stats = apply_cold_start_policy(
        view, relevance_threshold=relevance_threshold
    )
    return relevant


def _candidate_reports(
    item_item: Mapping[int, Sequence[tuple[int, float]]],
    two_tower: Mapping[int, Sequence[tuple[int, float]]],
    *,
    k: int,
    train: pd.DataFrame,
    labels: pd.DataFrame,
    relevance_threshold: float,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[int, list[int]]]]:
    relevant = _relevant_users(train, labels, relevance_threshold=relevance_threshold)
    lists_by_set: dict[str, dict[int, list[int]]] = {}
    reports: dict[str, dict[str, float]] = {}
    for name in CANDIDATE_SET_NAMES:
        lists = materialize_candidate_set(name, item_item, two_tower, k)
        lists_by_set[name] = lists
        raw = retrieval_recall(lists, relevant, ks=(100, 200))
        reports[name] = {key: _round(val) for key, val in raw.items()}
    return reports, lists_by_set


def _summary_from_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Compact ablation row. Full metrics stay in the top-level metrics map."""
    tail = (
        (metrics.get("segments") or {})
        .get("item_head_tail", {})
        .get("tail", {})
    )
    ci = (metrics.get("confidence_intervals") or {}).get("ndcg@10", {})
    return {
        "ndcg@10": metrics.get("ndcg@10"),
        "ndcg@10_ci": ci,
        "recall@10": metrics.get("recall@10"),
        "coverage@10": metrics.get("coverage@10"),
        "tail_ndcg@10": tail.get("ndcg@10"),
        "tail_ndcg@10_ci": (tail.get("confidence_intervals") or {}).get("ndcg@10"),
    }


def run_ranker_stage(
    *,
    split: SplitResult,
    movies: pd.DataFrame,
    config: RunConfig,
    tuned_item_hp: Mapping[str, Any],
    two_tower_hp: Mapping[str, Any],
    two_tower_epochs: int,
    full_train_item_model: ItemItemCosineRecommender,
    full_train_two_tower: Any,
    bar_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Validation candidate choice, ranker training, and one test evaluation.

    ``bar_metrics`` is the already-computed ``item_item_cosine`` test block
    (the S3 gate). ``full_train_*`` models are the refit retrievers.
    """
    if split.val is None or split.val.empty:
        raise ValueError("ranker requires a validation window")
    rank_cfg = config.models.ranker
    k = int(rank_cfg.candidate_k)
    threshold = float(config.eval.relevance_threshold)
    primary_seed = int(config.seed)
    seeds = [int(s) for s in rank_cfg.seeds]
    if primary_seed not in seeds:
        raise ValueError(
            f"ranker seeds {seeds} must include the config seed {primary_seed}"
        )
    t0 = time.perf_counter()
    demo_mode = str(rank_cfg.demographics)
    users = None
    if demo_mode != "off":
        if config.dataset != "ml-1m":
            raise ValueError(
                f"ranker demographics={demo_mode!r} is ml-1m only; "
                f"dataset is {config.dataset!r}"
            )
        users = load_users(config.dataset, config.data_dir)

    test_user_ids = _sample_user_ids(config, split.test)
    print(
        f"Scoring full-train two-tower top-{k} on the test sample...",
        flush=True,
    )
    tt_full = full_train_two_tower.topk_with_scores(k, user_ids=test_user_ids)
    _release_two_tower_pack(full_train_two_tower)
    release_memory()

    print(
        "Fitting item-item and two-tower on fit-train for ranker features...",
        flush=True,
    )
    item_fit = _fit_item_item(split.train, tuned_item_hp)
    tt_fit, _feat, _tr = _fit_two_tower(
        split.train,
        config=config,
        movies=movies,
        hyperparams=two_tower_hp,
        n_epochs=int(two_tower_epochs),
        seed=primary_seed,
    )
    print(f"Retrieving top-{k} candidates on the validation window...", flush=True)
    val_user_ids = _sample_user_ids(config, split.val)
    ii_val = item_fit.topk_with_scores(k, user_ids=val_user_ids)
    tt_val = tt_fit.topk_with_scores(k, user_ids=val_user_ids)
    del item_fit, tt_fit, _feat, _tr
    release_memory()
    reports, lists_val = _candidate_reports(
        ii_val,
        tt_val,
        k=k,
        train=split.train,
        labels=split.val,
        relevance_threshold=threshold,
    )
    winner = choose_candidate_set(reports)
    print(
        f"Validation candidate Recall@200 winner: {winner} "
        f"({reports[winner]['recall@200']})",
        flush=True,
    )

    ctx_fit = build_feature_context(
        split.train,
        movies,
        users=users,
        mode=demo_mode,
        group_frame=split.train,
    )
    assert_context_uses_only(ctx_fit, split.train, split.val)
    assert_context_uses_only(ctx_fit, split.train, split.test)
    if demo_mode in {"affinity", "both"}:
        assert users is not None
        assert_group_affinity_uses_only(
            ctx_fit,
            split.train,
            users,
            pd.concat([split.val, split.test], ignore_index=True),
        )
    ii_maps_val = score_rank_maps(ii_val)
    tt_maps_val = score_rank_maps(tt_val)
    positives = relevance_pairs(split.val, relevance_threshold=threshold)
    label_users = set(int(u) for u in split.val["user_id"].unique())

    rank_sets = list(RANKER_CANDIDATE_SETS)
    if winner not in rank_sets:
        rank_sets.append(winner)

    trained: dict[str, dict[int, Any]] = {}
    matrices_by_set: dict[str, RankerMatrices] = {}
    for name in rank_sets:
        blocks = _blocks_for_users(
            ctx_fit,
            lists_val[name],
            ii_maps_val,
            tt_maps_val,
            positives,
            users=label_users,
        )
        matrices = assemble_matrices(blocks, ctx_fit.names)
        matrices.categorical_features = categorical_feature_names(demo_mode)
        matrices_by_set[name] = matrices
        seed_list = seeds if name == winner else [primary_seed]
        trained[name] = {}
        for seed in seed_list:
            print(
                f"Training lambdarank on {name} (seed={seed}, "
                f"groups={len(matrices.group)})...",
                flush=True,
            )
            trained[name][seed] = train_with_early_stop_then_refit(
                matrices,
                split_seed=primary_seed,
                early_stop_fraction=float(rank_cfg.early_stop_fraction),
                ranker_seed=seed,
                num_boost_round=int(rank_cfg.num_boost_round),
                early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
                **_ranker_kwargs(config),
            )

    dropped_names: list[str] | None = None
    print(
        f"Training drop-retriever-features ablation on {winner} "
        f"(seed={primary_seed})...",
        flush=True,
    )
    _dropped_x, dropped_names = drop_retriever_score_rank(
        matrices_by_set[winner].x, matrices_by_set[winner].feature_names
    )
    dropped_matrices = _with_columns(matrices_by_set[winner], dropped_names)
    dropped = train_with_early_stop_then_refit(
        dropped_matrices,
        split_seed=primary_seed,
        early_stop_fraction=float(rank_cfg.early_stop_fraction),
        ranker_seed=primary_seed,
        num_boost_round=int(rank_cfg.num_boost_round),
        early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
        **_ranker_kwargs(config),
    )

    print("Rebuilding candidate features on full train for test scoring...", flush=True)
    ii_full = full_train_item_model.topk_with_scores(k, user_ids=test_user_ids)
    ctx_full = build_feature_context(
        split.full_train,
        movies,
        users=users,
        mode=demo_mode,
        group_frame=split.train,
    )
    assert_context_uses_only(ctx_full, split.full_train, split.test)
    if demo_mode in {"affinity", "both"}:
        assert users is not None
        assert_group_affinity_uses_only(ctx_full, split.train, users, split.test)
    ii_maps_full = score_rank_maps(ii_full)
    tt_maps_full = score_rank_maps(tt_full)
    lists_full = {
        name: materialize_candidate_set(name, ii_full, tt_full, k) for name in rank_sets
    }
    eval_split = SplitResult(
        train=split.full_train,
        test=split.test,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )

    ablation_metrics: dict[str, dict[str, Any]] = {}
    per_seed_rows: list[dict[str, Any]] = []
    for name in rank_sets:
        for seed, result in trained[name].items():
            ranked = _rerank(
                result.booster,
                ctx_full,
                lists_full[name],
                ii_maps_full,
                tt_maps_full,
                result.refit_info["feature_names"],
            )
            metrics = _evaluate_lists(
                ranked,
                split.full_train,
                split.test,
                config=config,
                split=eval_split,
                include_segments=True,
            )
            if seed == primary_seed:
                ablation_metrics[name] = metrics
            if name == winner:
                per_seed_rows.append(
                    {
                        "seed": seed,
                        "best_iteration": int(result.best_iteration),
                        "n_train_users": int(result.n_train_users),
                        "n_early_stop_users": int(result.n_early_stop_users),
                        "metrics": metrics,
                    }
                )

    per_seed_rows.sort(key=lambda row: int(row["seed"]))
    summary_keys = [
        "ndcg@10",
        "ndcg@20",
        "precision@10",
        "recall@10",
        "precision@20",
        "recall@20",
        "recall@100",
        "recall@200",
        "coverage@10",
        "mean_popularity@10",
    ]
    across = _seed_summary(per_seed_rows, summary_keys)
    primary_row = next(row for row in per_seed_rows if int(row["seed"]) == primary_seed)
    headline = dict(primary_row["metrics"])
    for key, block in across.items():
        headline[key] = block["mean"]
    headline["seed_summary"] = across
    headline["per_seed_confidence_intervals"] = {
        str(row["seed"]): row["metrics"].get("confidence_intervals", {})
        for row in per_seed_rows
    }
    headline["n_seeds"] = float(len(per_seed_rows))

    no_ranker_metrics = _evaluate_lists(
        lists_full[winner],
        split.full_train,
        split.test,
        config=config,
        split=eval_split,
        include_segments=True,
    )
    dropped_ranked = _rerank(
        dropped.booster,
        ctx_full,
        lists_full[winner],
        ii_maps_full,
        tt_maps_full,
        dropped.refit_info["feature_names"],
    )
    dropped_metrics = _evaluate_lists(
        dropped_ranked,
        split.full_train,
        split.test,
        config=config,
        split=eval_split,
        include_segments=True,
    )

    primary_result = trained[winner][primary_seed]
    gains = feature_importance_gain(
        primary_result.booster, primary_result.refit_info["feature_names"]
    )
    model_dir = Path(config.models_dir) / config.dataset
    save_ranker(
        primary_result.booster,
        model_dir,
        {
            "feature_names": list(primary_result.refit_info["feature_names"]),
            "best_iteration": int(primary_result.best_iteration),
            "num_boost_round": int(primary_result.best_iteration),
            "candidate_set": winner,
            "objective": "lambdarank",
            "dataset": config.dataset,
            "seed": primary_seed,
            "dropped_features": [],
            "recreate_command": recreate_command(config.dataset),
            "refit_semantics": REFIT_SEMANTICS,
        },
    )

    gate = gate_against_s3(
        ranker_mean=float(across["ndcg@10"]["mean"]),
        per_seed_cis=[
            row["metrics"]["confidence_intervals"]["ndcg@10"] for row in per_seed_rows
        ],
        metrics={"item_item_cosine": bar_metrics},
    )
    gate["ranker_mean_ndcg@10"] = _round(gate["ranker_mean_ndcg@10"])
    gate["bar_ndcg@10"] = _round(gate["bar_ndcg@10"])
    gate["bar_ci"] = {
        "low": _round(gate["bar_ci"]["low"]),
        "high": _round(gate["bar_ci"]["high"]),
    }

    metrics_out: dict[str, dict[str, Any]] = {
        "lambdarank": headline,
        "no_ranker": no_ranker_metrics,
    }
    ablation_summary: dict[str, Any] = {}
    best_iterations: dict[str, int] = {}
    for name, metrics in ablation_metrics.items():
        key = f"lambdarank_{name}"
        metrics_out[key] = metrics
        ablation_summary[key] = _summary_from_metrics(metrics)
        best_iterations[key] = int(trained[name][primary_seed].best_iteration)
    metrics_out["lambdarank_drop_retriever_features"] = dropped_metrics
    ablation_summary["no_ranker"] = _summary_from_metrics(no_ranker_metrics)
    ablation_summary["lambdarank_drop_retriever_features"] = _summary_from_metrics(
        dropped_metrics
    )
    ablation_summary["lambdarank"] = _summary_from_metrics(headline)
    best_iterations["lambdarank_drop_retriever_features"] = int(dropped.best_iteration)

    hyperparams = {
        "lambdarank": {
            "candidate_k": k,
            "candidate_set": winner,
            "objective": "lambdarank",
            "learning_rate": rank_cfg.learning_rate,
            "num_leaves": rank_cfg.num_leaves,
            "min_data_in_leaf": rank_cfg.min_data_in_leaf,
            "feature_fraction": rank_cfg.feature_fraction,
            "bagging_fraction": rank_cfg.bagging_fraction,
            "num_boost_round_cap": rank_cfg.num_boost_round,
            "early_stopping_rounds": rank_cfg.early_stopping_rounds,
            "early_stop_fraction": rank_cfg.early_stop_fraction,
            "early_stop_user_split_seed": primary_seed,
            "best_iteration_primary_seed": int(primary_result.best_iteration),
            "seeds": seeds,
            "retriever_item_item": dict(tuned_item_hp),
            "retriever_two_tower": dict(two_tower_hp),
            "retriever_two_tower_epochs": int(two_tower_epochs),
        },
        "no_ranker": {"candidate_set": winner, "candidate_k": k},
        "lambdarank_drop_retriever_features": {
            "candidate_set": winner,
            "dropped_features": list(dropped_names or []),
            "best_iteration": int(dropped.best_iteration),
        },
    }
    tuned_flags = {name: name != "no_ranker" for name in metrics_out}

    ranker_block = {
        "refit_semantics": REFIT_SEMANTICS,
        "candidate_k": k,
        "selection_metric": "recall@200",
        "selection_split": "validation",
        "tie_break": (
            "recall@200, then recall@100, then union_balanced, "
            "item_item, two_tower, union_unbalanced"
        ),
        "candidates": reports,
        "winner": winner,
        "early_stop_fraction": rank_cfg.early_stop_fraction,
        "early_stop_user_split_seed": primary_seed,
        "seeds": seeds,
        "primary_seed": primary_seed,
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "best_iteration": int(row["best_iteration"]),
                "n_train_users": int(row["n_train_users"]),
                "n_early_stop_users": int(row["n_early_stop_users"]),
                "ndcg@10": row["metrics"]["ndcg@10"],
                "ndcg@10_ci": row["metrics"]["confidence_intervals"]["ndcg@10"],
                "recall@10": row["metrics"].get("recall@10"),
                "coverage@10": row["metrics"].get("coverage@10"),
            }
            for row in per_seed_rows
        ],
        "across_seeds": across,
        "best_iteration": {
            "primary_seed": int(primary_result.best_iteration),
            "by_model": best_iterations,
        },
        "feature_importance_gain": gains,
        "ablations": ablation_summary,
        "ablation_note": (
            "Ablation metrics are the primary seed. The lambdarank headline "
            "point estimates are the mean over ranker seeds. Seeds share "
            "candidate sets and retriever models; only the ranker seed changes. "
            "no_ranker is the winning candidate set in its original retriever "
            "order after the full-train retriever refit. "
            "lambdarank_drop_retriever_features drops item-item and two-tower "
            "score and rank columns and keeps the other features, including in_both."
        ),
        "gate": gate,
        "model_artifact": {
            "schema_version": 1,
            "path": str(model_dir / "ranker.txt"),
            "meta_path": str(model_dir / "ranker_meta.json"),
            "gitignored": True,
            "recreate_command": recreate_command(config.dataset),
        },
        "runtime_sec": _round(time.perf_counter() - t0, 3),
    }
    return {
        "metrics": metrics_out,
        "hyperparameters": hyperparams,
        "tuned": tuned_flags,
        "ranker": ranker_block,
        "later_window": {
            "winner": winner,
            "best_iteration": int(primary_result.best_iteration),
            "tuned_item_hp": dict(tuned_item_hp),
            "two_tower_hp": dict(two_tower_hp),
            "two_tower_epochs": int(two_tower_epochs),
        },
    }


def evaluate_ranker_later_window(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    config: RunConfig,
    tuned_item_hp: Mapping[str, Any],
    two_tower_hp: Mapping[str, Any],
    two_tower_epochs: int,
    full_train_item_model: ItemItemCosineRecommender,
    full_train_two_tower: Any,
    winner: str,
    best_iteration: int,
    split: SplitResult,
) -> dict[str, dict[str, Any]]:
    """Fixed-round ranker on a later window (global cutoff).

    ``best_iteration`` and ``winner`` come from the headline protocol.
    ``test`` labels are not used for training. ``train`` is the pre-cutoff
    matrix; its chronological tail is the label window.
    """
    if int(best_iteration) < 1:
        raise ValueError("later-window ranker requires best_iteration >= 1")
    rank_cfg = config.models.ranker
    demo_mode = str(rank_cfg.demographics)
    users = None
    if demo_mode != "off":
        if config.dataset != "ml-1m":
            raise ValueError(
                f"ranker demographics={demo_mode!r} is ml-1m only; "
                f"dataset is {config.dataset!r}"
            )
        users = load_users(config.dataset, config.data_dir)
    k = int(rank_cfg.candidate_k)
    threshold = float(config.eval.relevance_threshold)
    head, tail = chronological_tail_holdout(train, float(config.split.val_fraction))
    if tail.empty:
        raise ValueError("later-window label tail is empty")

    print(
        f"[global_cutoff] Training lambdarank for fixed "
        f"num_boost_round={int(best_iteration)} on the pre-cutoff tail...",
        flush=True,
    )
    item_head = _fit_item_item(head, tuned_item_hp)
    tt_head, _feat, _tr = _fit_two_tower(
        head,
        config=config,
        movies=movies,
        hyperparams=two_tower_hp,
        n_epochs=int(two_tower_epochs),
        seed=int(config.seed),
    )
    ii_head = item_head.topk_with_scores(k)
    tt_head_scores = tt_head.topk_with_scores(k)
    lists_head = materialize_candidate_set(winner, ii_head, tt_head_scores, k)
    ctx_head = build_feature_context(
        head, movies, users=users, mode=demo_mode, group_frame=head
    )
    assert_context_uses_only(ctx_head, head, tail)
    assert_context_uses_only(ctx_head, head, test)
    if demo_mode in {"affinity", "both"}:
        assert users is not None
        assert_group_affinity_uses_only(
            ctx_head, head, users, pd.concat([tail, test], ignore_index=True)
        )
    positives = relevance_pairs(tail, relevance_threshold=threshold)
    label_users = set(int(u) for u in tail["user_id"].unique())
    blocks = _blocks_for_users(
        ctx_head,
        lists_head,
        score_rank_maps(ii_head),
        score_rank_maps(tt_head_scores),
        positives,
        users=label_users,
    )
    matrices = assemble_matrices(blocks, ctx_head.names)
    matrices.categorical_features = categorical_feature_names(demo_mode)
    booster, info = train_fixed_rounds(
        matrices,
        ranker_seed=int(config.seed),
        num_boost_round=int(best_iteration),
        **_ranker_kwargs(config),
    )
    if int(info["num_trees"]) != int(best_iteration):
        raise AssertionError(
            f"later-window ranker trained {info['num_trees']} trees, "
            f"expected fixed num_boost_round={int(best_iteration)}"
        )

    ctx_full = build_feature_context(
        train, movies, users=users, mode=demo_mode, group_frame=head
    )
    assert_context_uses_only(ctx_full, train, test)
    if demo_mode in {"affinity", "both"}:
        assert users is not None
        assert_group_affinity_uses_only(ctx_full, head, users, test)
    ii_full = full_train_item_model.topk_with_scores(k)
    tt_full = full_train_two_tower.topk_with_scores(k)
    lists_full = materialize_candidate_set(winner, ii_full, tt_full, k)
    ranked = _rerank(
        booster,
        ctx_full,
        lists_full,
        score_rank_maps(ii_full),
        score_rank_maps(tt_full),
        info["feature_names"],
    )
    lambdarank = _evaluate_lists(
        ranked,
        train,
        test,
        config=config,
        split=split,
        include_segments=False,
    )
    no_ranker = _evaluate_lists(
        lists_full,
        train,
        test,
        config=config,
        split=split,
        include_segments=False,
    )
    lambdarank["ranker_protocol"] = {
        "num_boost_round": int(best_iteration),
        "early_stopping": False,
        "candidate_set": winner,
        "labels": "chronological tail of pre-cutoff train",
        "training_features": "head of pre-cutoff train",
        "scoring_features": "full pre-cutoff train",
        "test_labels_used": False,
    }
    return {"lambdarank": lambdarank, "no_ranker": no_ranker}
