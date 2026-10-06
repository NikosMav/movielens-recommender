"""S4b demographic-feature experiment on the existing LambdaRank pipeline.

The candidate set, retrievers, seeds, and early-stopping protocol are the S4
ones. This module does not tune on test and does not change ml-latest-small.
Group-affinity statistics stay on fit-train even when other features are
rebuilt on full train for test scoring.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.config import RunConfig
from movielens_recommender.data import (
    AGE_BUCKET_LABELS,
    DATASET_SHA256,
    DATASET_VERSION_LABELS,
    load_ratings,
    load_users,
)
from movielens_recommender.evaluate import evaluate_recommender, format_metrics
from movielens_recommender.metrics import ndcg_at_k, paired_bootstrap_delta_ci
from movielens_recommender.movies import load_movies
from movielens_recommender.ranker.candidates import (
    PrecomputedRecommender,
    choose_candidate_set,
    materialize_candidate_set,
    score_rank_maps,
)
from movielens_recommender.ranker.features import (
    GROUP_PRIOR_STRENGTH,
    FeatureContext,
    assert_context_uses_only,
    assert_group_affinity_uses_only,
    build_feature_context,
    build_feature_matrix,
    categorical_feature_names,
    feature_names,
    history_user_stats,
    relevance_pairs,
)
from movielens_recommender.ranker.pipeline import (
    _blocks_for_users,
    _candidate_reports,
    _evaluate_lists,
    _fit_item_item,
    _fit_two_tower,
    _ranker_kwargs,
    _rerank,
    _round,
    _seed_summary,
    _with_columns,
)
from movielens_recommender.ranker.train import (
    assemble_matrices,
    feature_importance_gain,
    train_with_early_stop_then_refit,
)
from movielens_recommender.split import (
    SplitConfig,
    SplitResult,
    apply_cold_start_policy,
    time_based_split,
)

VARIANTS: tuple[tuple[str, str], ...] = (
    ("baseline", "off"),
    ("raw", "raw"),
    ("affinity", "affinity"),
    ("both", "both"),
)
COLD_START_NS: tuple[int, ...] = (5, 10)
ADOPTION_RULE = (
    "Keep demographic features as the ranker default only when the +both "
    "3-seed mean test NDCG@10 is strictly greater than the baseline mean and "
    "the primary-seed paired bootstrap CI of (both minus baseline) has low > 0. "
    "Otherwise leave models.ranker.demographics at off."
)


class DemographicGroupPopularity:
    """Most-popular within the user's age-bucket × gender group.

    Counts come from the frame passed to :meth:`fit` (fit-train in this
    experiment). Users with no group, or a group that never appears in that
    frame, fall back to the global ranking of the same frame.
    """

    def __init__(self) -> None:
        self._by_group: dict[tuple[int, str], np.ndarray] = {}
        self._global: np.ndarray = np.array([], dtype=np.int64)
        self._user_key: dict[int, tuple[int, str]] = {}

    def fit(
        self, train: pd.DataFrame, users: pd.DataFrame
    ) -> DemographicGroupPopularity:
        self._global = MostPopularRecommender().fit(train)._ranked_items
        side = users[["user_id", "age", "gender"]].drop_duplicates("user_id")
        merged = train.merge(side, on="user_id", how="inner")
        self._by_group = {}
        if not merged.empty:
            for key, group in merged.groupby(["age", "gender"], sort=True):
                age, gender = key
                ranked = MostPopularRecommender().fit(group)._ranked_items
                self._by_group[(int(age), str(gender))] = ranked
        self._user_key = {}
        for row in side.itertuples(index=False):
            try:
                age = int(row.age)
            except (TypeError, ValueError):
                continue
            self._user_key[int(row.user_id)] = (age, str(row.gender))
        return self

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0:
            return []
        key = self._user_key.get(int(user_id))
        ranked = self._by_group.get(key, self._global) if key is not None else self._global
        return [int(item) for item in ranked[:n]]


def _reference_ranker(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "ml-1m.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"S4 results not found at {path}. The demographic experiment "
            "reuses that file's tuned retriever hyperparameters."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        summary = payload["metrics"]["lambdarank"]["seed_summary"]["ndcg@10"]
        item_hp = payload["hyperparameters"]["item_item_cosine_tuned"]
        two_tower = payload["two_tower"]
        ranker = payload["ranker"]
    except KeyError as exc:
        raise ValueError(f"{path} is missing S4 fields needed for this experiment") from exc
    return {
        "path": str(path),
        "item_hp": dict(item_hp),
        "two_tower_hp": dict(two_tower["best_hyperparams"]),
        "two_tower_epochs": int(two_tower["best_epoch"]),
        "winner": ranker.get("winner"),
        "ndcg@10_mean": float(summary["mean"]),
        "ndcg@10_std": float(summary["std"]),
        "candidates": ranker.get("candidates") or {},
    }


def _aligned_ndcg(
    ranked: dict[int, list[int]],
    seen: dict[int, set[int]],
    relevant: dict[int, set[int]],
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    user_ids: list[int] = []
    values: list[float] = []
    for uid, rel in relevant.items():
        hide = seen.get(int(uid), set())
        recs = [int(item) for item in ranked.get(int(uid), []) if int(item) not in hide]
        user_ids.append(int(uid))
        values.append(ndcg_at_k(recs, rel, k))
    return np.asarray(user_ids, dtype=np.int64), np.asarray(values, dtype=np.float64)


def _activity_block(metrics: dict[str, Any]) -> dict[str, Any]:
    terciles = (metrics.get("segments") or {}).get("user_activity_terciles") or {}
    out: dict[str, Any] = {}
    for label in ("low", "mid", "high"):
        block = terciles.get(label) or {}
        ci = (block.get("confidence_intervals") or {}).get("ndcg@10") or {}
        out[label] = {
            "ndcg@10": block.get("ndcg@10"),
            "ndcg@10_ci": ci,
            "n_users": block.get("n_users"),
        }
    return out


def _tail_block(metrics: dict[str, Any]) -> tuple[Any, Any]:
    tail = (
        (metrics.get("segments") or {}).get("item_head_tail") or {}
    ).get("tail") or {}
    ci = (tail.get("confidence_intervals") or {}).get("ndcg@10")
    return tail.get("ndcg@10"), ci


def _variant_summary(
    per_seed: list[dict[str, Any]],
    primary_metrics: dict[str, Any],
    *,
    primary_seed: int,
) -> dict[str, Any]:
    across = _seed_summary(per_seed, ["ndcg@10", "recall@10", "coverage@10"])
    primary = next(row for row in per_seed if int(row["seed"]) == primary_seed)
    tail, tail_ci = _tail_block(primary_metrics)
    return {
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "best_iteration": int(row["best_iteration"]),
                "ndcg@10": row["metrics"]["ndcg@10"],
                "ndcg@10_ci": row["metrics"]["confidence_intervals"]["ndcg@10"],
                "recall@10": row["metrics"]["recall@10"],
                "coverage@10": row["metrics"]["coverage@10"],
            }
            for row in sorted(per_seed, key=lambda row: int(row["seed"]))
        ],
        "across_seeds": across,
        "summary": {
            "ndcg@10_mean": across["ndcg@10"]["mean"],
            "ndcg@10_std": across["ndcg@10"]["std"],
            "ndcg@10_ci": primary["metrics"]["confidence_intervals"]["ndcg@10"],
            "recall@10_mean": across["recall@10"]["mean"],
            "coverage@10_mean": across["coverage@10"]["mean"],
            "tail_ndcg@10": tail,
            "tail_ndcg@10_ci": tail_ci,
            "tail_seed": primary_seed,
            "ci_seed": primary_seed,
        },
        "activity": _activity_block(primary_metrics),
    }


def _rerank_histories(
    booster: Any,
    ctx: FeatureContext,
    lists: dict[int, list[int]],
    ii_maps: dict[int, dict[int, tuple[float, int]]],
    tt_maps: dict[int, dict[int, tuple[float, int]]],
    feature_names_wanted: list[str],
    histories: dict[int, pd.DataFrame],
) -> dict[int, list[int]]:
    """Score candidates with user-history features taken from ``histories``."""
    wanted = list(feature_names_wanted)
    index = None if wanted == ctx.names else [ctx.names.index(name) for name in wanted]
    blocks: list[np.ndarray] = []
    owners: list[tuple[int, list[int]]] = []
    ranked: dict[int, list[int]] = {}
    for uid in sorted(lists):
        items = [int(item) for item in lists[uid]]
        if not items:
            ranked[int(uid)] = []
            continue
        frame = histories[int(uid)]
        stats = history_user_stats(
            ctx, frame["item_id"].tolist(), frame["rating"].tolist()
        )
        matrix = build_feature_matrix(
            ctx,
            int(uid),
            items,
            ii_maps.get(int(uid), {}),
            tt_maps.get(int(uid), {}),
            user_stats=stats,
        )
        if index is not None:
            matrix = matrix[:, index]
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


def _earliest_histories(
    train: pd.DataFrame, user_ids: set[int], n: int
) -> tuple[
    dict[int, pd.DataFrame],
    dict[int, list[tuple[int, float]]],
    dict[int, list[int]],
    dict[int, list[int]],
    int,
    int,
]:
    """Earliest ``n`` train rows per user, plus the full seen-item mask."""
    sub = train.loc[train["user_id"].isin(user_ids)]
    ordered = sub.sort_values(
        ["user_id", "timestamp", "item_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    frames: dict[int, pd.DataFrame] = {}
    profiles: dict[int, list[tuple[int, float]]] = {}
    histories: dict[int, list[int]] = {}
    mask: dict[int, list[int]] = {}
    n_truncated = 0
    n_short = 0
    for uid, group in ordered.groupby("user_id", sort=False):
        uid_i = int(uid)
        head = group.iloc[: int(n)]
        if len(group) > n:
            n_truncated += 1
        else:
            n_short += 1
        frames[uid_i] = head
        profiles[uid_i] = list(
            zip(
                head["item_id"].astype(int).tolist(),
                head["rating"].astype(float).tolist(),
                strict=True,
            )
        )
        histories[uid_i] = [int(item) for item in head["item_id"].tolist()]
        mask[uid_i] = [int(item) for item in group["item_id"].tolist()]
    return frames, profiles, histories, mask, n_truncated, n_short


def _eval_at_10(
    ranked_or_fn: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    config: RunConfig,
    split: SplitResult,
    precomputed: bool,
) -> dict[str, Any]:
    if precomputed:
        recommend = PrecomputedRecommender(ranked_or_fn).recommend
    else:
        recommend = ranked_or_fn
    metrics = evaluate_recommender(
        recommend,
        train,
        test,
        relevance_threshold=config.eval.relevance_threshold,
        ks=(10,),
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
        split=split,
        include_segments=False,
    )
    return format_metrics(metrics)


def _fairness_table(
    user_ids: np.ndarray,
    baseline: np.ndarray,
    both: np.ndarray,
    users: pd.DataFrame,
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    by_id = users.drop_duplicates("user_id").set_index("user_id")
    gender: dict[str, Any] = {}
    age: dict[str, Any] = {}
    missing = 0
    buckets: dict[tuple[str, str], list[int]] = {}
    for pos, uid in enumerate(user_ids.tolist()):
        if int(uid) not in by_id.index:
            missing += 1
            continue
        row = by_id.loc[int(uid)]
        buckets.setdefault(("gender", str(row["gender"])), []).append(pos)
        age_key = str(int(row["age"])) if pd.notna(row["age"]) else "unknown"
        buckets.setdefault(("age", age_key), []).append(pos)

    def _one(indices: list[int]) -> dict[str, Any]:
        left = baseline[indices]
        right = both[indices]
        delta = paired_bootstrap_delta_ci(
            left, right, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
        )
        return {
            "n_users": int(len(indices)),
            "baseline_ndcg@10": _round(float(left.mean())),
            "both_ndcg@10": _round(float(right.mean())),
            "delta_ndcg@10": _round(float(delta["mean"])),
            "delta_ndcg@10_ci": {
                "low": _round(float(delta["low"])),
                "high": _round(float(delta["high"])),
            },
            "delta_excludes_zero": bool(delta["excludes_zero"]),
        }

    for key, indices in buckets.items():
        kind, label = key
        block = _one(indices)
        if kind == "gender":
            gender[label] = block
        else:
            block["label"] = AGE_BUCKET_LABELS.get(int(label), label) if label.isdigit() else label
            age[label] = block
    return {"by_gender": gender, "by_age": age, "n_eval_users_missing_demographics": missing}


def _demographic_gains(gains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, row in enumerate(gains, start=1):
        name = str(row["feature"])
        if name.startswith("demo_") or name.startswith("group_"):
            rows.append(
                {
                    "feature": name,
                    "gain": row["gain"],
                    "rank": rank,
                    "n_features": len(gains),
                }
            )
    return rows


def _cold_start_rows(
    *,
    config: RunConfig,
    split: SplitResult,
    eval_split: SplitResult,
    users: pd.DataFrame,
    item_full: Any,
    tt_full: Any,
    ctx_full: FeatureContext,
    winner: str,
    boosters: dict[str, Any],
    relevant_users: set[int],
) -> list[dict[str, Any]]:
    k = int(config.models.ranker.candidate_k)
    popular = MostPopularRecommender().fit(split.full_train)
    group_pop = DemographicGroupPopularity().fit(split.train, users)
    rows: list[dict[str, Any]] = []
    history_free = {
        "most_popular": popular.recommend,
        "group_most_popular": group_pop.recommend,
    }
    cached_free: dict[str, dict[str, Any]] = {}
    for name, fn in history_free.items():
        print(f"Cold-start {name} (history-free)...", flush=True)
        cached_free[name] = _eval_at_10(
            fn,
            split.full_train,
            split.test,
            config=config,
            split=eval_split,
            precomputed=False,
        )
    for n in COLD_START_NS:
        print(f"Cold-start truncated history N={n}...", flush=True)
        frames, profiles, histories, mask, n_truncated, n_short = _earliest_histories(
            split.full_train, relevant_users, n
        )
        ii = item_full.topk_for_profiles(profiles, k, mask_items=mask)
        tt = tt_full.topk_for_histories(histories, k, mask_items=mask)
        lists = materialize_candidate_set(winner, ii, tt, k)
        ii_maps = score_rank_maps(ii)
        tt_maps = score_rank_maps(tt)
        item_lists = {uid: [item for item, _score in ii.get(uid, [])] for uid in profiles}
        model_lists: dict[str, dict[int, list[int]]] = {"item_item": item_lists}
        for variant in ("baseline", "both"):
            result = boosters[variant]
            names = list(result.refit_info["feature_names"])
            model_lists[f"ranker_{variant}"] = _rerank_histories(
                result.booster,
                ctx_full,
                lists,
                ii_maps,
                tt_maps,
                names,
                frames,
            )
        counts = {
            "n_users_truncated": n_truncated,
            "n_users_shorter_than_n": n_short,
        }
        for name in ("most_popular", "group_most_popular"):
            metrics = cached_free[name]
            rows.append(
                {
                    "n": n,
                    "model": name,
                    "history_dependent": False,
                    "ndcg@10": metrics["ndcg@10"],
                    "ndcg@10_ci": metrics["confidence_intervals"]["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "coverage@10": metrics["coverage@10"],
                    "n_eval_users": metrics["n_eval_users"],
                    **counts,
                }
            )
        for name, ranked in model_lists.items():
            metrics = _eval_at_10(
                ranked,
                split.full_train,
                split.test,
                config=config,
                split=eval_split,
                precomputed=True,
            )
            rows.append(
                {
                    "n": n,
                    "model": name,
                    "history_dependent": True,
                    "ndcg@10": metrics["ndcg@10"],
                    "ndcg@10_ci": metrics["confidence_intervals"]["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "coverage@10": metrics["coverage@10"],
                    "n_eval_users": metrics["n_eval_users"],
                    **counts,
                }
            )
    return rows


def run_demographic_experiment(
    config: RunConfig,
    *,
    download: bool = False,
) -> Path:
    """Train the four feature variants on the S4 candidate set and write JSON.

    ml-latest-small is refused. Existing ``results/ml-1m.json`` is read and
    not modified.
    """
    if config.dataset != "ml-1m":
        raise ValueError(
            "The demographic experiment is ml-1m only "
            f"(got dataset {config.dataset!r}). ml-latest-small has no users.dat."
        )
    if download:
        from movielens_recommender.data import download_dataset

        download_dataset(config.dataset, config.data_dir)

    t0 = time.perf_counter()
    reference = _reference_ranker(Path(config.results_dir))
    ratings, _clean = load_ratings(config.dataset, config.data_dir, clean=True)
    movies = load_movies(config.dataset, config.data_dir)
    users = load_users(config.dataset, config.data_dir)
    split_cfg = SplitConfig(
        min_ratings=config.split.min_ratings,
        test_fraction=config.split.test_fraction,
        val_fraction=config.split.val_fraction,
        relevance_threshold=config.eval.relevance_threshold,
    )
    split = time_based_split(ratings, split_cfg)
    if split.val is None or split.val.empty:
        raise ValueError("demographic experiment requires a validation window")

    rank_cfg = config.models.ranker
    k = int(rank_cfg.candidate_k)
    threshold = float(config.eval.relevance_threshold)
    primary_seed = int(config.seed)
    seeds = [int(seed) for seed in rank_cfg.seeds]
    if primary_seed not in seeds:
        raise ValueError(f"ranker seeds {seeds} must include the config seed {primary_seed}")

    print("Fitting fit-train retrievers for ranker features...", flush=True)
    item_fit = _fit_item_item(split.train, reference["item_hp"])
    tt_fit, _feat, _tr = _fit_two_tower(
        split.train,
        config=config,
        movies=movies,
        hyperparams=reference["two_tower_hp"],
        n_epochs=int(reference["two_tower_epochs"]),
        seed=primary_seed,
    )
    print(f"Validation candidate sets at K={k}...", flush=True)
    ii_val = item_fit.topk_with_scores(k)
    tt_val = tt_fit.topk_with_scores(k)
    reports, lists_val = _candidate_reports(
        ii_val,
        tt_val,
        k=k,
        train=split.train,
        labels=split.val,
        relevance_threshold=threshold,
    )
    winner = choose_candidate_set(reports)
    print(f"Validation winner: {winner}", flush=True)

    ctx_fit = build_feature_context(
        split.train,
        movies,
        users=users,
        mode="both",
        group_frame=split.train,
        relevance_threshold=threshold,
        prior_strength=GROUP_PRIOR_STRENGTH,
    )
    assert_context_uses_only(ctx_fit, split.train, split.val)
    assert_context_uses_only(ctx_fit, split.train, split.test)
    assert_group_affinity_uses_only(
        ctx_fit,
        split.train,
        users,
        pd.concat([split.val, split.test], ignore_index=True),
    )

    # Fit-train retrievers are no longer needed; drop them before the full-train fit.
    del item_fit, tt_fit
    print("Fitting full-train retrievers for test scoring...", flush=True)
    item_full = _fit_item_item(split.full_train, reference["item_hp"])
    tt_full, _feat_full, _tr_full = _fit_two_tower(
        split.full_train,
        config=config,
        movies=movies,
        hyperparams=reference["two_tower_hp"],
        n_epochs=int(reference["two_tower_epochs"]),
        seed=primary_seed,
    )
    ctx_full = build_feature_context(
        split.full_train,
        movies,
        users=users,
        mode="both",
        group_frame=split.train,
        relevance_threshold=threshold,
        prior_strength=GROUP_PRIOR_STRENGTH,
    )
    assert_context_uses_only(ctx_full, split.full_train, split.test)
    assert_group_affinity_uses_only(ctx_full, split.train, users, split.test)

    positives = relevance_pairs(split.val, relevance_threshold=threshold)
    label_users = set(int(uid) for uid in split.val["user_id"].unique())
    ii_maps_val = score_rank_maps(ii_val)
    tt_maps_val = score_rank_maps(tt_val)
    blocks = _blocks_for_users(
        ctx_fit,
        lists_val[winner],
        ii_maps_val,
        tt_maps_val,
        positives,
        users=label_users,
    )
    matrices_both = assemble_matrices(blocks, ctx_fit.names)
    matrices_both.categorical_features = categorical_feature_names("both")

    ii_full = item_full.topk_with_scores(k)
    tt_scored = tt_full.topk_with_scores(k)
    lists_full = materialize_candidate_set(winner, ii_full, tt_scored, k)
    ii_maps_full = score_rank_maps(ii_full)
    tt_maps_full = score_rank_maps(tt_scored)
    eval_split = SplitResult(
        train=split.full_train,
        test=split.test,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )

    variants: dict[str, Any] = {}
    boosters: dict[str, Any] = {}
    ranked_primary: dict[str, dict[int, list[int]]] = {}
    per_user: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    seen, relevant, _stats = apply_cold_start_policy(
        eval_split, relevance_threshold=threshold
    )
    for variant, mode in VARIANTS:
        names = feature_names(mode)
        matrices = _with_columns(matrices_both, names)
        print(
            f"Training {variant} ({mode}) on {winner}, "
            f"features={len(names)}, groups={len(matrices.group)}...",
            flush=True,
        )
        per_seed: list[dict[str, Any]] = []
        primary_metrics: dict[str, Any] | None = None
        primary_result = None
        for seed in seeds:
            result = train_with_early_stop_then_refit(
                matrices,
                split_seed=primary_seed,
                early_stop_fraction=float(rank_cfg.early_stop_fraction),
                ranker_seed=seed,
                num_boost_round=int(rank_cfg.num_boost_round),
                early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
                **_ranker_kwargs(config),
            )
            ranked = _rerank(
                result.booster,
                ctx_full,
                lists_full,
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
                include_segments=seed == primary_seed,
            )
            per_seed.append(
                {
                    "seed": seed,
                    "best_iteration": int(result.best_iteration),
                    "metrics": metrics,
                }
            )
            if seed == primary_seed:
                primary_metrics = metrics
                primary_result = result
                boosters[variant] = result
                ranked_primary[variant] = ranked
                ids, values = _aligned_ndcg(ranked, seen, relevant, k=10)
                per_user[variant] = (ids, values)
                got = float(np.mean(values))
                reported = float(metrics["ndcg@10"])
                if abs(got - reported) > 1e-6:
                    raise AssertionError(
                        f"{variant} per-user NDCG@10 {got} != harness {reported}"
                    )
        assert primary_metrics is not None and primary_result is not None
        block = _variant_summary(per_seed, primary_metrics, primary_seed=primary_seed)
        block["mode"] = mode
        block["n_features"] = len(names)
        gains = feature_importance_gain(
            primary_result.booster, primary_result.refit_info["feature_names"]
        )
        if variant in {"baseline", "both"}:
            block["feature_importance_gain"] = gains
        block["demographic_feature_importance"] = _demographic_gains(gains)
        variants[variant] = block
        print(
            f"  {variant} NDCG@10 mean="
            f"{block['summary']['ndcg@10_mean']} "
            f"std={block['summary']['ndcg@10_std']}",
            flush=True,
        )

    del matrices_both
    base_ids, base_vals = per_user["baseline"]
    both_ids, both_vals = per_user["both"]
    if not np.array_equal(base_ids, both_ids):
        raise AssertionError("baseline and +both per-user ids are not aligned")
    paired = paired_bootstrap_delta_ci(
        base_vals,
        both_vals,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
    )
    paired_out = {
        "comparison": "both_minus_baseline",
        "seed": primary_seed,
        "metric": "ndcg@10",
        "mean": _round(float(paired["mean"])),
        "low": _round(float(paired["low"])),
        "high": _round(float(paired["high"])),
        "n_users": int(paired["n_users"]),
        "excludes_zero": bool(paired["excludes_zero"]),
        "n_bootstrap": int(config.eval.n_bootstrap),
        "alpha": float(config.eval.bootstrap_alpha),
    }
    base_mean = float(variants["baseline"]["summary"]["ndcg@10_mean"])
    both_mean = float(variants["both"]["summary"]["ndcg@10_mean"])
    mean_beats = both_mean > base_mean
    ci_positive = float(paired_out["low"]) > 0.0
    keep = bool(mean_beats and ci_positive)
    decision = {
        "rule": ADOPTION_RULE,
        "baseline_mean_ndcg@10": base_mean,
        "both_mean_ndcg@10": both_mean,
        "mean_beats_baseline": mean_beats,
        "paired_ci_low": paired_out["low"],
        "paired_ci_high": paired_out["high"],
        "paired_ci_excludes_zero": paired_out["excludes_zero"],
        "keep_as_default": keep,
        "negative_result": not keep,
    }

    print("Fairness slices (primary seed)...", flush=True)
    fairness = _fairness_table(
        base_ids,
        base_vals,
        both_vals,
        users,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
    )
    fairness["seed"] = primary_seed
    fairness["attributes"] = (
        "gender and age bucket from users.dat, used only for this comparison"
    )

    print("Simulated cold start...", flush=True)
    cold_rows = _cold_start_rows(
        config=config,
        split=split,
        eval_split=eval_split,
        users=users,
        item_full=item_full,
        tt_full=tt_full,
        ctx_full=ctx_full,
        winner=winner,
        boosters=boosters,
        relevant_users=set(int(uid) for uid in relevant),
    )

    s4_mean = float(reference["ndcg@10_mean"])
    s4_std = float(reference["ndcg@10_std"])
    reproduction = {
        "s4_source": reference["path"],
        "s4_ndcg@10_mean": s4_mean,
        "s4_ndcg@10_std": s4_std,
        "s4_winner": reference["winner"],
        "baseline_ndcg@10_mean": base_mean,
        "baseline_ndcg@10_std": float(variants["baseline"]["summary"]["ndcg@10_std"]),
        "abs_diff_mean": _round(abs(base_mean - s4_mean)),
        "matches_s4_at_4_decimals": round(base_mean, 4) == round(s4_mean, 4),
        "candidate_set": winner,
        "candidate_set_matches_s4": winner == reference["winner"],
    }

    from movielens_recommender.cli import library_versions

    payload: dict[str, Any] = {
        "experiment": "s4b_user_demographics",
        "dataset": config.dataset,
        "dataset_version": DATASET_VERSION_LABELS[config.dataset],
        "dataset_sha256": DATASET_SHA256[config.dataset],
        "seed": primary_seed,
        "seeds": seeds,
        "primary_seed": primary_seed,
        "candidate_k": k,
        "candidate_set": winner,
        "candidate_set_rule": (
            "validation recall@200, then recall@100, then union_balanced, "
            "item_item, two_tower, union_unbalanced"
        ),
        "validation_candidates": reports,
        "relevance_threshold": threshold,
        "group_prior_strength": GROUP_PRIOR_STRENGTH,
        "group_stats_matrix": (
            "fit-train only, including when other ranker features are rebuilt "
            "on full train for test scoring"
        ),
        "raw_features": list(feature_names("raw")[len(feature_names("off")) :]),
        "affinity_features": list(
            name for name in feature_names("affinity") if name.startswith("group_")
        ),
        "variants": variants,
        "paired_bootstrap": paired_out,
        "activity_segments": {name: variants[name]["activity"] for name, _mode in VARIANTS},
        "cold_start": {
            "ns": list(COLD_START_NS),
            "candidate_set": winner,
            "history": (
                "Each eval user's full-train ratings are sorted by timestamp "
                "and truncated to the earliest N. Retriever queries and "
                "user-history ranker features use only that prefix. The seen "
                "filter stays the full train history, so already-rated train "
                "items are not recommended. Test targets are unchanged."
            ),
            "most_popular_matrix": "full train",
            "group_popularity_matrix": "fit-train age-bucket x gender counts",
            "item_item": (
                "tuned item-item similarity fit on full train; the query "
                "vector is the truncated profile"
            ),
            "ranker": (
                "primary-seed refit boosters from this experiment; retriever "
                "scores and user-history features use the truncated profile; "
                "group-affinity tables stay on fit-train"
            ),
            "rows": cold_rows,
        },
        "fairness": fairness,
        "demographic_feature_importance": variants["both"]["demographic_feature_importance"],
        "decision": decision,
        "reproduction": reproduction,
        "protocol": {
            "labels": "validation window only",
            "training_features": "fit-train",
            "scoring_item_and_history_features": "full train",
            "scoring_group_affinity": "fit-train",
            "test_labels_used": False,
            "seeds_share_retrievers_and_candidates": True,
            "ml_latest_small_unchanged": True,
        },
        "library_versions": library_versions(),
        "runtime_sec": _round(time.perf_counter() - t0, 3),
    }
    out_dir = Path(config.results_dir) / "demographics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ml-1m.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {out_path}", flush=True)
    print(
        f"Decision keep_as_default={keep} "
        f"baseline={base_mean} both={both_mean} "
        f"paired=[{paired_out['low']}, {paired_out['high']}]",
        flush=True,
    )
    return out_path
