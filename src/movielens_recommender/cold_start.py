"""S5c new-user cold start on ml-1m.

Held-out users are chosen from ids only, then removed from every training
frame. Representation, candidate source, and ranker variant are chosen on the
remaining users' validation window. Held-out ratings are read only after those
choices are fixed, and only as the N-shot evaluation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.config import RunConfig
from movielens_recommender.data import (
    DATASET_SHA256,
    DATASET_VERSION_LABELS,
    load_ratings,
    load_users,
)
from movielens_recommender.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    ndcg_at_k,
    paired_bootstrap_delta_ci,
    recall_at_k,
)
from movielens_recommender.movies import load_movies
from movielens_recommender.ranker.candidates import retrieval_recall, score_rank_maps
from movielens_recommender.ranker.features import (
    FeatureContext,
    build_feature_context,
    build_feature_matrix,
    categorical_feature_names,
    cold_start_feature_names,
    history_user_stats,
    relevance_pairs,
)
from movielens_recommender.ranker.train import (
    RankerMatrices,
    assemble_matrices,
    fit_lambdarank,
    lgb_params,
    split_users,
)
from movielens_recommender.split import (
    SplitConfig,
    SplitResult,
    apply_cold_start_policy,
    time_based_split,
)
from movielens_recommender.two_tower.train import fit_two_tower_recommender

HOLDOUT_FRACTION = 0.1
DROPOUT_GRID: tuple[float, ...] = (0.25, 0.5, 0.75)
COLD_START_NS: tuple[int, ...] = (1, 3, 5, 10)
CANDIDATE_TIE_ORDER: tuple[str, ...] = (
    "item_item_fold_in",
    "ease_fold_in",
    "history_two_tower",
)
SIMPLE_BASELINES: tuple[str, ...] = (
    "most_popular",
    "item_item_fold_in",
    "ease_fold_in",
)
RANKER_TIE_WINNER = "demographics_off"

DIFFERENCE_FROM_S4B = (
    "This is not the S4b simulated cold start. S4b keeps each evaluated user "
    "in the training matrices, truncates the query to the earliest N full-train "
    "ratings (N of 5 and 10), and scores the original per-user test split. The "
    "user-id embedding is the one learned for that user, and item-item "
    "similarities include that user's later train ratings. Here the user is "
    "absent from every training row. The model sees only the first N "
    "chronological ratings, N in {1, 3, 5, 10}, and is scored on the later "
    "ratings with relevance at least 4. There is no user-id embedding at score "
    "time, and demographic features are not used."
)


def eligible_user_ids(ratings: pd.DataFrame, min_ratings: int) -> np.ndarray:
    """User ids with at least ``min_ratings`` rows. Counts only; not rating values."""
    counts = ratings.groupby("user_id", sort=True).size()
    kept = counts[counts >= int(min_ratings)]
    return np.sort(kept.index.to_numpy(dtype=np.int64))


def select_held_out_user_ids(
    user_ids: np.ndarray,
    *,
    fraction: float = HOLDOUT_FRACTION,
    seed: int = 42,
) -> np.ndarray:
    """Seeded shuffle of user ids. Rating values are not an input.

    ``fraction`` is rounded to a count, at least 1 and at most ``n - 1``.
    """
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("holdout fraction must be in (0, 1)")
    ids = np.unique(np.asarray(user_ids, dtype=np.int64))
    ids.sort()
    if len(ids) < 2:
        raise ValueError("holdout selection needs at least two users")
    rng = np.random.default_rng(int(seed))
    order = ids.copy()
    rng.shuffle(order)
    n_held = int(round(len(order) * float(fraction)))
    n_held = min(max(n_held, 1), len(order) - 1)
    return np.sort(order[:n_held])


def exclude_held_out_users(ratings: pd.DataFrame, held_out_ids: np.ndarray) -> pd.DataFrame:
    """Drop every row belonging to a held-out user."""
    held = {int(uid) for uid in np.asarray(held_out_ids, dtype=np.int64)}
    remain = ratings.loc[~ratings["user_id"].isin(held)].copy()
    if remain.empty:
        raise ValueError("no ratings left after removing held-out users")
    return remain


def assert_held_out_absent(held_out_ids: np.ndarray, *frames: pd.DataFrame) -> None:
    """Raise if any held-out user id appears in a training frame."""
    held = {int(uid) for uid in np.asarray(held_out_ids, dtype=np.int64)}
    for frame in frames:
        present = {int(uid) for uid in frame["user_id"].unique()}
        overlap = held & present
        if overlap:
            sample = sorted(overlap)[:5]
            raise AssertionError(f"held-out users are in a training frame: {sample}")


def choose_cold_candidate(reports: dict[str, dict[str, float]]) -> str:
    """Winner by validation Recall@200, then Recall@100, then a fixed order."""
    missing = [name for name in CANDIDATE_TIE_ORDER if name not in reports]
    if missing:
        raise ValueError(f"candidate reports missing: {missing}")

    def _key(name: str) -> tuple[float, float, int]:
        block = reports[name]
        return (
            -float(block["recall@200"]),
            -float(block["recall@100"]),
            CANDIDATE_TIE_ORDER.index(name),
        )

    return min(CANDIDATE_TIE_ORDER, key=_key)


def _round(value: float) -> float:
    return round(float(value), 6)


def _ci_dict(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": _round(mean), "low": _round(low), "high": _round(high)}


def _load_tuned_hyperparams(
    results_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    tower_path = results_dir / "tuning" / "two_tower_ml-1m.json"
    classic_path = results_dir / "tuning" / "ml-1m.json"
    tower = json.loads(tower_path.read_text(encoding="utf-8"))
    classic = json.loads(classic_path.read_text(encoding="utf-8"))
    tower_hp = dict(tower["two_tower"]["best_hyperparams"])
    item_hp = dict(classic["item_item_cosine"]["best_hyperparams"])
    ease_hp = dict(classic["ease"]["best_hyperparams"])
    return tower_hp, item_hp, ease_hp


def _fit_item_item(train: pd.DataFrame, hyperparams: dict[str, Any]) -> ItemItemCosineRecommender:
    return ItemItemCosineRecommender(
        min_common=int(hyperparams.get("min_common", 1)),
        k_neighbors=int(hyperparams.get("k_neighbors", 0)),
        shrinkage=float(hyperparams.get("shrinkage", 0.0)),
    ).fit(train)


def _fit_ease(train: pd.DataFrame, hyperparams: dict[str, Any]) -> EASERecommender:
    return EASERecommender(l2=float(hyperparams["l2"])).fit(train)


def _history_frames(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    ordered = frame.sort_values(
        ["user_id", "timestamp", "item_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    return {int(uid): group for uid, group in ordered.groupby("user_id", sort=False)}


def _profiles_from_frames(
    frames: dict[int, pd.DataFrame],
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[int]]]:
    profiles: dict[int, list[tuple[int, float]]] = {}
    histories: dict[int, list[int]] = {}
    for uid, group in frames.items():
        profiles[uid] = list(
            zip(
                group["item_id"].astype(int).tolist(),
                group["rating"].astype(float).tolist(),
                strict=True,
            )
        )
        histories[uid] = [int(item) for item in group["item_id"].tolist()]
    return profiles, histories


def _ids_only(retrieved: dict[int, list[tuple[int, float]]]) -> dict[int, list[int]]:
    return {int(uid): [int(item) for item, _score in rows] for uid, rows in retrieved.items()}


def _val_relevant(split: SplitResult, threshold: float) -> dict[int, set[int]]:
    val_view = SplitResult(
        train=split.train,
        test=split.val if split.val is not None else split.train.iloc[0:0],
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    _seen, relevant, _stats = apply_cold_start_policy(
        val_view, relevance_threshold=threshold
    )
    return {int(uid): set(int(i) for i in items) for uid, items in relevant.items()}


def _context_for(
    frame: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame | None,
    *,
    mode: str,
    group_frame: pd.DataFrame | None,
    threshold: float,
) -> FeatureContext:
    ctx = build_feature_context(
        frame,
        movies,
        users=users,
        mode=mode,
        group_frame=group_frame,
        relevance_threshold=threshold,
    )
    ctx.names = cold_start_feature_names(mode)
    return ctx


def _ranker_blocks(
    ctx: FeatureContext,
    lists: dict[int, list[int]],
    ii_maps: dict[int, dict[int, tuple[float, int]]],
    ease_maps: dict[int, dict[int, tuple[float, int]]],
    tt_maps: dict[int, dict[int, tuple[float, int]]],
    positives: set[tuple[int, int]],
    histories: dict[int, pd.DataFrame],
    *,
    demographics_missing: bool = False,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    blocks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for uid in sorted(lists):
        items = [int(item) for item in lists[uid]]
        if len(items) < 2:
            continue
        frame = histories.get(int(uid))
        if frame is None or frame.empty:
            continue
        stats = history_user_stats(ctx, frame["item_id"].tolist(), frame["rating"].tolist())
        matrix = build_feature_matrix(
            ctx,
            int(uid),
            items,
            ii_maps.get(int(uid), {}),
            tt_maps.get(int(uid), {}),
            user_stats=stats,
            ease=ease_maps.get(int(uid), {}),
            demographics_missing=demographics_missing,
        )
        labels = np.asarray(
            [1.0 if (int(uid), item) in positives else 0.0 for item in items],
            dtype=np.float64,
        )
        blocks.append((int(uid), matrix, labels))
    return blocks


def _nan_demographic_columns(matrices: RankerMatrices) -> RankerMatrices:
    x = np.array(matrices.x, copy=True)
    for index, name in enumerate(matrices.feature_names):
        if name.startswith("demo_") or name.startswith("group_"):
            x[:, index] = np.nan
    return RankerMatrices(
        x=x,
        y=matrices.y,
        group=matrices.group,
        user_ids=matrices.user_ids,
        feature_names=list(matrices.feature_names),
        categorical_features=list(matrices.categorical_features),
    )


def _rerank(
    booster: Any,
    ctx: FeatureContext,
    lists: dict[int, list[int]],
    ii_maps: dict[int, dict[int, tuple[float, int]]],
    ease_maps: dict[int, dict[int, tuple[float, int]]],
    tt_maps: dict[int, dict[int, tuple[float, int]]],
    histories: dict[int, pd.DataFrame],
    *,
    demographics_missing: bool,
) -> dict[int, list[int]]:
    blocks: list[np.ndarray] = []
    owners: list[tuple[int, list[int]]] = []
    ranked: dict[int, list[int]] = {}
    names = list(ctx.names)
    for uid in sorted(lists):
        items = [int(item) for item in lists[uid]]
        if not items:
            ranked[int(uid)] = []
            continue
        frame = histories[int(uid)]
        stats = history_user_stats(ctx, frame["item_id"].tolist(), frame["rating"].tolist())
        matrix = build_feature_matrix(
            ctx,
            int(uid),
            items,
            ii_maps.get(int(uid), {}),
            tt_maps.get(int(uid), {}),
            user_stats=stats,
            ease=ease_maps.get(int(uid), {}),
            demographics_missing=demographics_missing,
        )
        if list(ctx.names) != names:
            raise AssertionError("feature names changed while ranking")
        blocks.append(matrix)
        owners.append((int(uid), items))
    if not blocks:
        return ranked
    scores = np.asarray(booster.predict(np.vstack(blocks)), dtype=np.float64)
    offset = 0
    for uid, items in owners:
        n_items = len(items)
        order = np.argsort(-scores[offset : offset + n_items], kind="mergesort")
        ranked[uid] = [items[int(i)] for i in order]
        offset += n_items
    return ranked


def _score_ranking(
    lists: dict[int, list[int]],
    relevant: dict[int, set[int]],
    catalog: set[int],
    *,
    k: int,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    user_ids = np.asarray(sorted(relevant), dtype=np.int64)
    ndcgs = np.empty(len(user_ids), dtype=np.float64)
    recs: dict[int, list[int]] = {}
    for index, uid in enumerate(user_ids.tolist()):
        top = [int(item) for item in lists.get(int(uid), [])][:k]
        recs[int(uid)] = top
        ndcgs[index] = ndcg_at_k(top, relevant[int(uid)], k)
    mean, low, high = bootstrap_mean_ci(
        ndcgs, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
    )
    recall_vals = [
        recall_at_k(recs[int(uid)], relevant[int(uid)], k) for uid in user_ids.tolist()
    ]
    coverage = catalog_coverage(recs, catalog, k)
    metrics = {
        "ndcg@10": _round(mean),
        "ndcg@10_ci": _ci_dict(mean, low, high),
        "recall@10": _round(float(np.mean(recall_vals)) if recall_vals else 0.0),
        "coverage@10": _round(coverage),
        "n_eval_users": int(len(user_ids)),
    }
    return metrics, user_ids, ndcgs


def _best_simple(scores: dict[str, float]) -> str:
    return min(
        SIMPLE_BASELINES,
        key=lambda name: (-float(scores[name]), SIMPLE_BASELINES.index(name)),
    )


def _categorical_nan_failure(exc: BaseException) -> bool:
    """True when LightGBM rejected NaN demographic columns as categoricals."""
    text = str(exc).lower()
    return "categorical" in text or "nan" in text or "missing value" in text


def _known_user_reference(results_dir: Path) -> dict[str, Any]:
    """Copy headline test numbers. This file is not rewritten."""
    payload = json.loads((results_dir / "ml-1m.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    models: dict[str, Any] = {}
    for name in ("most_popular", "item_item_cosine"):
        block = metrics[name]
        models[name] = {
            "ndcg@10": block["ndcg@10"],
            "ndcg@10_ci": block["confidence_intervals"]["ndcg@10"],
            "recall@10": block["recall@10"],
            "coverage@10": block["coverage@10"],
            "aggregation": "single_run",
        }
    tower = payload["two_tower"]["across_seeds"]
    models["two_tower"] = {
        "ndcg@10": tower["ndcg@10"]["mean"],
        "ndcg@10_std": tower["ndcg@10"]["std"],
        "ndcg@10_ci": metrics["two_tower"]["confidence_intervals"]["ndcg@10"],
        "recall@10": tower["recall@10"]["mean"],
        "coverage@10": tower["coverage@10"]["mean"],
        "aggregation": "mean_over_seeds",
        "ci_note": "primary-seed user bootstrap; the point estimate is the seed mean",
    }
    ranker = payload["ranker"]["ablations"]["lambdarank"]
    across = payload["ranker"]["across_seeds"]["ndcg@10"]
    models["lambdarank"] = {
        "ndcg@10": ranker["ndcg@10"],
        "ndcg@10_std": across["std"],
        "ndcg@10_ci": ranker["ndcg@10_ci"],
        "recall@10": ranker["recall@10"],
        "coverage@10": ranker["coverage@10"],
        "aggregation": "mean_over_seeds",
        "ci_note": "primary-seed user bootstrap; the point estimate is the 3-seed mean",
    }
    return {
        "source": "results/ml-1m.json",
        "protocol": (
            "Headline per-user time split. These users were in training. "
            "Copied from the committed results file and not recomputed."
        ),
        "models": models,
    }


def _s4b_reference(results_dir: Path) -> dict[str, Any]:
    payload = json.loads((results_dir / "demographics" / "ml-1m.json").read_text(encoding="utf-8"))
    rows = [
        row
        for row in payload.get("cold_start", {}).get("rows", [])
        if int(row.get("n", -1)) == 5
    ]
    return {
        "source": "results/demographics/ml-1m.json",
        "n": 5,
        "rows": rows,
        "note": (
            "Copied for contrast. S4b does not remove the user from training. "
            "Those numbers are not the S5c result."
        ),
    }


def _held_out_targets(
    held_ratings: pd.DataFrame,
    n: int,
    catalog: set[int],
    threshold: float,
) -> tuple[
    dict[int, list[tuple[int, float]]],
    dict[int, list[int]],
    dict[int, pd.DataFrame],
    dict[int, set[int]],
    dict[str, int],
]:
    ordered = held_ratings.sort_values(
        ["user_id", "timestamp", "item_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    profiles: dict[int, list[tuple[int, float]]] = {}
    histories: dict[int, list[int]] = {}
    frames: dict[int, pd.DataFrame] = {}
    relevant: dict[int, set[int]] = {}
    n_short = 0
    n_no_warm = 0
    n_cold_relevant = 0
    for uid, group in ordered.groupby("user_id", sort=False):
        uid_i = int(uid)
        if len(group) <= n:
            n_short += 1
            continue
        head = group.iloc[:n]
        tail = group.iloc[n:]
        profiles[uid_i] = list(
            zip(
                head["item_id"].astype(int).tolist(),
                head["rating"].astype(float).tolist(),
                strict=True,
            )
        )
        histories[uid_i] = [int(item) for item in head["item_id"].tolist()]
        frames[uid_i] = head
        rel: set[int] = set()
        for item, rating in zip(
            tail["item_id"].astype(int).tolist(),
            tail["rating"].astype(float).tolist(),
            strict=True,
        ):
            if float(rating) < threshold:
                continue
            if int(item) not in catalog:
                n_cold_relevant += 1
                continue
            rel.add(int(item))
        if not rel:
            n_no_warm += 1
            continue
        relevant[uid_i] = rel
    counts = {
        "n_held_out_users": int(held_ratings["user_id"].nunique()),
        "n_users_shorter_than_or_equal_n": n_short,
        "n_users_no_warm_relevant": n_no_warm,
        "n_relevant_dropped_cold_item": n_cold_relevant,
        "n_eval_users": int(len(relevant)),
    }
    return profiles, histories, frames, relevant, counts


def _popular_lists(
    model: MostPopularRecommender,
    profiles: dict[int, list[tuple[int, float]]],
    n: int,
) -> dict[int, list[int]]:
    ranked = [int(item) for item in model._ranked_items.tolist()]
    out: dict[int, list[int]] = {}
    for uid, pairs in profiles.items():
        banned = {int(item) for item, _rating in pairs}
        picked: list[int] = []
        for item in ranked:
            if item in banned:
                continue
            picked.append(item)
            if len(picked) >= n:
                break
        out[int(uid)] = picked
    return out


def _representation_grid(base_hp: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [
        {
            **base_hp,
            "history_only": True,
            "user_id_dropout": 0.0,
            "score_without_user_id": True,
            "name": "history_only",
        }
    ]
    for p in DROPOUT_GRID:
        variants.append(
            {
                **base_hp,
                "history_only": False,
                "user_id_dropout": float(p),
                "score_without_user_id": True,
                "name": f"dropout_{p}",
            }
        )
    return variants


def _better_representation(
    score: float,
    history_only: bool,
    dropout: float,
    best_key: tuple[float, int, float],
) -> bool:
    return (float(score), 1 if history_only else 0, -float(dropout)) > best_key


def run_cold_start_experiment(
    config: RunConfig,
    *,
    download: bool = False,
) -> Path:
    """Choose on validation, then score held-out users. Writes one JSON file."""
    if config.dataset != "ml-1m":
        raise ValueError(
            "The cold-start evaluation is ml-1m only "
            f"(got dataset {config.dataset!r})."
        )
    if download:
        from movielens_recommender.data import download_dataset

        download_dataset(config.dataset, config.data_dir)

    t0 = time.perf_counter()
    results_dir = Path(config.results_dir)
    ratings, _clean = load_ratings(config.dataset, config.data_dir, clean=True)
    movies = load_movies(config.dataset, config.data_dir)
    users = load_users(config.dataset, config.data_dir)
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)
    k = int(config.models.ranker.candidate_k)
    rank_cfg = config.models.ranker

    eligible = eligible_user_ids(ratings, config.split.min_ratings)
    held_ids = select_held_out_user_ids(eligible, fraction=HOLDOUT_FRACTION, seed=seed)
    held_set = {int(uid) for uid in held_ids.tolist()}
    # Held-out rating values stay in this frame and are not passed to selection.
    held_ratings = ratings.loc[ratings["user_id"].isin(held_set)].copy()
    remain = exclude_held_out_users(ratings, held_ids)
    assert_held_out_absent(held_ids, remain)

    split = time_based_split(
        remain,
        SplitConfig(
            min_ratings=config.split.min_ratings,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            relevance_threshold=threshold,
        ),
    )
    if split.val is None or split.val.empty:
        raise ValueError("cold-start selection requires a validation window")
    full_train = split.full_train
    # The complement's per-user test tails are not training data and are not scored.
    assert_held_out_absent(held_ids, split.train, split.val, full_train)

    tower_hp, item_hp, ease_hp = _load_tuned_hyperparams(results_dir)
    print("Choosing a history-capable user tower on validation...", flush=True)
    winner_tower = None
    best_key = (float("-inf"), -1, float("-inf"))
    rep_trials: list[dict[str, Any]] = []
    val_split = SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    for variant in _representation_grid(tower_hp):
        name = str(variant["name"])
        print(f"  representation {name}", flush=True)
        rec, _feat, result = fit_two_tower_recommender(
            split.train,
            dataset=config.dataset,
            data_dir=config.data_dir,
            movies=movies,
            hyperparams=variant,
            seed=seed,
            relevance_threshold=threshold,
            val_split=val_split,
            show_progress=True,
        )
        score = float("-inf") if result.best_val_ndcg10 is None else float(result.best_val_ndcg10)
        trial = {
            "name": name,
            "history_only": bool(variant["history_only"]),
            "user_id_dropout": float(variant["user_id_dropout"]),
            "val_ndcg@10": None if score == float("-inf") else _round(score),
            "best_epoch": int(result.best_epoch),
            "epochs_trained": int(result.epochs_trained),
            "wall_time_sec": result.wall_time_sec,
        }
        rep_trials.append(trial)
        if _better_representation(
            score,
            bool(variant["history_only"]),
            float(variant["user_id_dropout"]),
            best_key,
        ):
            best_key = (
                score,
                1 if variant["history_only"] else 0,
                -float(variant["user_id_dropout"]),
            )
            winner_tower = rec
            winner_trial = trial
        else:
            del rec
    if winner_tower is None:
        raise RuntimeError("representation search produced no model")
    dropout_trials = [row for row in rep_trials if not row["history_only"]]
    best_dropout = max(
        dropout_trials,
        key=lambda row: (float(row["val_ndcg@10"]), -float(row["user_id_dropout"])),
    )
    dropout_at_edge = float(best_dropout["user_id_dropout"]) in (
        DROPOUT_GRID[0],
        DROPOUT_GRID[-1],
    )
    selected_at_edge = (not bool(winner_trial["history_only"])) and float(
        winner_trial["user_id_dropout"]
    ) in (DROPOUT_GRID[0], DROPOUT_GRID[-1])
    print(
        f"Representation winner: {winner_trial['name']} "
        f"val_ndcg@10={winner_trial['val_ndcg@10']}",
        flush=True,
    )

    print("Comparing candidate sources on validation recall...", flush=True)
    item_fit = _fit_item_item(split.train, item_hp)
    ease_fit = _fit_ease(split.train, ease_hp)
    relevant_val = _val_relevant(split, threshold)
    ii_val = item_fit.topk_with_scores(k)
    tt_val = winner_tower.topk_with_scores(k)
    fit_frames = _history_frames(split.train)
    ease_profiles, _ease_hist = _profiles_from_frames(fit_frames)
    ease_val = ease_fit.topk_for_profiles(ease_profiles, k)
    candidate_lists = {
        "item_item_fold_in": _ids_only(ii_val),
        "ease_fold_in": _ids_only(ease_val),
        "history_two_tower": _ids_only(tt_val),
    }
    candidate_reports: dict[str, dict[str, float]] = {}
    for name, lists in candidate_lists.items():
        report = retrieval_recall(lists, relevant_val, ks=(100, 200))
        candidate_reports[name] = {
            "recall@100": _round(report["recall@100"]),
            "recall@200": _round(report["recall@200"]),
            "mean_size": _round(report["mean_size"]),
            "n_eval_users": int(report["n_eval_users"]),
        }
    candidate_winner = choose_cold_candidate(candidate_reports)
    print(f"Candidate winner: {candidate_winner}", flush=True)

    print("Comparing rankers on the validation early-stop slice...", flush=True)
    train_users = {int(uid) for uid in split.train["user_id"].unique()}
    users_train = users.loc[users["user_id"].isin(train_users)].copy()
    positives = relevance_pairs(split.val, relevance_threshold=threshold)
    source_lists = candidate_lists[candidate_winner]
    ii_maps = score_rank_maps(ii_val)
    ease_maps = score_rank_maps(ease_val)
    tt_maps = score_rank_maps(tt_val)
    ranker_scores: dict[str, Any] = {}
    trained: dict[str, Any] = {}
    nan_variant_categoricals = True
    for mode, label in (("off", "demographics_off"), ("both", "demographics_nan")):
        ctx = _context_for(
            split.train,
            movies,
            users_train if mode == "both" else None,
            mode=mode,
            group_frame=split.train,
            threshold=threshold,
        )
        blocks = _ranker_blocks(
            ctx,
            source_lists,
            ii_maps,
            ease_maps,
            tt_maps,
            positives,
            fit_frames,
            demographics_missing=False,
        )
        matrices = assemble_matrices(blocks, ctx.names)
        if mode == "both":
            matrices.categorical_features = categorical_feature_names("both")
        train_ids, es_ids = split_users(
            matrices.user_ids, float(rank_cfg.early_stop_fraction), seed
        )
        train_m = matrices.subset({int(uid) for uid in train_ids.tolist()})
        early_m = matrices.subset({int(uid) for uid in es_ids.tolist()})
        valid_m = _nan_demographic_columns(early_m) if mode == "both" else early_m
        params = lgb_params(
            seed=seed,
            learning_rate=float(rank_cfg.learning_rate),
            num_leaves=int(rank_cfg.num_leaves),
            min_data_in_leaf=int(rank_cfg.min_data_in_leaf),
            feature_fraction=float(rank_cfg.feature_fraction),
            bagging_fraction=float(rank_cfg.bagging_fraction),
        )
        try:
            booster, info = fit_lambdarank(
                train_m,
                params=params,
                num_boost_round=int(rank_cfg.num_boost_round),
                valid=valid_m,
                early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
            )
        except Exception as exc:
            # Fixed before held-out scoring: if categorical columns cannot be
            # all-NaN on the early-stop slice, train that variant as numeric.
            if mode != "both" or not _categorical_nan_failure(exc):
                raise
            nan_variant_categoricals = False
            matrices.categorical_features = []
            train_m.categorical_features = []
            valid_m.categorical_features = []
            print(
                "  demographics_nan categorical columns rejected NaN; "
                "retrying with numeric splits",
                flush=True,
            )
            booster, info = fit_lambdarank(
                train_m,
                params=params,
                num_boost_round=int(rank_cfg.num_boost_round),
                valid=valid_m,
                early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
            )
        es_users = {int(uid) for uid in es_ids.tolist()}
        es_hist = {uid: fit_frames[uid] for uid in es_users if uid in fit_frames}
        es_lists = {uid: source_lists.get(uid, []) for uid in es_hist}
        missing_demos = mode == "both"
        ranked = _rerank(
            booster,
            ctx,
            es_lists,
            ii_maps,
            ease_maps,
            tt_maps,
            es_hist,
            demographics_missing=missing_demos,
        )
        relevant_es = {uid: items for uid, items in relevant_val.items() if uid in es_users}
        metrics, _ids, _vals = _score_ranking(
            ranked,
            relevant_es,
            set(int(i) for i in split.train["item_id"].unique()),
            k=10,
            n_bootstrap=int(config.eval.n_bootstrap),
            alpha=float(config.eval.bootstrap_alpha),
            seed=seed,
        )
        ranker_scores[label] = {
            "mode": mode,
            "best_iteration": int(info["best_iteration"]),
            "early_stop_ndcg@10": metrics["ndcg@10"],
            "early_stop_ndcg@10_ci": metrics["ndcg@10_ci"],
            "early_stop_n_eval_users": metrics["n_eval_users"],
            "demographics_missing_at_score_time": missing_demos,
        }
        trained[label] = {
            "matrices": matrices,
            "params": params,
            "best_iteration": int(info["best_iteration"]),
            "ctx_mode": mode,
        }
        print(
            f"  {label} early-stop NDCG@10={metrics['ndcg@10']} "
            f"best_iteration={info['best_iteration']}",
            flush=True,
        )
        del booster

    off_score = float(ranker_scores["demographics_off"]["early_stop_ndcg@10"])
    nan_score = float(ranker_scores["demographics_nan"]["early_stop_ndcg@10"])
    if nan_score > off_score:
        ranker_winner = "demographics_nan"
    else:
        # Tie, or off ahead: the new-user model does not keep demographic columns.
        ranker_winner = RANKER_TIE_WINNER if nan_score == off_score else "demographics_off"
    chosen_rank = trained[ranker_winner]
    refit_booster, refit_info = fit_lambdarank(
        chosen_rank["matrices"],
        params=chosen_rank["params"],
        num_boost_round=int(chosen_rank["best_iteration"]),
        valid=None,
        early_stopping_rounds=None,
    )
    print(f"Ranker winner: {ranker_winner}", flush=True)

    # Selection is done. Retrain retrievers on the 90% full train and score
    # held-out users. Nothing above read held_ratings.
    print("Refitting retrievers without held-out users...", flush=True)
    del item_fit, ease_fit, winner_tower
    final_hp = dict(tower_hp)
    final_hp["history_only"] = bool(winner_trial["history_only"])
    final_hp["user_id_dropout"] = float(winner_trial["user_id_dropout"])
    final_hp["score_without_user_id"] = True
    tower_full, _feat_full, _tower_fit = fit_two_tower_recommender(
        full_train,
        dataset=config.dataset,
        data_dir=config.data_dir,
        movies=movies,
        hyperparams=final_hp,
        seed=seed,
        relevance_threshold=threshold,
        n_epochs=int(winner_trial["best_epoch"]),
        show_progress=True,
    )
    item_full = _fit_item_item(full_train, item_hp)
    ease_full = _fit_ease(full_train, ease_hp)
    popular = MostPopularRecommender().fit(full_train)
    assert_held_out_absent(held_ids, full_train)
    tower_users = set(int(uid) for uid in tower_full._features.user_ids.tolist())
    if held_set & tower_users:
        raise AssertionError("held-out users are in the refit two-tower catalog")
    if held_set & set(item_full._user_index):
        raise AssertionError("held-out users are in the refit item-item model")
    if held_set & set(ease_full._user_index):
        raise AssertionError("held-out users are in the refit EASE model")

    score_mode = "both" if ranker_winner == "demographics_nan" else "off"
    demographics_missing = score_mode == "both"
    ctx_full = _context_for(
        full_train,
        movies,
        users_train if score_mode == "both" else None,
        mode=score_mode,
        group_frame=split.train,
        threshold=threshold,
    )
    catalog = {int(item) for item in full_train["item_id"].unique()}
    print("Scoring held-out users...", flush=True)
    by_n: list[dict[str, Any]] = []
    for n in COLD_START_NS:
        profiles, histories, frames, relevant, counts = _held_out_targets(
            held_ratings, n, catalog, threshold
        )
        ii = item_full.topk_for_profiles(profiles, k)
        ease_rows = ease_full.topk_for_profiles(profiles, k)
        tt_rows = tower_full.topk_for_new_histories(histories, k)
        ii_map = score_rank_maps(ii)
        ease_map = score_rank_maps(ease_rows)
        tt_map = score_rank_maps(tt_rows)
        lists_by_source = {
            "item_item_fold_in": _ids_only(ii),
            "ease_fold_in": _ids_only(ease_rows),
            "history_two_tower": _ids_only(tt_rows),
        }
        pipeline_in = {
            uid: lists_by_source[candidate_winner].get(uid, []) for uid in relevant
        }
        pipeline = _rerank(
            refit_booster,
            ctx_full,
            pipeline_in,
            ii_map,
            ease_map,
            tt_map,
            frames,
            demographics_missing=demographics_missing,
        )
        model_lists = {
            "most_popular": _popular_lists(popular, profiles, 10),
            "item_item_fold_in": {
                uid: lists_by_source["item_item_fold_in"].get(uid, [])[:10] for uid in relevant
            },
            "ease_fold_in": {
                uid: lists_by_source["ease_fold_in"].get(uid, [])[:10] for uid in relevant
            },
            "history_two_tower": {
                uid: lists_by_source["history_two_tower"].get(uid, [])[:10] for uid in relevant
            },
            "pipeline": {uid: pipeline.get(uid, [])[:10] for uid in relevant},
        }
        per_user: dict[str, np.ndarray] = {}
        user_order: np.ndarray | None = None
        model_metrics: dict[str, Any] = {}
        for name, ranked in model_lists.items():
            metrics, ids, values = _score_ranking(
                ranked,
                relevant,
                catalog,
                k=10,
                n_bootstrap=int(config.eval.n_bootstrap),
                alpha=float(config.eval.bootstrap_alpha),
                seed=seed,
            )
            model_metrics[name] = metrics
            per_user[name] = values
            if user_order is None:
                user_order = ids
            elif not np.array_equal(user_order, ids):
                raise AssertionError(f"{name} user order drifted")
        assert user_order is not None
        simple_scores = {name: float(model_metrics[name]["ndcg@10"]) for name in SIMPLE_BASELINES}
        best_name = _best_simple(simple_scores)
        paired = paired_bootstrap_delta_ci(
            per_user[best_name],
            per_user["pipeline"],
            n_bootstrap=int(config.eval.n_bootstrap),
            alpha=float(config.eval.bootstrap_alpha),
            seed=seed,
        )
        by_n.append(
            {
                "n": int(n),
                **counts,
                "models": model_metrics,
                "best_simple_baseline": best_name,
                "paired_bootstrap": {
                    "comparison": f"pipeline_minus_{best_name}",
                    "baseline": best_name,
                    "metric": "ndcg@10",
                    "mean": _round(float(paired["mean"])),
                    "low": _round(float(paired["low"])),
                    "high": _round(float(paired["high"])),
                    "n_users": int(paired["n_users"]),
                    "excludes_zero": bool(paired["excludes_zero"]),
                    "n_bootstrap": int(config.eval.n_bootstrap),
                    "alpha": float(config.eval.bootstrap_alpha),
                    "seed": seed,
                    "selection": (
                        "Highest NDCG@10 point estimate among most_popular, "
                        "item_item_fold_in, and ease_fold_in on this same table. "
                        "Ties prefer most_popular, then item_item_fold_in, then "
                        "ease_fold_in. The interval describes that gap."
                    ),
                },
            }
        )
        print(
            f"  N={n} pipeline NDCG@10={model_metrics['pipeline']['ndcg@10']} "
            f"best simple={best_name} {model_metrics[best_name]['ndcg@10']}",
            flush=True,
        )

    from movielens_recommender.serving.build import git_sha
    from movielens_recommender.serving.cold import measure_new_user_latency, save_cold_start_bundle

    artifact_dir = Path("artifacts") / config.dataset / "cold_start"
    save_cold_start_bundle(
        artifact_dir,
        movies=movies,
        item_item=item_full,
        ease=ease_full,
        two_tower=tower_full,
        booster=refit_booster,
        context=ctx_full,
        manifest={
            "dataset": config.dataset,
            "dataset_sha256": DATASET_SHA256[config.dataset],
            "git_sha": git_sha(),
            "candidate_source": candidate_winner,
            "candidate_k": k,
            "ranker_mode": ranker_winner,
            "demographics_missing": demographics_missing,
            "history_only": bool(winner_trial["history_only"]),
            "user_id_dropout": float(winner_trial["user_id_dropout"]),
            "best_iteration": int(chosen_rank["best_iteration"]),
            "feature_names": list(ctx_full.names),
            "relevance_threshold": threshold,
        },
    )
    latency = measure_new_user_latency(artifact_dir, full_train)
    print(f"Warmed new-user latency: {latency['latency_sec']:.4f}s", flush=True)

    from movielens_recommender.cli import library_versions

    payload: dict[str, Any] = {
        "experiment": "s5c_new_user_cold_start",
        "dataset": config.dataset,
        "dataset_version": DATASET_VERSION_LABELS[config.dataset],
        "dataset_sha256": DATASET_SHA256[config.dataset],
        "seed": seed,
        "holdout": {
            "fraction": HOLDOUT_FRACTION,
            "n_eligible_users": int(len(eligible)),
            "n_held_out_users": int(len(held_ids)),
            "held_out_user_ids": [int(uid) for uid in held_ids.tolist()],
            "selection": (
                "Seeded shuffle of user ids with at least min_ratings rows. "
                "Rating values and which rows would fall in a test tail are not inputs."
            ),
            "absent_from_training": True,
            "training_matrix": (
                "Per-user full train (fit-train union validation) of users who "
                "were not held out. Their per-user test tails are unused, matching "
                "the repo's full-train refit. Held-out users contribute no rows."
            ),
        },
        "representation": {
            "hyperparams_source": "results/tuning/two_tower_ml-1m.json",
            "fixed_hyperparams": {key: tower_hp[key] for key in sorted(tower_hp)},
            "note": (
                "Embedding size, learning rate, and temperature are the ADR-0006 "
                "winners. They were not searched again. Early stopping uses "
                "validation NDCG@10 with the id embedding removed."
            ),
            "dropout_grid": list(DROPOUT_GRID),
            "trials": rep_trials,
            "winner": winner_trial["name"],
            "history_only": bool(winner_trial["history_only"]),
            "user_id_dropout": float(winner_trial["user_id_dropout"]),
            "best_epoch": int(winner_trial["best_epoch"]),
            "val_ndcg@10": winner_trial["val_ndcg@10"],
            "tie_break": "higher validation NDCG@10, then history_only, then lower dropout p",
            "dropout_best_p": float(best_dropout["user_id_dropout"]),
            "dropout_best_val_ndcg@10": best_dropout["val_ndcg@10"],
            "dropout_best_at_edge": bool(dropout_at_edge),
            "selected_at_grid_edge": bool(selected_at_edge),
        },
        "candidates": {
            "k": k,
            "rule": (
                "validation recall@200, then recall@100, then "
                "item_item_fold_in, ease_fold_in, history_two_tower"
            ),
            "item_item_hyperparams_source": "results/tuning/ml-1m.json",
            "ease_hyperparams_source": "results/tuning/ml-1m.json",
            "item_item_hyperparams": item_hp,
            "ease_hyperparams": ease_hp,
            "reports": candidate_reports,
            "winner": candidate_winner,
            "query": (
                "Fit-train profile of each validation user. The history tower "
                "zeros the id embedding. Item-item uses ratings. EASE uses a "
                "binary fold-in. This is not the N-shot protocol."
            ),
        },
        "ranker": {
            "variants": ranker_scores,
            "winner": ranker_winner,
            "best_iteration": int(chosen_rank["best_iteration"]),
            "refit_num_trees": int(refit_info["num_trees"]),
            "seed": seed,
            "tie_break": (
                "Higher early-stop-slice NDCG@10 wins. An exact tie selects "
                "demographics_off."
            ),
            "selection_metric": (
                "NDCG@10 on the early-stop slice of validation users, scored "
                "with demographic and group columns missing. The nan variant is "
                "trained with those columns filled and early-stopped on a copy "
                "of the slice with the columns set to NaN. demographics_off never "
                "has the columns. One LightGBM seed."
            ),
            "feature_names": list(ctx_full.names),
            "demographics_missing_for_new_users": demographics_missing,
            "nan_variant_categoricals": bool(nan_variant_categoricals),
        },
        "evaluation": {
            "ns": list(COLD_START_NS),
            "relevance_threshold": threshold,
            "k": 10,
            "candidate_k": k,
            "catalog": "items in the refit training matrix",
            "seen_filter": "the N profile items are excluded; later ratings are the targets",
            "cold_items": "items absent from the refit catalog are dropped from relevance",
            "metrics": "ndcg@10 with user bootstrap CI, recall@10, coverage@10 (point only)",
            "n_bootstrap": int(config.eval.n_bootstrap),
            "bootstrap_alpha": float(config.eval.bootstrap_alpha),
            "bootstrap_seed": seed,
        },
        "known_users": _known_user_reference(results_dir),
        "s4b_simulated_cold_start_n5": _s4b_reference(results_dir),
        "difference_from_s4b": DIFFERENCE_FROM_S4B,
        "by_n": by_n,
        "serving": latency,
        "runtime_sec": _round(time.perf_counter() - t0),
        "library_versions": library_versions(),
    }
    out = results_dir / "cold-start" / f"{config.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return out
