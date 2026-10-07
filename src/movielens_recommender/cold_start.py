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
from movielens_recommender.data import load_ratings
from movielens_recommender.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    ndcg_at_k,
    recall_at_k,
)
from movielens_recommender.movies import load_movies
from movielens_recommender.ranker.features import (
    FeatureContext,
    build_feature_context,
    build_feature_matrix,
    cold_start_feature_names,
    history_user_stats,
)
from movielens_recommender.ranker.train import RankerMatrices
from movielens_recommender.split import (
    SplitConfig,
    SplitResult,
    _tail_count,
    apply_cold_start_policy,
    time_based_split,
)
from movielens_recommender.two_tower.train import fit_two_tower_recommender

HOLDOUT_FRACTION = 0.1
# p=0 is the same loop with the id mask never applied. Scoring still zeros the
# id embedding. 0.0 and 0.1 extend the round-1 grid, whose low edge was 0.25.
DROPOUT_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 0.75)
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
    *,
    target: str = "all_later",
    test_fraction: float = 0.2,
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
        if target == "all_later":
            tail = group.iloc[n:]
        elif target == "harness_tail":
            n_rows = len(group)
            n_tail = _tail_count(n_rows, test_fraction)
            start = max(n, n_rows - n_tail)
            tail = group.iloc[start:]
        else:
            raise ValueError(f"unknown cold-start target {target!r}")
        profiles[uid_i] = list(
            zip(
                head["item_id"].astype(int).tolist(),
                head["rating"].astype(float).tolist(),
                strict=True,
            )
        )
        histories[uid_i] = [int(item) for item in head["item_id"].tolist()]
        frames[uid_i] = head
        banned = set(histories[uid_i])
        rel: set[int] = set()
        for item, rating in zip(
            tail["item_id"].astype(int).tolist(),
            tail["rating"].astype(float).tolist(),
            strict=True,
        ):
            if float(rating) < threshold:
                continue
            if int(item) in banned:
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
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)

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
    # ``val`` must be set. train_two_tower early-stops only when val_split.val
    # is present. ``test`` is the same window because ndcg_point_estimate reads it.
    val_split = SplitResult(
        train=split.train,
        test=split.val,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    if val_split.val is None or val_split.val.empty:
        raise ValueError("representation early stopping requires a validation window")
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


    del winner_tower
    from movielens_recommender.cold_start_rank import finish_round2

    return finish_round2(
        config=config,
        results_dir=results_dir,
        t0=t0,
        winner_trial=winner_trial,
        rep_trials=rep_trials,
        dropout_at_edge=dropout_at_edge,
        selected_at_edge=selected_at_edge,
        best_dropout=best_dropout,
        split=split,
        full_train=full_train,
        remain=remain,
        held_ratings=held_ratings,
        held_ids=held_ids,
        n_eligible=int(len(eligible)),
        movies=movies,
        tower_hp=tower_hp,
        item_hp=item_hp,
        ease_hp=ease_hp,
    )
