"""Round-2 cold-start ranker. Choices use validation users from the 90% only.

The held-out 604 are not an argument of :func:`select_cold_start_ranker`.
Their ratings are scored once, after the candidate K and the per-N serving
rule have been printed.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.cold_start import (
    COLD_START_NS,
    DIFFERENCE_FROM_S4B,
    DROPOUT_GRID,
    SIMPLE_BASELINES,
    _best_simple,
    _context_for,
    _fit_ease,
    _fit_item_item,
    _held_out_targets,
    _known_user_reference,
    _popular_lists,
    _round,
    _s4b_reference,
    _score_ranking,
    assert_held_out_absent,
    select_held_out_user_ids,
)
from movielens_recommender.config import RunConfig
from movielens_recommender.data import DATASET_SHA256, DATASET_VERSION_LABELS
from movielens_recommender.metrics import ndcg_at_k, paired_bootstrap_delta_ci
from movielens_recommender.ranker.features import (
    annotate_cold_start_ranker_rows,
    build_feature_matrix,
    cold_start_ranker_feature_names,
    history_user_stats,
)
from movielens_recommender.ranker.train import (
    RankerMatrices,
    fit_lambdarank,
    lgb_params,
    split_users,
)
from movielens_recommender.split import SplitResult
from movielens_recommender.two_tower.recommender import TwoTowerRecommender
from movielens_recommender.two_tower.train import fit_two_tower_recommender

RANK_CANDIDATE_KS: tuple[int, ...] = (50, 100, 200)
SERVING_METHODS: tuple[str, ...] = (
    "cold_start_ranker",
    "most_popular",
    "item_item_fold_in",
    "history_two_tower",
)
# Simpler methods win a tie. The ranker is last.
SERVING_TIE_ORDER: tuple[str, ...] = (
    "most_popular",
    "item_item_fold_in",
    "history_two_tower",
    "cold_start_ranker",
)
PROFILE_BUCKETS: tuple[int, ...] = COLD_START_NS
POPULARITY_VS_KNOWN_USERS = (
    "Cold-start most-popular NDCG is higher than the known-user most-popular "
    "number because the targets are every later rating, a long tail, and the "
    "short profile has not consumed the popular titles. The known-user number "
    "is a short per-user test tail after a long history."
)


def profile_bucket(n_ratings: int) -> int:
    """Largest grid N that does not exceed the live profile length."""
    n = int(n_ratings)
    if n < 1:
        raise ValueError("profile length must be at least 1")
    chosen = PROFILE_BUCKETS[0]
    for bucket in PROFILE_BUCKETS:
        if bucket <= n:
            chosen = bucket
    return int(chosen)


def choose_serving_method(scores: Mapping[str, float]) -> str:
    """Highest validation NDCG@10. Ties follow :data:`SERVING_TIE_ORDER`."""
    missing = [name for name in SERVING_TIE_ORDER if name not in scores]
    if missing:
        raise ValueError(f"serving scores missing: {missing}")

    def _key(name: str) -> tuple[float, int]:
        return (-float(scores[name]), SERVING_TIE_ORDER.index(name))

    return min(SERVING_TIE_ORDER, key=_key)


def choose_candidate_k(mean_ndcg_by_k: Mapping[int, float]) -> int:
    """Highest mean validation NDCG@10. A tie keeps the smaller K."""
    if not mean_ndcg_by_k:
        raise ValueError("candidate K grid is empty")
    return min(mean_ndcg_by_k, key=lambda k: (-float(mean_ndcg_by_k[int(k)]), int(k)))


def carve_ranker_validation_users(
    train_user_ids: np.ndarray,
    held_out_ids: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the 90% into ranker-train and ranker-validation ids.

    ``held_out_ids`` must be disjoint from ``train_user_ids``. Rating values
    are not an input. The returned validation ids are the ones
    :func:`select_held_out_user_ids` would hold out of this pool.
    """
    pool = np.unique(np.asarray(train_user_ids, dtype=np.int64))
    held = {int(uid) for uid in np.asarray(held_out_ids, dtype=np.int64)}
    overlap = held & {int(uid) for uid in pool.tolist()}
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(f"ranker validation pool contains held-out users: {sample}")
    val_ids = select_held_out_user_ids(pool, fraction=fraction, seed=seed)
    val_set = {int(uid) for uid in val_ids.tolist()}
    fit_ids = np.asarray(
        [int(uid) for uid in pool.tolist() if int(uid) not in val_set],
        dtype=np.int64,
    )
    return fit_ids, val_ids


def union_candidate_ids(
    ease_ids: Sequence[int],
    popular_ids: Sequence[int],
    tower_ids: Sequence[int],
    *,
    banned: set[int] | None = None,
) -> tuple[list[int], set[int], set[int], set[int]]:
    """Union in order EASE, most-popular, history two-tower.

    Duplicates are kept once, in that order. Source sets still record every
    list that contained the item. Banned ids are dropped from the union and
    from the source sets.
    """
    skip = banned or set()

    def _clean(ids: Sequence[int]) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for item in ids:
            iid = int(item)
            if iid in skip or iid in seen:
                continue
            seen.add(iid)
            out.append(iid)
        return out

    ease = _clean(ease_ids)
    popular = _clean(popular_ids)
    tower = _clean(tower_ids)
    ease_set = set(ease)
    pop_set = set(popular)
    tower_set = set(tower)
    union: list[int] = []
    seen: set[int] = set()
    for item in ease + popular + tower:
        if item in seen:
            continue
        seen.add(item)
        union.append(item)
    return union, ease_set, pop_set, tower_set


@dataclass(frozen=True)
class SimQuery:
    """One truncated profile and the later relevant items."""

    uid: int
    profile_n: int
    profile: tuple[tuple[int, float], ...]
    history: tuple[int, ...]
    relevant: frozenset[int]


@dataclass
class Retrieved:
    ease: dict[int, list[tuple[int, float]]]
    item_item: dict[int, list[tuple[int, float]]]
    tower: dict[int, list[tuple[int, float]]]
    popular: dict[int, list[int]]


@dataclass
class RankerChoices:
    """Frozen before any held-out rating is scored."""

    candidate_k: int
    best_iteration: int
    serving_by_n: dict[int, str]
    serving_val_ndcg: dict[int, dict[str, float]]
    k_trials: list[dict[str, Any]]
    n_ranker_train_users: int
    n_ranker_val_users: int
    feature_names: list[str]
    hyperparams: dict[str, Any] = field(default_factory=dict)


def simulated_queries(
    ratings: pd.DataFrame,
    user_ids: np.ndarray,
    catalog: set[int],
    threshold: float,
    ns: Sequence[int] = COLD_START_NS,
) -> dict[int, list[SimQuery]]:
    """First-N profiles. Labels are later ratings with relevance at least 4.

    History features are computed later from ``profile`` only. ``ratings`` is
    the full chronology, including per-user test tails used as labels.
    """
    wanted = {int(uid) for uid in np.asarray(user_ids, dtype=np.int64).tolist()}
    frame = ratings.loc[ratings["user_id"].isin(wanted)]
    ordered = frame.sort_values(
        ["user_id", "timestamp", "item_id"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    by_n: dict[int, list[SimQuery]] = {int(n): [] for n in ns}
    for uid, group in ordered.groupby("user_id", sort=False):
        uid_i = int(uid)
        items = [int(item) for item in group["item_id"].tolist()]
        stars = [float(rating) for rating in group["rating"].tolist()]
        for n in ns:
            n_i = int(n)
            if len(items) <= n_i:
                continue
            banned = set(items[:n_i])
            relevant = {
                int(item)
                for item, rating in zip(items[n_i:], stars[n_i:], strict=True)
                if float(rating) >= threshold and int(item) in catalog and int(item) not in banned
            }
            if not relevant:
                continue
            by_n[n_i].append(
                SimQuery(
                    uid=uid_i,
                    profile_n=n_i,
                    profile=tuple(zip(items[:n_i], stars[:n_i], strict=True)),
                    history=tuple(items[:n_i]),
                    relevant=frozenset(relevant),
                )
            )
    return by_n


def _pair_map(rows: Sequence[tuple[int, float]]) -> dict[int, tuple[float, int]]:
    return {
        int(item): (float(score), rank)
        for rank, (item, score) in enumerate(rows, start=1)
    }


def _prefix_pairs(
    rows: Sequence[tuple[int, float]],
    k: int,
) -> list[tuple[int, float]]:
    return [(int(item), float(score)) for item, score in list(rows)[: int(k)]]


def _retrieve(
    queries: Sequence[SimQuery],
    *,
    item_item: ItemItemCosineRecommender,
    ease: EASERecommender,
    tower: TwoTowerRecommender,
    popular: MostPopularRecommender,
    k: int,
) -> Retrieved:
    if not queries:
        return Retrieved(ease={}, item_item={}, tower={}, popular={})
    profiles = {q.uid: list(q.profile) for q in queries}
    histories = {q.uid: list(q.history) for q in queries}
    return Retrieved(
        ease=ease.topk_for_profiles(profiles, k),
        item_item=item_item.topk_for_profiles(profiles, k),
        tower=tower.topk_for_new_histories(histories, k),
        popular=_popular_lists(popular, profiles, k),
    )


def _point_ndcg(
    lists: Mapping[int, Sequence[int]],
    queries: Sequence[SimQuery],
    *,
    k: int = 10,
) -> float:
    if not queries:
        return 0.0
    scores = [
        ndcg_at_k([int(item) for item in lists.get(q.uid, ())][:k], set(q.relevant), k)
        for q in queries
    ]
    return float(np.mean(np.asarray(scores, dtype=np.float64)))


def _ids_from_pairs(rows: Sequence[tuple[int, float]], k: int) -> list[int]:
    return [int(item) for item, _score in list(rows)[:k]]


def _baseline_lists(
    retrieved: Retrieved,
    queries: Sequence[SimQuery],
    *,
    k: int = 10,
) -> dict[str, dict[int, list[int]]]:
    return {
        "most_popular": {q.uid: list(retrieved.popular.get(q.uid, []))[:k] for q in queries},
        "item_item_fold_in": {
            q.uid: _ids_from_pairs(retrieved.item_item.get(q.uid, []), k) for q in queries
        },
        "history_two_tower": {
            q.uid: _ids_from_pairs(retrieved.tower.get(q.uid, []), k) for q in queries
        },
        "ease_fold_in": {
            q.uid: _ids_from_pairs(retrieved.ease.get(q.uid, []), k) for q in queries
        },
    }


def _candidate_spec(
    query: SimQuery,
    retrieved: Retrieved,
    k: int,
) -> tuple[list[int], set[int], set[int], set[int], dict, dict, dict] | None:
    banned = set(query.history)
    ease_rows = _prefix_pairs(retrieved.ease.get(query.uid, []), k)
    ii_rows = _prefix_pairs(retrieved.item_item.get(query.uid, []), k)
    tower_rows = _prefix_pairs(retrieved.tower.get(query.uid, []), k)
    popular_ids = [int(item) for item in retrieved.popular.get(query.uid, [])[:k]]
    items, ease_set, pop_set, tower_set = union_candidate_ids(
        [item for item, _score in ease_rows],
        popular_ids,
        [item for item, _score in tower_rows],
        banned=banned,
    )
    if len(items) < 2:
        return None
    return (
        items,
        ease_set,
        pop_set,
        tower_set,
        _pair_map(ii_rows),
        _pair_map(ease_rows),
        _pair_map(tower_rows),
    )


def _matrices_for_k(
    queries_by_n: Mapping[int, Sequence[SimQuery]],
    retrieved_by_n: Mapping[int, Retrieved],
    k: int,
    ctx: Any,
    names: list[str],
) -> RankerMatrices:
    """One LightGBM group per user per N. All N of a user share that user id.

    Specs are collected first so the feature matrix is allocated once.
    """
    specs: list[tuple] = []
    for n in COLD_START_NS:
        retrieved = retrieved_by_n[int(n)]
        grouped: list[tuple] = []
        for query in queries_by_n.get(int(n), []):
            spec = _candidate_spec(query, retrieved, k)
            if spec is None:
                continue
            items, ease_set, pop_set, tower_set, ii_map, ease_map, tt_map = spec
            labels = np.asarray(
                [1.0 if int(item) in query.relevant else 0.0 for item in items],
                dtype=np.float32,
            )
            if len(labels) < 2 or float(labels.min()) == float(labels.max()):
                continue
            grouped.append(
                (query, items, ease_set, pop_set, tower_set, ii_map, ease_map, tt_map, labels)
            )
        grouped.sort(key=lambda row: int(row[0].uid))
        specs.extend(grouped)
    if not specs:
        raise ValueError(f"K={k} produced no ranker groups")
    n_rows = sum(len(row[-1]) for row in specs)
    n_cols = len(names)
    x_out = np.empty((n_rows, n_cols), dtype=np.float32)
    y_out = np.empty(n_rows, dtype=np.float32)
    groups: list[int] = []
    owners: list[int] = []
    cursor = 0
    for query, items, ease_set, pop_set, tower_set, ii_map, ease_map, tt_map, labels in specs:
        stats = history_user_stats(
            ctx,
            [item for item, _rating in query.profile],
            [rating for _item, rating in query.profile],
        )
        matrix = build_feature_matrix(
            ctx,
            query.uid,
            items,
            ii_map,
            tt_map,
            user_stats=stats,
            ease=ease_map,
            demographics_missing=False,
        )
        matrix = annotate_cold_start_ranker_rows(
            matrix,
            names,
            items,
            profile_n=query.profile_n,
            ease_ids=ease_set,
            popular_ids=pop_set,
            tower_ids=tower_set,
        )
        width = len(labels)
        x_out[cursor : cursor + width] = matrix
        y_out[cursor : cursor + width] = labels
        cursor += width
        groups.append(width)
        owners.append(int(query.uid))
    del specs
    gc.collect()
    return RankerMatrices(
        x=x_out,
        y=y_out,
        group=np.asarray(groups, dtype=np.int32),
        user_ids=np.asarray(owners, dtype=np.int64),
        feature_names=list(names),
    )


def _predict_lists(
    booster: Any,
    ctx: Any,
    names: list[str],
    queries: Sequence[SimQuery],
    retrieved: Retrieved,
    k: int,
    *,
    num_iteration: int,
) -> dict[int, list[int]]:
    ranked: dict[int, list[int]] = {}
    chunk: list[tuple[int, list[int], np.ndarray]] = []

    def _flush() -> None:
        if not chunk:
            return
        blocks = [matrix for _uid, _items, matrix in chunk]
        scores = np.asarray(
            booster.predict(np.vstack(blocks), num_iteration=int(num_iteration)),
            dtype=np.float64,
        )
        offset = 0
        for uid, items, _matrix in chunk:
            n_items = len(items)
            order = np.argsort(-scores[offset : offset + n_items], kind="mergesort")
            ranked[uid] = [items[int(i)] for i in order]
            offset += n_items
        chunk.clear()

    for query in queries:
        spec = _candidate_spec(query, retrieved, k)
        if spec is None:
            ranked[query.uid] = []
            continue
        items, ease_set, pop_set, tower_set, ii_map, ease_map, tt_map = spec
        stats = history_user_stats(
            ctx,
            [item for item, _rating in query.profile],
            [rating for _item, rating in query.profile],
        )
        matrix = build_feature_matrix(
            ctx,
            query.uid,
            items,
            ii_map,
            tt_map,
            user_stats=stats,
            ease=ease_map,
            demographics_missing=False,
        )
        matrix = annotate_cold_start_ranker_rows(
            matrix,
            names,
            items,
            profile_n=query.profile_n,
            ease_ids=ease_set,
            popular_ids=pop_set,
            tower_ids=tower_set,
        )
        chunk.append((query.uid, items, np.asarray(matrix, dtype=np.float32)))
        if len(chunk) >= 256:
            _flush()
    _flush()
    return ranked


def _tower_hyperparams(
    tower_hp: Mapping[str, Any],
    winner_trial: Mapping[str, Any],
) -> dict[str, Any]:
    hp = dict(tower_hp)
    hp["history_only"] = bool(winner_trial["history_only"])
    hp["user_id_dropout"] = float(winner_trial["user_id_dropout"])
    hp["score_without_user_id"] = True
    return hp


def _fit_tower(
    frame: pd.DataFrame,
    *,
    config: RunConfig,
    movies: pd.DataFrame,
    hyperparams: Mapping[str, Any],
    epochs: int,
    threshold: float,
) -> TwoTowerRecommender:
    tower, _features, _result = fit_two_tower_recommender(
        frame,
        dataset=config.dataset,
        data_dir=config.data_dir,
        movies=movies,
        hyperparams=dict(hyperparams),
        seed=int(config.seed),
        relevance_threshold=threshold,
        n_epochs=max(1, int(epochs)),
        show_progress=True,
    )
    return tower


def select_cold_start_ranker(
    *,
    config: RunConfig,
    remain: pd.DataFrame,
    full_train: pd.DataFrame,
    held_ids: np.ndarray,
    movies: pd.DataFrame,
    tower_hp: Mapping[str, Any],
    item_hp: Mapping[str, Any],
    ease_hp: Mapping[str, Any],
    winner_trial: Mapping[str, Any],
) -> RankerChoices:
    """Choose K and the per-N rule on a user carve of the 90%.

    Retrievers used for this metric are fit on ranker-train full-train only,
    so ranker-validation users are absent. ``held_ids`` is an absence check,
    not a set of labels.
    """
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)
    fraction = float(config.models.ranker.early_stop_fraction)
    pool = np.unique(full_train["user_id"].to_numpy(dtype=np.int64))
    fit_ids, val_ids = carve_ranker_validation_users(
        pool, held_ids, fraction=fraction, seed=seed
    )
    fit_set = {int(uid) for uid in fit_ids.tolist()}
    val_set = {int(uid) for uid in val_ids.tolist()}
    if fit_set & val_set:
        raise AssertionError("ranker-train and ranker-validation users overlap")
    fit_frame = full_train.loc[full_train["user_id"].isin(fit_set)].copy()
    present = {int(uid) for uid in fit_frame["user_id"].unique()}
    if present & val_set:
        raise AssertionError("ranker-validation users are in the selection retrievers")
    if present & {int(uid) for uid in np.asarray(held_ids, dtype=np.int64).tolist()}:
        raise AssertionError("held-out users are in the selection retrievers")
    print(
        f"Ranker validation carve: {len(fit_ids)} train users, "
        f"{len(val_ids)} validation users, fraction {fraction}.",
        flush=True,
    )
    names = cold_start_ranker_feature_names()
    bad = [name for name in names if name.startswith(("demo_", "group_"))]
    if bad:
        raise AssertionError(f"cold-start ranker features include demographics: {bad}")
    hp = _tower_hyperparams(tower_hp, winner_trial)
    print("Fitting selection retrievers without ranker-validation users...", flush=True)
    tower = _fit_tower(
        fit_frame,
        config=config,
        movies=movies,
        hyperparams=hp,
        epochs=int(winner_trial["best_epoch"]),
        threshold=threshold,
    )
    item_item = _fit_item_item(fit_frame, dict(item_hp))
    ease = _fit_ease(fit_frame, dict(ease_hp))
    popular = MostPopularRecommender().fit(fit_frame)
    catalog = {int(item) for item in fit_frame["item_id"].unique()}
    ctx = _context_for(
        fit_frame,
        movies,
        None,
        mode="off",
        group_frame=None,
        threshold=threshold,
    )
    ctx.names = list(names)
    print("Building simulated short profiles...", flush=True)
    train_queries = simulated_queries(remain, fit_ids, catalog, threshold)
    val_queries = simulated_queries(remain, val_ids, catalog, threshold)
    max_k = max(RANK_CANDIDATE_KS)
    print(f"Retrieving validation candidates at K={max_k}...", flush=True)
    train_retrieved = {
        n: _retrieve(
            train_queries[n],
            item_item=item_item,
            ease=ease,
            tower=tower,
            popular=popular,
            k=max_k,
        )
        for n in COLD_START_NS
    }
    val_retrieved = {
        n: _retrieve(
            val_queries[n],
            item_item=item_item,
            ease=ease,
            tower=tower,
            popular=popular,
            k=max_k,
        )
        for n in COLD_START_NS
    }
    baseline_val: dict[int, dict[str, float]] = {}
    for n in COLD_START_NS:
        lists = _baseline_lists(val_retrieved[n], val_queries[n], k=10)
        baseline_val[n] = {
            name: _round(_point_ndcg(lists[name], val_queries[n]))
            for name in ("most_popular", "item_item_fold_in", "history_two_tower")
        }
        print(
            f"  val baselines N={n} "
            + " ".join(f"{name}={baseline_val[n][name]}" for name in baseline_val[n]),
            flush=True,
        )
    rank_cfg = config.models.ranker
    params = lgb_params(
        seed=seed,
        learning_rate=float(rank_cfg.learning_rate),
        num_leaves=int(rank_cfg.num_leaves),
        min_data_in_leaf=int(rank_cfg.min_data_in_leaf),
        feature_fraction=float(rank_cfg.feature_fraction),
        bagging_fraction=float(rank_cfg.bagging_fraction),
    )
    k_trials: list[dict[str, Any]] = []
    for k in RANK_CANDIDATE_KS:
        print(f"Training cold-start ranker at K={k}...", flush=True)
        matrices = _matrices_for_k(train_queries, train_retrieved, k, ctx, names)
        train_ids, es_ids = split_users(
            matrices.user_ids, fraction, seed
        )
        train_m = matrices.subset({int(uid) for uid in train_ids.tolist()})
        early_m = matrices.subset({int(uid) for uid in es_ids.tolist()})
        del matrices
        gc.collect()
        booster, info = fit_lambdarank(
            train_m,
            params=params,
            num_boost_round=int(rank_cfg.num_boost_round),
            valid=early_m,
            early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
        )
        del train_m, early_m
        gc.collect()
        by_n: dict[str, float] = {}
        for n in COLD_START_NS:
            ranked = _predict_lists(
                booster,
                ctx,
                names,
                val_queries[n],
                val_retrieved[n],
                k,
                num_iteration=int(info["best_iteration"]),
            )
            by_n[str(n)] = _round(_point_ndcg(ranked, val_queries[n]))
            del ranked
        mean = float(np.mean([by_n[str(n)] for n in COLD_START_NS]))
        trial = {
            "k": int(k),
            "mean_val_ndcg@10": _round(mean),
            "by_n": by_n,
            "best_iteration": int(info["best_iteration"]),
            "n_train_rows": int(info["n_rows"]),
            "n_train_groups": int(info["n_groups"]),
        }
        k_trials.append(trial)
        print(
            f"  K={k} mean val NDCG@10={trial['mean_val_ndcg@10']} "
            f"best_iteration={trial['best_iteration']}",
            flush=True,
        )
        del booster
        gc.collect()
    mean_by_k = {int(row["k"]): float(row["mean_val_ndcg@10"]) for row in k_trials}
    chosen_k = choose_candidate_k(mean_by_k)
    chosen = next(row for row in k_trials if int(row["k"]) == chosen_k)
    serving_by_n: dict[int, str] = {}
    serving_val: dict[int, dict[str, float]] = {}
    for n in COLD_START_NS:
        scores = dict(baseline_val[n])
        scores["cold_start_ranker"] = float(chosen["by_n"][str(n)])
        method = choose_serving_method(scores)
        serving_by_n[n] = method
        serving_val[n] = {name: _round(float(scores[name])) for name in SERVING_TIE_ORDER}
        print(f"  validation N={n} serve {method} ({scores[method]})", flush=True)
    del tower, item_item, ease, popular, train_retrieved, val_retrieved, ctx
    gc.collect()
    print(
        f"Validation choices frozen: K={chosen_k}, "
        f"best_iteration={chosen['best_iteration']}, "
        f"serving={ {n: serving_by_n[n] for n in COLD_START_NS} }",
        flush=True,
    )
    return RankerChoices(
        candidate_k=int(chosen_k),
        best_iteration=int(chosen["best_iteration"]),
        serving_by_n=serving_by_n,
        serving_val_ndcg=serving_val,
        k_trials=k_trials,
        n_ranker_train_users=int(len(fit_ids)),
        n_ranker_val_users=int(len(val_ids)),
        feature_names=list(names),
        hyperparams=hp,
    )


def _paired_block(
    left: np.ndarray,
    right: np.ndarray,
    *,
    baseline: str,
    comparison: str,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    paired = paired_bootstrap_delta_ci(
        left,
        right,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    mean = float(paired["mean"])
    return {
        "comparison": comparison,
        "baseline": baseline,
        "mean": _round(mean),
        "low": _round(float(paired["low"])),
        "high": _round(float(paired["high"])),
        "n_users": int(paired["n_users"]),
        "excludes_zero": bool(paired["excludes_zero"]),
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
        "seed": int(seed),
        "beats_baseline": bool(paired["excludes_zero"] and mean > 0.0),
    }


def _score_method_table(
    lists: Mapping[str, Mapping[int, Sequence[int]]],
    relevant: Mapping[int, set[int]],
    catalog: set[int],
    *,
    served_method: str,
    config: RunConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    seed = int(config.seed)
    n_bootstrap = int(config.eval.n_bootstrap)
    alpha = float(config.eval.bootstrap_alpha)
    metrics: dict[str, Any] = {}
    per_user: dict[str, np.ndarray] = {}
    for name, rows in lists.items():
        block, _ids, ndcgs = _score_ranking(
            {int(uid): [int(item) for item in rows.get(int(uid), [])] for uid in relevant},
            {int(uid): set(items) for uid, items in relevant.items()},
            catalog,
            k=10,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        metrics[name] = block
        per_user[name] = ndcgs
    metrics["served"] = dict(metrics[served_method])
    per_user["served"] = per_user[served_method]
    simple_scores = {name: float(metrics[name]["ndcg@10"]) for name in SIMPLE_BASELINES}
    best_name = _best_simple(simple_scores)
    paired = _paired_block(
        per_user[best_name],
        per_user["served"],
        baseline=best_name,
        comparison="served_minus_best_simple",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    versus_pop = _paired_block(
        per_user["most_popular"],
        per_user["served"],
        baseline="most_popular",
        comparison="served_minus_most_popular",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    ranker_paired = _paired_block(
        per_user[best_name],
        per_user["cold_start_ranker"],
        baseline=best_name,
        comparison="cold_start_ranker_minus_best_simple",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    ranker_vs_pop = _paired_block(
        per_user["most_popular"],
        per_user["cold_start_ranker"],
        baseline="most_popular",
        comparison="cold_start_ranker_minus_most_popular",
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    table = {
        "models": metrics,
        "best_simple_baseline": best_name,
        "paired_bootstrap": paired,
        "paired_vs_most_popular": versus_pop,
        "ranker_paired_vs_best_simple": ranker_paired,
        "ranker_paired_vs_most_popular": ranker_vs_pop,
        "served_method": served_method,
        "beats_most_popular": bool(versus_pop["beats_baseline"]),
    }
    return table, per_user


def _held_out_lists(
    *,
    profiles: Mapping[int, list[tuple[int, float]]],
    histories: Mapping[int, list[int]],
    frames: Mapping[int, pd.DataFrame],
    item_item: ItemItemCosineRecommender,
    ease: EASERecommender,
    tower: TwoTowerRecommender,
    popular: MostPopularRecommender,
    booster: Any,
    ctx: Any,
    names: list[str],
    k: int,
    num_iteration: int,
) -> dict[str, dict[int, list[int]]]:
    ii = item_item.topk_for_profiles(profiles, k)
    ease_rows = ease.topk_for_profiles(profiles, k)
    tt = tower.topk_for_new_histories(histories, k)
    pop = _popular_lists(popular, profiles, k)
    retrieved = Retrieved(ease=ease_rows, item_item=ii, tower=tt, popular=pop)
    queries = [
        SimQuery(
            uid=int(uid),
            profile_n=len(frames[int(uid)]),
            profile=tuple(profiles[int(uid)]),
            history=tuple(histories[int(uid)]),
            relevant=frozenset(),
        )
        for uid in frames
    ]
    ranked = _predict_lists(
        booster,
        ctx,
        names,
        queries,
        retrieved,
        k,
        num_iteration=num_iteration,
    )
    lists = _baseline_lists(retrieved, queries, k=10)
    lists["cold_start_ranker"] = {int(uid): ranked.get(int(uid), []) for uid in frames}
    return lists


def _assert_same_holdout(payload: Mapping[str, Any], held_ids: np.ndarray) -> None:
    recorded = [int(uid) for uid in payload["holdout"]["held_out_user_ids"]]
    fresh = [int(uid) for uid in np.asarray(held_ids, dtype=np.int64).tolist()]
    if recorded != fresh:
        raise AssertionError(
            "held-out ids changed. Round 2 must score the same 604 users as round 1."
        )


def _round1_record(payload: Mapping[str, Any]) -> dict[str, Any]:
    nested = payload.get("round1")
    if isinstance(nested, dict) and "by_n" in nested:
        return dict(nested)
    record = {
        "note": (
            "Round 1 pipeline. Copied from the first cold-start measurement and "
            "not recomputed. It loses to a simple baseline at every N."
        )
    }
    for key in ("representation", "candidates", "ranker", "by_n", "serving", "evaluation"):
        if key in payload:
            record[key] = payload[key]
    return record


def _attach_pipeline_v1(
    by_n: list[dict[str, Any]],
    round1_by_n: Sequence[Mapping[str, Any]],
) -> None:
    old = {int(block["n"]): block for block in round1_by_n}
    for block in by_n:
        source = old[int(block["n"])]["models"]["pipeline"]
        copied = dict(source)
        copied["copied_from"] = "round1"
        block["models"]["pipeline_v1"] = copied


def finish_round2(
    *,
    config: RunConfig,
    results_dir: Path,
    t0: float,
    winner_trial: Mapping[str, Any],
    rep_trials: list[dict[str, Any]],
    dropout_at_edge: bool,
    selected_at_edge: bool,
    best_dropout: Mapping[str, Any],
    split: SplitResult,
    full_train: pd.DataFrame,
    remain: pd.DataFrame,
    held_ratings: pd.DataFrame,
    held_ids: np.ndarray,
    n_eligible: int,
    movies: pd.DataFrame,
    tower_hp: Mapping[str, Any],
    item_hp: Mapping[str, Any],
    ease_hp: Mapping[str, Any],
) -> Path:
    """Select on validation, then score the held-out users once."""
    assert_held_out_absent(held_ids, full_train, split.train)
    if int(n_eligible) < 2:
        raise ValueError("cold-start ranker selection needs the eligible user pool")
    path = results_dir / "cold-start" / f"{config.dataset}.json"
    existing = json.loads(path.read_text(encoding="utf-8"))
    _assert_same_holdout(existing, held_ids)
    round1 = _round1_record(existing)
    for block in round1["by_n"]:
        if "pipeline" not in block.get("models", {}):
            raise RuntimeError("round 1 is missing pipeline metrics")

    choices = select_cold_start_ranker(
        config=config,
        remain=remain,
        full_train=full_train,
        held_ids=held_ids,
        movies=movies,
        tower_hp=tower_hp,
        item_hp=item_hp,
        ease_hp=ease_hp,
        winner_trial=winner_trial,
    )
    print("Choices frozen. Scoring held-out users once.", flush=True)
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)
    print("Refitting retrievers and the short-profile ranker on the 90%...", flush=True)
    tower = _fit_tower(
        full_train,
        config=config,
        movies=movies,
        hyperparams=choices.hyperparams,
        epochs=int(winner_trial["best_epoch"]),
        threshold=threshold,
    )
    item_item = _fit_item_item(full_train, dict(item_hp))
    ease = _fit_ease(full_train, dict(ease_hp))
    popular = MostPopularRecommender().fit(full_train)
    catalog = {int(item) for item in full_train["item_id"].unique()}
    names = list(choices.feature_names)
    ctx = _context_for(
        full_train,
        movies,
        None,
        mode="off",
        group_frame=None,
        threshold=threshold,
    )
    ctx.names = names
    pool = np.unique(full_train["user_id"].to_numpy(dtype=np.int64))
    refit_queries = simulated_queries(remain, pool, catalog, threshold)
    max_k = max(RANK_CANDIDATE_KS)
    refit_retrieved = {
        n: _retrieve(
            refit_queries[n],
            item_item=item_item,
            ease=ease,
            tower=tower,
            popular=popular,
            k=max_k,
        )
        for n in COLD_START_NS
    }
    print(
        f"Refitting LambdaRank for {choices.best_iteration} trees, no second early stop...",
        flush=True,
    )
    matrices = _matrices_for_k(refit_queries, refit_retrieved, choices.candidate_k, ctx, names)
    rank_cfg = config.models.ranker
    params = lgb_params(
        seed=seed,
        learning_rate=float(rank_cfg.learning_rate),
        num_leaves=int(rank_cfg.num_leaves),
        min_data_in_leaf=int(rank_cfg.min_data_in_leaf),
        feature_fraction=float(rank_cfg.feature_fraction),
        bagging_fraction=float(rank_cfg.bagging_fraction),
    )
    booster, refit_info = fit_lambdarank(
        matrices,
        params=params,
        num_boost_round=int(choices.best_iteration),
        valid=None,
        early_stopping_rounds=None,
    )
    del matrices, refit_queries, refit_retrieved
    gc.collect()
    if int(refit_info["num_trees"]) != int(choices.best_iteration):
        raise AssertionError("refit did not keep the validation iteration count")

    by_n: list[dict[str, Any]] = []
    sensitivity_by_n: list[dict[str, Any]] = []
    test_fraction = float(config.split.test_fraction)
    for n in COLD_START_NS:
        _profiles, histories, frames, relevant, counts = _held_out_targets(
            held_ratings, n, catalog, threshold
        )
        _profiles_s, _hist_s, _frames_s, relevant_s, counts_s = _held_out_targets(
            held_ratings,
            n,
            catalog,
            threshold,
            target="harness_tail",
            test_fraction=test_fraction,
        )
        profiles = {
            int(uid): list(
                zip(
                    frames[int(uid)]["item_id"].astype(int).tolist(),
                    frames[int(uid)]["rating"].astype(float).tolist(),
                    strict=True,
                )
            )
            for uid in frames
        }
        lists = _held_out_lists(
            profiles=profiles,
            histories=histories,
            frames=frames,
            item_item=item_item,
            ease=ease,
            tower=tower,
            popular=popular,
            booster=booster,
            ctx=ctx,
            names=names,
            k=choices.candidate_k,
            num_iteration=int(choices.best_iteration),
        )
        method = choices.serving_by_n[n]
        table, _per_user = _score_method_table(
            lists, relevant, catalog, served_method=method, config=config
        )
        table["n"] = int(n)
        table["n_eval_users"] = int(counts["n_eval_users"])
        table["counts"] = counts
        table["target"] = "all_later_ratings"
        by_n.append(table)
        sens, _sens_users = _score_method_table(
            lists, relevant_s, catalog, served_method=method, config=config
        )
        sens["n"] = int(n)
        sens["n_eval_users"] = int(counts_s["n_eval_users"])
        sens["counts"] = counts_s
        sens["target"] = "harness_tail"
        sensitivity_by_n.append(sens)
        print(
            f"  held-out N={n} served={method} "
            f"NDCG@10={table['models']['served']['ndcg@10']} "
            f"ranker={table['models']['cold_start_ranker']['ndcg@10']} "
            f"most_popular={table['models']['most_popular']['ndcg@10']}",
            flush=True,
        )

    _attach_pipeline_v1(by_n, round1["by_n"])

    from movielens_recommender.serving.build import git_sha
    from movielens_recommender.serving.cold import measure_new_user_latency, save_cold_start_bundle

    artifact_dir = Path("artifacts") / config.dataset / "cold_start"
    save_cold_start_bundle(
        artifact_dir,
        movies=movies,
        item_item=item_item,
        ease=ease,
        two_tower=tower,
        booster=booster,
        context=ctx,
        popular_item_ids=np.asarray(popular._ranked_items, dtype=np.int64),
        manifest={
            "dataset": config.dataset,
            "dataset_sha256": DATASET_SHA256[config.dataset],
            "git_sha": git_sha(),
            "candidate_source": "cold_start_union",
            "candidate_k": int(choices.candidate_k),
            "ranker_mode": "cold_start_ranker",
            "demographics_missing": False,
            "history_only": bool(winner_trial["history_only"]),
            "user_id_dropout": float(winner_trial["user_id_dropout"]),
            "best_iteration": int(choices.best_iteration),
            "feature_names": list(names),
            "relevance_threshold": threshold,
            "serving_by_n": {str(n): choices.serving_by_n[n] for n in COLD_START_NS},
            "profile_buckets": list(PROFILE_BUCKETS),
        },
    )
    latency = measure_new_user_latency(artifact_dir, full_train)
    print(f"Warmed new-user latency: {latency['latency_sec']:.4f}s", flush=True)

    from movielens_recommender.cli import host_info, library_versions

    payload: dict[str, Any] = {
        "experiment": "s5c_new_user_cold_start",
        "round": 2,
        "dataset": config.dataset,
        "dataset_version": DATASET_VERSION_LABELS[config.dataset],
        "dataset_sha256": DATASET_SHA256[config.dataset],
        "seed": seed,
        "holdout": existing["holdout"],
        "round1": round1,
        "round2": {
            "dropout_grid": list(DROPOUT_GRID),
            "representation_winner": winner_trial["name"],
            "representation_val_ndcg@10": winner_trial["val_ndcg@10"],
            "representation_best_epoch": int(winner_trial["best_epoch"]),
            "history_only": bool(winner_trial["history_only"]),
            "user_id_dropout": float(winner_trial["user_id_dropout"]),
            "dropout_best_p": float(best_dropout["user_id_dropout"]),
            "dropout_best_at_edge": bool(dropout_at_edge),
            "selected_at_grid_edge": bool(selected_at_edge),
            "ranker_validation": {
                "fraction": float(config.models.ranker.early_stop_fraction),
                "seed": seed,
                "n_ranker_train_users": choices.n_ranker_train_users,
                "n_ranker_val_users": choices.n_ranker_val_users,
                "source": (
                    "Carved with select_held_out_user_ids from the 90% full-train "
                    "users. Disjoint from the held-out 604. Selection retrievers "
                    "are fit on the ranker-train users only."
                ),
            },
            "candidate_k_grid": list(RANK_CANDIDATE_KS),
            "candidate_k_trials": choices.k_trials,
            "candidate_k": choices.candidate_k,
            "candidate_k_rule": (
                "Highest mean validation NDCG@10 of the cold-start ranker across "
                "N in {1, 3, 5, 10}. Ties keep the smaller K."
            ),
            "candidate_pool": [
                "ease_fold_in",
                "most_popular",
                "history_two_tower",
            ],
            "serving_by_n": {str(n): choices.serving_by_n[n] for n in COLD_START_NS},
            "serving_val_ndcg@10": {
                str(n): choices.serving_val_ndcg[n] for n in COLD_START_NS
            },
            "serving_rule": (
                "For each N, the highest validation NDCG@10 among the cold-start "
                "ranker, most-popular, item-item fold-in, and the history two-tower. "
                "Ties prefer most-popular, then item-item fold-in, then the history "
                "two-tower, then the ranker."
            ),
            "profile_bucket_rule": (
                "A live profile of length n uses the largest grid N that is <= n."
            ),
            "best_iteration": choices.best_iteration,
            "feature_names": list(names),
            "demographics": "off",
            "protocol": (
                "Simulated profiles truncate each ranker-train user to the first "
                "N ratings. History features are recomputed from that prefix. "
                "Labels are later ratings with relevance at least 4. The ranker "
                "is fit once per K on all N stacked as separate groups. The "
                "held-out 604 are scored after these choices, on all later ratings."
            ),
            "leakage_note": (
                "Ranker-train users' later full-train ratings can sit in the item "
                "similarity. That can bias the ranker fit. It does not enter the "
                "selection metric: ranker-validation users are absent from those "
                "retrievers. Per-user test tails are labels only."
            ),
            "popularity_vs_known_users": POPULARITY_VS_KNOWN_USERS,
        },
        "representation": {
            "hyperparams_source": "results/tuning/two_tower_ml-1m.json",
            "fixed_hyperparams": {key: tower_hp[key] for key in sorted(tower_hp)},
            "note": (
                "Round 2 grid, including p=0.0 and p=0.1. Round 1's representation "
                "search is under round1 and was not recomputed. p=0.0 uses the same "
                "loop with the id mask never applied. Scoring still zeros the id."
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
            "k": int(choices.candidate_k),
            "k_grid": list(RANK_CANDIDATE_KS),
            "pool": ["ease_fold_in", "most_popular", "history_two_tower"],
            "winner": "union",
            "note": (
                "Round 2 does not pick one source. The ranker sees the union. "
                "Item-item fold-in is a serving option and a feature when the "
                "item is in its top K. It does not add candidates. Round 1's "
                "candidate choice is under round1."
            ),
        },
        "ranker": {
            "winner": "cold_start_ranker",
            "demographics": "off",
            "best_iteration": int(choices.best_iteration),
            "feature_names": list(names),
            "num_trees": int(refit_info["num_trees"]),
            "note": (
                "Trained on simulated short profiles from the 90%. Round 1's "
                "ranker is under round1 and was not rerun."
            ),
        },
        "evaluation": {
            "primary": (
                "All later ratings after the first N, relevance at least 4, "
                "item in the refit catalog, profile items excluded."
            ),
            "sensitivity_added_after_first_results": True,
            "known_users": "copied from results/ml-1m.json",
        },
        "known_users": _known_user_reference(results_dir),
        "s4b_simulated_cold_start_n5": _s4b_reference(results_dir),
        "difference_from_s4b": DIFFERENCE_FROM_S4B,
        "by_n": by_n,
        "sensitivity_last_20_percent": {
            "added_after_first_results": True,
            "serving_rule_rechosen": False,
            "test_fraction": test_fraction,
            "note": (
                "Added after the first held-out results. Targets are each "
                "held-out user's harness tail: the last max(1, int(n_ratings * "
                "test_fraction)) ratings, leaving at least one head row, and only "
                "where that tail is after the first N. The serving rule was not "
                "re-chosen on this view. Pipeline v1 is not scored here."
            ),
            "by_n": sensitivity_by_n,
        },
        "serving": latency,
        "popularity_vs_known_users": POPULARITY_VS_KNOWN_USERS,
        "runtime_sec": _round(time.perf_counter() - t0),
        "library_versions": library_versions(),
        "host": host_info(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    return path
