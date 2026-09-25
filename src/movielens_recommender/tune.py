"""Validation-grid hyperparameter search (never tunes on test)."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from movielens_recommender.baselines import ALSRecommender, ItemItemCosineRecommender
from movielens_recommender.evaluate import ndcg_point_estimate
from movielens_recommender.split import SplitResult

# Small grids sized for a reasonable ml-1m run. Primary selection metric: NDCG@10.
ALS_GRID: list[dict[str, Any]] = [
    {"factors": f, "regularization": r, "alpha": a, "iterations": 15}
    for f, r, a in itertools.product([32, 64, 128], [0.01, 0.1], [20.0, 40.0])
]

ITEM_KNN_GRID: list[dict[str, Any]] = [
    {"k_neighbors": k, "shrinkage": s, "min_common": 1}
    for k, s in itertools.product([40, 100, 200], [0.0, 100.0])
]


@dataclass(frozen=True)
class TuningResult:
    """Outcome of a validation grid search for one model family."""

    model: str
    primary_metric: str
    grid: list[dict[str, Any]]
    trials: list[dict[str, Any]]
    best_hyperparams: dict[str, Any]
    best_val_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "primary_metric": self.primary_metric,
            "grid": self.grid,
            "trials": self.trials,
            "best_hyperparams": self.best_hyperparams,
            "best_val_score": round(float(self.best_val_score), 6),
        }


def _iter_with_progress(
    items: Sequence[dict[str, Any]], label: str
) -> Iterator[tuple[int, dict[str, Any]]]:
    total = len(items)
    for i, cfg in enumerate(items, start=1):
        print(f"  [{label}] trial {i}/{total}: {cfg}", flush=True)
        yield i, cfg


def tune_als(
    split: SplitResult,
    *,
    relevance_threshold: float = 4.0,
    seed: int = 42,
    grid: Sequence[Mapping[str, Any]] | None = None,
) -> TuningResult:
    """Grid-search ALS on validation NDCG@10; never touches test."""
    if split.val is None or split.val.empty:
        raise ValueError("tune_als requires a non-empty validation split")
    configs = [dict(c) for c in (grid if grid is not None else ALS_GRID)]
    trials: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_hp: dict[str, Any] = dict(configs[0])

    # Val eval uses fit-train as the seen/catalog matrix.
    val_split = SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )

    for _, hp in _iter_with_progress(configs, "als"):
        model = ALSRecommender(
            factors=int(hp["factors"]),
            regularization=float(hp["regularization"]),
            iterations=int(hp["iterations"]),
            alpha=float(hp["alpha"]),
            confidence_threshold=relevance_threshold,
            random_state=seed,
        ).fit(split.train)
        score = ndcg_point_estimate(
            model.recommend,
            split.train,
            split.val,
            relevance_threshold=relevance_threshold,
            k=10,
            split=val_split,
        )
        trials.append({"hyperparams": hp, "val_ndcg@10": round(score, 6)})
        if score > best_score:
            best_score = score
            best_hp = dict(hp)

    return TuningResult(
        model="als",
        primary_metric="ndcg@10",
        grid=configs,
        trials=trials,
        best_hyperparams=best_hp,
        best_val_score=best_score,
    )


def tune_item_knn(
    split: SplitResult,
    *,
    relevance_threshold: float = 4.0,
    grid: Sequence[Mapping[str, Any]] | None = None,
) -> TuningResult:
    """Grid-search item-item CF on validation NDCG@10; never touches test."""
    if split.val is None or split.val.empty:
        raise ValueError("tune_item_knn requires a non-empty validation split")
    configs = [dict(c) for c in (grid if grid is not None else ITEM_KNN_GRID)]
    trials: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_hp: dict[str, Any] = dict(configs[0])

    val_split = SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )

    for _, hp in _iter_with_progress(configs, "item_item_cosine"):
        model = ItemItemCosineRecommender(
            min_common=int(hp["min_common"]),
            k_neighbors=int(hp["k_neighbors"]),
            shrinkage=float(hp["shrinkage"]),
        ).fit(split.train)
        score = ndcg_point_estimate(
            model.recommend,
            split.train,
            split.val,
            relevance_threshold=relevance_threshold,
            k=10,
            split=val_split,
        )
        trials.append({"hyperparams": hp, "val_ndcg@10": round(score, 6)})
        if score > best_score:
            best_score = score
            best_hp = dict(hp)

    return TuningResult(
        model="item_item_cosine",
        primary_metric="ndcg@10",
        grid=configs,
        trials=trials,
        best_hyperparams=best_hp,
        best_val_score=best_score,
    )


def build_model(
    name: str,
    train: pd.DataFrame,
    *,
    hyperparams: Mapping[str, Any],
    relevance_threshold: float,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Fit a named baseline and return (model, recorded hyperparams)."""
    if name == "most_popular":
        from movielens_recommender.baselines import MostPopularRecommender

        model = MostPopularRecommender().fit(train)
        return model, {}
    if name in {"item_item_cosine", "item_item_cosine_tuned"}:
        model = ItemItemCosineRecommender(
            min_common=int(hyperparams.get("min_common", 1)),
            k_neighbors=int(hyperparams.get("k_neighbors", 0)),
            shrinkage=float(hyperparams.get("shrinkage", 0.0)),
        ).fit(train)
        return model, model.hyperparams()
    if name in {"als", "als_tuned"}:
        model = ALSRecommender(
            factors=int(hyperparams.get("factors", 64)),
            regularization=float(hyperparams.get("regularization", 0.01)),
            iterations=int(hyperparams.get("iterations", 15)),
            alpha=float(hyperparams.get("alpha", 40.0)),
            confidence_threshold=relevance_threshold,
            random_state=seed,
        ).fit(train)
        return model, model.hyperparams()
    raise ValueError(f"Unknown model: {name}")


ModelFactory = Callable[..., Any]
