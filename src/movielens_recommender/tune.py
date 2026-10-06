"""Validation-grid hyperparameter search (never tunes on test)."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from movielens_recommender.baselines import (
    ALSRecommender,
    EASERecommender,
    ItemItemCosineRecommender,
    RP3betaRecommender,
)
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

# First search was {10, 50, 100, 500, 1000}. ml-1m selected the top edge, so the
# grid was extended after that test run (ADR-0008). The new ml-1m winner, 5000,
# is interior.
EASE_L2_VALUES: tuple[float, ...] = (10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0)
EASE_GRID: list[dict[str, Any]] = [{"l2": lam} for lam in EASE_L2_VALUES]

# First search was alpha {0.6, 1.0} × beta {0.0, 0.3, 0.6} × top_k {50, 200}.
# Those winners sat on non-natural edges, so each axis was extended after the
# first test run. top_k=1000 still won that extended grid on both datasets, so
# one further step added 2000. That value is still short of the fit-train
# catalog; the search stops there (ADR-0008). alpha=0 was the same one step on
# ml-latest-small only. The winner moved to 0.2, and that trial stays in
# results/tuning/ml-latest-small.json rather than this shared product.
RP3BETA_ALPHAS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
RP3BETA_BETAS: tuple[float, ...] = (0.0, 0.3, 0.6, 0.8, 1.0)
RP3BETA_TOPKS: tuple[int, ...] = (50, 200, 500, 1000, 2000)
RP3BETA_GRID: list[dict[str, Any]] = [
    {"alpha": alpha, "beta": beta, "top_k": top_k}
    for alpha, beta, top_k in itertools.product(RP3BETA_ALPHAS, RP3BETA_BETAS, RP3BETA_TOPKS)
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


def _val_split(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def tune_ease(
    split: SplitResult,
    *,
    relevance_threshold: float = 4.0,
    grid: Sequence[Mapping[str, Any]] | None = None,
) -> TuningResult:
    """Grid-search EASE λ on validation NDCG@10; never touches test."""
    if split.val is None or split.val.empty:
        raise ValueError("tune_ease requires a non-empty validation split")
    configs = [dict(c) for c in (grid if grid is not None else EASE_GRID)]
    trials: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_hp: dict[str, Any] = dict(configs[0])
    val_split = _val_split(split)

    for _, hp in _iter_with_progress(configs, "ease"):
        model = EASERecommender(l2=float(hp["l2"])).fit(split.train)
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
        model="ease",
        primary_metric="ndcg@10",
        grid=configs,
        trials=trials,
        best_hyperparams=best_hp,
        best_val_score=best_score,
    )


def tune_rp3beta(
    split: SplitResult,
    *,
    relevance_threshold: float = 4.0,
    grid: Sequence[Mapping[str, Any]] | None = None,
) -> TuningResult:
    """Grid-search RP3beta (alpha, beta, top_k) on validation NDCG@10."""
    if split.val is None or split.val.empty:
        raise ValueError("tune_rp3beta requires a non-empty validation split")
    configs = [dict(c) for c in (grid if grid is not None else RP3BETA_GRID)]
    trials: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_hp: dict[str, Any] = dict(configs[0])
    val_split = _val_split(split)

    for _, hp in _iter_with_progress(configs, "rp3beta"):
        model = RP3betaRecommender(
            alpha=float(hp["alpha"]),
            beta=float(hp["beta"]),
            top_k=int(hp["top_k"]),
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
        model="rp3beta",
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
    if name == "ease":
        model = EASERecommender(l2=float(hyperparams["l2"])).fit(train)
        return model, model.hyperparams()
    if name == "rp3beta":
        model = RP3betaRecommender(
            alpha=float(hyperparams["alpha"]),
            beta=float(hyperparams["beta"]),
            top_k=int(hyperparams["top_k"]),
        ).fit(train)
        return model, model.hyperparams()
    raise ValueError(f"Unknown model: {name}")


ModelFactory = Callable[..., Any]
