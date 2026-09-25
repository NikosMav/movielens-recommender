"""Evaluation harness with cold-start filtering, diagnostics, and bootstrap CIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from movielens_recommender.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    mean_popularity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from movielens_recommender.segments import evaluate_segments
from movielens_recommender.split import SplitConfig, SplitResult, apply_cold_start_policy

RecommenderFn = Callable[[int, int], Sequence[int]]
"""(user_id, n) -> ranked item ids (may include seen items; harness filters)."""

# Catalog coverage is a set-union over users. Resampling users with replacement
# shrinks the unique-user set and systematically underestimates coverage, so the
# point estimate often falls above a naive user-bootstrap CI. We therefore report
# coverage as a point estimate only (see ADR-0003).
COVERAGE_CI_POLICY = "point_estimate_only"


def recommend_filtered(
    recommend_fn: RecommenderFn,
    user_id: int,
    k: int,
    seen: set[int],
    *,
    candidate_pool: int | None = None,
) -> list[int]:
    """Get recommendations excluding already-seen train items."""
    pool = candidate_pool if candidate_pool is not None else k + len(seen)
    pool = max(pool, k)
    raw = recommend_fn(user_id, pool)
    out: list[int] = []
    for item in raw:
        if item in seen:
            continue
        out.append(int(item))
        if len(out) >= k:
            break
    return out


def _per_user_popularity(
    recommendations: Mapping[int, Sequence[int]],
    item_popularity: Mapping[int, float],
    k: int,
) -> list[float]:
    """Per-user mean popularity of top-k recommendations (unknown items → 0)."""
    values: list[float] = []
    for recs in recommendations.values():
        top = list(recs[:k])
        if not top:
            values.append(0.0)
            continue
        values.append(sum(float(item_popularity.get(int(i), 0.0)) for i in top) / len(top))
    return values


def _prepare_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    relevance_threshold: float,
    split: SplitResult | None,
) -> tuple[SplitResult, dict[int, set[int]], dict[int, set[int]]]:
    if split is None:
        split = SplitResult(
            train=train,
            test=test,
            config=SplitConfig(relevance_threshold=relevance_threshold),
            n_users_kept=int(train["user_id"].nunique()),
            n_users_dropped=0,
        )
    # Ensure cold-start uses the train matrix the model was fit on.
    split.train = train
    split.test = test
    seen, relevant, _stats = apply_cold_start_policy(
        split, relevance_threshold=relevance_threshold
    )
    if not relevant:
        raise ValueError("No test users with warm relevant items to evaluate.")
    return split, seen, relevant


def collect_recommendations(
    recommend_fn: RecommenderFn,
    seen: Mapping[int, set[int]],
    relevant: Mapping[int, set[int]],
    *,
    max_k: int,
) -> dict[int, list[int]]:
    """Generate filtered top-``max_k`` lists for every eval user."""
    recommendations: dict[int, list[int]] = {}
    for user_id in relevant:
        user_seen = seen.get(user_id, set())
        recommendations[user_id] = recommend_filtered(
            recommend_fn, user_id, max_k, user_seen
        )
    return recommendations


def ndcg_point_estimate(
    recommend_fn: RecommenderFn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    relevance_threshold: float = 4.0,
    k: int = 10,
    split: SplitResult | None = None,
) -> float:
    """Mean NDCG@k without bootstrap — for validation-grid selection only."""
    split, seen, relevant = _prepare_eval(
        train, test, relevance_threshold=relevance_threshold, split=split
    )
    recommendations = collect_recommendations(recommend_fn, seen, relevant, max_k=k)
    values = [
        ndcg_at_k(recommendations[uid], rel, k) for uid, rel in relevant.items()
    ]
    return float(sum(values) / len(values))


def evaluate_recommender(
    recommend_fn: RecommenderFn,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    relevance_threshold: float = 4.0,
    ks: Sequence[int] = (10, 20),
    n_bootstrap: int = 1000,
    bootstrap_alpha: float = 0.05,
    seed: int = 42,
    split: SplitResult | None = None,
    include_segments: bool = False,
) -> dict[str, Any]:
    """Evaluate with cold-start policy, diagnostics, and bootstrap CIs.

    Ranking metrics and mean popularity are user-level means → percentile
    bootstrap CIs over users. Catalog coverage is a catalog-level set statistic
    → point estimate only (no user-bootstrap CI).

    When ``include_segments`` is True, also compute activity-tercile and
    head/tail NDCG@10 breakdowns (see :mod:`movielens_recommender.segments`).
    """
    split, seen, relevant = _prepare_eval(
        train, test, relevance_threshold=relevance_threshold, split=split
    )
    cold_stats = split.cold_start
    assert cold_stats is not None

    max_k = max(ks)
    catalog = set(train["item_id"].astype(int))
    item_pop = {int(i): float(c) for i, c in train.groupby("item_id").size().items()}

    recommendations = collect_recommendations(
        recommend_fn, seen, relevant, max_k=max_k
    )

    per_user: dict[str, list[float]] = {
        f"{m}@{k}": [] for k in ks for m in ("precision", "recall", "ndcg")
    }
    for user_id, rel in relevant.items():
        recs = recommendations[user_id]
        for k in ks:
            per_user[f"precision@{k}"].append(precision_at_k(recs, rel, k))
            per_user[f"recall@{k}"].append(recall_at_k(recs, rel, k))
            per_user[f"ndcg@{k}"].append(ndcg_at_k(recs, rel, k))

    metrics: dict[str, Any] = {"n_eval_users": float(len(relevant))}
    cis: dict[str, dict[str, float]] = {}

    for name, values in per_user.items():
        mean, low, high = bootstrap_mean_ci(
            values, n_bootstrap=n_bootstrap, alpha=bootstrap_alpha, seed=seed
        )
        metrics[name] = mean
        cis[name] = {"mean": mean, "low": low, "high": high}

    for k in ks:
        cov = catalog_coverage(recommendations, catalog, k)
        metrics[f"coverage@{k}"] = cov
        # No CI entry for coverage — see COVERAGE_CI_POLICY / ADR-0003.

        pop_vals = _per_user_popularity(recommendations, item_pop, k)
        pop_mean, pop_low, pop_high = bootstrap_mean_ci(
            pop_vals, n_bootstrap=n_bootstrap, alpha=bootstrap_alpha, seed=seed
        )
        # mean_popularity() must match the mean of pop_vals.
        assert abs(pop_mean - mean_popularity(recommendations, item_pop, k)) < 1e-12
        metrics[f"mean_popularity@{k}"] = pop_mean
        cis[f"mean_popularity@{k}"] = {"mean": pop_mean, "low": pop_low, "high": pop_high}

    # Sanity: every reported CI must contain its point estimate.
    for name, bounds in cis.items():
        if not (bounds["low"] <= bounds["mean"] <= bounds["high"]):
            raise AssertionError(
                f"CI for {name} does not contain the point estimate: {bounds}"
            )

    metrics["confidence_intervals"] = cis
    metrics["cold_start"] = cold_stats.to_dict()
    metrics["uncertainty_policy"] = {
        "bootstrap_over_users": [
            "precision@k",
            "recall@k",
            "ndcg@k",
            "mean_popularity@k",
        ],
        "point_estimate_only": ["coverage@k"],
        "coverage_ci_policy": COVERAGE_CI_POLICY,
        "coverage_ci_rationale": (
            "Coverage is |union of top-k lists| / |catalog|. "
            "User bootstrap with replacement reduces unique users and biases "
            "the union downward, so CIs would not contain the full-sample estimate."
        ),
    }

    if include_segments:
        metrics["segments"] = evaluate_segments(
            recommendations,
            relevant,
            train,
            k=10,
            n_bootstrap=n_bootstrap,
            bootstrap_alpha=bootstrap_alpha,
            seed=seed,
        )
    return metrics


def format_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Round floating metrics / CI bounds for stable JSON output."""

    def _round_num(value: float) -> float:
        return round(float(value), 6)

    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "confidence_intervals":
            cis_out: dict[str, dict[str, float]] = {}
            for metric_name, bounds in value.items():
                cis_out[metric_name] = {b: _round_num(v) for b, v in bounds.items()}
            out[key] = cis_out
        elif key in {"cold_start", "uncertainty_policy", "segments"}:
            out[key] = value
        elif key == "n_eval_users":
            out[key] = float(value)
        elif isinstance(value, int | float):
            out[key] = _round_num(value)
        else:
            out[key] = value
    return out
