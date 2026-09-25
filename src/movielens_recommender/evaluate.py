"""Evaluation harness with cold-start filtering, diagnostics, and bootstrap CIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    mean_popularity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from movielens_recommender.split import SplitResult, apply_cold_start_policy

RecommenderFn = Callable[[int, int], Sequence[int]]
"""(user_id, n) -> ranked item ids (may include seen items; harness filters)."""


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
) -> dict[str, Any]:
    """Evaluate with cold-start policy, diagnostics, and bootstrap CIs.

    Ranking metrics are averaged over users with ≥1 warm relevant test item.
    Catalog coverage and mean popularity are diagnostics (not stage gates).
    """
    if split is None:
        # Build a minimal SplitResult so cold-start policy can run.
        from movielens_recommender.split import SplitConfig

        split = SplitResult(
            train=train,
            test=test,
            config=SplitConfig(relevance_threshold=relevance_threshold),
            n_users_kept=int(train["user_id"].nunique()),
            n_users_dropped=0,
        )

    seen, relevant, cold_stats = apply_cold_start_policy(
        split, relevance_threshold=relevance_threshold
    )
    if not relevant:
        raise ValueError("No test users with warm relevant items to evaluate.")

    max_k = max(ks)
    catalog = set(train["item_id"].astype(int))
    item_pop = train.groupby("item_id").size().astype(float).to_dict()
    item_pop = {int(k): float(v) for k, v in item_pop.items()}

    per_user: dict[str, list[float]] = {
        f"{m}@{k}": [] for k in ks for m in ("precision", "recall", "ndcg")
    }
    recommendations: dict[int, list[int]] = {}

    for user_id, rel in relevant.items():
        user_seen = seen.get(user_id, set())
        recs = recommend_filtered(recommend_fn, user_id, max_k, user_seen)
        recommendations[user_id] = recs
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

    # Diagnostics + bootstrap over users by resampling recommendation lists.
    user_ids = list(recommendations.keys())
    rng = np.random.default_rng(seed)
    for k in ks:
        cov = catalog_coverage(recommendations, catalog, k)
        pop = mean_popularity(recommendations, item_pop, k)
        metrics[f"coverage@{k}"] = cov
        metrics[f"mean_popularity@{k}"] = pop

        cov_samples: list[float] = []
        pop_samples: list[float] = []
        n = len(user_ids)
        for _ in range(max(n_bootstrap, 1)):
            sample_ids = rng.choice(user_ids, size=n, replace=True)
            # Coverage uses the unique user set in the bootstrap sample (union of lists).
            uniq = {int(uid): recommendations[int(uid)] for uid in set(int(x) for x in sample_ids)}
            cov_samples.append(catalog_coverage(uniq, catalog, k))
            # Popularity averages over the sampled users (with replacement).
            pop_vals = []
            for uid in sample_ids:
                top = recommendations[int(uid)][:k]
                if not top:
                    pop_vals.append(0.0)
                else:
                    pop_vals.append(float(np.mean([item_pop.get(int(i), 0.0) for i in top])))
            pop_samples.append(float(np.mean(pop_vals)))
        cis[f"coverage@{k}"] = {
            "mean": cov,
            "low": float(np.quantile(cov_samples, bootstrap_alpha / 2)),
            "high": float(np.quantile(cov_samples, 1 - bootstrap_alpha / 2)),
        }
        cis[f"mean_popularity@{k}"] = {
            "mean": pop,
            "low": float(np.quantile(pop_samples, bootstrap_alpha / 2)),
            "high": float(np.quantile(pop_samples, 1 - bootstrap_alpha / 2)),
        }

    metrics["confidence_intervals"] = cis
    metrics["cold_start"] = cold_stats.to_dict()
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
        elif key == "cold_start":
            out[key] = dict(value)
        elif key == "n_eval_users":
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = _round_num(value)
        else:
            out[key] = value
    return out
