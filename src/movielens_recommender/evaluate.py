"""Evaluation harness: recommend, filter seen items, average metrics over users."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from movielens_recommender.metrics import ndcg_at_k, precision_at_k, recall_at_k

RecommenderFn = Callable[[int, int], Sequence[int]]
"""(user_id, n) -> ranked item ids (may include seen items; harness filters)."""


def build_user_maps(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    relevance_threshold: float,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Return (seen_train_items, relevant_test_items) keyed by user_id."""
    seen: dict[int, set[int]] = {
        int(uid): set(g["item_id"].astype(int)) for uid, g in train.groupby("user_id")
    }
    relevant: dict[int, set[int]] = {}
    for uid, g in test.groupby("user_id"):
        rel = set(g.loc[g["rating"] >= relevance_threshold, "item_id"].astype(int))
        relevant[int(uid)] = rel
    return seen, relevant


def recommend_filtered(
    recommend_fn: RecommenderFn,
    user_id: int,
    k: int,
    seen: set[int],
    *,
    candidate_pool: int | None = None,
) -> list[int]:
    """Get recommendations excluding already-seen train items.

    Requests extra candidates so that after filtering we still have ``k`` items
    when possible. ``candidate_pool`` defaults to ``k + len(seen)``.
    """
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
) -> dict[str, float]:
    """Average precision/recall/NDCG@k over test users with ≥1 relevant item.

    Users with no relevant test items (rating >= threshold) are skipped for
    averaging so recall/NDCG are well-defined. Metrics are micro-averaged over
    the remaining users (equal weight per user).
    """
    seen, relevant = build_user_maps(train, test, relevance_threshold=relevance_threshold)
    max_k = max(ks)
    metric_sums: dict[str, float] = {}
    n_eval = 0

    for user_id, rel in relevant.items():
        if not rel:
            continue
        user_seen = seen.get(user_id, set())
        recs = recommend_filtered(recommend_fn, user_id, max_k, user_seen)
        n_eval += 1
        for k in ks:
            metric_sums[f"precision@{k}"] = metric_sums.get(f"precision@{k}", 0.0) + precision_at_k(
                recs, rel, k
            )
            metric_sums[f"recall@{k}"] = metric_sums.get(f"recall@{k}", 0.0) + recall_at_k(
                recs, rel, k
            )
            metric_sums[f"ndcg@{k}"] = metric_sums.get(f"ndcg@{k}", 0.0) + ndcg_at_k(recs, rel, k)

    if n_eval == 0:
        raise ValueError("No test users with relevant items to evaluate.")

    return {name: value / n_eval for name, value in metric_sums.items()} | {
        "n_eval_users": float(n_eval)
    }


def format_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Round floating metrics for stable JSON output."""
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if key == "n_eval_users":
            out[key] = float(value)
        else:
            out[key] = round(float(value), 6)
    return out
