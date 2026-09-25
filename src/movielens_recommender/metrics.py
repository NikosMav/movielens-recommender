"""Ranking and diagnostic metrics for recommendation evaluation."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the top-k recommendations that are relevant.

    ``precision@k = |recommended[:k] ∩ relevant| / k``
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in rel)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of relevant items recovered in the top-k.

    ``recall@k = |recommended[:k] ∩ relevant| / |relevant|``
    Returns 0.0 when there are no relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in rel)
    return hits / len(rel)


def dcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Discounted cumulative gain at k with binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    score = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in rel:
            score += 1.0 / math.log2(rank + 1)
    return score


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Normalized DCG at k with binary relevance.

    Ideal DCG uses ``min(k, |relevant|)`` relevant items in the top ranks.
    Returns 0.0 when there are no relevant items.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = dcg_at_k(recommended, rel, k)
    ideal_hits = min(k, len(rel))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def catalog_coverage(
    recommendations: Mapping[int, Sequence[int]],
    catalog: Iterable[int],
    k: int,
) -> float:
    """Fraction of the catalog that appears in at least one user's top-k list.

    ``coverage@k = |⋃_u recs_u[:k]| / |catalog|``
    """
    if k <= 0:
        raise ValueError("k must be positive")
    cat = set(catalog)
    if not cat:
        return 0.0
    recommended: set[int] = set()
    for recs in recommendations.values():
        recommended.update(int(i) for i in recs[:k])
    return len(recommended & cat) / len(cat)


def mean_popularity(
    recommendations: Mapping[int, Sequence[int]],
    item_popularity: Mapping[int, float],
    k: int,
) -> float:
    """Mean train popularity of recommended items (averaged over users).

    For each user, average popularity of ``recs[:k]`` (unknown items contribute
    0.0); then average across users. Higher = more popularity-biased.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not recommendations:
        return 0.0
    user_means: list[float] = []
    for recs in recommendations.values():
        top = list(recs[:k])
        if not top:
            user_means.append(0.0)
            continue
        user_means.append(
            float(np.mean([float(item_popularity.get(int(i), 0.0)) for i in top]))
        )
    return float(np.mean(user_means))


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of ``values``.

    Returns ``(mean, low, high)``. With fewer than 2 values, low=high=mean.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("values must be non-empty")
    mean = float(arr.mean())
    if arr.size < 2 or n_bootstrap < 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    n = arr.size
    for i in range(n_bootstrap):
        draw = rng.choice(arr, size=n, replace=True)
        samples[i] = draw.mean()
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return mean, low, high
