"""Ranking metrics for recommendation evaluation (binary relevance)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


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
