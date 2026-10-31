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


def per_user_popularity(
    recommendations: Mapping[int, Sequence[int]],
    item_popularity: Mapping[int, float],
    k: int,
) -> list[float]:
    """Per-user mean train popularity of the top-k list, in dict order.

    Unknown items contribute 0. An empty list scores 0. The mean of this list
    is :func:`mean_popularity`.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    values: list[float] = []
    for recs in recommendations.values():
        top = [int(i) for i in list(recs)[:k]]
        if not top:
            values.append(0.0)
            continue
        values.append(float(np.mean([float(item_popularity.get(i, 0.0)) for i in top])))
    return values


def self_information(item_popularity: Mapping[int, float]) -> dict[int, float]:
    """``-log2(count(i) / total)`` for items with a positive train count.

    ``total`` is the sum of the positive counts. That is the self-information
    of the item's share of train interactions. Non-positive counts are omitted
    because ``-log2(0)`` is undefined.
    """
    counts = {int(i): float(c) for i, c in item_popularity.items() if float(c) > 0.0}
    total = float(sum(counts.values()))
    if total <= 0.0:
        raise ValueError("item popularity must contain a positive count")
    return {item: float(-math.log2(count / total)) for item, count in counts.items()}


def per_user_novelty(
    recommendations: Mapping[int, Sequence[int]],
    item_popularity: Mapping[int, float],
    k: int,
) -> list[float]:
    """Per-user mean self-information of the top-k list, in dict order.

    Items with no positive train count are left out of that user's average.
    If none of the top-k items have a positive count, or the list is empty,
    the user scores 0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    info = self_information(item_popularity)
    values: list[float] = []
    for recs in recommendations.values():
        top = [int(i) for i in list(recs)[:k]]
        known = [info[i] for i in top if i in info]
        if not known:
            values.append(0.0)
            continue
        values.append(float(np.mean(known)))
    return values


def novelty_at_k(
    recommendations: Mapping[int, Sequence[int]],
    item_popularity: Mapping[int, float],
    k: int,
) -> float:
    """Mean over users of mean top-k self-information. Empty input is 0."""
    values = per_user_novelty(recommendations, item_popularity, k)
    if not values:
        return 0.0
    return float(np.mean(values))


def _list_intra_diversity(ids: Sequence[int], vectors: Mapping[int, np.ndarray]) -> float:
    """``1 - mean pairwise cosine`` over positions ``i < j``. Fewer than 2 items is 0.

    A zero-norm vector has cosine 0 with every item, including another zero vector.
    """
    if len(ids) < 2:
        return 0.0
    rows: list[np.ndarray] = []
    for item in ids:
        if item not in vectors:
            raise KeyError(f"no item vector for {item}")
        rows.append(np.asarray(vectors[item], dtype=np.float64).ravel())
    mat = np.vstack(rows)
    norms = np.linalg.norm(mat, axis=1)
    upper = np.triu_indices(len(rows), k=1)
    denom = np.outer(norms, norms)[upper]
    dots = (mat @ mat.T)[upper]
    sims = np.zeros(dots.shape, dtype=np.float64)
    np.divide(dots, denom, out=sims, where=denom > 0.0)
    return float(1.0 - sims.mean())


def per_user_intra_list_diversity(
    recommendations: Mapping[int, Sequence[int]],
    item_vectors: Mapping[int, np.ndarray],
    k: int,
) -> list[float]:
    """Per-user intra-list diversity of the top-k list, in dict order.

    Diversity is 1 minus the mean pairwise cosine similarity of the item
    vectors. Lists with fewer than two items score 0.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    values: list[float] = []
    for recs in recommendations.values():
        top = [int(i) for i in list(recs)[:k]]
        values.append(_list_intra_diversity(top, item_vectors))
    return values


def intra_list_diversity(
    recommendations: Mapping[int, Sequence[int]],
    item_vectors: Mapping[int, np.ndarray],
    k: int,
) -> float:
    """Mean over users of top-k intra-list diversity. Empty input is 0."""
    values = per_user_intra_list_diversity(recommendations, item_vectors, k)
    if not values:
        return 0.0
    return float(np.mean(values))


def per_user_long_tail_share(
    recommendations: Mapping[int, Sequence[int]],
    tail_items: Iterable[int],
    k: int,
) -> list[float]:
    """Per-user fraction of the top-k list that lies in ``tail_items``.

    An empty list scores 0. Dict order is preserved.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    tail = {int(i) for i in tail_items}
    values: list[float] = []
    for recs in recommendations.values():
        top = [int(i) for i in list(recs)[:k]]
        if not top:
            values.append(0.0)
            continue
        values.append(sum(1 for item in top if item in tail) / len(top))
    return values


def long_tail_share(
    recommendations: Mapping[int, Sequence[int]],
    tail_items: Iterable[int],
    k: int,
) -> float:
    """Mean over users of the top-k long-tail share. Empty input is 0."""
    values = per_user_long_tail_share(recommendations, tail_items, k)
    if not values:
        return 0.0
    return float(np.mean(values))


def paired_bootstrap_delta_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float | int | bool]:
    """Percentile bootstrap CI for the mean of ``right - left`` on paired users.

    ``left`` and ``right`` are aligned per-user scores. ``excludes_zero`` is
    true when the interval lies entirely above or entirely below 0.
    """
    a = np.asarray(list(left), dtype=np.float64)
    b = np.asarray(list(right), dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("paired bootstrap requires two equal-length 1-d sequences")
    if a.size == 0:
        raise ValueError("paired bootstrap requires at least one user")
    mean, low, high = bootstrap_mean_ci(
        b - a, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
    )
    return {
        "mean": mean,
        "low": low,
        "high": high,
        "n_users": int(a.size),
        "excludes_zero": bool(low > 0.0 or high < 0.0),
    }


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
