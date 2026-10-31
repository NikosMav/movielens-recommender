"""Beyond-accuracy scores on lists the ranking harness already built.

Definitions are fixed in ADR-0017. Catalog coverage stays a point estimate.
The other metrics are user-level means and use :func:`bootstrap_mean_ci`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    intra_list_diversity,
    long_tail_share,
    mean_popularity,
    ndcg_at_k,
    novelty_at_k,
    per_user_intra_list_diversity,
    per_user_long_tail_share,
    per_user_novelty,
    per_user_popularity,
)
from movielens_recommender.movies import build_item_side_features
from movielens_recommender.segments import head_tail_items

# One-line summaries of ADR-0017. The ADR is the definition; these travel
# with the results file so a reader does not have to open the doc first.
METRIC_DEFINITIONS: dict[str, str] = {
    "coverage": (
        "Share of the train catalog that appears in at least one top-k list. "
        "Point estimate only: a user bootstrap of this set-union is biased low "
        "and does not contain the point estimate (ADR-0003)."
    ),
    "novelty": (
        "Mean over users of the mean self-information of the top-k items. "
        "Self-information of item i is -log2(count(i) / n_train_interactions), "
        "where count(i) is the number of train rows for i."
    ),
    "intra_list_diversity": (
        "Mean over users of 1 minus the mean pairwise cosine similarity of the "
        "top-k items. The item vector is the genre multi-hot from movies.dat."
    ),
    "mean_popularity": (
        "Existing diagnostic: mean over users of the mean train interaction "
        "count of the top-k items. Not a second popularity metric."
    ),
    "long_tail_share": (
        "Mean over users of the fraction of the top-k list that falls in the "
        "train tail. The tail is every train item outside the top 20% head."
    ),
    "ndcg": (
        "Existing binary NDCG@k (relevant when the test rating is at least "
        "the harness threshold). Recomputed on the lists scored here."
    ),
}

ITEM_REPRESENTATION = (
    "Genre multi-hot from movies.dat, the same vector as "
    "build_item_side_features: one bit per entry of GENRES, ml-1m "
    "Children's mapped to Children, and a single (no genres listed) bit "
    "when the item is missing or has no known genre. Cosine is computed "
    "on these raw 0/1 vectors. A zero norm would count as cosine 0; the "
    "movies.dat builder does not emit an all-zero vector."
)


def long_tail_items(train: pd.DataFrame) -> set[int]:
    """Train items outside the top-20% head (ADR-0017).

    The cut is :func:`head_tail_items`: ``n_head = max(1, int(n_items * 0.2))``,
    ordered by interaction count descending and item id ascending. The tail is
    the rest of the train catalog. ``int`` truncates, so an item that does not
    fit in the head stays in the tail. On a catalog whose size is divisible
    by 5, the tail is exactly the bottom 80%.
    """
    return set(head_tail_items(train, head_fraction=0.2)["tail"])


def genre_item_vectors(
    movies: pd.DataFrame,
    item_ids: Sequence[int],
) -> dict[int, np.ndarray]:
    """Genre multi-hot for each id, using the two-tower item representation."""
    genres, _years, _mean, _std = build_item_side_features(movies, list(item_ids))
    return {
        int(item_id): np.asarray(genres[row], dtype=np.float64)
        for row, item_id in enumerate(item_ids)
    }


def score_recommendation_lists(
    recommendations: Mapping[int, Sequence[int]],
    *,
    relevant: Mapping[int, set[int]],
    catalog: Iterable[int],
    item_popularity: Mapping[int, float],
    item_vectors: Mapping[int, np.ndarray],
    tail_items: Iterable[int],
    k: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Score one model's lists.

    ``recommendations`` iteration order is the user order for every mean and
    for the bootstrap. Callers that want the harness interval must pass the
    dict built by :func:`collect_recommendations`.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not recommendations:
        raise ValueError("recommendations must be non-empty")
    missing = [uid for uid in recommendations if uid not in relevant]
    if missing:
        raise KeyError(f"recommendations missing relevance for {missing[:5]}")

    pop_vals = per_user_popularity(recommendations, item_popularity, k)
    nov_vals = per_user_novelty(recommendations, item_popularity, k)
    div_vals = per_user_intra_list_diversity(recommendations, item_vectors, k)
    tail_vals = per_user_long_tail_share(recommendations, tail_items, k)
    ndcg_vals = [
        ndcg_at_k(recommendations[uid], relevant[uid], k) for uid in recommendations
    ]

    # The popularity point estimate has to match the existing function.
    pop_mean = float(np.mean(pop_vals))
    if abs(pop_mean - mean_popularity(recommendations, item_popularity, k)) > 1e-12:
        raise AssertionError("mean popularity drifted from the existing metric")
    if abs(float(np.mean(nov_vals)) - novelty_at_k(recommendations, item_popularity, k)) > 1e-12:
        raise AssertionError("novelty mean drifted")
    div_mean = float(np.mean(div_vals))
    div_point = intra_list_diversity(recommendations, item_vectors, k)
    if abs(div_mean - div_point) > 1e-12:
        raise AssertionError("intra-list diversity mean drifted")
    if abs(float(np.mean(tail_vals)) - long_tail_share(recommendations, tail_items, k)) > 1e-12:
        raise AssertionError("long-tail share mean drifted")

    user_metrics = {
        f"novelty@{k}": nov_vals,
        f"intra_list_diversity@{k}": div_vals,
        f"mean_popularity@{k}": pop_vals,
        f"long_tail_share@{k}": tail_vals,
        f"ndcg@{k}": ndcg_vals,
    }
    metrics: dict[str, Any] = {"n_eval_users": len(recommendations)}
    cis: dict[str, dict[str, float]] = {}
    for name, values in user_metrics.items():
        mean, low, high = bootstrap_mean_ci(
            values, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
        )
        metrics[name] = mean
        cis[name] = {"mean": mean, "low": low, "high": high}
        if not (low <= mean <= high):
            raise AssertionError(f"CI for {name} does not contain the point estimate")

    metrics[f"coverage@{k}"] = catalog_coverage(recommendations, catalog, k)
    metrics["confidence_intervals"] = cis
    metrics["uncertainty_policy"] = {
        "bootstrap_over_users": list(user_metrics),
        "point_estimate_only": [f"coverage@{k}"],
        "coverage_ci_policy": "point_estimate_only",
        "bootstrap": "bootstrap_mean_ci",
        "n_bootstrap": n_bootstrap,
        "alpha": alpha,
        "seed": seed,
    }
    return metrics
