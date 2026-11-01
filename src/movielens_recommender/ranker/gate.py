"""Historical fixed-baseline gate and descriptive prior-model comparisons.

The bar is item–item cosine at its default hyperparameters. The value is read
from that model's metrics block. It is not a literal in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Historical ADR-0007 comparator; it is not the best model on every dataset.
S3_GATE_MODEL = "item_item_cosine"
PRIOR_MODELS = (
    "most_popular", "item_item_cosine", "item_item_cosine_tuned",
    "als", "als_tuned", "ease", "rp3beta", "two_tower",
)


def gate_against_s3(
    *,
    ranker_mean: float,
    per_seed_cis: Sequence[Mapping[str, float]],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare a ranker mean to ``metrics[item_item_cosine]['ndcg@10']``.

    ``negative_result`` is true when the 3-seed mean does not strictly exceed
    that point estimate. CI separation (every seed CI low above the bar CI
    high) is recorded alongside, the same way S3b reported seed intervals.
    """
    if S3_GATE_MODEL not in metrics:
        raise KeyError(
            f"S3 gate model {S3_GATE_MODEL!r} is missing from metrics; "
            "refusing to fall back to a hard-coded NDCG value"
        )
    block = metrics[S3_GATE_MODEL]
    bar = float(block["ndcg@10"])
    ci = block["confidence_intervals"]["ndcg@10"]
    bar_low = float(ci["low"])
    bar_high = float(ci["high"])
    mean_exceeds = bool(float(ranker_mean) > bar)
    clears_ci = bool(per_seed_cis) and all(
        float(bounds["low"]) > bar_high for bounds in per_seed_cis
    )
    return {
        "scope": "fixed_item_item_baseline",
        "bar_model": S3_GATE_MODEL,
        "bar_ndcg@10": bar,
        "bar_ci": {"low": bar_low, "high": bar_high},
        "bar_source": f"metrics[{S3_GATE_MODEL!r}]['ndcg@10']",
        "ranker_mean_ndcg@10": float(ranker_mean),
        "mean_exceeds_bar_point": mean_exceeds,
        "all_seed_ci_low_above_bar_ci_high": clears_ci,
        "negative_result": not mean_exceeds,
        "comparisons": compare_prior_points(ranker_mean, metrics),
    }


def compare_prior_points(ranker_mean: float, metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Describe test points, without selecting a model or claiming significance.

    Comparator families are fixed here before future runs. The largest prior
    point is descriptive; test scores must not choose training settings.
    Missing comparisons remain unknown, rather than counting as wins.
    """
    available = [name for name in PRIOR_MODELS if name in metrics]
    best = max(available, key=lambda name: float(metrics[name]["ndcg@10"])) if available else None
    prior = None if best is None else float(metrics[best]["ndcg@10"])
    unranked = metrics.get("no_ranker", {}).get("ndcg@10")
    unranked = None if unranked is None else float(unranked)
    beats_prior = None if prior is None else bool(ranker_mean > prior)
    beats_unranked = None if unranked is None else bool(ranker_mean > unranked)
    return {
        "scope": "descriptive_test_points_not_model_selection",
        "prior_model_families": list(PRIOR_MODELS),
        "available_prior_models": available,
        "best_prior_model": best,
        "best_prior_ndcg@10": prior,
        "ranker_mean_ndcg@10": float(ranker_mean),
        "ranker_minus_best_prior": None if prior is None else round(ranker_mean - prior, 6),
        "beats_best_prior_point": beats_prior,
        "no_ranker_ndcg@10": unranked,
        "ranker_minus_no_ranker": (
            None if unranked is None else round(ranker_mean - unranked, 6)
        ),
        "beats_no_ranker_point": beats_unranked,
        "beats_both_points": (
            None if prior is None or unranked is None else beats_prior and beats_unranked
        ),
        "paired_vs_best_prior": None,
        "paired_vs_best_prior_note": "Aligned per-user prior-model scores are not retained.",
    }
