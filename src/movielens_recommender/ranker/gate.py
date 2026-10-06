"""Stage-4 gate against the best S3 test NDCG@10.

The bar is item–item cosine at its default hyperparameters. The value is read
from that model's metrics block. It is not a literal in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

# Best S3 model on the headline test NDCG@10 (see results JSON / ADR-0006).
S3_GATE_MODEL = "item_item_cosine"


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
        "bar_model": S3_GATE_MODEL,
        "bar_ndcg@10": bar,
        "bar_ci": {"low": bar_low, "high": bar_high},
        "bar_source": f"metrics[{S3_GATE_MODEL!r}]['ndcg@10']",
        "ranker_mean_ndcg@10": float(ranker_mean),
        "mean_exceeds_bar_point": mean_exceeds,
        "all_seed_ci_low_above_bar_ci_high": clears_ci,
        "negative_result": not mean_exceeds,
    }
