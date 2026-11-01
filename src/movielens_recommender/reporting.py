"""Separate means across training seeds from user uncertainty within a seed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np


def primary_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Read primary-run diagnostics from new summaries or historical records."""
    return metrics.get("primary_seed_metrics", metrics)


def summarize_seed_metrics(
    rows: Sequence[Mapping[str, Any]], *, primary_seed: int,
) -> dict[str, Any]:
    """Average every scalar metric; never attach one seed's CI to that mean.

    Top-level scalar metrics remain aliases for existing consumers. Complete
    per-run metrics, intervals, and segments live under explicit seed fields.
    No confidence interval for the across-seed mean is estimated here.
    """
    per_seed = {str(row["seed"]): deepcopy(dict(row["metrics"])) for row in rows}
    if len(per_seed) != len(rows) or str(primary_seed) not in per_seed:
        raise ValueError("Unique seeds including the primary seed are required")
    primary = per_seed[str(primary_seed)]
    keys = {key for key, value in primary.items() if "@" in key and isinstance(value, int | float)}
    for metrics in per_seed.values():
        if metrics.get("n_eval_users") != primary.get("n_eval_users"):
            raise ValueError("Cannot combine different evaluation populations")
        other = {k for k, v in metrics.items() if "@" in k and isinstance(v, int | float)}
        if other != keys:
            raise ValueError("Every seed must report the same scalar metrics")
    spread = {}
    for key in sorted(keys):
        values = np.asarray([m[key] for m in per_seed.values()], dtype=np.float64)
        spread[key] = {
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std()), 6),
            "min": round(float(values.min()), 6),
            "max": round(float(values.max()), 6),
            "values": values.tolist(),
        }
    means = {key: block["mean"] for key, block in spread.items()}
    return {
        **means,
        "n_eval_users": primary.get("n_eval_users"),
        "n_seeds": len(rows),
        "aggregation": "mean_over_seeds",
        "seed_mean_metrics": dict(means),
        "seed_summary": spread,
        "confidence_intervals": {},
        "uncertainty_policy": {
            "seed_mean_confidence_interval": "not_computed",
            "user_bootstrap_intervals": "per_seed_metrics[seed].confidence_intervals",
        },
        "primary_seed": int(primary_seed),
        "primary_seed_metrics": deepcopy(primary),
        "per_seed_metrics": per_seed,
        "per_seed_confidence_intervals": {
            seed: deepcopy(metrics.get("confidence_intervals", {}))
            for seed, metrics in per_seed.items()
        },
    }
