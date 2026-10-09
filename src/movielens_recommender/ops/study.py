"""Run the ADR-0016 replay on ml-1m and write ``results/ops/ml-1m.json``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from movielens_recommender.config import RunConfig
from movielens_recommender.data import DATASET_SHA256, DATASET_VERSION_LABELS, load_ratings
from movielens_recommender.ops.drift import ml1m_periods
from movielens_recommender.ops.replay import run_replay

TAUS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
N_TUNING = 3
SAMPLE_SIZE = 2000
MIN_HISTORY = 5
MIN_ITEM_COUNT = 5
TOLERANCE = 0.005


def tuned_models(results_dir: Path) -> dict[str, dict[str, Any]]:
    """The ml-1m validation winners; not searched again (ADR-0016)."""
    tuning = json.loads((results_dir / "tuning" / "ml-1m.json").read_text(encoding="utf-8"))
    return {
        "most_popular": {},
        "item_item": dict(tuning["item_item_cosine"]["best_hyperparams"]),
        "ease": dict(tuning["ease"]["best_hyperparams"]),
    }


def run_drift_study(config: RunConfig, *, download: bool = False) -> Path:
    if config.dataset != "ml-1m":
        raise ValueError(f"The drift study is ml-1m only (got dataset {config.dataset!r}).")
    if download:
        from movielens_recommender.data import download_dataset

        download_dataset(config.dataset, config.data_dir)
    from movielens_recommender.cli import host_info, library_versions

    started = time.perf_counter()
    results_dir = Path(config.results_dir)
    ratings, _clean = load_ratings(config.dataset, config.data_dir, clean=True)
    models = tuned_models(results_dir)
    replay = run_replay(
        ratings,
        ml1m_periods(),
        models=models,
        taus=TAUS,
        n_tuning=N_TUNING,
        sample_size=SAMPLE_SIZE,
        min_history=MIN_HISTORY,
        relevance=float(config.eval.relevance_threshold),
        min_item_count=MIN_ITEM_COUNT,
        tolerance=TOLERANCE,
        n_bootstrap=int(config.eval.n_bootstrap),
        seed=int(config.seed),
    )
    payload = {
        "dataset": config.dataset,
        "dataset_version": DATASET_VERSION_LABELS[config.dataset],
        "dataset_sha256": DATASET_SHA256[config.dataset],
        "adr": "docs/adr/0016-drift-refresh-monitoring.md",
        "protocol": {
            "first_model_trained_before": "2000-08-01",
            "metric": "ndcg@10",
            "relevance_threshold": float(config.eval.relevance_threshold),
            "min_history": MIN_HISTORY,
            "drift_sample_size": SAMPLE_SIZE,
            "new_item_min_count": MIN_ITEM_COUNT,
            "taus": list(TAUS),
            "n_tuning_periods": N_TUNING,
            "tolerance": TOLERANCE,
            "n_bootstrap": int(config.eval.n_bootstrap),
            "seed": int(config.seed),
        },
        **replay,
        "runtime_sec": round(time.perf_counter() - started, 3),
        "library_versions": library_versions(),
        "host": host_info(),
    }
    out = results_dir / "ops" / f"{config.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out
