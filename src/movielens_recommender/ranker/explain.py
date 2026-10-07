"""Per-candidate LightGBM contributions (pred_contrib) and model I/O.

The saved artifact is the refit ranker: feature names, schema version, and a
LightGBM text model under a gitignored ``models/`` directory. Recreate it with
the documented ``movielens-recommender run`` command; do not commit the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from movielens_recommender.ranker.train import require_lightgbm

SCHEMA_VERSION = 1
MODEL_FILENAME = "ranker.txt"
META_FILENAME = "ranker_meta.json"


def explain_candidates(
    booster: Any,
    features: np.ndarray,
    feature_names: list[str],
    *,
    item_ids: list[int],
    retrievers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map ``pred_contrib=True`` values onto feature names, plus retriever provenance.

    LightGBM returns one contribution per feature and a final bias column.
    Those values sum to the raw score (the sum is the caller's check; this
    function also stores both the raw score and the bias).

    ``retrievers[i]`` describes which retriever supplied candidate ``i``::

        {
          "item_item": {"score": float, "rank": int} | None,
          "two_tower": {"score": float, "rank": int} | None,
        }
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("features must be a 2-d matrix")
    if matrix.shape[0] != len(item_ids) or matrix.shape[0] != len(retrievers):
        raise ValueError("item_ids, retrievers, and feature rows must align")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature columns must match feature_names")

    contrib = np.asarray(booster.predict(matrix, pred_contrib=True), dtype=np.float64)
    raw = np.asarray(booster.predict(matrix, raw_score=True), dtype=np.float64)
    if contrib.ndim == 1:
        contrib = contrib.reshape(1, -1)
    if contrib.shape != (matrix.shape[0], len(feature_names) + 1):
        raise ValueError(
            f"pred_contrib shape {contrib.shape} != "
            f"({matrix.shape[0]}, {len(feature_names) + 1})"
        )

    rows: list[dict[str, Any]] = []
    for i, item_id in enumerate(item_ids):
        contributions = {
            feature_names[j]: float(contrib[i, j]) for j in range(len(feature_names))
        }
        rows.append(
            {
                "item_id": int(item_id),
                "raw_score": float(raw[i]),
                "bias": float(contrib[i, -1]),
                "contributions": contributions,
                "retrievers": retrievers[i],
            }
        )
    return rows


def save_ranker(booster: Any, directory: Path | str, meta: dict[str, Any]) -> Path:
    """Write ``ranker.txt`` and ``ranker_meta.json`` (schema version included)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILENAME
    booster.save_model(str(model_path))
    payload = {
        "schema_version": SCHEMA_VERSION,
        **meta,
    }
    payload["schema_version"] = SCHEMA_VERSION
    if "feature_names" not in payload:
        raise ValueError("meta must include feature_names")
    meta_path = directory / META_FILENAME
    meta_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model_path


def load_ranker(directory: Path | str) -> tuple[Any, dict[str, Any]]:
    """Reload a refit ranker saved by :func:`save_ranker`."""
    lgb = require_lightgbm()
    directory = Path(directory)
    meta_path = directory / META_FILENAME
    model_path = directory / MODEL_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = int(meta.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"ranker schema_version {version} does not match supported {SCHEMA_VERSION}"
        )
    booster = lgb.Booster(model_file=str(model_path))
    return booster, meta


def recreate_command(dataset: str) -> str:
    """CLI command that retrains the gitignored refit ranker for ``dataset``."""
    if dataset == "ml-1m":
        config = "configs/ml-1m.yaml"
    elif dataset == "ml-32m":
        config = "configs/ml-32m.yaml"
    else:
        config = "configs/default.yaml"
    return f"movielens-recommender run --config {config}"
