"""Prepare isolated reruns from the config and selected settings in a result record."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

from movielens_recommender.config import RunConfig, load_config
from movielens_recommender.data import DATASET_SHA256
from movielens_recommender.serving.provenance import file_sha256


def prepare_reproduction(
    source: Path | str, output_root: Path | str, *, run_id: str | None = None,
) -> tuple[RunConfig, Path]:
    """Freeze a historical headline protocol without overwriting records or models.

    Selected hyperparameters come from the record itself, never today's YAML
    or a mutable tuning file. Training still uses the current implementation;
    this does not promise byte-identical results across code or math libraries.
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run_id must contain only letters, digits, underscores, or hyphens")
    source = Path(source).resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if "config" not in record:
        raise ValueError("Reproduction requires a headline result record with an embedded config")
    raw = deepcopy(record["config"])
    dataset = raw.get("dataset")
    if dataset != record.get("dataset") or dataset not in DATASET_SHA256:
        raise ValueError("Result and config must name the same supported dataset")
    if record.get("dataset_sha256") != DATASET_SHA256[dataset]:
        raise ValueError("The result dataset checksum does not match the supported dataset")
    selections = {}
    if raw.get("tune", True):
        for family, model, score_key in (
            ("als", "als_tuned", "als"),
            ("item_item_cosine", "item_item_cosine_tuned", "item_item_cosine"),
            ("ease", "ease", "ease"), ("rp3beta", "rp3beta", "rp3beta"),
        ):
            selections[family] = {
                "best_hyperparams": record["hyperparameters"][model],
                "best_val_score": record["tuning_summary"][f"{score_key}_best_val_ndcg@10"],
            }
    tower_selection = None
    if raw.get("models", {}).get("two_tower", {}).get("enabled", True):
        tower = record["two_tower"]
        tower_selection = {"two_tower": {
            "best_hyperparams": tower["best_hyperparams"], "best_epoch": tower["best_epoch"],
            "best_val_score": tower["best_val_ndcg@10"],
        }}
    directory = Path(output_root).resolve() / run_id
    # Atomic reservation: repeating a run id fails before any existing output changes.
    directory.mkdir(parents=True, exist_ok=False)
    raw["results_dir"] = str(directory / "results")
    raw["models_dir"] = str(directory / "models")
    raw["data_dir"] = str(Path(raw.get("data_dir", "data")).resolve())
    config_path = directory / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)
    config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    tuning = directory / "results" / "tuning"
    tuning.mkdir(parents=True)
    provenance = {"source_record_sha256": file_sha256(source), "dataset": dataset}
    for filename, selection in (
        (f"{dataset}.json", selections), (f"two_tower_{dataset}.json", tower_selection),
    ):
        if selection is not None:
            (tuning / filename).write_text(
                json.dumps({**selection, **provenance}, indent=2) + "\n", encoding="utf-8",
            )
    (directory / "source-results.json").write_bytes(source.read_bytes())
    (directory / "run.json").write_text(json.dumps({
        **provenance, "run_id": run_id, "source_record": str(source), "status": "prepared",
        "effective_config_sha256": file_sha256(config_path),
        "note": "Same recorded protocol and selected settings, using current code and environment.",
    }, indent=2) + "\n", encoding="utf-8")
    return config, directory
