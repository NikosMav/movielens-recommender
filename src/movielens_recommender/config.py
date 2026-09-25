"""YAML run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SplitYAML:
    min_ratings: int = 5
    test_fraction: float = 0.2
    val_fraction: float = 0.1


@dataclass
class EvalYAML:
    ks: list[int] = field(default_factory=lambda: [10, 20])
    relevance_threshold: float = 4.0
    n_bootstrap: int = 1000
    bootstrap_alpha: float = 0.05


@dataclass
class ALSYAML:
    factors: int = 64
    regularization: float = 0.01
    iterations: int = 15
    alpha: float = 40.0


@dataclass
class ItemKNNYAML:
    min_common: int = 1
    k_neighbors: int = 0
    shrinkage: float = 0.0


@dataclass
class GlobalCutoffYAML:
    enabled: bool = False
    timestamp_quantile: float = 0.8
    min_train_ratings: int = 5


@dataclass
class ModelsYAML:
    als: ALSYAML = field(default_factory=ALSYAML)
    item_item_cosine: ItemKNNYAML = field(default_factory=ItemKNNYAML)


@dataclass
class RunConfig:
    """Top-level experiment config loaded from YAML."""

    seed: int = 42
    dataset: str = "ml-latest-small"
    data_dir: str = "data"
    results_dir: str = "results"
    tune: bool = True
    split: SplitYAML = field(default_factory=SplitYAML)
    eval: EvalYAML = field(default_factory=EvalYAML)
    models: ModelsYAML = field(default_factory=ModelsYAML)
    global_cutoff: GlobalCutoffYAML = field(default_factory=GlobalCutoffYAML)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: Path | str | None = None) -> RunConfig:
    """Load a :class:`RunConfig` from YAML, or return defaults when path is None."""
    if path is None:
        return RunConfig()
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    split_raw = raw.get("split", {})
    eval_raw = raw.get("eval", {})
    models_raw = raw.get("models", {})
    als_raw = models_raw.get("als", {})
    knn_raw = models_raw.get("item_item_cosine", {})
    gc_raw = raw.get("global_cutoff", {})
    return RunConfig(
        seed=int(raw.get("seed", 42)),
        dataset=str(raw.get("dataset", "ml-latest-small")),
        data_dir=str(raw.get("data_dir", "data")),
        results_dir=str(raw.get("results_dir", "results")),
        tune=bool(raw.get("tune", True)),
        split=SplitYAML(
            min_ratings=int(split_raw.get("min_ratings", 5)),
            test_fraction=float(split_raw.get("test_fraction", 0.2)),
            val_fraction=float(split_raw.get("val_fraction", 0.1)),
        ),
        eval=EvalYAML(
            ks=[int(k) for k in eval_raw.get("ks", [10, 20])],
            relevance_threshold=float(eval_raw.get("relevance_threshold", 4.0)),
            n_bootstrap=int(eval_raw.get("n_bootstrap", 1000)),
            bootstrap_alpha=float(eval_raw.get("bootstrap_alpha", 0.05)),
        ),
        models=ModelsYAML(
            als=ALSYAML(
                factors=int(als_raw.get("factors", 64)),
                regularization=float(als_raw.get("regularization", 0.01)),
                iterations=int(als_raw.get("iterations", 15)),
                alpha=float(als_raw.get("alpha", 40.0)),
            ),
            item_item_cosine=ItemKNNYAML(
                min_common=int(knn_raw.get("min_common", 1)),
                k_neighbors=int(knn_raw.get("k_neighbors", 0)),
                shrinkage=float(knn_raw.get("shrinkage", 0.0)),
            ),
        ),
        global_cutoff=GlobalCutoffYAML(
            enabled=bool(gc_raw.get("enabled", False)),
            timestamp_quantile=float(gc_raw.get("timestamp_quantile", 0.8)),
            min_train_ratings=int(gc_raw.get("min_train_ratings", 5)),
        ),
    )
