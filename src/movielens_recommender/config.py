"""YAML run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from movielens_recommender.ranker.features import DEMO_MODES


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
    # Extra recall cutoffs for retrieval models (S3b); reported when computed.
    retrieval_ks: list[int] = field(default_factory=lambda: [100, 200])
    # 0 = score every eligible user. A positive size is a seeded sample
    # shared by every model (ADR-0011). Histories are not truncated.
    user_sample_size: int = 0
    # None uses the run seed.
    user_sample_seed: int | None = None


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
class EaseYAML:
    """EASE^R. ``max_items`` restricts the closed form to the head catalog.

    ``None`` uses every train item (ml-1m / ml-latest-small).
    """

    max_items: int | None = None


@dataclass
class RankerYAML:
    """LightGBM LambdaRank (S4). Optional extra: ``pip install -e '.[rank]'``."""

    enabled: bool = True
    candidate_k: int = 200
    early_stop_fraction: float = 0.2
    num_boost_round: int = 200
    early_stopping_rounds: int = 30
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.8
    # Ranker seeds only. Retrievers stay on the config seed.
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    # S4b. ``off`` is the S4 feature set. ``raw`` / ``affinity`` / ``both``
    # add ml-1m demographic features. The dataclass default stays ``off`` so
    # ml-latest-small cannot enable them. configs/ml-1m.yaml sets ``both``
    # because ADR-0009's pre-registered rule passed.
    demographics: str = "off"


@dataclass
class TwoTowerYAML:
    enabled: bool = True
    embedding_dim: int = 64
    learning_rate: float = 1e-3
    temperature: float = 0.1
    batch_size: int = 1024
    weight_decay: float = 1e-4
    max_epochs: int = 20
    patience: int = 3
    max_history: int = 50
    # Test-time seeds for variance reporting (ADR-0006).
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44])


@dataclass
class GlobalCutoffYAML:
    enabled: bool = False
    timestamp_quantile: float = 0.8
    min_train_ratings: int = 5


@dataclass
class TuningYAML:
    """Optional validation grids. ``None`` keeps the module-level default grid."""

    als: list[dict[str, Any]] | None = None
    item_item_cosine: list[dict[str, Any]] | None = None
    ease: list[dict[str, Any]] | None = None
    rp3beta: list[dict[str, Any]] | None = None
    two_tower: list[dict[str, Any]] | None = None


@dataclass
class ModelsYAML:
    als: ALSYAML = field(default_factory=ALSYAML)
    item_item_cosine: ItemKNNYAML = field(default_factory=ItemKNNYAML)
    ease: EaseYAML = field(default_factory=EaseYAML)
    two_tower: TwoTowerYAML = field(default_factory=TwoTowerYAML)
    ranker: RankerYAML = field(default_factory=RankerYAML)


@dataclass
class RunConfig:
    """Top-level experiment config loaded from YAML."""

    seed: int = 42
    dataset: str = "ml-latest-small"
    data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "models"
    tune: bool = True
    split: SplitYAML = field(default_factory=SplitYAML)
    eval: EvalYAML = field(default_factory=EvalYAML)
    models: ModelsYAML = field(default_factory=ModelsYAML)
    global_cutoff: GlobalCutoffYAML = field(default_factory=GlobalCutoffYAML)
    tuning: TuningYAML = field(default_factory=TuningYAML)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _demographic_mode(value: Any) -> str:
    """Normalize ``models.ranker.demographics`` to a :data:`DEMO_MODES` value.

    PyYAML 1.1 parses a bare ``off`` as boolean ``False``. ``None`` is a null
    or omitted value. Both mean the S4 feature set.
    """
    if value is False or value is None:
        return "off"
    if isinstance(value, bool):
        raise ValueError(
            "models.ranker.demographics must be one of "
            f"{DEMO_MODES}; got boolean {value!r}. "
            'Quote the value in YAML, for example demographics: "off".'
        )
    text = str(value)
    if text not in DEMO_MODES:
        raise ValueError(
            f"models.ranker.demographics must be one of {DEMO_MODES}; got {value!r}"
        )
    return text


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
    tt_raw = models_raw.get("two_tower", {})
    rank_raw = models_raw.get("ranker", {})
    gc_raw = raw.get("global_cutoff", {})
    ease_raw = models_raw.get("ease") or {}
    tune_raw = raw.get("tuning") or {}
    sample_seed = eval_raw.get("user_sample_seed")
    return RunConfig(
        seed=int(raw.get("seed", 42)),
        dataset=str(raw.get("dataset", "ml-latest-small")),
        data_dir=str(raw.get("data_dir", "data")),
        results_dir=str(raw.get("results_dir", "results")),
        models_dir=str(raw.get("models_dir", "models")),
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
            retrieval_ks=[int(k) for k in eval_raw.get("retrieval_ks", [100, 200])],
            user_sample_size=int(eval_raw.get("user_sample_size", 0) or 0),
            user_sample_seed=None if sample_seed is None else int(sample_seed),
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
            ease=EaseYAML(
                max_items=(
                    None
                    if ease_raw.get("max_items") is None
                    else int(ease_raw.get("max_items"))
                ),
            ),
            two_tower=TwoTowerYAML(
                enabled=bool(tt_raw.get("enabled", True)),
                embedding_dim=int(tt_raw.get("embedding_dim", 64)),
                learning_rate=float(tt_raw.get("learning_rate", 1e-3)),
                temperature=float(tt_raw.get("temperature", 0.1)),
                batch_size=int(tt_raw.get("batch_size", 1024)),
                weight_decay=float(tt_raw.get("weight_decay", 1e-4)),
                max_epochs=int(tt_raw.get("max_epochs", 20)),
                patience=int(tt_raw.get("patience", 3)),
                max_history=int(tt_raw.get("max_history", 50)),
                seeds=[int(s) for s in tt_raw.get("seeds", [42, 43, 44])],
            ),
            ranker=RankerYAML(
                enabled=bool(rank_raw.get("enabled", True)),
                candidate_k=int(rank_raw.get("candidate_k", 200)),
                early_stop_fraction=float(rank_raw.get("early_stop_fraction", 0.2)),
                num_boost_round=int(rank_raw.get("num_boost_round", 200)),
                early_stopping_rounds=int(rank_raw.get("early_stopping_rounds", 30)),
                learning_rate=float(rank_raw.get("learning_rate", 0.05)),
                num_leaves=int(rank_raw.get("num_leaves", 31)),
                min_data_in_leaf=int(rank_raw.get("min_data_in_leaf", 20)),
                feature_fraction=float(rank_raw.get("feature_fraction", 0.9)),
                bagging_fraction=float(rank_raw.get("bagging_fraction", 0.8)),
                seeds=[int(s) for s in rank_raw.get("seeds", [42, 43, 44])],
                demographics=_demographic_mode(rank_raw.get("demographics", "off")),
            ),
        ),
        global_cutoff=GlobalCutoffYAML(
            enabled=bool(gc_raw.get("enabled", False)),
            timestamp_quantile=float(gc_raw.get("timestamp_quantile", 0.8)),
            min_train_ratings=int(gc_raw.get("min_train_ratings", 5)),
        ),
        tuning=TuningYAML(
            als=_optional_grid(tune_raw, "als"),
            item_item_cosine=_optional_grid(tune_raw, "item_item_cosine"),
            ease=_optional_grid(tune_raw, "ease"),
            rp3beta=_optional_grid(tune_raw, "rp3beta"),
            two_tower=_optional_grid(tune_raw, "two_tower"),
        ),
    )


def _optional_grid(tune_raw: dict[str, Any], key: str) -> list[dict[str, Any]] | None:
    """Return a copied grid, or ``None`` when the YAML omits it."""
    if key not in tune_raw or tune_raw[key] is None:
        return None
    return [dict(item) for item in tune_raw[key]]
