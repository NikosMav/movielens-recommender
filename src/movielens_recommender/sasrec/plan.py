"""Validation grid, edge step, and budget cut for SASRec (ADR-0019).

Pure functions. They do not read ratings and they do not compute a ranking metric.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

# Searched axes. Order is the tie-break order (ADR-0019).
MAX_SEQ_LENS: tuple[int, ...] = (50, 200)
N_LAYERS: tuple[int, ...] = (1, 2)
DROPOUTS: tuple[float, ...] = (0.2, 0.5)

# Fixed beside the grid.
D_MODEL = 64
N_HEADS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
WEIGHT_DECAY = 0.0
MAX_EPOCHS = 30
PATIENCE = 3
LOSS = "full_softmax"

BUDGET_SEC = 3 * 60 * 60
COMPARISON_RESERVE_SEC = 1800
TEST_SEEDS: tuple[int, ...] = (42, 43, 44)
PRIMARY_SEED = 42
SHUFFLE_SEED = 42
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 1000
BOOTSTRAP_ALPHA = 0.05

# Two-tower ADR-0006 grid and the EASE grid. The SASRec search matches that count.
TWO_TOWER_GRID_CONFIGS = 8
EASE_CONFIGS = 8

# torch.Generator seed for epoch ``e`` is ``model_seed + e * EPOCH_SHUFFLE``.
EPOCH_SHUFFLE = 1_000_003

# Dropout high step is the gap between the two grid values, 0.5 - 0.2.
DROPOUT_HIGH_STEP = 0.3
LAYER_HIGH_STEP = 3


def fixed_hyperparams(*, max_epochs: int = MAX_EPOCHS) -> dict[str, Any]:
    """Hyperparameters that are not grid axes."""
    return {
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "learning_rate": LEARNING_RATE,
        "loss": LOSS,
        "max_epochs": int(max_epochs),
        "n_heads": N_HEADS,
        "patience": PATIENCE,
        "weight_decay": WEIGHT_DECAY,
    }


def base_grid(
    *,
    max_epochs: int = MAX_EPOCHS,
    drop_len_200: bool = False,
    drop_layers_2: bool = False,
) -> list[dict[str, Any]]:
    """Grid in tie-break order: length, layers, dropout, each ascending."""
    lengths = tuple(v for v in MAX_SEQ_LENS if not (drop_len_200 and v == 200))
    layers = tuple(v for v in N_LAYERS if not (drop_layers_2 and v == 2))
    rows: list[dict[str, Any]] = []
    for max_seq_len, n_layers, dropout in itertools.product(lengths, layers, DROPOUTS):
        row = fixed_hyperparams(max_epochs=max_epochs)
        row["dropout"] = float(dropout)
        row["max_seq_len"] = int(max_seq_len)
        row["n_layers"] = int(n_layers)
        rows.append(row)
    return rows


def config_key(cfg: Mapping[str, Any]) -> str:
    """Stable identity of a config. Floats are rounded at 6 decimals."""
    payload = {}
    for key in sorted(cfg):
        value = cfg[key]
        if isinstance(value, float):
            payload[key] = round(float(value), 6)
        else:
            payload[key] = value
    return json.dumps(payload, sort_keys=True)


def projected_trial_seconds(
    cfg: Mapping[str, Any],
    *,
    t50: float,
    t200: float,
    max_epochs: int | None = None,
) -> float:
    """Worst-case seconds: ``max_epochs`` times the probe epoch, scaled by layers.

    ``t50`` and ``t200`` are one epoch plus one validation encode at 2 layers.
    One layer is half of that. Dropout does not change the projection.
    """
    epochs = int(cfg["max_epochs"] if max_epochs is None else max_epochs)
    length = int(cfg["max_seq_len"])
    if length == 50:
        rate = float(t50)
    elif length == 200:
        rate = float(t200)
    else:
        # Edge steps do not change length. Anything else is a programming error.
        raise ValueError(f"no probe time for max_seq_len={length}")
    return epochs * rate * (int(cfg["n_layers"]) / 2.0)


def _slowest(grid: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return max(grid, key=lambda cfg: (int(cfg["max_seq_len"]), int(cfg["n_layers"])))


def edge_reserve_seconds(
    grid: Sequence[Mapping[str, Any]],
    *,
    t50: float,
    t200: float,
    max_epochs: int,
) -> float:
    """Worst-case seconds for the edge steps the rule can still add."""
    if not grid:
        return 0.0
    lengths = {int(cfg["max_seq_len"]) for cfg in grid}
    layers = {int(cfg["n_layers"]) for cfg in grid}
    dropouts = {round(float(cfg["dropout"]), 6) for cfg in grid}
    slow_len = max(lengths)
    slow_layers = max(layers)
    reserve = 0.0
    if 2 in layers:
        reserve += projected_trial_seconds(
            {"max_seq_len": slow_len, "n_layers": LAYER_HIGH_STEP, "max_epochs": max_epochs},
            t50=t50,
            t200=t200,
        )
    if 0.5 in dropouts:
        reserve += projected_trial_seconds(
            {"max_seq_len": slow_len, "n_layers": slow_layers, "max_epochs": max_epochs},
            t50=t50,
            t200=t200,
        )
    return reserve


def _cut_name(
    *,
    drop_len_200: bool,
    max_epochs: int,
    test_seeds: Sequence[int],
    drop_layers_2: bool,
) -> str | None:
    parts: list[str] = []
    if drop_len_200:
        parts.append("dropped max_seq_len=200")
    if max_epochs != MAX_EPOCHS:
        parts.append(f"max_epochs={max_epochs}")
    if list(test_seeds) != list(TEST_SEEDS):
        parts.append("test seeds " + ",".join(str(s) for s in test_seeds))
    if drop_layers_2:
        parts.append("dropped n_layers=2")
    if not parts:
        return None
    return "; ".join(parts)


def lock_budget(t50: float, t200: float) -> dict[str, Any]:
    """First pre-registered cut whose worst-case total fits in the 3 hour cap.

    ``t200`` may be infinite when the length-200 probe ran out of memory.
    ``fits`` is false when even the last cut is over the cap. The caller must
    not score validation in that case.
    """
    if t50 <= 0 or math.isnan(t50) or math.isnan(t200):
        raise ValueError("probe times must be positive and finite for t50")
    stages = (
        {
            "drop_len_200": False,
            "max_epochs": MAX_EPOCHS,
            "seeds": TEST_SEEDS,
            "drop_layers_2": False,
        },
        {
            "drop_len_200": True,
            "max_epochs": MAX_EPOCHS,
            "seeds": TEST_SEEDS,
            "drop_layers_2": False,
        },
        {
            "drop_len_200": True,
            "max_epochs": 15,
            "seeds": TEST_SEEDS,
            "drop_layers_2": False,
        },
        {
            "drop_len_200": True,
            "max_epochs": 15,
            "seeds": (PRIMARY_SEED,),
            "drop_layers_2": False,
        },
        {
            "drop_len_200": True,
            "max_epochs": 15,
            "seeds": (PRIMARY_SEED,),
            "drop_layers_2": True,
        },
    )
    last: dict[str, Any] | None = None
    for stage in stages:
        grid = base_grid(
            max_epochs=int(stage["max_epochs"]),
            drop_len_200=bool(stage["drop_len_200"]),
            drop_layers_2=bool(stage["drop_layers_2"]),
        )
        epochs = int(stage["max_epochs"])
        seeds = tuple(int(s) for s in stage["seeds"])  # type: ignore[union-attr]
        slow = _slowest(grid)
        grid_sec = sum(
            projected_trial_seconds(cfg, t50=t50, t200=t200) for cfg in grid
        )
        edge_sec = edge_reserve_seconds(grid, t50=t50, t200=t200, max_epochs=epochs)
        one = projected_trial_seconds(slow, t50=t50, t200=t200)
        test_sec = len(seeds) * one
        shuffle_sec = one
        total = grid_sec + edge_sec + test_sec + shuffle_sec + COMPARISON_RESERVE_SEC
        decision = {
            "fits": total <= BUDGET_SEC and math.isfinite(total),
            "cut": _cut_name(
                drop_len_200=bool(stage["drop_len_200"]),
                max_epochs=epochs,
                test_seeds=seeds,
                drop_layers_2=bool(stage["drop_layers_2"]),
            ),
            "max_epochs": epochs,
            "test_seeds": list(seeds),
            "grid": grid,
            "n_configs": len(grid),
            "projected_sec": None if not math.isfinite(total) else round(total, 3),
            "grid_sec": None if not math.isfinite(grid_sec) else round(grid_sec, 3),
            "edge_reserve_sec": None if not math.isfinite(edge_sec) else round(edge_sec, 3),
            "test_reserve_sec": None if not math.isfinite(test_sec) else round(test_sec, 3),
            "shuffle_reserve_sec": (
                None if not math.isfinite(shuffle_sec) else round(shuffle_sec, 3)
            ),
            "comparison_reserve_sec": COMPARISON_RESERVE_SEC,
            "cap_sec": BUDGET_SEC,
            "t50": t50,
            "t200": None if not math.isfinite(t200) else t200,
        }
        last = decision
        if decision["fits"]:
            return decision
    assert last is not None
    return last


def find_edges(
    hyperparams: Mapping[str, Any],
    scored: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Edges of searched axes. A one-value axis is not an edge.

    Fixed axes (``d_model``, ``n_heads``) are not considered.
    """
    edges: list[dict[str, Any]] = []
    for parameter in ("max_seq_len", "n_layers", "dropout"):
        values = [cfg[parameter] for cfg in scored if parameter in cfg]
        distinct = {round(float(v), 6) for v in values}
        if len(distinct) < 2:
            continue
        lo = min(values)
        hi = max(values)
        value = hyperparams[parameter]
        if value == lo:
            edges.append({"parameter": parameter, "side": "low", "value": value})
        elif value == hi:
            edges.append({"parameter": parameter, "side": "high", "value": value})
    return edges


def edge_extension_configs(
    hyperparams: Mapping[str, Any],
    scored: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """At most one step on layers and one on dropout. Length is never extended."""
    known = {config_key(cfg) for cfg in scored}
    extensions: list[dict[str, Any]] = []
    for edge in find_edges(hyperparams, scored):
        stepped: dict[str, Any] | None = None
        if (
            edge["parameter"] == "n_layers"
            and edge["side"] == "high"
            and int(edge["value"]) == 2
        ):
            stepped = dict(hyperparams)
            stepped["n_layers"] = LAYER_HIGH_STEP
        elif (
            edge["parameter"] == "dropout"
            and edge["side"] == "high"
            and round(float(edge["value"]), 6) == 0.5
        ):
            stepped = dict(hyperparams)
            stepped["dropout"] = round(0.5 + DROPOUT_HIGH_STEP, 6)
        if stepped is None:
            continue
        if config_key(stepped) in known:
            continue
        extensions.append(stepped)
        known.add(config_key(stepped))
    return extensions


def select_winner(trials: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Strict maximum of validation NDCG@10 at 6 decimals. Ties keep the earlier trial."""
    if not trials:
        raise ValueError("no trials to select from")
    winner = trials[0]
    best = round(float(winner["val_ndcg@10"]), 6)
    for trial in trials[1:]:
        score = round(float(trial["val_ndcg@10"]), 6)
        if score > best:
            winner = trial
            best = score
    return winner
