"""Pre-registered compute plan for the S3e two-tower losses.

The rules in this module turn a timing probe into a grid. They do not read
ranking metrics. ADR-0013 records the plan that these rules emit.
"""

from __future__ import annotations

import itertools
from typing import Any

LOSS_FULL = "full_softmax"
LOSS_SAMPLED = "sampled_softmax"
LOSSES = (LOSS_FULL, LOSS_SAMPLED)

ML1M_TUNE_CAP_SEC = 6 * 3600.0
# 21h. One warmed full-softmax epoch on this host is about two hours, so an
# 8h cap cannot both train that loss for more than two epochs and tune the
# sampled-softmax alternative. The shrink below picks the first grid that fits.
ML32M_TUNE_CAP_SEC = 21 * 3600.0
TEST_BOTH_CAP_SEC = 6 * 3600.0
THREE_SEED_CAP_SEC = 4 * 3600.0

# Fixed sampled-softmax width. It is not a tuned axis.
N_NEGATIVES = 256


def _base(
    loss: str, dim: int, lr: float, temp: float, max_epochs: int, patience: int
) -> dict[str, Any]:
    hp: dict[str, Any] = {
        "batch_size": 1024,
        "embedding_dim": int(dim),
        "learning_rate": float(lr),
        "loss": loss,
        "max_epochs": int(max_epochs),
        "max_history": 50,
        "patience": int(patience),
        "temperature": float(temp),
        "weight_decay": 0.0001,
    }
    if loss == LOSS_SAMPLED:
        hp["n_negatives"] = N_NEGATIVES
    return hp


def factorial(
    *,
    losses: tuple[str, ...] = LOSSES,
    dims: tuple[int, ...] = (32, 64),
    learning_rates: tuple[float, ...] = (0.0003, 0.001, 0.003),
    temperatures: tuple[float, ...] = (0.05, 0.1, 0.2),
    max_epochs: int,
    patience: int,
) -> list[dict[str, Any]]:
    grid = []
    for loss, dim, lr, temp in itertools.product(losses, dims, learning_rates, temperatures):
        grid.append(
            _base(loss, dim, lr, temp, max_epochs, patience)
        )
    return grid


def one_factor(
    *,
    losses: tuple[str, ...] = LOSSES,
    max_epochs: int,
    patience: int,
) -> list[dict[str, Any]]:
    """Reference point plus one step on learning rate, temperature, and dim."""
    points = (
        (64, 0.0003, 0.1),
        (64, 0.001, 0.1),
        (64, 0.003, 0.1),
        (64, 0.001, 0.05),
        (64, 0.001, 0.2),
        (32, 0.001, 0.1),
    )
    grid = []
    for loss in losses:
        for dim, lr, temp in points:
            grid.append(_base(loss, dim, lr, temp, max_epochs, patience))
    return grid


def extrapolation_cap_sec(
    grid: list[dict[str, Any]], epoch_sec: dict[str, float], val_sec: float
) -> float:
    """Worst-case tuning seconds: every trial runs ``max_epochs`` with no early stop."""
    total = 0.0
    for hp in grid:
        total += (float(epoch_sec[hp["loss"]]) + float(val_sec)) * int(hp["max_epochs"])
    return total


def grid_edges(hp: dict[str, Any], grid: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flag axes where ``hp`` sits on the edge of the same-loss slice of ``grid``."""
    same = [row for row in grid if row["loss"] == hp["loss"]]
    edges = []
    for key in ("embedding_dim", "learning_rate", "temperature"):
        vals = sorted({float(row[key]) for row in same})
        if not vals:
            continue
        value = float(hp[key])
        at_low = value == vals[0]
        at_high = value == vals[-1]
        if not (at_low or at_high):
            continue
        if at_low and at_high:
            which = "only_value"
        elif at_low:
            which = "low"
        else:
            which = "high"
        edges.append(
            {
                "parameter": key,
                "value": value,
                "grid_values": vals,
                "edge": which,
            }
        )
    return edges


def _epoch_map(block: dict[str, Any]) -> dict[str, float]:
    losses = block["losses"]
    return {
        name: float(losses[name]["extrapolated_epoch_train_sec"])
        for name in ("in_batch", LOSS_FULL, LOSS_SAMPLED)
    }


def _shrink_ml1m(
    epoch_sec: dict[str, float], val_sec: float
) -> tuple[list[dict[str, Any]], list[str]]:
    steps: list[str] = []
    specs = [
        ((32, 64), (0.0003, 0.001, 0.003), (0.05, 0.1, 0.2), "full factorial"),
        ((32, 64), (0.0003, 0.001, 0.003), (0.05, 0.1), "dropped temperature 0.2"),
        ((64,), (0.0003, 0.001, 0.003), (0.05, 0.1), "dropped embedding_dim 32"),
        ((64,), (0.0003, 0.001), (0.05, 0.1), "dropped learning_rate 0.003"),
    ]
    chosen: list[dict[str, Any]] | None = None
    for dims, lrs, temps, label in specs:
        grid = factorial(
            dims=dims,
            learning_rates=lrs,
            temperatures=temps,
            max_epochs=20,
            patience=3,
        )
        cap = extrapolation_cap_sec(grid, epoch_sec, val_sec)
        steps.append(f"{label}: {len(grid)} trials, cap {cap:.1f}s")
        chosen = grid
        if cap <= ML1M_TUNE_CAP_SEC:
            break
    assert chosen is not None
    return chosen, steps


def _ml32m_balanced(
    full_epochs: int,
    full_patience: int,
    sampled_epochs: int,
    sampled_patience: int,
) -> list[dict[str, Any]]:
    """Two full-softmax learning rates plus five sampled-softmax points."""
    full = [
        _base(LOSS_FULL, 64, lr, 0.1, full_epochs, full_patience)
        for lr in (0.0003, 0.001)
    ]
    sampled_points = (
        (64, 0.0003, 0.1),
        (64, 0.001, 0.1),
        (64, 0.003, 0.1),
        (64, 0.001, 0.05),
        (32, 0.001, 0.1),
    )
    sampled = [
        _base(LOSS_SAMPLED, dim, lr, temp, sampled_epochs, sampled_patience)
        for dim, lr, temp in sampled_points
    ]
    return full + sampled


def _shrink_ml32m(
    epoch_sec: dict[str, float], val_sec: float
) -> tuple[list[dict[str, Any]], list[str]]:
    steps: list[str] = []
    factorial_grid = factorial(max_epochs=6, patience=2)
    cap = extrapolation_cap_sec(factorial_grid, epoch_sec, val_sec)
    steps.append(f"full factorial: {len(factorial_grid)} trials, cap {cap:.1f}s")
    if cap <= ML32M_TUNE_CAP_SEC:
        return factorial_grid, steps

    factor_grid = one_factor(max_epochs=6, patience=2)
    cap = extrapolation_cap_sec(factor_grid, epoch_sec, val_sec)
    steps.append(
        "one factor around dim 64 / lr 0.001 / temp 0.1: "
        f"{len(factor_grid)} trials, cap {cap:.1f}s"
    )
    if cap <= ML32M_TUNE_CAP_SEC:
        return factor_grid, steps

    # Keep every sampled-softmax one-factor point. Shrink only full softmax.
    sampled = [hp for hp in factor_grid if hp["loss"] == LOSS_SAMPLED]
    full = [
        _base(LOSS_FULL, dim, lr, temp, 6, 2)
        for dim, lr, temp in (
            (64, 0.0003, 0.1),
            (64, 0.001, 0.1),
            (64, 0.003, 0.1),
            (64, 0.001, 0.05),
            (32, 0.001, 0.1),
        )
    ]
    grid = full + sampled
    cap = extrapolation_cap_sec(grid, epoch_sec, val_sec)
    steps.append(f"full softmax reduced to 5, sampled softmax unchanged: cap {cap:.1f}s")
    if cap <= ML32M_TUNE_CAP_SEC:
        return grid, steps

    full = [
        _base(LOSS_FULL, 64, lr, 0.1, 6, 2)
        for lr in (0.0003, 0.001, 0.003)
    ]
    sampled = [
        hp for hp in factor_grid if hp["loss"] == LOSS_SAMPLED and float(hp["temperature"]) != 0.2
    ]
    grid = full + sampled
    cap = extrapolation_cap_sec(grid, epoch_sec, val_sec)
    steps.append(
        "full softmax keeps three learning rates at temperature 0.1; "
        f"sampled softmax drops temperature 0.2: cap {cap:.1f}s"
    )
    if cap <= ML32M_TUNE_CAP_SEC:
        return grid, steps

    # Prefer four full-softmax epochs over a two-epoch run. Temperature and
    # embedding dim stay at the published point for that loss; sampled softmax
    # keeps learning rate, one lower temperature, and embedding dim 32.
    balanced = (
        (4, 2, 4, 2),
        (4, 2, 3, 1),
        (3, 1, 4, 2),
        (3, 1, 3, 1),
    )
    for full_epochs, full_patience, sampled_epochs, sampled_patience in balanced:
        grid = _ml32m_balanced(full_epochs, full_patience, sampled_epochs, sampled_patience)
        cap = extrapolation_cap_sec(grid, epoch_sec, val_sec)
        steps.append(
            "balanced: full softmax two learning rates, "
            f"max_epochs {full_epochs}; sampled softmax five points, "
            f"max_epochs {sampled_epochs}: cap {cap:.1f}s"
        )
        if cap <= ML32M_TUNE_CAP_SEC:
            return grid, steps

    sampled = [
        hp for hp in factor_grid if hp["loss"] == LOSS_SAMPLED and float(hp["temperature"]) != 0.2
    ]
    for epochs, patience in ((6, 2), (4, 2), (3, 1), (2, 1)):
        full = [
            _base(LOSS_FULL, 64, lr, 0.1, epochs, patience)
            for lr in (0.0003, 0.001)
        ]
        grid = full + sampled
        cap = extrapolation_cap_sec(grid, epoch_sec, val_sec)
        steps.append(
            "full softmax keeps lr 0.0003 and 0.001 at dim 64, temperature 0.1, "
            f"max_epochs {epochs}: cap {cap:.1f}s"
        )
        if cap <= ML32M_TUNE_CAP_SEC:
            return grid, steps
    return grid, steps


def _test_plan(
    dataset: str,
    epoch_sec: dict[str, float],
    *,
    max_epochs: int,
    reference_epochs: int,
) -> dict[str, Any]:
    slower = max(epoch_sec[LOSS_FULL], epoch_sec[LOSS_SAMPLED])
    three_seed_sec = (slower * max_epochs + epoch_sec["in_batch"] * reference_epochs) * 3
    seeds = [42, 43, 44] if three_seed_sec <= THREE_SEED_CAP_SEC else [42]
    both_sec = (
        epoch_sec[LOSS_FULL] + epoch_sec[LOSS_SAMPLED]
    ) * max_epochs * len(seeds) + epoch_sec["in_batch"] * reference_epochs * len(seeds)
    test_both = both_sec <= TEST_BOTH_CAP_SEC
    return {
        "dataset": dataset,
        "seeds": seeds,
        "three_seed_cap_sec": round(three_seed_sec, 1),
        "three_seed_cap_limit_sec": THREE_SEED_CAP_SEC,
        "test_both_losses": test_both,
        "test_both_cap_sec": round(both_sec, 1),
        "test_both_cap_limit_sec": TEST_BOTH_CAP_SEC,
        "reference_epochs": int(reference_epochs),
        "test_epoch_cap": int(max_epochs),
    }


def lock_plan(probe: dict[str, Any], reference_epochs: dict[str, int]) -> dict[str, Any]:
    """Choose grids and seed counts from probe extrapolations.

    ``probe["datasets"]`` maps a dataset name to its timing block.
    ``reference_epochs`` is the published in-batch ``best_epoch`` per dataset.
    """
    datasets_out: dict[str, Any] = {}
    for dataset, block in probe["datasets"].items():
        epoch_sec = _epoch_map(block)
        val_sec = float(block.get("val_score_sec_per_epoch", 0.0))
        if dataset == "ml-1m":
            grid, steps = _shrink_ml1m(epoch_sec, val_sec)
            cap_limit = ML1M_TUNE_CAP_SEC
            max_epochs = 20
        elif dataset == "ml-32m":
            grid, steps = _shrink_ml32m(epoch_sec, val_sec)
            cap_limit = ML32M_TUNE_CAP_SEC
            max_epochs = 6
        else:
            raise ValueError(f"no S3e plan for dataset {dataset!r}")
        if dataset not in reference_epochs:
            raise ValueError(f"missing published reference epoch for {dataset}")
        cap = extrapolation_cap_sec(grid, epoch_sec, val_sec)
        datasets_out[dataset] = {
            "grid": grid,
            "n_trials": len(grid),
            "shrink_steps": steps,
            "tune_cap_sec": round(cap, 1),
            "tune_cap_limit_sec": cap_limit,
            "cap_within_limit": bool(cap <= cap_limit),
            "epoch_train_sec": {key: round(val, 1) for key, val in epoch_sec.items()},
            "val_score_sec_per_epoch": round(val_sec, 1),
            "test": _test_plan(
                dataset,
                epoch_sec,
                max_epochs=max_epochs,
                reference_epochs=int(reference_epochs[dataset]),
            ),
            "fixed": {
                "n_negatives": N_NEGATIVES,
                "batch_size": 1024,
                "weight_decay": 0.0001,
                "max_history": 50,
                "temperature_is": "tuned hyperparameter, not a learned parameter",
            },
        }
    return {
        "experiment": "s3e_full_softmax_two_tower",
        "status": "locked",
        "ranking_metrics_used": False,
        "datasets": datasets_out,
    }
