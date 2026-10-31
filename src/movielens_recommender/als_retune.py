"""Validation grid and edge rule for the ml-1m implicit ALS retune (ADR-0018).

The grid, the budget cut, and the one-step edge extension are pure functions.
They do not read test labels.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from movielens_recommender.baselines.als import BM25_B, BM25_K1

# Base search. ADR-0018. Order is the tie-break order.
FACTORS: tuple[int, ...] = (64, 128, 256)
REGULARIZATION: tuple[float, ...] = (0.01, 0.1, 1.0)
ITERATIONS: tuple[int, ...] = (15, 30)
ALPHAS: tuple[float, ...] = (10.0, 20.0, 40.0)
CONFIDENCE_WITH_ALPHA: tuple[str, ...] = ("linear", "log")
CONFIDENCE_LIBRARY: tuple[str, ...] = ("bm25", "tfidf")

# Committed winner in results/tuning/ml-1m.json (als.best_hyperparams),
# plus the linear scheme that code already used.
OLD_ALS_HYPERPARAMS: dict[str, Any] = {
    "alpha": 20.0,
    "confidence": "linear",
    "factors": 128,
    "iterations": 15,
    "regularization": 0.1,
}

# Committed item_item_cosine defaults (results/ml-1m.json hyperparameters).
ITEM_ITEM_HYPERPARAMS: dict[str, Any] = {
    "k_neighbors": 0,
    "min_common": 1,
    "shrinkage": 0.0,
}

# Counted from results/tuning/ml-1m.json trials.
EASE_CONFIGS = 8
RP3BETA_CONFIGS = 125
ORIGINAL_ALS_CONFIGS = 12

BUDGET_SEC = 3 * 60 * 60
RESERVE_SEC = 1200
SEARCH_WALL_SEC = 9900

# Probe corners. factors * iterations.
PROBE_SMALL = {"factors": 64, "iterations": 15}
PROBE_LARGE = {"factors": 256, "iterations": 30}

ORDERED_AXES: tuple[str, ...] = ("factors", "regularization", "iterations", "alpha")

CUT_FACTORS_64 = "dropped factors=64"
CUT_BOTH = "dropped factors=64 and iterations=30"


def als_retune_grid() -> list[dict[str, Any]]:
    """144-config base grid. See ADR-0018 for the axis values."""
    configs: list[dict[str, Any]] = []
    for confidence in CONFIDENCE_WITH_ALPHA:
        for factors, regularization, iterations, alpha in itertools.product(
            FACTORS, REGULARIZATION, ITERATIONS, ALPHAS
        ):
            configs.append(
                {
                    "alpha": float(alpha),
                    "confidence": confidence,
                    "factors": int(factors),
                    "iterations": int(iterations),
                    "regularization": float(regularization),
                }
            )
    for confidence in CONFIDENCE_LIBRARY:
        for factors, regularization, iterations in itertools.product(
            FACTORS, REGULARIZATION, ITERATIONS
        ):
            row: dict[str, Any] = {
                "confidence": confidence,
                "factors": int(factors),
                "iterations": int(iterations),
                "regularization": float(regularization),
            }
            if confidence == "bm25":
                row["bm25_b"] = BM25_B
                row["bm25_k1"] = BM25_K1
            configs.append(row)
    return configs


def config_key(cfg: Mapping[str, Any]) -> tuple:
    """Identity of a grid row. Ignores key order. Floats compare at 6 decimals."""
    items = []
    for key, value in cfg.items():
        if isinstance(value, float) and value.is_integer():
            items.append((key, float(int(value))))
        elif isinstance(value, float):
            items.append((key, round(float(value), 6)))
        else:
            items.append((key, value))
    return tuple(sorted(items))


def is_old_als(cfg: Mapping[str, Any]) -> bool:
    """True when ``cfg`` is the committed tuned ALS config (linear, alpha 20)."""
    return (
        cfg.get("confidence", "linear") == "linear"
        and int(cfg.get("factors", -1)) == 128
        and int(cfg.get("iterations", -1)) == 15
        and float(cfg.get("alpha", -1)) == 20.0
        and float(cfg.get("regularization", -1)) == 0.1
    )


def _axis_values(
    configs: Sequence[Mapping[str, Any]], axis: str, confidence: str | None
) -> list[Any]:
    values = []
    for cfg in configs:
        if axis not in cfg:
            continue
        if axis == "alpha" and confidence is not None and cfg.get("confidence") != confidence:
            continue
        values.append(cfg[axis])
    return values


def find_grid_edges(
    hyperparams: Mapping[str, Any], configs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Edges of ordered axes. A one-value axis is not an edge."""
    edges: list[dict[str, Any]] = []
    confidence = str(hyperparams.get("confidence", "linear"))
    for axis in ORDERED_AXES:
        if axis not in hyperparams:
            continue
        values = _axis_values(configs, axis, confidence)
        if len(values) < 2:
            continue
        lo = min(values)
        hi = max(values)
        value = hyperparams[axis]
        if value == lo and value == hi:
            continue
        if value == lo:
            edges.append({"parameter": axis, "side": "low", "value": value})
        elif value == hi:
            edges.append({"parameter": axis, "side": "high", "value": value})
    return edges


def extension_value(axis: str, side: str, value: float) -> float | int | None:
    """One pre-registered step past an edge, or None when that side is not extended.

    The low side of iterations is not extended (ADR-0018).
    """
    if axis == "iterations" and side == "low":
        return None
    if axis in {"factors", "iterations"}:
        current = int(value)
        stepped = current * 2 if side == "high" else current // 2
        if stepped < 1 or stepped == current:
            return None
        return stepped
    factor = 10.0 if axis == "regularization" else 2.0
    if side == "low":
        factor = 1.0 / factor
    return float(f"{float(value) * factor:.6g}")


def edge_extension_configs(
    hyperparams: Mapping[str, Any], configs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """One new config per extendable edge. Other hyperparameters stay put."""
    known = {config_key(cfg) for cfg in configs}
    extensions: list[dict[str, Any]] = []
    for edge in find_grid_edges(hyperparams, configs):
        stepped = extension_value(edge["parameter"], edge["side"], edge["value"])
        if stepped is None:
            continue
        row = dict(hyperparams)
        row[edge["parameter"]] = stepped
        if config_key(row) in known:
            continue
        known.add(config_key(row))
        extensions.append(row)
    return extensions


def projected_seconds(
    grid: Sequence[Mapping[str, Any]], t_small: float, t_large: float
) -> float:
    """Sum of ``a + b * factors * iterations`` fit from the two probe corners.

    Corners are factors 64 / 15 iterations and factors 256 / 30 iterations.
    A negative intercept or slope falls back to scaling from the slower corner.
    """
    p_small = PROBE_SMALL["factors"] * PROBE_SMALL["iterations"]
    p_large = PROBE_LARGE["factors"] * PROBE_LARGE["iterations"]
    b = (float(t_large) - float(t_small)) / float(p_large - p_small)
    a = float(t_small) - b * p_small
    products = [int(cfg["factors"]) * int(cfg["iterations"]) for cfg in grid]
    if a < 0.0 or b < 0.0:
        if t_large >= t_small:
            rate = float(t_large) / float(p_large)
        else:
            rate = float(t_small) / float(p_small)
        return float(sum(rate * p for p in products))
    return float(sum(a + b * p for p in products))


def lock_grid(t_small: float, t_large: float) -> tuple[list[dict[str, Any]], str | None, float]:
    """Apply the ADR-0018 budget cuts. Returns grid, cut label, projected seconds."""
    grid = als_retune_grid()
    projected = projected_seconds(grid, t_small, t_large)
    if projected + RESERVE_SEC <= BUDGET_SEC:
        return grid, None, projected
    grid = [cfg for cfg in grid if int(cfg["factors"]) != 64]
    projected = projected_seconds(grid, t_small, t_large)
    if projected + RESERVE_SEC <= BUDGET_SEC:
        return grid, CUT_FACTORS_64, projected
    grid = [cfg for cfg in grid if int(cfg["iterations"]) != 30]
    projected = projected_seconds(grid, t_small, t_large)
    return grid, CUT_BOTH, projected


def batch_ndcg_point_estimate(
    model: Any,
    train: Any,
    test: Any,
    *,
    relevance_threshold: float,
    k: int,
    split: Any,
) -> float:
    """Mean NDCG@k from :meth:`ALSRecommender.recommend_many`.

    Same cold-start users as ``ndcg_point_estimate``. Seen items are dropped
    again, which is a no-op when ``recommend`` already removed them.
    """
    from movielens_recommender.evaluate import _prepare_eval
    from movielens_recommender.metrics import ndcg_at_k

    _prepared, seen, relevant = _prepare_eval(
        train, test, relevance_threshold=relevance_threshold, split=split
    )
    lists = model.recommend_many(list(relevant), k)
    values: list[float] = []
    for uid, rel in relevant.items():
        user_seen = seen.get(uid, set())
        ranked = [item for item in lists.get(uid, []) if item not in user_seen][:k]
        values.append(ndcg_at_k(ranked, rel, k))
    if not values:
        raise ValueError("No test users with warm relevant items to evaluate.")
    return float(sum(values) / len(values))


def trails_item_item(selected_ndcg: float, item_item_ndcg: float) -> bool:
    """Negative result: selected test NDCG@10 is strictly below item–item.

    Compared at the stored values the caller passes (6 decimals in the results
    file). A tie does not trail.
    """
    return float(selected_ndcg) < float(item_item_ndcg)


def select_best(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Highest ``val_ndcg@10``. Ties keep the earlier trial."""
    if not trials:
        raise ValueError("select_best requires at least one trial")
    best = trials[0]
    best_score = float(best["val_ndcg@10"])
    for trial in trials[1:]:
        score = float(trial["val_ndcg@10"])
        if score > best_score:
            best = trial
            best_score = score
    return dict(best)
