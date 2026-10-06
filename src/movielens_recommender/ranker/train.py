"""LightGBM LambdaRank training, early stopping, and fixed-round refit.

The early-stopped run records ``best_iteration``. A refit trains a new booster
for exactly that many rounds on the requested rows, with no validation callback.
Test labels are not an input to this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

RANKER_OBJECTIVE = "lambdarank"


def require_lightgbm():
    """Import LightGBM or raise an install hint for the ``[rank]`` extra."""
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "LightGBM is required for the ranker. "
            "Install with: pip install -e '.[rank]' (pins lightgbm==4.6.0)."
        ) from exc
    return lgb


def lgb_params(
    *,
    seed: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
) -> dict[str, Any]:
    """LambdaRank parameters. ``seed`` is the only per-run stochastic input."""
    return {
        "objective": RANKER_OBJECTIVE,
        "metric": "ndcg",
        "eval_at": [10],
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "feature_fraction": float(feature_fraction),
        "bagging_fraction": float(bagging_fraction),
        "bagging_freq": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": int(seed),
        "bagging_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "data_random_seed": int(seed),
    }


def split_users(
    user_ids: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split group-level user ids into (train, early-stop).

    ``seed`` is the experiment seed, not the ranker seed, so every ranker seed
    shares the same held-out users. At least one user is held out and at least
    one user remains for training.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("early-stop fraction must be in (0, 1)")
    users = np.unique(np.asarray(user_ids, dtype=np.int64))
    if len(users) < 2:
        raise ValueError("early stopping needs at least two users")
    rng = np.random.default_rng(int(seed))
    order = users.copy()
    rng.shuffle(order)
    n_es = int(round(len(order) * float(fraction)))
    n_es = min(max(n_es, 1), len(order) - 1)
    early = np.sort(order[:n_es])
    train = np.sort(order[n_es:])
    return train, early


def validate_groups(y: np.ndarray, group: np.ndarray) -> None:
    """LambdaRank groups must be positive and sum to the number of rows."""
    if y.ndim != 1:
        raise ValueError("labels must be a 1-d array")
    if group.ndim != 1 or len(group) == 0:
        raise ValueError("group must be a non-empty 1-d array of group sizes")
    if np.any(group <= 0):
        raise ValueError("group sizes must be positive")
    if int(group.sum()) != len(y):
        raise ValueError(
            f"group sizes sum to {int(group.sum())} but there are {len(y)} rows"
        )


@dataclass
class RankerMatrices:
    """Contiguous per-user blocks for LambdaRank."""

    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    user_ids: np.ndarray  # one id per group, same order as ``group``
    feature_names: list[str]

    def subset(self, keep: set[int]) -> RankerMatrices:
        """Keep groups whose user id is in ``keep``, preserving order."""
        mask = np.array([int(uid) in keep for uid in self.user_ids], dtype=bool)
        if not mask.any():
            return RankerMatrices(
                x=np.zeros((0, self.x.shape[1]), dtype=self.x.dtype),
                y=np.zeros(0, dtype=self.y.dtype),
                group=np.zeros(0, dtype=self.group.dtype),
                user_ids=np.zeros(0, dtype=self.user_ids.dtype),
                feature_names=list(self.feature_names),
            )
        sizes = self.group
        # Row spans of the kept groups.
        starts = np.cumsum(sizes) - sizes
        rows: list[np.ndarray] = []
        for i, take in enumerate(mask):
            if take:
                start = int(starts[i])
                rows.append(np.arange(start, start + int(sizes[i])))
        idx = np.concatenate(rows)
        return RankerMatrices(
            x=self.x[idx],
            y=self.y[idx],
            group=self.group[mask],
            user_ids=self.user_ids[mask],
            feature_names=list(self.feature_names),
        )


def assemble_matrices(
    blocks: list[tuple[int, np.ndarray, np.ndarray]],
    feature_names: list[str],
) -> RankerMatrices:
    """Stack ``(user_id, X, y)`` blocks that each contain both labels.

    Blocks are sorted by user id so group order is deterministic. A block is
    dropped when every label is the same (LambdaRank has no pairwise signal)
    or when it has fewer than two rows.
    """
    usable = []
    for uid, x, y in blocks:
        y = np.asarray(y, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if len(y) < 2 or float(y.min()) == float(y.max()):
            continue
        usable.append((int(uid), x, y))
    if not usable:
        raise ValueError("no ranker groups with both a positive and a negative label")
    usable.sort(key=lambda row: row[0])
    xs = [row[1] for row in usable]
    ys = [row[2] for row in usable]
    return RankerMatrices(
        x=np.vstack(xs),
        y=np.concatenate(ys),
        group=np.asarray([len(y) for y in ys], dtype=np.int32),
        user_ids=np.asarray([uid for uid, _, _ in usable], dtype=np.int64),
        feature_names=list(feature_names),
    )


def fit_lambdarank(
    matrices: RankerMatrices,
    *,
    params: dict[str, Any],
    num_boost_round: int,
    valid: RankerMatrices | None = None,
    early_stopping_rounds: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train LambdaRank.

    With ``valid`` and ``early_stopping_rounds``, stop on validation NDCG and
    record ``best_iteration``. Without them, train exactly ``num_boost_round``
    trees and do not consult a validation set.
    """
    lgb = require_lightgbm()
    if num_boost_round < 1:
        raise ValueError("num_boost_round must be >= 1")
    validate_groups(matrices.y, matrices.group)
    train_set = lgb.Dataset(
        matrices.x,
        label=matrices.y,
        group=[int(g) for g in matrices.group.tolist()],
        feature_name=list(matrices.feature_names),
        free_raw_data=False,
    )
    callbacks = []
    valid_sets = None
    valid_names = None
    if early_stopping_rounds is not None:
        if valid is None:
            raise ValueError("early stopping requires a validation matrix")
        validate_groups(valid.y, valid.group)
        valid_set = lgb.Dataset(
            valid.x,
            label=valid.y,
            group=[int(g) for g in valid.group.tolist()],
            feature_name=list(valid.feature_names),
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets = [valid_set]
        valid_names = ["early_stop"]
        callbacks.append(
            lgb.early_stopping(int(early_stopping_rounds), first_metric_only=True, verbose=False)
        )
    callbacks.append(lgb.log_evaluation(period=0))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(num_boost_round),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    best = int(getattr(booster, "best_iteration", 0) or 0)
    if early_stopping_rounds is not None and best < 1:
        best = 1
    info = {
        "objective": RANKER_OBJECTIVE,
        "num_boost_round": int(num_boost_round),
        "early_stopping": early_stopping_rounds is not None,
        "early_stopping_rounds": (
            None if early_stopping_rounds is None else int(early_stopping_rounds)
        ),
        "best_iteration": best if early_stopping_rounds is not None else int(num_boost_round),
        "num_trees": int(booster.num_trees()),
        "n_rows": int(len(matrices.y)),
        "n_groups": int(len(matrices.group)),
        "group_sizes": [int(g) for g in matrices.group.tolist()],
        "feature_names": list(matrices.feature_names),
    }
    return booster, info


@dataclass
class RefitResult:
    """Early-stopped iteration count plus the fixed-round refit booster."""

    booster: Any
    best_iteration: int
    n_train_users: int
    n_early_stop_users: int
    early_stop_info: dict[str, Any]
    refit_info: dict[str, Any]


def train_with_early_stop_then_refit(
    matrices: RankerMatrices,
    *,
    split_seed: int,
    early_stop_fraction: float,
    ranker_seed: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
) -> RefitResult:
    """Early-stop on held-out users, then refit on every user for that many rounds.

    The refit uses all groups in ``matrices`` (the whole label window that
    produced those groups). It passes ``num_boost_round=best_iteration`` and
    no validation set. ``split_seed`` chooses the user slice; ``ranker_seed``
    is the LightGBM seed.
    """
    train_users, es_users = split_users(
        matrices.user_ids, early_stop_fraction, split_seed
    )
    train = matrices.subset(set(int(u) for u in train_users))
    early = matrices.subset(set(int(u) for u in es_users))
    if len(train.group) == 0 or len(early.group) == 0:
        raise ValueError("early-stop split left one side without usable groups")
    params = lgb_params(
        seed=ranker_seed,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
    )
    _es_booster, es_info = fit_lambdarank(
        train,
        params=params,
        num_boost_round=num_boost_round,
        valid=early,
        early_stopping_rounds=early_stopping_rounds,
    )
    best = int(es_info["best_iteration"])
    if best < 1:
        best = 1
    refit, refit_info = fit_lambdarank(
        matrices,
        params=params,
        num_boost_round=best,
        valid=None,
        early_stopping_rounds=None,
    )
    return RefitResult(
        booster=refit,
        best_iteration=best,
        n_train_users=int(len(train.group)),
        n_early_stop_users=int(len(early.group)),
        early_stop_info=es_info,
        refit_info=refit_info,
    )


def train_fixed_rounds(
    matrices: RankerMatrices,
    *,
    ranker_seed: int,
    num_boost_round: int,
    learning_rate: float,
    num_leaves: int,
    min_data_in_leaf: int,
    feature_fraction: float,
    bagging_fraction: float,
) -> tuple[Any, dict[str, Any]]:
    """Train for a fixed round count. No early stopping and no validation set.

    Used when the ranker is retrained on a later window: ``num_boost_round``
    is the headline ``best_iteration``.
    """
    params = lgb_params(
        seed=ranker_seed,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
    )
    return fit_lambdarank(
        matrices,
        params=params,
        num_boost_round=int(num_boost_round),
        valid=None,
        early_stopping_rounds=None,
    )


def feature_importance_gain(booster: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    """Gain importance aligned to ``feature_names``, highest gain first."""
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
    model_names = list(booster.feature_name())
    if model_names and model_names != list(feature_names):
        # Prefer the booster's own names when it was trained with them.
        feature_names = model_names
    if len(gains) != len(feature_names):
        raise ValueError(
            f"importance length {len(gains)} != {len(feature_names)} feature names"
        )
    rows = [
        {"feature": name, "gain": round(float(gain), 6)}
        for name, gain in zip(feature_names, gains, strict=True)
    ]
    rows.sort(key=lambda row: (-row["gain"], row["feature"]))
    return rows
