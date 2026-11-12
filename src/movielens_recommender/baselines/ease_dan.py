"""EASE-DAN: Data-Adaptive Normalized Linear Autoencoder (Park et al., SIGIR 2025; ADR-0028)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg, sparse

from movielens_recommender.baselines.common import (
    binary_user_item,
    mask_seen,
    recommend_from_scores,
    top_n_indices,
)
from movielens_recommender.baselines.ease import _popular_item_ids
from movielens_recommender.scale import should_cache_user_scores

L2_VALUES: tuple[float, ...] = (10.0, 100.0, 500.0, 2000.0)
DROPOUT_VALUES: tuple[float, ...] = (0.0, 0.3, 0.6)
ALPHA_VALUES: tuple[float, ...] = (0.0, 0.2, 0.4)
BETA_VALUES: tuple[float, ...] = (0.0, 0.3, 0.6)

ORDERED_AXES: tuple[str, ...] = ("l2", "dropout_p", "alpha", "beta")

BUDGET_SEC = 3600
RESERVE_SEC = 600
SEARCH_WALL_SEC = 3000

CUT_L2_2000 = "dropped l2=2000.0"
CUT_BOTH = "dropped l2=2000.0 and dropout_p=0.6"


class EASEDANRecommender:
    """EASE^R equipped with Data-Adaptive Normalization (DAN).

    Binary user-item matrix ``X`` (any observed train rating is 1). With user
    degrees ``d_U = X 1`` and item degrees ``d_I = Xᵀ 1`` (clamped below at 1):

        G = Xᵀ diag(d_U^{-beta}) X
        Λ_ii = l2 + (dropout_p / (1 - dropout_p)) * d_{I,i}
        P = (G + diag(Λ))⁻¹
        B̃ = I − P · diag(1 / diag(P)),  diag(B̃) = 0
        B = diag(d_I^{alpha}) · B̃ · diag(d_I^{-alpha}),  diag(B) = 0

    When ``alpha = 0``, ``beta = 0``, and ``dropout_p = 0``, ``B`` reduces
    exactly to :class:`EASERecommender` with regularization ``l2``.
    """

    def __init__(
        self,
        *,
        l2: float = 100.0,
        dropout_p: float = 0.3,
        alpha: float = 0.2,
        beta: float = 0.3,
        max_items: int | None = None,
        cache_user_scores: bool | None = None,
    ) -> None:
        if l2 <= 0.0:
            raise ValueError(f"l2 must be positive, got {l2}")
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(f"dropout_p must be in [0.0, 1.0), got {dropout_p}")
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if max_items is not None and int(max_items) <= 0:
            raise ValueError(f"max_items must be positive, got {max_items}")
        self.l2 = float(l2)
        self.dropout_p = float(dropout_p)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_items = None if max_items is None else int(max_items)
        self.cache_user_scores = cache_user_scores
        self.n_items_before_restriction = 0
        self.n_items_fit = 0
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._item_index: dict[int, int] = {}
        self._user_index: dict[int, int] = {}
        self._similarity: np.ndarray | None = None
        self._user_scores: np.ndarray | None = None
        self._interactions = None

    def fit(self, train: pd.DataFrame) -> EASEDANRecommender:
        self.n_items_before_restriction = int(train["item_id"].nunique())
        fit_frame = train
        if self.max_items is not None and self.n_items_before_restriction > self.max_items:
            keep = _popular_item_ids(train, self.max_items)
            fit_frame = train.loc[train["item_id"].isin(pd.Index(keep))]
        interactions, item_ids, user_index, item_index = binary_user_item(fit_frame)
        self._item_ids = item_ids
        self._item_index = item_index
        self._user_index = user_index
        self.n_items_fit = int(len(item_ids))

        n_users, n_items = interactions.shape
        item_counts = np.maximum(
            np.asarray(interactions.sum(axis=0)).ravel().astype(np.float64), 1.0
        )
        if self.beta > 0.0:
            user_counts = np.maximum(
                np.asarray(interactions.sum(axis=1)).ravel().astype(np.float64), 1.0
            )
            weighted_x = sparse.diags(np.power(user_counts, -self.beta)) @ interactions
            gram = (weighted_x.T @ interactions).toarray()
        else:
            gram = (interactions.T @ interactions).toarray()

        diag_reg = np.full(n_items, self.l2, dtype=np.float64)
        if self.dropout_p > 0.0:
            diag_reg = diag_reg + (self.dropout_p / (1.0 - self.dropout_p)) * item_counts
        gram.flat[:: n_items + 1] += diag_reg

        if n_items >= 4000:
            precision = linalg.inv(gram, overwrite_a=True, check_finite=False)
        else:
            precision = np.linalg.inv(gram)
        diag = np.diag(precision).copy()
        if np.any(diag == 0.0):
            raise ValueError("EASE-DAN precision matrix has a zero diagonal entry.")

        weights = -precision / diag[np.newaxis, :]
        np.fill_diagonal(weights, 0.0)
        if self.alpha > 0.0:
            scale = np.power(item_counts, self.alpha)
            weights = weights * scale[:, np.newaxis] * (1.0 / scale)[np.newaxis, :]
            np.fill_diagonal(weights, 0.0)
        self._similarity = np.ascontiguousarray(weights, dtype=np.float64)

        if should_cache_user_scores(n_users, n_items, self.cache_user_scores):
            raw_scores = interactions @ self._similarity
            self._user_scores = mask_seen(np.asarray(raw_scores, dtype=np.float64), interactions)
            self._interactions = None
        else:
            self._user_scores = None
            self._interactions = interactions
        return self

    @property
    def similarity(self) -> np.ndarray:
        """Item-item weight matrix B, shape ``(n_items, n_items)``."""
        if self._similarity is None:
            raise RuntimeError("Call fit() before reading similarity.")
        return self._similarity

    def topk_for_profiles(
        self,
        profiles: Mapping[int, Sequence[tuple[int, float]]],
        n: int,
        *,
        mask_items: Mapping[int, Sequence[int]] | None = None,
    ) -> dict[int, list[tuple[int, float]]]:
        """Fold a new binary profile in through ``B`` and return top-n."""
        if n <= 0 or self._similarity is None or not profiles:
            return {}
        uids = [int(uid) for uid in profiles]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row, uid in enumerate(uids):
            seen_cols: set[int] = set()
            for item_id, _rating in profiles[uid]:
                col = self._item_index.get(int(item_id))
                if col is None or col in seen_cols:
                    continue
                seen_cols.add(col)
                rows.append(row)
                cols.append(col)
                data.append(1.0)
        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(uids), len(self._item_ids)),
            dtype=np.float64,
        )
        scores = np.asarray(mat @ self._similarity, dtype=np.float64)
        seen_rows, seen_cols_idx = mat.nonzero()
        if len(seen_rows):
            scores[seen_rows, seen_cols_idx] = -np.inf
        if mask_items:
            row_of = {uid: row for row, uid in enumerate(uids)}
            for uid, items in mask_items.items():
                row = row_of.get(int(uid))
                if row is None:
                    continue
                for item_id in items:
                    col = self._item_index.get(int(item_id))
                    if col is not None:
                        scores[row, col] = -np.inf
        out: dict[int, list[tuple[int, float]]] = {}
        for row, uid in enumerate(uids):
            order = top_n_indices(scores[row], n)
            out[uid] = [
                (int(self._item_ids[j]), float(scores[row, j]))
                for j in order
                if np.isfinite(scores[row, j])
            ]
        return out

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0:
            return []
        uidx = self._user_index.get(int(user_id))
        if uidx is None:
            return []
        if self._user_scores is not None:
            return recommend_from_scores(self._user_scores[uidx], self._item_ids, n)
        if self._interactions is None or self._similarity is None:
            return []
        row = self._interactions.getrow(uidx)
        scores = np.asarray(row @ self._similarity, dtype=np.float64).ravel()
        if row.nnz:
            scores[row.indices] = -np.inf
        return recommend_from_scores(scores, self._item_ids, n)

    def topk_with_scores(
        self,
        n: int,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, list[tuple[int, float]]]:
        """Batched top-n ``(item_id, score)`` for fit users, excluding seen train items."""
        if n <= 0 or self._similarity is None:
            return {}
        if user_ids is None:
            wanted = list(self._user_index.keys())
        else:
            wanted = [int(uid) for uid in user_ids if int(uid) in self._user_index]
        if not wanted:
            return {}
        out: dict[int, list[tuple[int, float]]] = {}
        if self._user_scores is not None:
            for uid in wanted:
                row_scores = self._user_scores[self._user_index[uid]]
                order = top_n_indices(row_scores, n)
                out[uid] = [
                    (int(self._item_ids[int(j)]), float(row_scores[int(j)]))
                    for j in order
                    if np.isfinite(row_scores[int(j)])
                ]
            return out
        if self._interactions is None:
            return {}
        chunk = 256
        for start in range(0, len(wanted), chunk):
            batch = wanted[start : start + chunk]
            idx = np.asarray([self._user_index[uid] for uid in batch], dtype=np.int64)
            sub = self._interactions[idx]
            scores = np.asarray(sub @ self._similarity, dtype=np.float64)
            seen_r, seen_c = sub.nonzero()
            if len(seen_r):
                scores[seen_r, seen_c] = -np.inf
            for r_i, uid in enumerate(batch):
                order = top_n_indices(scores[r_i], n)
                out[uid] = [
                    (int(self._item_ids[int(j)]), float(scores[r_i, int(j)]))
                    for j in order
                    if np.isfinite(scores[r_i, int(j)])
                ]
        return out

    def hyperparams(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "l2": self.l2,
            "dropout_p": self.dropout_p,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        if self.max_items is not None:
            out["max_items"] = int(self.max_items)
        if self.n_items_fit:
            out["n_items_before_restriction"] = int(self.n_items_before_restriction)
            out["n_items_fit"] = int(self.n_items_fit)
        return out


def ease_dan_grid() -> list[dict[str, float]]:
    """108-config base grid in ADR-0028 tie-break order."""
    return [
        {
            "l2": float(l2),
            "dropout_p": float(dropout_p),
            "alpha": float(alpha),
            "beta": float(beta),
        }
        for l2, dropout_p, alpha, beta in itertools.product(
            L2_VALUES, DROPOUT_VALUES, ALPHA_VALUES, BETA_VALUES
        )
    ]


def config_key(cfg: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    """Deterministic key for deduplicating grid configs."""
    return tuple(
        (axis, round(float(cfg[axis]), 6))
        for axis in ORDERED_AXES
        if axis in cfg
    )


def lock_grid(t_probe_sec: float) -> tuple[list[dict[str, float]], str | None, float]:
    """Apply ADR-0028 budget cuts from the timing probe."""
    grid = ease_dan_grid()
    projected = len(grid) * float(t_probe_sec)
    if projected + RESERVE_SEC <= BUDGET_SEC:
        return grid, None, projected
    grid = [cfg for cfg in grid if float(cfg["l2"]) != 2000.0]
    projected = len(grid) * float(t_probe_sec)
    if projected + RESERVE_SEC <= BUDGET_SEC:
        return grid, CUT_L2_2000, projected
    grid = [cfg for cfg in grid if float(cfg["dropout_p"]) != 0.6]
    projected = len(grid) * float(t_probe_sec)
    return grid, CUT_BOTH, projected


def find_grid_edges(
    hyperparams: Mapping[str, Any], configs: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Edges of ordered axes. Natural zero boundaries on dropout_p/alpha/beta are not edges."""
    edges: list[dict[str, Any]] = []
    for axis in ORDERED_AXES:
        if axis not in hyperparams:
            continue
        values = [float(cfg[axis]) for cfg in configs if axis in cfg]
        if len(set(values)) < 2:
            continue
        lo = min(values)
        hi = max(values)
        val = float(hyperparams[axis])
        if val == lo and axis == "l2":
            edges.append({"parameter": axis, "side": "low", "value": val})
        elif val == hi:
            edges.append({"parameter": axis, "side": "high", "value": val})
    return edges


def extension_value(axis: str, side: str, value: float) -> float | None:
    """One pre-registered step past an extendable edge (ADR-0028)."""
    if axis == "l2":
        if side == "low":
            return 2.0 if value > 2.0 else None
        if side == "high":
            return 5000.0 if value < 5000.0 else float(value * 2.5)
    if side != "high":
        return None
    if axis == "dropout_p":
        return 0.8 if value < 0.8 else None
    if axis == "alpha":
        return round(value + 0.2, 6)
    if axis == "beta":
        return round(value + 0.3, 6)
    return None


def edge_extension_configs(
    hyperparams: Mapping[str, Any], configs: Sequence[Mapping[str, Any]]
) -> list[dict[str, float]]:
    """At most four one-step edge configs, holding other axes at the winner."""
    known = {config_key(cfg) for cfg in configs}
    extensions: list[dict[str, float]] = []
    for edge in find_grid_edges(hyperparams, configs):
        stepped = extension_value(edge["parameter"], edge["side"], float(edge["value"]))
        if stepped is None:
            continue
        row = {axis: float(hyperparams[axis]) for axis in ORDERED_AXES}
        row[edge["parameter"]] = float(stepped)
        if config_key(row) in known:
            continue
        known.add(config_key(row))
        extensions.append(row)
    return extensions


def select_best(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Highest ``val_ndcg@10``; ties keep the earlier trial."""
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
