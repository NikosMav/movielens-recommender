"""RP3beta: 3-step random-walk item similarity (Christoffel 2015; Paudel 2017)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from movielens_recommender.baselines.common import (
    binary_user_item,
    mask_seen,
    recommend_from_scores,
)


class RP3betaRecommender:
    """Item-item similarity from a 3-step walk on the user-item graph.

    ``X`` is the binary interaction matrix. ``P_ui`` is the row-normalized
    user→item transition and ``P_iu`` is the row-normalized item→user
    transition (boolean transpose). Both are raised elementwise to ``alpha``.
    The item-item matrix of the walk is ``P_iu @ P_ui`` (item→user→item);
    scoring ``X @ S`` is the third step from the user. Target items are then
    divided by ``degree(item) ** beta``, the diagonal is cleared, and each
    row keeps at most ``top_k`` neighbors.

    This follows the Dacrema / Anelli graph construction without the optional
    extra L1 renormalization of ``S``.
    """

    def __init__(self, *, alpha: float = 1.0, beta: float = 0.6, top_k: int = 100) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.top_k = int(top_k)
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._user_index: dict[int, int] = {}
        self._similarity: sparse.csr_matrix | None = None
        self._user_scores: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> RP3betaRecommender:
        interactions, item_ids, user_index, _item_index = binary_user_item(train)
        self._item_ids = item_ids
        self._user_index = user_index
        n_items = len(item_ids)

        p_ui = _l1_normalize_rows(interactions)
        # Boolean item→user: rating magnitudes do not change this transition.
        x_bool_t = interactions.T.tocsr().copy()
        x_bool_t.data = np.ones(x_bool_t.data.shape, dtype=np.float64)
        p_iu = _l1_normalize_rows(x_bool_t)

        if self.alpha != 1.0:
            p_ui = _power_stored(p_ui, self.alpha)
            p_iu = _power_stored(p_iu, self.alpha)

        item_degree = np.asarray(interactions.sum(axis=0)).ravel()
        penalty = np.zeros(n_items, dtype=np.float64)
        nonzero = item_degree > 0.0
        penalty[nonzero] = np.power(item_degree[nonzero], -self.beta)

        product = p_iu @ p_ui
        dense = product.toarray() if sparse.issparse(product) else np.asarray(product)
        dense = np.asarray(dense, dtype=np.float64)
        dense *= penalty  # scale columns: destination item degree^beta
        np.fill_diagonal(dense, 0.0)
        dense = _top_k_per_row(dense, self.top_k)

        self._similarity = sparse.csr_matrix(dense)
        raw_scores = interactions @ self._similarity
        if sparse.issparse(raw_scores):
            raw_scores = raw_scores.toarray()
        self._user_scores = mask_seen(np.asarray(raw_scores, dtype=np.float64), interactions)
        return self

    @property
    def similarity(self) -> np.ndarray:
        """Item-item weights, shape ``(n_items, n_items)``, at most ``top_k`` nnz/row."""
        if self._similarity is None:
            raise RuntimeError("Call fit() before reading similarity.")
        return self._similarity.toarray()

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0 or self._user_scores is None:
            return []
        uidx = self._user_index.get(int(user_id))
        if uidx is None:
            return []
        return recommend_from_scores(self._user_scores[uidx], self._item_ids, n)

    def hyperparams(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "top_k": self.top_k}


def _power_stored(mat: sparse.csr_matrix, alpha: float) -> sparse.csr_matrix:
    """Elementwise power on stored entries. ``alpha=0`` maps those entries to 1.

    SciPy refuses ``power(0)`` because a dense ``0**0`` fill would materialize
    the structural zeros. On this graph only existing edges are transitions.
    """
    if alpha == 0.0:
        out = mat.copy()
        out.data = np.ones(out.data.shape, dtype=np.float64)
        return out
    return mat.power(alpha)


def _l1_normalize_rows(mat: sparse.csr_matrix) -> sparse.csr_matrix:
    """Divide each row by its sum. Empty rows stay zero."""
    totals = np.asarray(mat.sum(axis=1)).ravel()
    totals[totals == 0.0] = 1.0
    return sparse.diags(1.0 / totals) @ mat


def _top_k_per_row(sim: np.ndarray, k: int) -> np.ndarray:
    """Keep the ``k`` largest entries per row. Non-positive values are dropped."""
    n_rows, n_cols = sim.shape
    if k >= n_cols:
        out = sim.copy()
        out[out <= 0.0] = 0.0
        return out
    out = np.zeros_like(sim)
    for i in range(n_rows):
        row = sim[i]
        part = np.argpartition(-row, k)[:k]
        keep = part[row[part] > 0.0]
        out[i, keep] = row[keep]
    return out
