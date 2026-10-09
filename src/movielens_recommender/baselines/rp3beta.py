"""RP3beta: 3-step random-walk item similarity (Christoffel 2015; Paudel 2017)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from movielens_recommender.baselines.common import (
    binary_user_item,
    mask_seen,
    recommend_from_scores,
    top_n_indices,
)
from movielens_recommender.scale import SPARSE_CATALOG_ITEMS, should_cache_user_scores


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

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        beta: float = 0.6,
        top_k: int = 100,
        block_topk: bool | None = None,
        cache_user_scores: bool | None = None,
    ) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.top_k = int(top_k)
        self.block_topk = block_topk
        self.cache_user_scores = cache_user_scores
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._user_index: dict[int, int] = {}
        self._similarity: sparse.csr_matrix | None = None
        self._user_scores: np.ndarray | None = None
        self._interactions: sparse.csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> RP3betaRecommender:
        interactions, item_ids, user_index, _item_index = binary_user_item(train)
        self._item_ids = item_ids
        self._user_index = user_index
        n_items = len(item_ids)
        use_block = self.block_topk
        if use_block is None:
            use_block = n_items > SPARSE_CATALOG_ITEMS
        if use_block:
            self._fit_blocked(interactions)
            return self

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
        self._store_scores(interactions)
        return self

    def _fit_blocked(self, interactions: sparse.csr_matrix) -> None:
        """Same walk as :meth:`fit`, without a dense item-item matrix."""
        n_items = interactions.shape[1]
        p_ui = _l1_normalize_rows(interactions)
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

        row_bytes = max(n_items * 8, 1)
        block = max(1, min(n_items, (256 * 1024 * 1024) // row_bytes))
        if n_items > SPARSE_CATALOG_ITEMS:
            print(
                f"  RP3beta blocked top-k: n_items={n_items} top_k={self.top_k} "
                f"block_rows={block}",
                flush=True,
            )
        indptr = [0]
        indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        nnz = 0
        for start in range(0, n_items, block):
            end = min(start + block, n_items)
            dense = np.asarray((p_iu[start:end] @ p_ui).toarray(), dtype=np.float64)
            dense *= penalty
            for local, item in enumerate(range(start, end)):
                dense[local, item] = 0.0
            for local in range(end - start):
                kept = _positive_topk_indices(dense[local], self.top_k)
                indices.append(kept)
                data.append(dense[local, kept])
                nnz += len(kept)
                indptr.append(nnz)
        if nnz == 0:
            self._similarity = sparse.csr_matrix((n_items, n_items), dtype=np.float64)
        else:
            self._similarity = sparse.csr_matrix(
                (
                    np.concatenate(data).astype(np.float64, copy=False),
                    np.concatenate(indices).astype(np.int32, copy=False),
                    np.asarray(indptr, dtype=np.int32),
                ),
                shape=(n_items, n_items),
                dtype=np.float64,
            )
        self._store_scores(interactions)

    def _store_scores(self, interactions: sparse.csr_matrix) -> None:
        n_users, n_items = interactions.shape
        if should_cache_user_scores(n_users, n_items, self.cache_user_scores):
            raw_scores = interactions @ self._similarity
            if sparse.issparse(raw_scores):
                raw_scores = raw_scores.toarray()
            self._user_scores = mask_seen(np.asarray(raw_scores, dtype=np.float64), interactions)
            self._interactions = None
        else:
            self._user_scores = None
            self._interactions = interactions

    @property
    def similarity(self) -> np.ndarray:
        """Item-item weights, shape ``(n_items, n_items)``, at most ``top_k`` nnz/row."""
        if self._similarity is None:
            raise RuntimeError("Call fit() before reading similarity.")
        return self._similarity.toarray()

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
        product = row @ self._similarity
        if sparse.issparse(product):
            product = product.toarray()
        scores = np.asarray(product, dtype=np.float64).ravel()
        if row.nnz:
            scores[row.indices] = -np.inf
        return recommend_from_scores(scores, self._item_ids, n)

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


def _positive_topk_indices(row: np.ndarray, k: int) -> np.ndarray:
    """Column indices of the ``k`` largest positive entries, matching ``_top_k_per_row``."""
    n_cols = row.shape[0]
    if k >= n_cols:
        return np.flatnonzero(row > 0.0).astype(np.int32)
    part = np.sort(top_n_indices(row, k))
    return part[row[part] > 0.0].astype(np.int32)


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
        part = top_n_indices(row, k)
        keep = part[row[part] > 0.0]
        out[i, keep] = row[keep]
    return out
