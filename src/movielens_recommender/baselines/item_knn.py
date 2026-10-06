"""Item-item cosine similarity collaborative filtering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


class ItemItemCosineRecommender:
    """Score candidate items by cosine similarity to the user's train items.

    Builds a user-item CSR matrix of ratings, L2-normalizes item columns, and
    computes the dense item-item Gram matrix ``S = X.T @ X`` (cosine since
    columns are unit-norm). Optional shrinkage and neighbourhood truncation:

    - ``shrinkage``: ``sim' = (n_common / (n_common + shrinkage)) * sim``
    - ``k_neighbors``: keep only the top-k similar items per item (0 = all)

    For a user with train ratings ``r``, scores are ``S @ r`` (with train items
    zeroed so they are not re-recommended).

    For large catalogs this is O(|I|^2); fine for ml-latest-small / ml-1m.
    """

    def __init__(
        self,
        *,
        min_common: int = 1,
        k_neighbors: int = 0,
        shrinkage: float = 0.0,
    ) -> None:
        self.min_common = min_common
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._item_index: dict[int, int] = {}
        self._user_index: dict[int, int] = {}
        self._similarity: np.ndarray | None = None
        self._user_item: sparse.csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> ItemItemCosineRecommender:
        users = np.sort(train["user_id"].unique())
        items = np.sort(train["item_id"].unique())
        self._user_index = {int(u): i for i, u in enumerate(users)}
        self._item_index = {int(it): i for i, it in enumerate(items)}
        self._item_ids = items.astype(np.int64)

        rows = train["user_id"].map(self._user_index).to_numpy()
        cols = train["item_id"].map(self._item_index).to_numpy()
        data = train["rating"].to_numpy(dtype=np.float64)

        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(items)),
            dtype=np.float64,
        )
        self._user_item = mat

        # Binary co-occurrence for min_common filter and shrinkage.
        binary = mat.copy()
        binary.data = np.ones_like(binary.data)
        common = (binary.T @ binary).toarray()

        # Column-normalize for cosine.
        norms = np.sqrt(mat.power(2).sum(axis=0)).A1
        norms[norms == 0.0] = 1.0
        mat_norm = mat @ sparse.diags(1.0 / norms)
        sim = (mat_norm.T @ mat_norm).toarray()
        np.fill_diagonal(sim, 0.0)
        if self.min_common > 1:
            sim[common < self.min_common] = 0.0
        if self.shrinkage > 0.0:
            sim = sim * (common / (common + self.shrinkage))
            np.fill_diagonal(sim, 0.0)

        if self.k_neighbors > 0 and self.k_neighbors < sim.shape[0]:
            sim = _top_k_neighbors(sim, self.k_neighbors)

        self._similarity = sim.astype(np.float64)
        return self

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0 or self._similarity is None or self._user_item is None:
            return []
        uidx = self._user_index.get(int(user_id))
        if uidx is None:
            # Cold-start user: no personalization; empty (eval filters/handles).
            return []

        user_vec = self._user_item.getrow(uidx).toarray().ravel()
        scores = self._similarity @ user_vec
        # Never recommend items the user already interacted with in train.
        scores[user_vec != 0] = -np.inf

        if n >= len(scores):
            order = np.argsort(-scores, kind="mergesort")
        else:
            # Partial top-n then stable sort those.
            part = np.argpartition(-scores, n - 1)[:n]
            order = part[np.argsort(-scores[part], kind="mergesort")]
        return [int(self._item_ids[i]) for i in order[:n] if np.isfinite(scores[i])]

    def topk_with_scores(self, n: int) -> dict[int, list[tuple[int, float]]]:
        """Batched top-n ``(item_id, score)`` for every fit user.

        Same score as :meth:`recommend` (``S @ r``, seen items masked). The
        matmul is batched; ties use a stable mergesort, matching ``recommend``.
        """
        if n <= 0 or self._similarity is None or self._user_item is None:
            return {}
        # scores[u, i] = (S @ r_u)_i, with S stored row-wise (not always symmetric).
        scores = np.asarray(self._user_item @ self._similarity.T, dtype=np.float64)
        seen_rows, seen_cols = self._user_item.nonzero()
        scores[seen_rows, seen_cols] = -np.inf

        inv_users = np.empty(len(self._user_index), dtype=np.int64)
        for uid, uidx in self._user_index.items():
            inv_users[uidx] = int(uid)

        out: dict[int, list[tuple[int, float]]] = {}
        for uidx in range(scores.shape[0]):
            chosen = _topk_indices(scores[uidx], n)
            uid = int(inv_users[uidx])
            out[uid] = [
                (int(self._item_ids[j]), float(scores[uidx, j]))
                for j in chosen
                if np.isfinite(scores[uidx, j])
            ]
        return out

    def topk_for_profiles(
        self,
        profiles: Mapping[int, Sequence[tuple[int, float]]],
        n: int,
        *,
        mask_items: Mapping[int, Sequence[int]] | None = None,
    ) -> dict[int, list[tuple[int, float]]]:
        """Top-n from explicit rating profiles, using the fitted item similarity.

        ``profiles`` maps user id to ``(item_id, rating)`` pairs. Items outside
        the fit catalog are ignored. ``mask_items`` are set to -inf before the
        top-n so a truncated profile can still hide the rest of the user's
        train history. Ties use the same stable mergesort as :meth:`recommend`.
        """
        if n <= 0 or self._similarity is None or not profiles:
            return {}
        uids = [int(uid) for uid in profiles]
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row, uid in enumerate(uids):
            for item_id, rating in profiles[uid]:
                col = self._item_index.get(int(item_id))
                if col is None:
                    continue
                rows.append(row)
                cols.append(col)
                data.append(float(rating))
        mat = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(uids), len(self._item_ids)),
            dtype=np.float64,
        )
        scores = np.asarray(mat @ self._similarity.T, dtype=np.float64)
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
            chosen = _topk_indices(scores[row], n)
            out[uid] = [
                (int(self._item_ids[j]), float(scores[row, j]))
                for j in chosen
                if np.isfinite(scores[row, j])
            ]
        return out

    def hyperparams(self) -> dict:
        return {
            "min_common": self.min_common,
            "k_neighbors": self.k_neighbors,
            "shrinkage": self.shrinkage,
        }

    def topk_for_user(self, user_id: int, n: int) -> list[tuple[int, float]]:
        """Top-n ``(item_id, score)`` for one fit user.

        Same score as :meth:`recommend` (``S @ r``, seen items masked). Ties
        use a stable mergesort.
        """
        if n <= 0 or self._similarity is None or self._user_item is None:
            return []
        uidx = self._user_index.get(int(user_id))
        if uidx is None:
            return []
        user_vec = self._user_item.getrow(uidx).toarray().ravel()
        scores = self._similarity @ user_vec
        scores[user_vec != 0] = -np.inf
        chosen = _topk_indices(scores, n)
        return [
            (int(self._item_ids[j]), float(scores[j]))
            for j in chosen
            if np.isfinite(scores[j])
        ]

    def neighbor_similarities(
        self, item_id: int, others: Sequence[int]
    ) -> dict[int, float]:
        """Similarity of ``item_id`` to each catalog id in ``others``.

        This is the row of ``S`` that :meth:`recommend` dots with the user's
        ratings: ``S[candidate, history]``. Ids outside the fit catalog are
        omitted. The item itself is omitted.
        """
        if self._similarity is None:
            return {}
        row = self._item_index.get(int(item_id))
        if row is None:
            return {}
        out: dict[int, float] = {}
        for other in others:
            oid = int(other)
            if oid == int(item_id):
                continue
            col = self._item_index.get(oid)
            if col is None:
                continue
            out[oid] = float(self._similarity[row, col])
        return out


def _topk_indices(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the top-n finite scores, ties broken by mergesort."""
    finite = np.isfinite(scores)
    if not finite.any():
        return np.array([], dtype=np.int64)
    idx = np.flatnonzero(finite)
    sc = scores[idx]
    if n >= len(idx):
        order = np.argsort(-sc, kind="mergesort")
        return idx[order]
    part = np.argpartition(-sc, n - 1)[:n]
    order = part[np.argsort(-sc[part], kind="mergesort")]
    return idx[order]


def _top_k_neighbors(sim: np.ndarray, k: int) -> np.ndarray:
    """Zero all but the top-k absolute similarities per row (stable ties)."""
    n = sim.shape[0]
    out = np.zeros_like(sim)
    # argpartition on -sim keeps largest similarities.
    # For each row, keep top-k; ties broken by original column index via mergesort.
    for i in range(n):
        row = sim[i]
        if k >= n:
            out[i] = row
            continue
        part = np.argpartition(-row, k)[:k]
        # Stable order among the selected for determinism of which ties survive
        # is not required for scoring; keep the partitioned top-k values.
        out[i, part] = row[part]
    return out
