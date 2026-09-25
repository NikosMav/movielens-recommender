"""Item-item cosine similarity collaborative filtering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


class ItemItemCosineRecommender:
    """Score candidate items by cosine similarity to the user's train items.

    Builds a user-item CSR matrix of ratings, L2-normalizes item columns, and
    computes the dense item-item Gram matrix ``S = X.T @ X`` (cosine since
    columns are unit-norm). For a user with train ratings ``r``, scores are
    ``S @ r`` (with train items zeroed so they are not re-recommended).

    For large catalogs this is O(|I|^2); fine for ml-latest-small / ml-1m.
    """

    def __init__(self, *, min_common: int = 1) -> None:
        self.min_common = min_common
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

        # Binary co-occurrence for optional min_common filter.
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
