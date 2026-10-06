"""EASE^R: closed-form shallow autoencoder (Steck 2019)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movielens_recommender.baselines.common import (
    binary_user_item,
    mask_seen,
    recommend_from_scores,
)


class EASERecommender:
    """Embarrassingly Shallow Autoencoder (EASE^R).

    Binary user-item matrix ``X`` (any observed train rating is 1). With Gram
    matrix ``G = XᵀX`` and regularization ``λ`` (``l2``):

        P = (G + λ I)⁻¹
        B = I − P · diag(1 / diag(P))

    which sets ``diag(B) = 0``. Scores are ``X B``; already-seen items are
    excluded at recommend time.
    """

    def __init__(self, *, l2: float = 100.0) -> None:
        if l2 <= 0.0:
            raise ValueError(f"l2 must be positive, got {l2}")
        self.l2 = float(l2)
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._user_index: dict[int, int] = {}
        self._similarity: np.ndarray | None = None
        self._user_scores: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> EASERecommender:
        interactions, item_ids, user_index, _item_index = binary_user_item(train)
        self._item_ids = item_ids
        self._user_index = user_index

        gram = (interactions.T @ interactions).toarray()
        n_items = gram.shape[0]
        gram.flat[:: n_items + 1] += self.l2

        precision = np.linalg.inv(gram)
        diag = np.diag(precision).copy()
        if np.any(diag == 0.0):
            raise ValueError("EASE precision matrix has a zero diagonal entry.")
        # B = I − P @ diag(1/diag(P)); the identity cancels the diagonal.
        weights = -precision / diag[np.newaxis, :]
        np.fill_diagonal(weights, 0.0)
        self._similarity = np.ascontiguousarray(weights, dtype=np.float64)

        raw_scores = interactions @ self._similarity
        self._user_scores = mask_seen(np.asarray(raw_scores, dtype=np.float64), interactions)
        return self

    @property
    def similarity(self) -> np.ndarray:
        """Item-item weight matrix B, shape ``(n_items, n_items)``."""
        if self._similarity is None:
            raise RuntimeError("Call fit() before reading similarity.")
        return self._similarity

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0 or self._user_scores is None:
            return []
        uidx = self._user_index.get(int(user_id))
        if uidx is None:
            return []
        return recommend_from_scores(self._user_scores[uidx], self._item_ids, n)

    def hyperparams(self) -> dict:
        return {"l2": self.l2}
