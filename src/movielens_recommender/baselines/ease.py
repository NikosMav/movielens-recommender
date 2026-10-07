"""EASE^R: closed-form shallow autoencoder (Steck 2019)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import linalg

from movielens_recommender.baselines.common import (
    binary_user_item,
    mask_seen,
    recommend_from_scores,
)
from movielens_recommender.scale import should_cache_user_scores


class EASERecommender:
    """Embarrassingly Shallow Autoencoder (EASE^R).

    Binary user-item matrix ``X`` (any observed train rating is 1). With Gram
    matrix ``G = XᵀX`` and regularization ``λ`` (``l2``):

        P = (G + λ I)⁻¹
        B = I − P · diag(1 / diag(P))

    which sets ``diag(B) = 0``. Scores are ``X B``; already-seen items are
    excluded at recommend time.
    """

    def __init__(
        self,
        *,
        l2: float = 100.0,
        max_items: int | None = None,
        cache_user_scores: bool | None = None,
    ) -> None:
        if l2 <= 0.0:
            raise ValueError(f"l2 must be positive, got {l2}")
        if max_items is not None and int(max_items) <= 0:
            raise ValueError(f"max_items must be positive, got {max_items}")
        self.l2 = float(l2)
        self.max_items = None if max_items is None else int(max_items)
        self.cache_user_scores = cache_user_scores
        self.n_items_before_restriction = 0
        self.n_items_fit = 0
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._user_index: dict[int, int] = {}
        self._similarity: np.ndarray | None = None
        self._user_scores: np.ndarray | None = None
        self._interactions = None

    def fit(self, train: pd.DataFrame) -> EASERecommender:
        self.n_items_before_restriction = int(train["item_id"].nunique())
        fit_frame = train
        if self.max_items is not None and self.n_items_before_restriction > self.max_items:
            keep = _popular_item_ids(train, self.max_items)
            fit_frame = train.loc[train["item_id"].isin(pd.Index(keep))]
        interactions, item_ids, user_index, _item_index = binary_user_item(fit_frame)
        self._item_ids = item_ids
        self._user_index = user_index
        self.n_items_fit = int(len(item_ids))

        gram = (interactions.T @ interactions).toarray()
        n_items = gram.shape[0]
        gram.flat[:: n_items + 1] += self.l2

        # Small catalogs stay on numpy.linalg.inv so the closed-form test matches.
        # Larger catalogs use a LAPACK inverse that can reuse the Gram buffer.
        if n_items >= 4000:
            precision = linalg.inv(gram, overwrite_a=True, check_finite=False)
        else:
            precision = np.linalg.inv(gram)
        diag = np.diag(precision).copy()
        if np.any(diag == 0.0):
            raise ValueError("EASE precision matrix has a zero diagonal entry.")
        # B = I − P @ diag(1/diag(P)); the identity cancels the diagonal.
        weights = -precision / diag[np.newaxis, :]
        np.fill_diagonal(weights, 0.0)
        self._similarity = np.ascontiguousarray(weights, dtype=np.float64)

        n_users = interactions.shape[0]
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

    def hyperparams(self) -> dict:
        out: dict = {"l2": self.l2}
        if self.max_items is not None:
            out["max_items"] = int(self.max_items)
        if self.n_items_fit:
            out["n_items_before_restriction"] = int(self.n_items_before_restriction)
            out["n_items_fit"] = int(self.n_items_fit)
        return out


def _popular_item_ids(train: pd.DataFrame, max_items: int) -> np.ndarray:
    """Top-``max_items`` item ids by interaction count.

    Ties break toward the smaller item id (stable mergesort).
    """
    counts = train.groupby("item_id", sort=False).size().rename("n").reset_index()
    counts = counts.sort_values(
        ["n", "item_id"], ascending=[False, True], kind="mergesort"
    )
    return counts["item_id"].to_numpy(dtype=np.int64)[:max_items]
