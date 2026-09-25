"""Matrix-factorization baseline via Alternating Least Squares (implicit library)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import sparse


class ALSRecommender:
    """Implicit ALS on confidence-weighted ratings.

    Uses the ``implicit`` library's ``AlternatingLeastSquares`` on a CSR
    user-item matrix. Ratings at or above ``confidence_threshold`` are treated
    as positive with confidence ``1 + alpha * rating``; lower ratings are
    dropped (implicit feedback framing).
    """

    def __init__(
        self,
        *,
        factors: int = 64,
        regularization: float = 0.01,
        iterations: int = 15,
        alpha: float = 40.0,
        confidence_threshold: float = 4.0,
        random_state: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state

        self._model = None
        self._user_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._user_items: sparse.csr_matrix | None = None
        self._seen: dict[int, set[int]] = {}

    def fit(self, train: pd.DataFrame) -> ALSRecommender:
        # Import here so collecting tests stays light if implicit is missing.
        from implicit.als import AlternatingLeastSquares

        # Avoid OpenBLAS oversubscription warnings / slowdowns inside ALS.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        positives = train[train["rating"] >= self.confidence_threshold].copy()
        if positives.empty:
            raise ValueError("No positive interactions for ALS (check confidence_threshold).")

        users = np.sort(positives["user_id"].unique())
        items = np.sort(positives["item_id"].unique())
        self._user_index = {int(u): i for i, u in enumerate(users)}
        self._item_index = {int(it): i for i, it in enumerate(items)}
        self._item_ids = items.astype(np.int64)

        # Track all train items per user (not only positives) so recommend can
        # exclude the full seen set even when filter_already_liked_items pads.
        self._seen = {
            int(uid): set(g["item_id"].astype(int)) for uid, g in train.groupby("user_id")
        }

        rows = positives["user_id"].map(self._user_index).to_numpy()
        cols = positives["item_id"].map(self._item_index).to_numpy()
        confidence = 1.0 + self.alpha * positives["rating"].to_numpy(dtype=np.float64)

        mat = sparse.csr_matrix(
            (confidence, (rows, cols)),
            shape=(len(users), len(items)),
            dtype=np.float64,
        )
        self._user_items = mat

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            use_gpu=False,
        )
        # implicit >=0.5 fit expects a user-item CSR confidence matrix.
        model.fit(mat, show_progress=False)
        self._model = model
        return self

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0 or self._model is None or self._user_items is None:
            return []
        uidx = self._user_index.get(int(user_id))
        if uidx is None:
            return []

        # Score all items via the user factor; filter seen ourselves.
        # (implicit's filter_already_liked_items can error / pad on small sets.)
        user_factor = np.asarray(self._model.user_factors[uidx], dtype=np.float64)
        item_factors = np.asarray(self._model.item_factors, dtype=np.float64)
        scores = item_factors @ user_factor

        seen = self._seen.get(int(user_id), set())
        for item_id in seen:
            col = self._item_index.get(item_id)
            if col is not None:
                scores[col] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            return []
        # Rank only among unscored-as-seen items.
        candidates = np.where(finite)[0]
        cand_scores = scores[candidates]
        if n >= len(candidates):
            order = candidates[np.argsort(-cand_scores, kind="mergesort")]
        else:
            part = np.argpartition(-cand_scores, n - 1)[:n]
            order = candidates[part[np.argsort(-cand_scores[part], kind="mergesort")]]
        return [int(self._item_ids[int(i)]) for i in order[:n]]

    def hyperparams(self) -> dict:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "confidence_threshold": self.confidence_threshold,
            "random_state": self.random_state,
        }
