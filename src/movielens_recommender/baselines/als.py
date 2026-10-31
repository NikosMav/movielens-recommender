"""Matrix-factorization baseline via Alternating Least Squares (implicit library)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from movielens_recommender.baselines.common import top_n_indices

CONFIDENCE_SCHEMES = ("linear", "log", "bm25", "tfidf")
# implicit 0.7.3 defaults for bm25_weight. Alpha is not applied on top.
BM25_K1 = 100.0
BM25_B = 0.8


def confidence_values(ratings: np.ndarray, *, scheme: str, alpha: float) -> np.ndarray:
    """Per-rating confidence before any BM25 or TF-IDF reweight.

    ``linear`` is ``1 + alpha * r`` (the committed ALS formula). ``log`` is
    ``1 + alpha * ln(1 + r)``. ``bm25`` and ``tfidf`` keep the raw rating; the
    library weighting is applied to the matrix afterwards (ADR-0018).
    """
    values = np.asarray(ratings, dtype=np.float64)
    if scheme == "linear":
        return 1.0 + float(alpha) * values
    if scheme == "log":
        return 1.0 + float(alpha) * np.log1p(values)
    if scheme in {"bm25", "tfidf"}:
        return np.array(values, dtype=np.float64, copy=True)
    raise ValueError(f"Unknown confidence scheme: {scheme!r}. Choose from {CONFIDENCE_SCHEMES}.")


def user_item_confidence(
    train: pd.DataFrame,
    *,
    confidence_threshold: float,
    scheme: str,
    alpha: float,
    bm25_k1: float = BM25_K1,
    bm25_b: float = BM25_B,
) -> tuple[sparse.csr_matrix, dict[int, int], dict[int, int], np.ndarray]:
    """CSR confidence matrix for implicit ALS, users and items sorted.

    Ratings below ``confidence_threshold`` are dropped. Duplicate pairs sum the
    per-rating confidence (``linear`` / ``log``) or the raw rating (``bm25`` /
    ``tfidf``, before the library weighting).
    """
    if scheme not in CONFIDENCE_SCHEMES:
        raise ValueError(
            f"Unknown confidence scheme: {scheme!r}. Choose from {CONFIDENCE_SCHEMES}."
        )
    positives = train[train["rating"] >= confidence_threshold]
    if positives.empty:
        raise ValueError("No positive interactions for ALS (check confidence_threshold).")

    users = np.sort(positives["user_id"].unique())
    items = np.sort(positives["item_id"].unique())
    user_index = {int(u): i for i, u in enumerate(users)}
    item_index = {int(it): i for i, it in enumerate(items)}
    rows = positives["user_id"].map(user_index).to_numpy()
    cols = positives["item_id"].map(item_index).to_numpy()
    ratings = positives["rating"].to_numpy(dtype=np.float64)
    data = confidence_values(ratings, scheme=scheme, alpha=alpha)
    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(items)),
        dtype=np.float64,
    )
    if scheme == "bm25":
        from implicit.nearest_neighbours import bm25_weight

        mat = bm25_weight(mat, K1=bm25_k1, B=bm25_b).tocsr()
    elif scheme == "tfidf":
        from implicit.nearest_neighbours import tfidf_weight

        mat = tfidf_weight(mat).tocsr()
    return mat, user_index, item_index, items.astype(np.int64)


class ALSRecommender:
    """Implicit ALS on confidence-weighted ratings.

    Uses the ``implicit`` library's ``AlternatingLeastSquares`` on a CSR
    user-item matrix. Ratings at or above ``confidence_threshold`` are kept.
    ``confidence`` selects the weighting (ADR-0018): ``linear`` (default,
    ``1 + alpha * rating``), ``log``, ``bm25``, or ``tfidf``.
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
        confidence: str = "linear",
        bm25_k1: float = BM25_K1,
        bm25_b: float = BM25_B,
    ) -> None:
        if confidence not in CONFIDENCE_SCHEMES:
            raise ValueError(
                f"Unknown confidence scheme: {confidence!r}. Choose from {CONFIDENCE_SCHEMES}."
            )
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.confidence = confidence
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self._model = None
        self._user_index: dict[int, int] = {}
        self._item_index: dict[int, int] = {}
        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._user_items: sparse.csr_matrix | None = None
        self._seen: dict[int, set[int]] = {}
        self._user_factors: np.ndarray | None = None
        self._item_factors: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> ALSRecommender:
        # Import here so collecting tests stays light if implicit is missing.
        from implicit.als import AlternatingLeastSquares

        # Avoid OpenBLAS oversubscription warnings / slowdowns inside ALS.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        mat, user_index, item_index, item_ids = user_item_confidence(
            train,
            confidence_threshold=self.confidence_threshold,
            scheme=self.confidence,
            alpha=self.alpha,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
        )
        self._user_index = user_index
        self._item_index = item_index
        self._item_ids = item_ids
        self._user_items = mat

        # Track all train items per user (not only positives) so recommend can
        # exclude the full seen set even when filter_already_liked_items pads.
        self._seen = {
            int(uid): set(g["item_id"].astype(int)) for uid, g in train.groupby("user_id")
        }

        # Library alpha stays at its default of 1. Passing the confidence alpha
        # here would scale the matrix a second time (ADR-0018).
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
        self._user_factors = np.asarray(model.user_factors, dtype=np.float64)
        self._item_factors = np.asarray(model.item_factors, dtype=np.float64)
        return self

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0 or self._model is None or self._user_factors is None:
            return []
        if self._item_factors is None:
            return []
        uidx = self._user_index.get(int(user_id))
        if uidx is None:
            return []

        # Score all items via the user factor; filter seen ourselves.
        # (implicit's filter_already_liked_items can error / pad on small sets.)
        scores = self._item_factors @ self._user_factors[uidx]
        self._mask_seen(scores, int(user_id))
        if not np.isfinite(scores).any():
            return []
        order = top_n_indices(scores, n)
        return [int(self._item_ids[int(i)]) for i in order[:n]]

    def recommend_many(self, user_ids: Sequence[int], n: int) -> dict[int, list[int]]:
        """Top-n for many users. Same lists as :meth:`recommend`, one GEMM."""
        out = {int(uid): [] for uid in user_ids}
        if n <= 0 or self._model is None or self._user_factors is None:
            return out
        if self._item_factors is None:
            return out
        keep: list[int] = []
        rows: list[int] = []
        for uid in user_ids:
            uidx = self._user_index.get(int(uid))
            if uidx is not None:
                keep.append(int(uid))
                rows.append(uidx)
        if not rows:
            return out
        scores = self._user_factors[np.asarray(rows, dtype=np.int64)] @ self._item_factors.T
        for i, uid in enumerate(keep):
            row = scores[i]
            self._mask_seen(row, uid)
            if not np.isfinite(row).any():
                continue
            order = top_n_indices(row, n)
            out[uid] = [int(self._item_ids[int(j)]) for j in order[:n]]
        return out

    def _mask_seen(self, scores: np.ndarray, user_id: int) -> None:
        for item_id in self._seen.get(int(user_id), ()):
            col = self._item_index.get(item_id)
            if col is not None:
                scores[col] = -np.inf

    def hyperparams(self) -> dict:
        out: dict = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
            "confidence": self.confidence,
            "confidence_threshold": self.confidence_threshold,
            "random_state": self.random_state,
        }
        if self.confidence in {"linear", "log"}:
            out["alpha"] = self.alpha
        if self.confidence == "bm25":
            out["bm25_k1"] = self.bm25_k1
            out["bm25_b"] = self.bm25_b
        return out


def als_kwargs(hyperparams: Mapping) -> dict:
    """Constructor kwargs for a grid row. Missing keys keep the class defaults."""
    scheme = str(hyperparams.get("confidence", "linear"))
    kwargs: dict = {
        "factors": int(hyperparams.get("factors", 64)),
        "regularization": float(hyperparams.get("regularization", 0.01)),
        "iterations": int(hyperparams.get("iterations", 15)),
        "confidence": scheme,
        "bm25_k1": float(hyperparams.get("bm25_k1", BM25_K1)),
        "bm25_b": float(hyperparams.get("bm25_b", BM25_B)),
    }
    if "alpha" in hyperparams or scheme in {"linear", "log"}:
        kwargs["alpha"] = float(hyperparams.get("alpha", 40.0))
    return kwargs
