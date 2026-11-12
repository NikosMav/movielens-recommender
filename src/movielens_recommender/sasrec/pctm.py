"""Probabilistic Collaborative Transition Model (PCTM; Petrov et al., RecSys 2026; ADR-0030).

Pairwise recency-weighted sequential transition probe with Dirichlet smoothing
and log-expert pooling. Uses zero learned embeddings or neural layers.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from movielens_recommender.baselines.common import top_n_indices

DECAY_SCHEMES: tuple[str, ...] = ("harmonic", "exp_0.7")


def distance_weights(max_distance: int, scheme: str) -> np.ndarray:
    """Weights ``γ_d`` for ``d = 1 .. max_distance`` with ``γ_1 = 1``."""
    if max_distance < 1:
        raise ValueError("max_distance must be >= 1")
    steps = np.arange(1, int(max_distance) + 1, dtype=np.float64)
    if scheme == "harmonic":
        return 1.0 / steps
    if scheme == "exp_0.7":
        return np.power(0.7, steps - 1.0)
    raise ValueError(f"unknown decay scheme {scheme!r}; choose from {DECAY_SCHEMES}")


def head_tail_kernel(
    length: int,
    *,
    head_k: int = 7,
    head_mass: float = 0.8,
    head_decay: float = 0.6,
) -> np.ndarray:
    """Normalized recency weights ``w_1 .. w_L`` (`j=1` is newest; Petrov et al. Eq. 5)."""
    if length < 1:
        return np.zeros(0, dtype=np.float64)
    if not (0.0 < head_mass <= 1.0) or not (0.0 < head_decay <= 1.0) or head_k < 1:
        raise ValueError("invalid head-tail kernel parameters")
    if length <= head_k:
        raw = np.power(head_decay, np.arange(length, dtype=np.float64))
        return raw / raw.sum()
    head_raw = np.power(head_decay, np.arange(head_k, dtype=np.float64))
    head = head_mass * (head_raw / head_raw.sum())
    tail = np.full(length - head_k, (1.0 - head_mass) / float(length - head_k), dtype=np.float64)
    return np.concatenate([head, tail])


class PCTMRecommender:
    """Pairwise causal transition model with Dirichlet smoothing and log-expert pooling."""

    def __init__(
        self,
        *,
        max_distance: int = 20,
        decay: str = "harmonic",
        smoothing_tau: float = 150.0,
        max_history: int = 200,
        pop_beta: float = -0.25,
        relevance_threshold: float = 4.0,
        head_k: int = 7,
        head_mass: float = 0.8,
        head_decay: float = 0.6,
    ) -> None:
        if max_distance < 1:
            raise ValueError("max_distance must be >= 1")
        if decay not in DECAY_SCHEMES:
            raise ValueError(f"decay must be in {DECAY_SCHEMES}; got {decay!r}")
        if smoothing_tau <= 0.0:
            raise ValueError("smoothing_tau must be > 0")
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        self.max_distance = int(max_distance)
        self.decay = str(decay)
        self.smoothing_tau = float(smoothing_tau)
        self.max_history = int(max_history)
        self.pop_beta = float(pop_beta)
        self.relevance_threshold = float(relevance_threshold)
        self.head_k = int(head_k)
        self.head_mass = float(head_mass)
        self.head_decay = float(head_decay)

        self._item_ids: np.ndarray = np.array([], dtype=np.int64)
        self._item_index: dict[int, int] = {}
        self._user_histories: dict[int, np.ndarray] = {}
        self._seen: dict[int, np.ndarray] = {}
        self._counts: sparse.csr_matrix | None = None
        self._log_evidence: sparse.csr_matrix | None = None
        self._log_pop: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> PCTMRecommender:
        """Count forward causal transitions on ``train`` and build sparse log-evidence."""
        ordered = train.sort_values(["user_id", "timestamp"], kind="mergesort")
        items = np.sort(ordered["item_id"].unique().astype(np.int64))
        self._item_ids = items
        self._item_index = {int(it): idx for idx, it in enumerate(items.tolist())}
        n_items = len(items)
        gammas = distance_weights(self.max_distance, self.decay)

        freq = np.zeros(n_items, dtype=np.float64)
        rows_all: list[np.ndarray] = []
        cols_all: list[np.ndarray] = []
        vals_all: list[np.ndarray] = []
        histories: dict[int, np.ndarray] = {}
        seen: dict[int, np.ndarray] = {}

        for uid, group in ordered.groupby("user_id", sort=False):
            uid_i = int(uid)
            seq = group["item_id"].map(self._item_index).to_numpy(dtype=np.int32)
            ratings = group["rating"].to_numpy(dtype=np.float64)
            seen[uid_i] = np.unique(seq)
            histories[uid_i] = seq[-self.max_history :].copy()
            pos_all = ratings >= self.relevance_threshold
            if np.any(pos_all):
                np.add.at(freq, seq[pos_all], 1.0)
            m = len(seq)
            upper = min(self.max_distance, m - 1)
            for d in range(1, upper + 1):
                pos_target = ratings[d:] >= self.relevance_threshold
                if np.any(pos_target):
                    src = seq[:-d][pos_target]
                    dst = seq[d:][pos_target]
                    rows_all.append(src)
                    cols_all.append(dst)
                    vals_all.append(
                        np.full(int(pos_target.sum()), gammas[d - 1], dtype=np.float64)
                    )

        if rows_all:
            counts = sparse.csr_matrix(
                (
                    np.concatenate(vals_all),
                    (np.concatenate(rows_all), np.concatenate(cols_all)),
                ),
                shape=(n_items, n_items),
                dtype=np.float64,
            )
            counts.sum_duplicates()
        else:
            counts = sparse.csr_matrix((n_items, n_items), dtype=np.float64)

        self._counts = counts
        self._user_histories = histories
        self._seen = seen
        pop = (freq + 1.0) / (float(freq.sum()) + float(n_items))
        self._log_pop = np.log(pop)
        self._rebuild_log_evidence()
        return self

    def _rebuild_log_evidence(self) -> None:
        assert self._counts is not None
        n_items = len(self._item_ids)
        ev = self._counts.copy()
        if ev.nnz:
            ev.data = np.log1p((float(n_items) / self.smoothing_tau) * ev.data)
        self._log_evidence = ev

    def set_scoring_params(self, *, smoothing_tau: float, pop_beta: float) -> PCTMRecommender:
        """Reuse counted transitions ``C`` while varying ``smoothing_tau`` or ``pop_beta``."""
        if smoothing_tau <= 0.0:
            raise ValueError("smoothing_tau must be > 0")
        if abs(float(smoothing_tau) - self.smoothing_tau) > 1e-12:
            self.smoothing_tau = float(smoothing_tau)
            self._rebuild_log_evidence()
        self.pop_beta = float(pop_beta)
        return self

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0 or self._log_evidence is None or self._log_pop is None:
            return []
        uid = int(user_id)
        hist = self._user_histories.get(uid)
        if hist is None or len(hist) == 0:
            return []
        # Newest item is hist[-1], corresponding to j=1 in head_tail_kernel.
        recent_first = hist[::-1]
        weights = head_tail_kernel(
            len(recent_first),
            head_k=self.head_k,
            head_mass=self.head_mass,
            head_decay=self.head_decay,
        )
        # Aggregate weights if a user rated the same item twice (cleaned data has unique pairs).
        n_items = len(self._item_ids)
        query = sparse.csr_matrix(
            (weights, (np.zeros(len(recent_first), dtype=np.int32), recent_first)),
            shape=(1, n_items),
            dtype=np.float64,
        )
        scores = np.asarray((query @ self._log_evidence).toarray(), dtype=np.float64).ravel()
        if self.pop_beta != 0.0:
            scores = scores + self.pop_beta * self._log_pop
        seen_cols = self._seen.get(uid)
        if seen_cols is not None and len(seen_cols):
            scores[seen_cols] = -np.inf
        chosen = top_n_indices(scores, n)
        return [int(self._item_ids[int(i)]) for i in chosen]

    def hyperparams(self) -> dict[str, Any]:
        return {
            "max_distance": self.max_distance,
            "decay": self.decay,
            "smoothing_tau": self.smoothing_tau,
            "max_history": self.max_history,
            "pop_beta": self.pop_beta,
            "head_k": self.head_k,
            "head_mass": self.head_mass,
            "head_decay": self.head_decay,
        }


def pctm_grid() -> list[dict[str, Any]]:
    """36-config validation grid (ADR-0030)."""
    return [
        {
            "max_distance": int(w),
            "decay": str(decay),
            "smoothing_tau": float(tau),
            "pop_beta": float(pop_beta),
            "max_history": 200,
        }
        for w, decay, tau, pop_beta in itertools.product(
            (10, 20),
            DECAY_SCHEMES,
            (50.0, 150.0, 500.0),
            (0.0, -0.25, -0.5),
        )
    ]


def esasrec_grid() -> list[dict[str, Any]]:
    """8-config validation grid for eSASRec (ADR-0030), matching SASRec's 8-config budget."""
    return [
        {
            "block_type": "ligr",
            "ff_mult": 2,
            "d_model": 64,
            "n_heads": 2,
            "batch_size": 128,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 30,
            "patience": 3,
            "max_seq_len": int(seq_len),
            "n_layers": int(n_layers),
            "dropout": float(dropout),
        }
        for seq_len, n_layers, dropout in itertools.product(
            (50, 200),
            (2, 3),
            (0.2, 0.5),
        )
    ]


def select_best_trial(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("select_best_trial requires at least one trial")
    best = trials[0]
    best_score = round(float(best["val_ndcg@10"]), 6)
    for row in trials[1:]:
        score = round(float(row["val_ndcg@10"]), 6)
        if score > best_score:
            best = row
            best_score = score
    return dict(best)
