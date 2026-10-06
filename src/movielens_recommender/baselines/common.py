"""Shared helpers for item-similarity baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


def binary_user_item(
    train: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, dict[int, int], dict[int, int]]:
    """Build a 0/1 user-item CSR from every observed train interaction.

    Any rating is an edge. Duplicate ``(user_id, item_id)`` rows collapse to 1.
    Users and items are sorted so the matrix layout is deterministic.
    """
    if train.empty:
        raise ValueError("Cannot fit on an empty interaction frame.")

    users = np.sort(train["user_id"].unique())
    items = np.sort(train["item_id"].unique())
    user_index = {int(u): i for i, u in enumerate(users)}
    item_index = {int(it): i for i, it in enumerate(items)}

    rows = train["user_id"].map(user_index).to_numpy()
    cols = train["item_id"].map(item_index).to_numpy()
    data = np.ones(len(train), dtype=np.float64)
    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(items)),
        dtype=np.float64,
    )
    mat.sum_duplicates()
    mat.data = np.ones(mat.data.shape, dtype=np.float64)
    return mat, items.astype(np.int64), user_index, item_index


def mask_seen(scores: np.ndarray, interactions: sparse.csr_matrix) -> np.ndarray:
    """Set scores of observed interactions to -inf (copy)."""
    masked = np.array(scores, dtype=np.float64, copy=True)
    coo = interactions.tocoo()
    if coo.nnz:
        masked[coo.row, coo.col] = -np.inf
    return masked


def recommend_from_scores(
    scores: np.ndarray,
    item_ids: np.ndarray,
    n: int,
) -> list[int]:
    """Top-n finite scores. Ties break by item column index (mergesort)."""
    if n <= 0:
        return []
    scores = np.asarray(scores, dtype=np.float64)
    if n >= len(scores):
        order = np.argsort(-scores, kind="mergesort")
    else:
        part = np.argpartition(-scores, n - 1)[:n]
        order = part[np.argsort(-scores[part], kind="mergesort")]
    return [int(item_ids[i]) for i in order[:n] if np.isfinite(scores[i])]
