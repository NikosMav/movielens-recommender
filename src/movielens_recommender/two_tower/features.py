"""Feature matrices for the two-tower model (protocol-safe history)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from movielens_recommender.movies import GENRES, build_item_side_features, load_movies


@dataclass
class TwoTowerFeatures:
    """Indexed feature pack derived from a fit/train matrix + movie metadata."""

    user_ids: np.ndarray  # sorted unique user ids
    item_ids: np.ndarray  # sorted unique item ids in train catalog
    user_index: dict[int, int]
    item_index: dict[int, int]
    # Per-user history item *indices* (variable length lists, capped).
    user_history: dict[int, np.ndarray]
    # Full seen item *ids* per raw user_id (uncapped; for recommend filtering).
    seen_item_ids: dict[int, set[int]]
    # Positive (user_idx, item_idx) pairs for training.
    pos_user_idx: np.ndarray
    pos_item_idx: np.ndarray
    # Item side features aligned to item_ids order.
    genres: np.ndarray  # (n_items, n_genres)
    years: np.ndarray  # (n_items,)
    year_mean: float
    year_std: float
    # Sampling distribution for log-q correction over catalog indices.
    item_q: np.ndarray  # (n_items,) probabilities
    relevance_threshold: float

    @property
    def n_users(self) -> int:
        return int(len(self.user_ids))

    @property
    def n_items(self) -> int:
        return int(len(self.item_ids))

    @property
    def n_genres(self) -> int:
        return int(self.genres.shape[1])


def build_features(
    train: pd.DataFrame,
    *,
    dataset: str,
    data_dir: str | None = None,
    movies: pd.DataFrame | None = None,
    relevance_threshold: float = 4.0,
    max_history: int = 50,
) -> TwoTowerFeatures:
    """Build features from the matrix the model is allowed to see.

    History and positives come **only** from ``train`` (fit-train or full-train).
    """
    if train.empty:
        raise ValueError("train is empty")

    user_ids = np.sort(train["user_id"].unique().astype(np.int64))
    item_ids = np.sort(train["item_id"].unique().astype(np.int64))
    user_index = {int(u): i for i, u in enumerate(user_ids)}
    item_index = {int(it): i for i, it in enumerate(item_ids)}

    if movies is None:
        movies = load_movies(dataset, data_dir or "data")
    genres, years, year_mean, year_std = build_item_side_features(movies, item_ids)

    # Full history (all train interactions), capped to most recent max_history
    # for the tower input. Keep an uncapped seen set for recommend filtering.
    user_history: dict[int, np.ndarray] = {}
    seen_item_ids: dict[int, set[int]] = {}
    ordered = train.sort_values(
        ["user_id", "timestamp"], ascending=[True, True], kind="mergesort"
    )
    for uid, group in ordered.groupby("user_id", sort=False):
        raw_uid = int(uid)
        item_id_list = [int(i) for i in group["item_id"].tolist() if int(i) in item_index]
        seen_item_ids[raw_uid] = set(item_id_list)
        idxs = [item_index[i] for i in item_id_list]
        if max_history > 0 and len(idxs) > max_history:
            idxs = idxs[-max_history:]
        user_history[int(user_index[raw_uid])] = np.asarray(idxs, dtype=np.int64)

    positives = train[train["rating"] >= relevance_threshold]
    if positives.empty:
        raise ValueError("No positive interactions for two-tower training")

    pos_user_idx = positives["user_id"].map(user_index).to_numpy(dtype=np.int64)
    pos_item_idx = positives["item_id"].map(item_index).to_numpy(dtype=np.int64)

    # q(i) ∝ positive count in the fit matrix (catalog-aligned).
    counts = np.bincount(pos_item_idx, minlength=len(item_ids)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    item_q = counts / counts.sum()

    return TwoTowerFeatures(
        user_ids=user_ids,
        item_ids=item_ids,
        user_index=user_index,
        item_index=item_index,
        user_history=user_history,
        seen_item_ids=seen_item_ids,
        pos_user_idx=pos_user_idx,
        pos_item_idx=pos_item_idx,
        genres=genres,
        years=years,
        year_mean=year_mean,
        year_std=year_std,
        item_q=item_q.astype(np.float64),
        relevance_threshold=relevance_threshold,
    )


def pad_histories(
    user_indices: np.ndarray,
    features: TwoTowerFeatures,
    *,
    exclude_item_idx: np.ndarray | None = None,
    max_history: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad histories for a batch of user indices.

    When ``exclude_item_idx`` is provided (training), that item is removed from
    each row's history so the positive cannot trivially reconstruct itself.
    """
    batch = len(user_indices)
    hist = np.zeros((batch, max_history), dtype=np.int64)
    mask = np.zeros((batch, max_history), dtype=np.float32)
    for row, uidx in enumerate(user_indices):
        items = features.user_history.get(int(uidx))
        if items is None or len(items) == 0:
            continue
        seq = items
        if exclude_item_idx is not None:
            excl = int(exclude_item_idx[row])
            seq = seq[seq != excl]
        if len(seq) == 0:
            continue
        if len(seq) > max_history:
            seq = seq[-max_history:]
        n = len(seq)
        hist[row, :n] = seq
        mask[row, :n] = 1.0
    return hist, mask


# Re-export genre count for model construction without importing movies elsewhere.
N_GENRES = len(GENRES)
