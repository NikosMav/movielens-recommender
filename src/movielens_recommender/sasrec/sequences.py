"""Allowed-slice windows for SASRec (ADR-0019).

Fit-train windows are the validation-time sequences. Full-train windows are
the test-time sequences. Test interactions are never tokens. Validation
interactions are tokens only in the full-train refit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def events_by_user(frame: pd.DataFrame) -> dict[int, list[tuple[int, float]]]:
    """``user -> [(item_id, rating), ...]`` in timestamp order.

    Ties keep the original row order, matching the split's mergesort.
    """
    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("cannot build sequences from an empty frame")
    ordered = frame.sort_values(["user_id", "timestamp"], kind="mergesort")
    out: dict[int, list[tuple[int, float]]] = {}
    for uid, group in ordered.groupby("user_id", sort=False):
        items = group["item_id"].to_numpy()
        ratings = group["rating"].to_numpy()
        out[int(uid)] = [
            (int(item), float(rating))
            for item, rating in zip(items, ratings, strict=True)
        ]
    return out


def take_window(
    events: list[tuple[int, float]], max_seq_len: int
) -> list[tuple[int, float]]:
    """Most recent ``max_seq_len`` events. The list is copied."""
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1")
    if len(events) > max_seq_len:
        return list(events[-max_seq_len:])
    return list(events)


def windows_by_user(
    frame: pd.DataFrame,
    max_seq_len: int,
    *,
    shuffle_seed: int | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """One window per user.

    With ``shuffle_seed`` set, each window is permuted by one
    ``numpy.random.default_rng(shuffle_seed)`` stream, users in ascending id
    order. The stream is re-seeded on every call. The item set is unchanged.
    """
    events = events_by_user(frame)
    rng = None if shuffle_seed is None else np.random.default_rng(int(shuffle_seed))
    out: dict[int, list[tuple[int, float]]] = {}
    for uid in sorted(events):
        window = take_window(events[uid], max_seq_len)
        if rng is not None and window:
            order = rng.permutation(len(window))
            window = [window[int(i)] for i in order]
        out[uid] = window
    return out


def window_item_ids(
    frame: pd.DataFrame,
    max_seq_len: int,
    *,
    shuffle_seed: int | None = None,
) -> dict[int, list[int]]:
    """Item ids in window order. Tests use this for the leakage checks."""
    return {
        uid: [item for item, _rating in window]
        for uid, window in windows_by_user(
            frame, max_seq_len, shuffle_seed=shuffle_seed
        ).items()
    }


def training_target_ids(
    frame: pd.DataFrame,
    max_seq_len: int,
    *,
    shuffle_seed: int | None = None,
) -> dict[int, list[int]]:
    """Next-item targets inside each window (``window[1:]``).

    A window of length 1 has no target. Targets are not filtered by rating;
    the loss mask does that later.
    """
    out: dict[int, list[int]] = {}
    for uid, items in window_item_ids(
        frame, max_seq_len, shuffle_seed=shuffle_seed
    ).items():
        out[uid] = list(items[1:])
    return out


@dataclass
class SequenceData:
    """Tensors for one allowed slice.

    ``tokens`` are embedding indices (0 is pad). ``targets`` are class indices
    into the catalog (the pad row of the embedding is not a class). Rows with
    no positive target are omitted from the training tensors and kept for
    inference.
    """

    train_user_ids: np.ndarray
    tokens: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    infer_user_ids: np.ndarray
    infer_tokens: np.ndarray
    item_ids: np.ndarray
    item_index: dict[int, int]
    seen: dict[int, np.ndarray]


def build_sequence_data(
    frame: pd.DataFrame,
    *,
    max_seq_len: int,
    relevance_threshold: float,
    shuffle_seed: int | None = None,
) -> SequenceData:
    """Left-padded windows and next-item targets for ``frame`` only."""
    if max_seq_len < 1:
        raise ValueError("max_seq_len must be >= 1")
    events = events_by_user(frame)
    item_ids = np.sort(frame["item_id"].unique().astype(np.int64))
    item_index = {int(item): idx for idx, item in enumerate(item_ids)}
    windows = windows_by_user(frame, max_seq_len, shuffle_seed=shuffle_seed)

    seen: dict[int, np.ndarray] = {}
    for uid, ev in events.items():
        seen[uid] = np.unique(np.asarray([item for item, _rating in ev], dtype=np.int64))

    train_users: list[int] = []
    token_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    mask_rows: list[list[bool]] = []
    infer_users: list[int] = []
    infer_rows: list[list[int]] = []

    for uid in sorted(windows):
        window = windows[uid]
        infer_users.append(uid)
        infer_index = [item_index[item] + 1 for item, _rating in window]
        infer_rows.append(_left_pad(infer_index, max_seq_len))
        if len(window) < 2:
            continue
        inputs = [item_index[item] + 1 for item, _rating in window[:-1]]
        targets = [item_index[item] for item, _rating in window[1:]]
        ratings = [rating for _item, rating in window[1:]]
        mask = [rating >= relevance_threshold for rating in ratings]
        if not any(mask):
            continue
        train_users.append(uid)
        token_rows.append(_left_pad(inputs, max_seq_len))
        target_rows.append(_left_pad(targets, max_seq_len))
        mask_rows.append(_left_pad_bool(mask, max_seq_len))

    if not infer_users:
        raise ValueError("sequence frame produced no users")

    empty_int = np.zeros((0, max_seq_len), dtype=np.int64)
    empty_mask = np.zeros((0, max_seq_len), dtype=bool)
    return SequenceData(
        train_user_ids=np.asarray(train_users, dtype=np.int64),
        tokens=np.asarray(token_rows, dtype=np.int64) if token_rows else empty_int,
        targets=np.asarray(target_rows, dtype=np.int64) if target_rows else empty_int,
        loss_mask=np.asarray(mask_rows, dtype=bool) if mask_rows else empty_mask,
        infer_user_ids=np.asarray(infer_users, dtype=np.int64),
        infer_tokens=np.asarray(infer_rows, dtype=np.int64),
        item_ids=item_ids,
        item_index=item_index,
        seen=seen,
    )


def _left_pad(values: list[int], length: int) -> list[int]:
    if len(values) > length:
        raise ValueError("window is longer than max_seq_len")
    return [0] * (length - len(values)) + list(values)


def _left_pad_bool(values: list[bool], length: int) -> list[bool]:
    if len(values) > length:
        raise ValueError("window is longer than max_seq_len")
    return [False] * (length - len(values)) + list(values)
