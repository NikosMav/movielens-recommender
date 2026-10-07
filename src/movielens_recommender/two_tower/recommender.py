"""Two-tower recommender wrapper with exact brute-force top-k."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from movielens_recommender.two_tower.features import TwoTowerFeatures, pad_histories
from movielens_recommender.two_tower.model import TwoTowerModel


class TwoTowerRecommender:
    """Exact top-k retrieval via user·item dots over the train catalog."""

    def __init__(self) -> None:
        self._model: TwoTowerModel | None = None
        self._features: TwoTowerFeatures | None = None
        self._item_vectors: np.ndarray | None = None
        self._max_history = 50
        self._device = "cpu"
        self._seen: dict[int, np.ndarray | set[int]] = {}
        self._hyperparams: dict[str, Any] = {}
        # New-user scoring zeros the id embedding. The headline model leaves this off.
        self.score_without_user_id = False

    @classmethod
    def from_trained(
        cls,
        model: TwoTowerModel,
        features: TwoTowerFeatures,
        *,
        max_history: int = 50,
        device: str = "cpu",
        hyperparams: dict[str, Any] | None = None,
        train_for_seen: pd.DataFrame | None = None,
    ) -> TwoTowerRecommender:
        obj = cls()
        obj._model = model
        obj._features = features
        obj._max_history = max_history
        obj._device = device
        obj._hyperparams = dict(hyperparams or {})
        obj.score_without_user_id = bool(
            obj._hyperparams.get("score_without_user_id", False)
        ) or bool(getattr(model, "history_only", False))
        obj._item_vectors = obj._encode_all_items()
        if train_for_seen is not None:
            obj._seen = {
                int(uid): np.asarray(g["item_id"].to_numpy(), dtype=np.int64)
                for uid, g in train_for_seen.groupby("user_id")
            }
        else:
            # Prefer uncapped seen ids from features (history pooling may truncate).
            # Share the mapping; do not copy into Python sets.
            obj._seen = features.seen_item_ids
        return obj

    def _encode_all_items(self) -> np.ndarray:
        assert self._model is not None and self._features is not None
        model = self._model
        feat = self._features
        device = torch.device(self._device)
        model.eval()
        with torch.no_grad():
            idx = torch.arange(feat.n_items, dtype=torch.long, device=device)
            genres = torch.from_numpy(feat.genres).to(device)
            years = torch.from_numpy(feat.years).to(device)
            vecs = model.encode_items(idx, genres, years)
        return vecs.detach().cpu().numpy().astype(np.float32)

    def encode_user(self, user_id: int) -> np.ndarray | None:
        assert self._model is not None and self._features is not None
        uidx = self._features.user_index.get(int(user_id))
        if uidx is None:
            return None
        hist, mask = pad_histories(
            np.asarray([uidx], dtype=np.int64),
            self._features,
            max_history=self._max_history,
        )
        vecs = self._encode_user_batch(
            np.asarray([uidx], dtype=np.int64), hist, mask
        )
        return vecs[0]

    def topk_for_user(self, user_id: int, n: int) -> list[tuple[int, float]]:
        """Top-n ``(item_id, score)`` for one fit user.

        Score is the same unscaled dot product :meth:`recommend` ranks by.
        Seen train items are masked. Ties use a stable mergesort. Unknown
        user ids return an empty list: there is no embedding to encode.
        """
        if n <= 0 or self._item_vectors is None or self._features is None:
            return []
        user_vec = self.encode_user(user_id)
        if user_vec is None:
            return []

        scores = self._item_vectors @ user_vec
        seen = self._seen.get(int(user_id), set())
        for item_id in seen:
            col = self._features.item_index.get(item_id)
            if col is not None:
                scores[col] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            return []
        candidates = np.where(finite)[0]
        cand_scores = scores[candidates]
        if n >= len(candidates):
            order = candidates[np.argsort(-cand_scores, kind="mergesort")]
        else:
            part = np.argpartition(-cand_scores, n - 1)[:n]
            order = candidates[part[np.argsort(-cand_scores[part], kind="mergesort")]]
        return [
            (int(self._features.item_ids[i]), float(scores[i]))
            for i in order
            if np.isfinite(scores[i])
        ]

    def recommend(self, user_id: int, n: int) -> list[int]:
        return [item_id for item_id, _score in self.topk_for_user(user_id, n)]

    def topk_with_scores(
        self,
        n: int,
        user_ids: Sequence[int] | None = None,
    ) -> dict[int, list[tuple[int, float]]]:
        """Batched exact top-n ``(item_id, score)`` for fit users.

        Score is the same unscaled dot product :meth:`recommend` ranks by.
        Seen train items are masked. Ties use a stable mergesort.
        ``user_ids`` scores only those users, in chunks, so a large catalog
        does not allocate a users × items score matrix.
        """
        if (
            n <= 0
            or self._model is None
            or self._features is None
            or self._item_vectors is None
        ):
            return {}
        if user_ids is not None:
            return self._topk_user_ids(n, user_ids)
        feat = self._features
        user_idx = np.arange(feat.n_users, dtype=np.int64)
        hist, mask = pad_histories(user_idx, feat, max_history=self._max_history)
        user_vecs = self._encode_user_batch(user_idx, hist, mask)
        scores = user_vecs @ self._item_vectors.T
        for uid, seen in self._seen.items():
            uidx = feat.user_index.get(int(uid))
            if uidx is None:
                continue
            for item_id in seen:
                col = feat.item_index.get(int(item_id))
                if col is not None:
                    scores[uidx, col] = -np.inf

        out: dict[int, list[tuple[int, float]]] = {}
        for uidx, uid in enumerate(feat.user_ids):
            row = scores[uidx]
            chosen = _topk_indices(row, n)
            out[int(uid)] = [
                (int(feat.item_ids[j]), float(row[j]))
                for j in chosen
                if np.isfinite(row[j])
            ]
        return out

    def _topk_user_ids(
        self,
        n: int,
        user_ids: Sequence[int],
    ) -> dict[int, list[tuple[int, float]]]:
        assert self._model is not None and self._features is not None
        assert self._item_vectors is not None
        feat = self._features
        uids = [int(uid) for uid in user_ids if int(uid) in feat.user_index]
        if not uids:
            return {}
        device = torch.device(self._device)
        self._model.eval()
        out: dict[int, list[tuple[int, float]]] = {}
        chunk = 256
        for start in range(0, len(uids), chunk):
            batch = uids[start : start + chunk]
            user_idx = np.asarray(
                [feat.user_index[uid] for uid in batch], dtype=np.int64
            )
            hist, mask = pad_histories(
                user_idx, feat, max_history=self._max_history
            )
            with torch.no_grad():
                user_vecs = (
                    self._model.encode_users(
                        torch.from_numpy(user_idx).to(device),
                        torch.from_numpy(hist).to(device),
                        torch.from_numpy(mask).to(device),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
            scores = user_vecs @ self._item_vectors.T
            for row, uid in enumerate(batch):
                for item_id in self._seen.get(uid, ()):
                    col = feat.item_index.get(int(item_id))
                    if col is not None:
                        scores[row, col] = -np.inf
                chosen = _topk_indices(scores[row], n)
                out[uid] = [
                    (int(feat.item_ids[j]), float(scores[row, j]))
                    for j in chosen
                    if np.isfinite(scores[row, j])
                ]
        return out

    def topk_for_histories(
        self,
        histories: Mapping[int, Sequence[int]],
        n: int,
        *,
        mask_items: Mapping[int, Sequence[int]] | None = None,
    ) -> dict[int, list[tuple[int, float]]]:
        """Exact top-n using the given item-id histories instead of stored ones.

        History order is chronological (oldest first). Only the last
        ``max_history`` ids that exist in the fit catalog are pooled. The
        user-id embedding is still the one learned at fit time. ``mask_items``
        are hidden before the top-n. Users absent from the fit catalog are
        omitted.
        """
        if (
            n <= 0
            or self._model is None
            or self._features is None
            or self._item_vectors is None
            or not histories
        ):
            return {}
        feat = self._features
        uids = [int(uid) for uid in histories if int(uid) in feat.user_index]
        if not uids:
            return {}
        width = int(self._max_history)
        hist = np.zeros((len(uids), width), dtype=np.int64)
        mask = np.zeros((len(uids), width), dtype=np.float32)
        user_idx = np.empty(len(uids), dtype=np.int64)
        for row, uid in enumerate(uids):
            user_idx[row] = feat.user_index[uid]
            seq = [
                feat.item_index[int(item)]
                for item in histories[uid]
                if int(item) in feat.item_index
            ]
            if len(seq) > width:
                seq = seq[-width:]
            if seq:
                hist[row, : len(seq)] = seq
                mask[row, : len(seq)] = 1.0
        device = torch.device(self._device)
        self._model.eval()
        with torch.no_grad():
            user_vecs = (
                self._model.encode_users(
                    torch.from_numpy(user_idx).to(device),
                    torch.from_numpy(hist).to(device),
                    torch.from_numpy(mask).to(device),
                )
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        scores = user_vecs @ self._item_vectors.T
        if mask_items:
            for row, uid in enumerate(uids):
                for item_id in mask_items.get(uid, ()):
                    col = feat.item_index.get(int(item_id))
                    if col is not None:
                        scores[row, col] = -np.inf
        out: dict[int, list[tuple[int, float]]] = {}
        for row, uid in enumerate(uids):
            chosen = _topk_indices(scores[row], n)
            out[uid] = [
                (int(feat.item_ids[j]), float(scores[row, j]))
                for j in chosen
                if np.isfinite(scores[row, j])
            ]
        return out

    def topk_for_new_histories(
        self,
        histories: Mapping[int, Sequence[int]],
        n: int,
        *,
        mask_items: Mapping[int, Sequence[int]] | None = None,
    ) -> dict[int, list[tuple[int, float]]]:
        """Exact top-n for users who have no id row.

        History is chronological item ids (oldest first). Only the last
        ``max_history`` ids in the fit catalog are pooled. The id embedding
        is zero. Profile items are excluded, and so is anything in
        ``mask_items``. Unknown catalog ids in the history are ignored.
        """
        if (
            n <= 0
            or self._model is None
            or self._features is None
            or self._item_vectors is None
            or not histories
        ):
            return {}
        feat = self._features
        uids = [int(uid) for uid in histories]
        width = int(self._max_history)
        hist = np.zeros((len(uids), width), dtype=np.int64)
        mask = np.zeros((len(uids), width), dtype=np.float32)
        profile_cols: list[list[int]] = []
        for row, uid in enumerate(uids):
            cols: list[int] = []
            for item in histories[uid]:
                col = feat.item_index.get(int(item))
                if col is not None:
                    cols.append(int(col))
            profile_cols.append(cols)
            seq = cols[-width:] if len(cols) > width else cols
            if seq:
                hist[row, : len(seq)] = seq
                mask[row, : len(seq)] = 1.0
        # Dummy ids. history_only ignores them; otherwise zero_user_id drops them.
        user_idx = np.zeros(len(uids), dtype=np.int64)
        user_vecs = self._encode_user_batch(user_idx, hist, mask, force_zero_user_id=True)
        scores = user_vecs @ self._item_vectors.T
        for row, cols in enumerate(profile_cols):
            if cols:
                scores[row, cols] = -np.inf
        if mask_items:
            for row, uid in enumerate(uids):
                for item_id in mask_items.get(uid, ()):
                    col = feat.item_index.get(int(item_id))
                    if col is not None:
                        scores[row, col] = -np.inf
        out: dict[int, list[tuple[int, float]]] = {}
        for row, uid in enumerate(uids):
            chosen = _topk_indices(scores[row], n)
            out[uid] = [
                (int(feat.item_ids[j]), float(scores[row, j]))
                for j in chosen
                if np.isfinite(scores[row, j])
            ]
        return out

    def _encode_user_batch(
        self,
        user_idx: np.ndarray,
        hist: np.ndarray,
        mask: np.ndarray,
        *,
        force_zero_user_id: bool = False,
    ) -> np.ndarray:
        assert self._model is not None
        device = torch.device(self._device)
        self._model.eval()
        zero = None
        if force_zero_user_id or self.score_without_user_id or self._model.history_only:
            zero = torch.ones(len(user_idx), dtype=torch.float32, device=device)
        with torch.no_grad():
            vec = self._model.encode_users(
                torch.as_tensor(user_idx, dtype=torch.long, device=device),
                torch.from_numpy(np.ascontiguousarray(hist)).to(device),
                torch.from_numpy(np.ascontiguousarray(mask)).to(device),
                zero_user_id=zero,
            )
        return vec.detach().cpu().numpy().astype(np.float32)

    def hyperparams(self) -> dict[str, Any]:
        return dict(self._hyperparams)


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
