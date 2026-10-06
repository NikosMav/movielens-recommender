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
        self._seen: dict[int, set[int]] = {}
        self._hyperparams: dict[str, Any] = {}

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
        obj._item_vectors = obj._encode_all_items()
        if train_for_seen is not None:
            obj._seen = {
                int(uid): set(g["item_id"].astype(int))
                for uid, g in train_for_seen.groupby("user_id")
            }
        else:
            # Prefer uncapped seen ids from features (history pooling may truncate).
            obj._seen = {
                int(uid): set(items) for uid, items in features.seen_item_ids.items()
            }
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
        device = torch.device(self._device)
        self._model.eval()
        with torch.no_grad():
            user_t = torch.tensor([uidx], dtype=torch.long, device=device)
            hist_t = torch.from_numpy(hist).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            vec = self._model.encode_users(user_t, hist_t, mask_t)
        return vec.detach().cpu().numpy().astype(np.float32)[0]

    def recommend(self, user_id: int, n: int) -> list[int]:
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
        return [int(self._features.item_ids[i]) for i in order]

    def topk_with_scores(self, n: int) -> dict[int, list[tuple[int, float]]]:
        """Batched exact top-n ``(item_id, score)`` for every fit user.

        Score is the same unscaled dot product :meth:`recommend` ranks by.
        Seen train items are masked. Ties use a stable mergesort.
        """
        if (
            n <= 0
            or self._model is None
            or self._features is None
            or self._item_vectors is None
        ):
            return {}
        feat = self._features
        user_idx = np.arange(feat.n_users, dtype=np.int64)
        hist, mask = pad_histories(user_idx, feat, max_history=self._max_history)
        device = torch.device(self._device)
        self._model.eval()
        with torch.no_grad():
            user_t = torch.from_numpy(user_idx).to(device)
            hist_t = torch.from_numpy(hist).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            user_vecs = (
                self._model.encode_users(user_t, hist_t, mask_t)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
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
