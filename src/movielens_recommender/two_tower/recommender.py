"""Two-tower recommender wrapper with exact brute-force top-k."""

from __future__ import annotations

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

    def hyperparams(self) -> dict[str, Any]:
        return dict(self._hyperparams)
