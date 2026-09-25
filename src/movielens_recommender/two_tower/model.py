"""PyTorch two-tower network."""

from __future__ import annotations

import torch
from torch import nn


class TwoTowerModel(nn.Module):
    """User and item towers with shared item-id embeddings for history pooling.

    User = user_id emb + mean-pooled item_id embs of history.
    Item = item_id emb + genre projection + year projection.
    Both sides are LayerNorm'd then L2-normalized; score is temperature-scaled
    dot product.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_genres: int,
        embedding_dim: int = 64,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.n_users = n_users
        self.n_items = n_items

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.genre_proj = nn.Linear(n_genres, embedding_dim, bias=False)
        self.year_proj = nn.Linear(1, embedding_dim, bias=False)
        self.user_norm = nn.LayerNorm(embedding_dim)
        self.item_norm = nn.LayerNorm(embedding_dim)

        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        nn.init.xavier_uniform_(self.genre_proj.weight)
        nn.init.xavier_uniform_(self.year_proj.weight)

    def encode_users(
        self,
        user_idx: torch.Tensor,
        history_idx: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode users.

        Parameters
        ----------
        user_idx : (B,) long
        history_idx : (B, H) long — padded item indices
        history_mask : (B, H) bool/float — True/1 where history is valid
        """
        u = self.user_emb(user_idx)
        hist = self.item_emb(history_idx)  # (B, H, D)
        mask = history_mask.to(dtype=hist.dtype).unsqueeze(-1)  # (B, H, 1)
        summed = (hist * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denom
        # Users with empty history rely on the id embedding alone.
        has_hist = (mask.sum(dim=1) > 0).to(dtype=u.dtype)
        combined = u + pooled * has_hist
        combined = self.user_norm(combined)
        return nn.functional.normalize(combined, dim=-1)

    def encode_items(
        self,
        item_idx: torch.Tensor,
        genres: torch.Tensor,
        years: torch.Tensor,
    ) -> torch.Tensor:
        """Encode items.

        Parameters
        ----------
        item_idx : (N,) long
        genres : (N, G) float
        years : (N,) or (N, 1) float
        """
        if years.dim() == 1:
            years = years.unsqueeze(-1)
        combined = (
            self.item_emb(item_idx)
            + self.genre_proj(genres)
            + self.year_proj(years)
        )
        combined = self.item_norm(combined)
        return nn.functional.normalize(combined, dim=-1)

    def score(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        """Temperature-scaled cosine (dot of L2-normalized vectors)."""
        return (user_vec * item_vec).sum(dim=-1) / self.temperature
