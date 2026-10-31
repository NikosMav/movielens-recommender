"""Causal SASRec encoder (ADR-0019). Written here; no recommender framework."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def build_attention_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Boolean mask, True where a query must not attend.

    Position ``i`` may attend to ``j <= i`` when ``j`` is not padding.
    A query whose every key would be blocked (a pad slot whose past is also
    pad) may attend to itself so the softmax is defined. Its output is not
    a training target. Shape is ``(batch, length, length)``.
    """
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape (batch, length)")
    _batch, length = tokens.shape
    future = torch.triu(
        torch.ones(length, length, dtype=torch.bool, device=tokens.device),
        diagonal=1,
    )
    blocked = future.unsqueeze(0).expand(tokens.shape[0], -1, -1).clone()
    pad = tokens.eq(0)
    blocked.masked_fill_(pad.unsqueeze(1), True)
    none_allowed = blocked.all(dim=-1)
    if bool(none_allowed.any()):
        rows, queries = none_allowed.nonzero(as_tuple=True)
        blocked[rows, queries, queries] = False
    return blocked


class SASRecBlock(nn.Module):
    """Pre-norm residual block. Inner feed-forward width equals ``d_model``."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model)
        self.ff2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        normed = self.ln1(hidden)
        attended, _weights = self.attn(
            normed,
            normed,
            normed,
            attn_mask=attn_mask,
            need_weights=False,
        )
        hidden = hidden + self.dropout(attended)
        normed = self.ln2(hidden)
        forwarded = self.ff2(F.relu(self.ff1(normed)))
        return hidden + self.dropout(forwarded)


class SASRecModel(nn.Module):
    """Item sequence encoder with a tied output embedding.

    ``forward`` returns hidden states ``(batch, length, d_model)``. The last
    slot is the inference state. Logits are a separate full-softmax call so
    the dot product can be chunked.
    """

    def __init__(
        self,
        n_items: int,
        *,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 1,
        dropout: float = 0.2,
        max_seq_len: int = 50,
    ) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError("n_items must be >= 1")
        if d_model < 1 or n_layers < 1 or n_heads < 1 or max_seq_len < 1:
            raise ValueError("d_model, n_layers, n_heads, and max_seq_len must be >= 1")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = int(n_items)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.max_seq_len = int(max_seq_len)
        self.item_emb = nn.Embedding(self.n_items + 1, self.d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(self.d_model, self.n_heads, dropout) for _ in range(self.n_layers)
        )
        self.final_norm = nn.LayerNorm(self.d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != self.max_seq_len:
            raise ValueError(
                f"tokens must have shape (batch, {self.max_seq_len}); got {tuple(tokens.shape)}"
            )
        positions = torch.arange(self.max_seq_len, device=tokens.device).unsqueeze(0)
        hidden = self.emb_dropout(self.item_emb(tokens) + self.pos_emb(positions))
        blocked = build_attention_mask(tokens)
        if self.n_heads > 1:
            attn_mask = blocked.repeat_interleave(self.n_heads, dim=0)
        else:
            attn_mask = blocked
        for block in self.blocks:
            hidden = block(hidden, attn_mask)
        return self.final_norm(hidden)

    def full_softmax_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        *,
        chunk_positions: int = 2048,
    ) -> torch.Tensor | None:
        """Mean cross-entropy over valid positions. ``None`` when none are valid.

        Chunks the valid positions and backpropagates each chunk so the logit
        tensor stays bounded. The sum of the chunks is the mean over all valid
        positions in ``hidden``, not a mean of chunk means.
        """
        if hidden.shape[:2] != targets.shape or targets.shape != mask.shape:
            raise ValueError("hidden, targets, and mask must share batch and length")
        weight = self.item_emb.weight[1:]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        flat_targets = targets.reshape(-1)
        index = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        count = int(index.numel())
        if count == 0:
            return None
        loss_value: torch.Tensor | None = None
        for start in range(0, count, chunk_positions):
            take = index[start : start + chunk_positions]
            logits = F.linear(flat_hidden.index_select(0, take), weight)
            chunk = F.cross_entropy(logits, flat_targets.index_select(0, take), reduction="sum")
            chunk = chunk / count
            last = start + chunk_positions >= count
            if hidden.requires_grad:
                chunk.backward(retain_graph=not last)
            loss_value = chunk.detach() if loss_value is None else loss_value + chunk.detach()
        return loss_value


def zero_padding_row(model: SASRecModel) -> None:
    """Keep the pad embedding at 0 after an optimizer step."""
    with torch.no_grad():
        model.item_emb.weight[0].zero_()
