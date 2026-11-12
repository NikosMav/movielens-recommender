"""Unit tests for eSASRec LiGR blocks and PCTM pairwise transition model (ADR-0030)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from movielens_recommender.sasrec.model import SASRecModel
from movielens_recommender.sasrec.pctm import (
    PCTMRecommender,
    distance_weights,
    esasrec_grid,
    head_tail_kernel,
    pctm_grid,
)
from movielens_recommender.sasrec.train import train_sasrec


def _synthetic_sequences() -> pd.DataFrame:
    rows = []
    # Clear directional pattern: 1 -> 2 -> 3 -> 4 -> 5
    for uid in range(1, 20):
        base = [1, 2, 3, 4, 5, (uid % 5) + 6]
        for step, iid in enumerate(base, start=1):
            rows.append(
                {
                    "user_id": uid,
                    "item_id": iid,
                    "rating": 5.0,
                    "timestamp": uid * 100 + step,
                }
            )
    # Add one user currently at [1, 2, 3] so next expected item is 4 (then 5)
    for step, iid in enumerate([1, 2, 3], start=1):
        rows.append({"user_id": 99, "item_id": iid, "rating": 5.0, "timestamp": 9900 + step})
    return pd.DataFrame(rows)


def test_ligr_block_preserves_causal_invariance():
    torch.manual_seed(42)
    model = SASRecModel(
        n_items=12,
        d_model=16,
        n_layers=2,
        n_heads=2,
        dropout=0.0,
        max_seq_len=6,
        block_type="ligr",
        ff_mult=2,
    )
    model.eval()
    tokens_a = torch.tensor([[0, 0, 1, 2, 3, 4]], dtype=torch.long)
    tokens_b = torch.tensor([[0, 0, 1, 2, 3, 9]], dtype=torch.long)
    with torch.no_grad():
        out_a = model(tokens_a)
        out_b = model(tokens_b)
    # Changing slot 5 must not alter causal representations at slots 0..4.
    torch.testing.assert_close(out_a[:, :5, :], out_b[:, :5, :])
    assert not torch.allclose(out_a[:, 5, :], out_b[:, 5, :])


def test_esasrec_trains_and_recommends_without_seen_items():
    df = _synthetic_sequences()
    hp = esasrec_grid()[0]
    hp = {**hp, "d_model": 16, "max_seq_len": 8, "max_epochs": 3}
    rec, _data, result = train_sasrec(df, hyperparams=hp, seed=42, n_epochs=3)
    assert result.epochs_trained == 3
    recs = rec.recommend(99, 3)
    assert len(recs) == 3
    assert not (set(recs) & {1, 2, 3})


def test_pctm_captures_directional_transitions_and_excludes_seen():
    df = _synthetic_sequences()
    w = head_tail_kernel(10, head_k=4, head_mass=0.8, head_decay=0.6)
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w[:4].sum(), 0.8)
    assert distance_weights(4, "harmonic")[0] == 1.0

    model = PCTMRecommender(
        max_distance=4,
        decay="harmonic",
        smoothing_tau=5.0,
        pop_beta=0.0,
    ).fit(df)
    recs = model.recommend(99, 3)
    assert len(recs) == 3
    assert not (set(recs) & {1, 2, 3})
    # Item 4 immediately follows 3 across 19 users, so it must rank #1 for user 99.
    assert recs[0] == 4
    assert model.recommend(999999, 3) == []
    assert len(pctm_grid()) == 36
    assert len(esasrec_grid()) == 8
