"""Unit tests for EASE-DAN (ADR-0028) on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.ease_dan import (
    EASEDANRecommender,
    ease_dan_grid,
    edge_extension_configs,
    find_grid_edges,
    lock_grid,
    select_best,
)
from movielens_recommender.tune import build_model


def _synthetic_train() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(7)
    for uid in range(1, 25):
        n = int(rng.integers(4, 10))
        items = rng.choice(np.arange(1, 16), size=n, replace=False)
        for step, iid in enumerate(items, start=1):
            rows.append(
                {
                    "user_id": int(uid),
                    "item_id": int(iid),
                    "rating": float(rng.choice([3.0, 4.0, 5.0])),
                    "timestamp": int(1000 + uid * 10 + step),
                }
            )
    return pd.DataFrame(rows)


def test_ease_dan_reduces_exactly_to_ease_at_zero_normalization():
    train = _synthetic_train()
    ease = EASERecommender(l2=250.0).fit(train)
    dan = EASEDANRecommender(l2=250.0, dropout_p=0.0, alpha=0.0, beta=0.0).fit(train)
    np.testing.assert_allclose(dan.similarity, ease.similarity, rtol=1e-12, atol=1e-12)
    for uid in (1, 5, 12, 24):
        assert dan.recommend(uid, 5) == ease.recommend(uid, 5)


def test_ease_dan_excludes_seen_items_and_folds_profiles():
    train = _synthetic_train()
    dan = EASEDANRecommender(l2=50.0, dropout_p=0.3, alpha=0.2, beta=0.3).fit(train)
    assert np.all(np.diag(dan.similarity) == 0.0)
    seen = set(train.loc[train["user_id"] == 3, "item_id"].astype(int))
    recs = dan.recommend(3, 5)
    assert len(recs) == 5
    assert not (set(recs) & seen)
    assert dan.recommend(99999, 5) == []

    folded = dan.topk_for_profiles({0: [(1, 5.0), (2, 4.0)]}, 4, mask_items={0: [3]})
    returned_ids = [iid for iid, _score in folded[0]]
    assert len(returned_ids) == 4
    assert not (set(returned_ids) & {1, 2, 3})


def test_ease_dan_parameter_validation_and_build_model():
    with pytest.raises(ValueError):
        EASEDANRecommender(l2=0.0)
    with pytest.raises(ValueError):
        EASEDANRecommender(dropout_p=1.0)
    with pytest.raises(ValueError):
        EASEDANRecommender(alpha=-0.1)
    with pytest.raises(ValueError):
        EASEDANRecommender(beta=-0.1)

    train = _synthetic_train()
    model, hp = build_model(
        "ease_dan",
        train,
        hyperparams={"l2": 100.0, "dropout_p": 0.3, "alpha": 0.2, "beta": 0.3},
        relevance_threshold=4.0,
        seed=42,
    )
    assert hp["l2"] == 100.0
    assert hp["dropout_p"] == 0.3
    assert hp["alpha"] == 0.2
    assert hp["beta"] == 0.3
    assert len(model.recommend(1, 3)) == 3


def test_ease_dan_grid_budget_and_edge_rules():
    grid = ease_dan_grid()
    assert len(grid) == 108

    full_grid, cut, _proj = lock_grid(2.5)
    assert cut is None and len(full_grid) == 108

    cut1, label1, _ = lock_grid(30.0)
    assert label1 == "dropped l2=2000.0" and len(cut1) == 81

    cut2, label2, _ = lock_grid(40.0)
    assert label2 == "dropped l2=2000.0 and dropout_p=0.6" and len(cut2) == 54

    # Zero on dropout_p/alpha/beta is a natural boundary, not an extendable edge.
    zero_hp = {"l2": 100.0, "dropout_p": 0.0, "alpha": 0.0, "beta": 0.0}
    assert find_grid_edges(zero_hp, grid) == []
    assert edge_extension_configs(zero_hp, grid) == []

    # High edges step once; low l2 steps once.
    high_hp = {"l2": 10.0, "dropout_p": 0.6, "alpha": 0.4, "beta": 0.6}
    edges = find_grid_edges(high_hp, grid)
    assert {e["parameter"] for e in edges} == {"l2", "dropout_p", "alpha", "beta"}
    exts = edge_extension_configs(high_hp, grid)
    assert len(exts) == 4
    assert any(row["l2"] == 2.0 for row in exts)
    assert any(row["dropout_p"] == 0.8 for row in exts)
    assert any(row["alpha"] == 0.6 for row in exts)
    assert any(row["beta"] == 0.9 for row in exts)

    winner = select_best(
        [
            {"hyperparams": zero_hp, "val_ndcg@10": 0.081},
            {"hyperparams": high_hp, "val_ndcg@10": 0.081},
        ]
    )
    assert winner["hyperparams"] == zero_hp


def test_ease_dan_readme_and_quality_panel_match_committed_json(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result_path = root / "results" / "ease-dan" / "ml-1m.json"
    before = result_path.read_bytes()
    payload = json.loads(before.decode("utf-8"))
    assert payload["decision"]["beats_ease"] is True
    assert payload["paired"]["ease_dan_minus_ease"]["low"] > 0.0

    table_spec = importlib.util.spec_from_file_location(
        "make_results_table_ease_dan_test", root / "scripts" / "make_results_table.py"
    )
    assert table_spec is not None and table_spec.loader is not None
    table_mod = importlib.util.module_from_spec(table_spec)
    table_spec.loader.exec_module(table_mod)

    assert table_mod.render_ease_dan(tmp_path / "missing.json") == ""
    panel = table_mod.render_ease_dan(result_path)
    assert panel == table_mod.render_ease_dan(result_path)
    assert "### Data-Adaptive Normalized EASE (`EASE-DAN`, `ml-1m`)" in panel
    assert table_mod._fmt(payload["models"]["ease_dan"]["metrics"]["ndcg@10"]) in panel

    q_spec = importlib.util.spec_from_file_location(
        "space_quality_ease_dan_test", root / "space" / "quality.py"
    )
    assert q_spec is not None and q_spec.loader is not None
    q_mod = importlib.util.module_from_spec(q_spec)
    q_spec.loader.exec_module(q_mod)

    view = q_mod.load_quality(root / "results")
    assert "ease_dan" in view
    block = view["ease_dan"]
    assert block["beats_ease"] == payload["decision"]["beats_ease"]
    assert block["paired_ease_mean"] == payload["paired"]["ease_dan_minus_ease"]["mean"]
    assert q_mod.rounded(block["paired_ease_mean"]) in q_mod.ease_dan_lead(block)
    assert q_mod.fmt(block["paired_ease_mean"]) in q_mod.ease_dan_caption(view)
    assert result_path.read_bytes() == before

