"""ALS confidence schemes, the ADR-0018 grid, and the README panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from movielens_recommender.als_retune import (
    BUDGET_SEC,
    RESERVE_SEC,
    als_retune_grid,
    batch_ndcg_point_estimate,
    edge_extension_configs,
    extension_value,
    find_grid_edges,
    is_old_als,
    lock_grid,
    projected_seconds,
    select_best,
    trails_item_item,
)
from movielens_recommender.baselines.als import (
    ALSRecommender,
    confidence_values,
    user_item_confidence,
)
from movielens_recommender.evaluate import ndcg_point_estimate
from movielens_recommender.split import SplitConfig, SplitResult

ROOT = Path(__file__).resolve().parents[1]


def _train() -> pd.DataFrame:
    rows = []
    observed = {
        1: [(10, 5.0), (20, 4.0), (30, 5.0)],
        2: [(10, 4.0), (20, 5.0), (40, 4.0)],
        3: [(20, 5.0), (30, 4.0), (40, 5.0)],
        4: [(10, 5.0), (30, 4.0), (50, 5.0)],
    }
    for user_id, items in observed.items():
        for timestamp, (item_id, rating) in enumerate(items, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": rating,
                    "timestamp": timestamp,
                }
            )
    rows.append({"user_id": 1, "item_id": 50, "rating": 3.0, "timestamp": 9})
    return pd.DataFrame(rows)


def _fit(scheme: str, **kwargs) -> ALSRecommender:
    return ALSRecommender(
        factors=4,
        iterations=2,
        regularization=0.1,
        alpha=10.0,
        random_state=0,
        confidence=scheme,
        **kwargs,
    ).fit(_train())


def test_linear_matches_the_committed_formula_and_log_uses_log1p():
    ratings = np.array([4.0, 5.0])
    np.testing.assert_allclose(
        confidence_values(ratings, scheme="linear", alpha=10.0), [41.0, 51.0]
    )
    np.testing.assert_allclose(
        confidence_values(ratings, scheme="log", alpha=10.0),
        1.0 + 10.0 * np.log1p(ratings),
    )
    mat, users, items, _ids = user_item_confidence(
        _train(), confidence_threshold=4.0, scheme="linear", alpha=10.0
    )
    assert mat[users[1], items[10]] == pytest.approx(1.0 + 10.0 * 5.0)
    # Rating 3 is below the positive mask, so it is not a stored confidence.
    assert mat[users[1], items[50]] == 0.0


def test_bm25_and_tfidf_match_the_implicit_library():
    from implicit.nearest_neighbours import bm25_weight, tfidf_weight

    raw, _users, _items, _ids = user_item_confidence(
        _train(), confidence_threshold=4.0, scheme="bm25", alpha=0.0, bm25_k1=50.0, bm25_b=0.5
    )
    # Rebuild the raw rating matrix the same way the linear path would, then weight it.
    ratings = _train()
    ratings = ratings[ratings["rating"] >= 4.0]
    expect_bm25 = bm25_weight(
        _raw_rating_matrix(ratings), K1=50.0, B=0.5
    ).tocsr()
    assert raw.shape == expect_bm25.shape
    np.testing.assert_allclose(raw.toarray(), expect_bm25.toarray())

    tfidf, *_rest = user_item_confidence(
        _train(), confidence_threshold=4.0, scheme="tfidf", alpha=0.0
    )
    expect_tfidf = tfidf_weight(_raw_rating_matrix(ratings)).tocsr()
    np.testing.assert_allclose(tfidf.toarray(), expect_tfidf.toarray())


def _raw_rating_matrix(positives: pd.DataFrame) -> sparse.csr_matrix:
    users = np.sort(positives["user_id"].unique())
    items = np.sort(positives["item_id"].unique())
    user_index = {int(u): i for i, u in enumerate(users)}
    item_index = {int(it): i for i, it in enumerate(items)}
    rows = positives["user_id"].map(user_index).to_numpy()
    cols = positives["item_id"].map(item_index).to_numpy()
    data = positives["rating"].to_numpy(dtype=np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))


def test_unknown_scheme_is_rejected_and_library_schemes_omit_alpha():
    with pytest.raises(ValueError):
        ALSRecommender(confidence="counts")
    for scheme in ("bm25", "tfidf", "log", "linear"):
        model = _fit(scheme)
        recs = model.recommend(1, 3)
        assert 10 not in recs
        assert 20 not in recs
        assert 30 not in recs
        assert model.recommend_many([1, 999], 3)[1] == recs
        assert model.recommend_many([1, 999], 3)[999] == []
        recorded = model.hyperparams()
        assert recorded["confidence"] == scheme
        if scheme in {"bm25", "tfidf"}:
            assert "alpha" not in recorded
        else:
            assert recorded["alpha"] == 10.0
        if scheme == "bm25":
            assert recorded["bm25_k1"] == 100.0
            assert recorded["bm25_b"] == 0.8


def test_recommend_many_matches_recommend_on_a_tie():
    model = _fit("linear")
    n_items = len(model._item_ids)
    n_users = len(model._user_index)
    model._item_factors = np.zeros((n_items, 2))
    model._user_factors = np.zeros((n_users, 2))
    model._item_factors[:, 0] = np.arange(n_items, dtype=np.float64)
    # Two trailing items share the top score. The lower index wins the tie.
    model._item_factors[-1, 0] = 100.0
    model._item_factors[-2, 0] = 100.0
    user_id = int(next(iter(model._user_index)))
    uidx = model._user_index[user_id]
    model._user_factors[uidx, 0] = 1.0
    # Clear seen so the tie is not masked away.
    model._seen[user_id] = set()
    assert model.recommend(user_id, 2) == model.recommend_many([user_id], 2)[user_id]


def test_batch_validation_ndcg_matches_the_harness():
    train = _train()
    val = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 40, "rating": 5.0, "timestamp": 10},
            {"user_id": 2, "item_id": 30, "rating": 5.0, "timestamp": 10},
            {"user_id": 3, "item_id": 10, "rating": 4.0, "timestamp": 10},
        ]
    )
    view = SplitResult(
        train=train,
        test=val,
        config=SplitConfig(relevance_threshold=4.0),
        n_users_kept=3,
        n_users_dropped=0,
    )
    model = _fit("log")
    harness = ndcg_point_estimate(
        model.recommend, train, val, relevance_threshold=4.0, k=10, split=view
    )
    fresh = SplitResult(
        train=train,
        test=val,
        config=SplitConfig(relevance_threshold=4.0),
        n_users_kept=3,
        n_users_dropped=0,
    )
    batched = batch_ndcg_point_estimate(
        model, train, val, relevance_threshold=4.0, k=10, split=fresh
    )
    assert batched == pytest.approx(harness)


def test_grid_contains_the_committed_winner_and_the_stated_budget():
    grid = als_retune_grid()
    assert len(grid) == 144
    assert sum(1 for row in grid if row["confidence"] == "linear") == 54
    assert sum(1 for row in grid if row["confidence"] == "bm25") == 18
    assert all("alpha" not in row for row in grid if row["confidence"] in {"bm25", "tfidf"})
    assert any(is_old_als(row) for row in grid)
    tuning = json.loads((ROOT / "results" / "tuning" / "ml-1m.json").read_text(encoding="utf-8"))
    assert len(tuning["ease"]["trials"]) == 8
    assert len(tuning["rp3beta"]["trials"]) == 125
    assert len(tuning["als"]["trials"]) == 12
    assert is_old_als(tuning["als"]["best_hyperparams"])


def test_edges_and_one_step_extensions():
    grid = als_retune_grid()
    winner = {
        "alpha": 10.0,
        "confidence": "linear",
        "factors": 256,
        "iterations": 15,
        "regularization": 0.1,
    }
    edges = find_grid_edges(winner, grid)
    assert [(e["parameter"], e["side"]) for e in edges] == [
        ("factors", "high"),
        ("iterations", "low"),
        ("alpha", "low"),
    ]
    extended = edge_extension_configs(winner, grid)
    assert {row["factors"] for row in extended} == {256, 512}
    assert any(row["alpha"] == 5.0 and row["factors"] == 256 for row in extended)
    assert all(row["iterations"] == 15 for row in extended)
    assert extension_value("iterations", "low", 15) is None
    high_iter = dict(winner, iterations=30, factors=128, alpha=20.0)
    assert any(row["iterations"] == 60 for row in edge_extension_configs(high_iter, grid))
    # Iterations has two values, so 30 is the high edge and 15 is the low edge.
    assert find_grid_edges(
        {"confidence": "tfidf", "factors": 128, "iterations": 30, "regularization": 0.1},
        grid,
    ) == [{"parameter": "iterations", "side": "high", "value": 30}]


def test_budget_cut_drops_factors_then_iterations():
    full = als_retune_grid()
    grid, cut, projected = lock_grid(1.0, 8.0)
    assert cut is None
    assert len(grid) == 144
    assert projected + RESERVE_SEC <= BUDGET_SEC

    base = projected_seconds(full, 1.0, 8.0)
    scale = (BUDGET_SEC - RESERVE_SEC + 1) / base
    grid, cut, projected = lock_grid(scale, 8 * scale)
    assert cut == "dropped factors=64"
    assert len(grid) == 96
    assert projected + RESERVE_SEC <= BUDGET_SEC

    grid, cut, _projected = lock_grid(10_000.0, 80_000.0)
    assert cut == "dropped factors=64 and iterations=30"
    assert len(grid) == 48
    assert {row["factors"] for row in grid} == {128, 256}
    assert {row["iterations"] for row in grid} == {15}


def test_selection_tie_keeps_the_earlier_config_and_the_decision_is_strict():
    chosen = select_best(
        [
            {"hyperparams": {"factors": 64}, "val_ndcg@10": 0.2},
            {"hyperparams": {"factors": 128}, "val_ndcg@10": 0.2},
            {"hyperparams": {"factors": 256}, "val_ndcg@10": 0.1},
        ]
    )
    assert chosen["hyperparams"]["factors"] == 64
    assert trails_item_item(0.090620, 0.120135) is True
    assert trails_item_item(0.120135, 0.120135) is False
    assert trails_item_item(0.120136, 0.120135) is False


def _results_payload(*, negative: bool, edges: list[dict]) -> dict:
    def metrics(point: float) -> dict:
        return {
            "confidence_intervals": {
                "ndcg@10": {"high": point + 0.003, "low": point - 0.003, "mean": point}
            },
            "ndcg@10": point,
        }

    return {
        "budget": {
            "cut": None,
            "ease_configs": 8,
            "n_configs": 144,
            "n_configs_planned": 144,
            "n_edge_extensions": 1,
            "n_trials_scored": 145,
            "rp3beta_configs": 125,
            "truncated": False,
        },
        "decision": {"negative_result": negative},
        "models": {
            "als_old": {
                "metrics": metrics(0.09062),
                "published_test_ndcg@10": 0.09062,
                "val_ndcg@10": 0.064625,
            },
            "als_retuned": {"metrics": metrics(0.091), "val_ndcg@10": 0.07},
            "item_item_cosine": {
                "metrics": metrics(0.120135),
                "published_test_ndcg@10": 0.120135,
                "val_ndcg@10": None,
            },
        },
        "paired": {
            "retuned_minus_item_item": {
                "excludes_zero": True,
                "high": -0.02,
                "low": -0.04,
                "mean": -0.029,
            },
            "retuned_minus_old_als": {
                "excludes_zero": False,
                "high": 0.004,
                "low": -0.002,
                "mean": 0.001,
            },
        },
        "selected": {
            "grid_edges": edges,
            "hyperparams": {
                "alpha": 10.0,
                "confidence": "linear",
                "factors": 256,
                "iterations": 30,
                "regularization": 0.1,
            },
            "val_ndcg@10": 0.07,
        },
    }


def test_readme_panel_is_three_sentences_from_the_json(tmp_path: Path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_results_table import _fmt, als_retune_reading, render_als_retune

    payload = _results_payload(
        negative=True,
        edges=[{"parameter": "factors", "side": "high", "value": 256}],
    )
    reading = als_retune_reading(payload)
    assert len(reading) == 3
    assert all(sentence.endswith(".") for sentence in reading)
    text = "\n".join(reading)
    assert _fmt(payload["models"]["als_retuned"]["metrics"]["ndcg@10"]) in text
    assert _fmt(payload["models"]["als_old"]["metrics"]["ndcg@10"]) in text
    assert _fmt(payload["models"]["item_item_cosine"]["metrics"]["ndcg@10"]) in text
    assert _fmt(payload["paired"]["retuned_minus_old_als"]["mean"]) in text
    assert _fmt(payload["paired"]["retuned_minus_item_item"]["mean"]) in text
    assert "Negative result" in reading[2]
    assert "high factors edge (256)" in reading[2]
    assert "linear confidence" in reading[0]

    path = tmp_path / "ml-1m.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    panel = render_als_retune(path)
    assert panel == render_als_retune(path)
    for sentence in reading:
        assert sentence in panel
    assert "Validation configs in the locked grid: 144." in panel
    assert "EASE had 8 and RP3beta had 125" in panel
    assert "Edge-extension trials scored: 1." in panel
    assert render_als_retune(tmp_path / "missing.json") == ""

    stepped_payload = _results_payload(negative=True, edges=[])
    stepped_payload["budget"]["n_edge_extensions"] = 2
    stepped_payload["edge_extension_trials"] = [
        {
            "hyperparams": {
                "confidence": "linear",
                "factors": 16,
                "iterations": 30,
                "regularization": 0.1,
                "alpha": 10.0,
            },
            "val_ndcg@10": 0.05,
        },
        {
            "hyperparams": {
                "confidence": "linear",
                "factors": 256,
                "iterations": 30,
                "regularization": 5.0,
                "alpha": 10.0,
            },
            "val_ndcg@10": 0.06,
        },
    ]
    stepped_path = tmp_path / "with-edges.json"
    stepped_path.write_text(json.dumps(stepped_payload), encoding="utf-8")
    stepped_panel = render_als_retune(stepped_path)
    assert (
        "Edge-extension trials scored after the base grid, under the ADR-0018 "
        "one-step rule: 2 (factors 16, regularization 5; all lower on validation)."
        in stepped_panel
    )

    clear = _results_payload(negative=False, edges=[])
    assert "does not trail" in als_retune_reading(clear)[2]
    assert "no edge of the searched axes" in als_retune_reading(clear)[2]
    stepped = _results_payload(negative=True, edges=[])
    stepped["selected"]["base_grid_edges"] = [
        {"parameter": "factors", "side": "low", "value": 64},
        {"parameter": "regularization", "side": "high", "value": 1.0},
        {"parameter": "iterations", "side": "high", "value": 30},
    ]
    stepped_text = als_retune_reading(stepped)[2]
    assert "no edge of the values scored" in stepped_text
    assert "low factors edge (64)" in stepped_text
    assert "high regularization edge (1)" in stepped_text
    assert "high iterations edge (30)" in stepped_text
