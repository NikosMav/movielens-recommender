"""S6 drift and refresh replay. Synthetic data only; no MovieLens download."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.ops.drift import (
    Period,
    item_divergence,
    js_divergence,
    ml1m_periods,
    new_item_share,
    new_user_share,
    select_tau,
)
from movielens_recommender.ops.replay import run_replay


def _ts(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def test_ml1m_periods_are_contiguous_and_match_the_adr():
    periods = ml1m_periods()
    assert [p.name for p in periods][:6] == [
        "2000-08",
        "2000-09",
        "2000-10",
        "2000-11",
        "2000-12",
        "2001-Q1",
    ]
    assert periods[-1].name == "2003-Q1"
    assert len(periods) == 14
    assert periods[0].start == _ts("2000-08-01")
    for left, right in zip(periods, periods[1:], strict=False):
        assert left.end == right.start
    assert periods[-1].end == _ts("2003-04-01")


def test_js_divergence_bounds():
    assert js_divergence({1: 5, 2: 5}, {1: 5, 2: 5}) == pytest.approx(0.0)
    assert js_divergence({1: 3}, {2: 7}) == pytest.approx(1.0)
    middle = js_divergence({1: 1, 2: 1}, {1: 1})
    assert 0.0 < middle < 1.0
    assert js_divergence({1: 2, 2: 1}, {1: 1}) == pytest.approx(
        js_divergence({1: 1}, {1: 2, 2: 1})
    )


def test_item_divergence_samples_equal_sizes_and_is_seeded():
    rng = np.random.default_rng(0)
    last = rng.integers(0, 50, size=5000)
    train = rng.integers(0, 50, size=20000)
    first = item_divergence(last, train, sample_size=1000, seed=42)
    assert first == item_divergence(last, train, sample_size=1000, seed=42)
    shifted = item_divergence(last + 100, train, sample_size=1000, seed=42)
    assert shifted == pytest.approx(1.0)
    assert first < 0.2
    small = item_divergence(last[:10], train, sample_size=1000, seed=42)
    assert 0.0 <= small <= 1.0


def test_new_item_and_new_user_shares():
    train_counts = {1: 10, 2: 4, 3: 5}
    assert new_item_share([1, 2, 3, 9], train_counts, min_count=5) == pytest.approx(0.5)
    assert new_item_share([], train_counts, min_count=5) == 0.0
    period = Period("p", 100, 200)
    frame = pd.DataFrame({"user_id": [1, 1, 2, 3], "timestamp": [120, 130, 150, 160]})
    first_seen = {1: 50, 2: 140, 3: 155}
    assert new_user_share(frame, first_seen, period) == pytest.approx(0.5)


def test_select_tau_prefers_fewest_retrains_within_tolerance():
    tuning = {
        0.1: {"mean_ndcg": 0.300, "retrains": 2},
        0.2: {"mean_ndcg": 0.297, "retrains": 1},
        0.3: {"mean_ndcg": 0.290, "retrains": 0},
        0.4: {"mean_ndcg": 0.296, "retrains": 1},
    }
    chosen = select_tau(tuning, periodic_mean=0.300, tolerance=0.005)
    assert chosen["tau"] == 0.4
    assert chosen["qualifying"] == [0.1, 0.2, 0.4]
    nothing = select_tau(tuning, periodic_mean=0.400, tolerance=0.005)
    assert nothing["tau"] == 0.1
    assert nothing["qualifying"] == []


def _synthetic_ratings() -> tuple[pd.DataFrame, list[Period]]:
    """Users rate in two eras; era-2 users prefer a block of items era 1 barely saw."""
    rng = np.random.default_rng(3)
    rows = []
    for user_id in range(1, 61):
        for step in range(30):
            era_two = step >= 15
            ts = 1_000 + step * 100 + user_id
            if era_two:
                item = int(rng.integers(40, 60)) if rng.random() < 0.8 else int(rng.integers(0, 40))
            else:
                item = int(rng.integers(0, 40))
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item,
                    "rating": float(rng.choice([3.0, 4.0, 5.0])),
                    "timestamp": ts,
                }
            )
    ratings = pd.DataFrame(rows).drop_duplicates(["user_id", "item_id"], keep="last")
    periods = [Period(f"p{i}", 1_000 + 1_000 * i, 1_000 + 1_000 * (i + 1)) for i in range(1, 3)]
    return ratings.reset_index(drop=True), periods


def test_replay_runs_end_to_end_and_pairs_users():
    ratings, periods = _synthetic_ratings()
    out = run_replay(
        ratings,
        periods,
        models={
            "most_popular": {},
            "item_item": {"k_neighbors": 20, "shrinkage": 0.0, "min_common": 1},
            "ease": {"l2": 10.0},
        },
        taus=[0.1, 0.9],
        n_tuning=1,
        sample_size=200,
        n_bootstrap=20,
        seed=42,
    )
    assert [p["name"] for p in out["periods"]] == ["p1", "p2"]
    for period in out["periods"]:
        assert period["n_eval_users"] > 0
        for key in ("item_divergence", "new_item_share", "new_user_share"):
            assert 0.0 <= period["drift_vs_frozen"][key] <= 1.0
    assert out["divergence_floor"] >= 0.0
    for name in ("most_popular", "item_item", "ease"):
        block = out["models"][name]
        frozen = block["policies"]["frozen"]
        periodic = block["policies"]["periodic"]
        assert frozen["retrains"] == 0
        assert periodic["retrains"] == len(periods) - 1
        assert len(frozen["per_period"]) == len(periods)
        first_gap = block["paired_periodic_minus_frozen"][0]
        assert first_gap["mean"] == pytest.approx(0.0)
        assert first_gap["n_users"] == out["periods"][0]["n_eval_users"]
        assert block["policies"]["drift"]["tau"] in (0.1, 0.9)
        for row in frozen["per_period"]:
            assert 0.0 <= row["ndcg@10"] <= 1.0
        rho = block["spearman_vs_gap"]["item_divergence"]
        assert rho is None or (isinstance(rho, float) and not math.isnan(rho))
    assert out["tau_selection"]["chosen_on"] == "item_item"
    gap = out["models"]["item_item"]["paired_periodic_minus_frozen"][1]["mean"]
    assert gap > 0.0
