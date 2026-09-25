"""Unit tests for the time-based split and cold-start policy."""

from __future__ import annotations

import pandas as pd
import pytest

from movielens_recommender.split import (
    SplitConfig,
    apply_cold_start_policy,
    time_based_split,
)


def _toy_ratings() -> pd.DataFrame:
    rows = []
    for t in range(1, 6):
        rows.append({"user_id": 1, "item_id": t, "rating": 4.0, "timestamp": t})
    for t in range(1, 5):
        rows.append({"user_id": 2, "item_id": t, "rating": 5.0, "timestamp": t})
    for t in range(10, 20):
        rows.append({"user_id": 3, "item_id": t, "rating": 3.0 + (t % 3) * 0.5, "timestamp": t})
    return pd.DataFrame(rows)


def test_no_within_user_timestamp_leakage():
    split = time_based_split(_toy_ratings(), SplitConfig(min_ratings=5, test_fraction=0.2))
    for uid in split.train["user_id"].unique():
        train_ts = split.train.loc[split.train["user_id"] == uid, "timestamp"]
        test_ts = split.test.loc[split.test["user_id"] == uid, "timestamp"]
        assert not test_ts.empty
        assert not train_ts.empty
        assert test_ts.min() >= train_ts.max()


def test_min_ratings_drops_sparse_users():
    split = time_based_split(_toy_ratings(), SplitConfig(min_ratings=5, test_fraction=0.2))
    assert 2 not in set(split.train["user_id"])
    assert 2 not in set(split.test["user_id"])
    assert split.n_users_dropped >= 1
    assert set(split.train["user_id"]) == {1, 3}


def test_test_fraction_counts():
    split = time_based_split(_toy_ratings(), SplitConfig(min_ratings=5, test_fraction=0.2))
    assert len(split.train[split.train["user_id"] == 1]) == 4
    assert len(split.test[split.test["user_id"] == 1]) == 1
    assert len(split.train[split.train["user_id"] == 3]) == 8
    assert len(split.test[split.test["user_id"] == 3]) == 2


def test_deterministic():
    cfg = SplitConfig(min_ratings=5, test_fraction=0.2)
    a = time_based_split(_toy_ratings(), cfg)
    b = time_based_split(_toy_ratings(), cfg)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_all_users_dropped_raises():
    tiny = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "rating": 5.0, "timestamp": 1},
            {"user_id": 1, "item_id": 2, "rating": 5.0, "timestamp": 2},
        ]
    )
    with pytest.raises(ValueError, match="No users remaining"):
        time_based_split(tiny, SplitConfig(min_ratings=10, test_fraction=0.2))


def test_cold_start_drops_cold_relevant_items():
    # Train has items 1,2; test has relevant cold item 99 and warm item 2.
    train = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "rating": 5.0, "timestamp": 1},
            {"user_id": 1, "item_id": 2, "rating": 4.0, "timestamp": 2},
            {"user_id": 2, "item_id": 1, "rating": 5.0, "timestamp": 1},
        ]
    )
    test = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 99, "rating": 5.0, "timestamp": 3},
            {"user_id": 1, "item_id": 2, "rating": 5.0, "timestamp": 4},
            {"user_id": 2, "item_id": 99, "rating": 5.0, "timestamp": 3},
        ]
    )
    split = time_based_split(
        pd.concat([train, test], ignore_index=True),
        SplitConfig(min_ratings=2, test_fraction=0.5, relevance_threshold=4.0),
    )
    # Force known train/test for the policy unit test.
    split.train = train
    split.test = test
    _seen, relevant, stats = apply_cold_start_policy(split)
    assert 99 not in relevant.get(1, set())
    assert 2 in relevant[1]
    # user 2 only had cold relevant → excluded
    assert 2 not in relevant
    assert stats.n_relevant_dropped_cold_item >= 1
    assert stats.n_users_excluded_no_warm_relevant >= 1
    assert stats.n_cold_items_in_test >= 1
