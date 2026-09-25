"""Unit tests for validation holdout, global cutoff, and segment helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from movielens_recommender.segments import (
    assign_activity_terciles,
    filter_recs_to_segment,
    head_tail_items,
    ndcg_for_item_segment,
)
from movielens_recommender.split import (
    GlobalCutoffConfig,
    SplitConfig,
    global_time_cutoff_split,
    time_based_split,
)


def _user_ratings(uid: int, n: int, start_ts: int = 1, start_item: int = 1) -> list[dict]:
    return [
        {
            "user_id": uid,
            "item_id": start_item + i,
            "rating": 4.0 + (i % 2) * 0.5,
            "timestamp": start_ts + i,
        }
        for i in range(n)
    ]


def _multi_user_frame() -> pd.DataFrame:
    # User 1: 20 ratings → test=4, train_full=16 → val=1, fit=15 (fractions 0.2 / 0.1)
    # User 2: 10 ratings → test=2, train_full=8 → val=1, fit=7
    # User 3: 5 ratings → test=1, train_full=4 → val=1, fit=3
    rows = (
        _user_ratings(1, 20, start_ts=100, start_item=1)
        + _user_ratings(2, 10, start_ts=200, start_item=100)
        + _user_ratings(3, 5, start_ts=300, start_item=200)
    )
    return pd.DataFrame(rows)


def test_validation_split_no_within_user_leakage_across_three_slices():
    split = time_based_split(
        _multi_user_frame(),
        SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=0.1),
    )
    assert split.val is not None
    for uid in split.train["user_id"].unique():
        train_ts = split.train.loc[split.train["user_id"] == uid, "timestamp"]
        val_ts = split.val.loc[split.val["user_id"] == uid, "timestamp"]
        test_ts = split.test.loc[split.test["user_id"] == uid, "timestamp"]
        assert not train_ts.empty and not val_ts.empty and not test_ts.empty
        assert train_ts.max() <= val_ts.min()
        assert val_ts.max() <= test_ts.min()
        # Pairwise disjoint item-time rows: no shared timestamps within user across slices.
        assert set(train_ts).isdisjoint(set(val_ts))
        assert set(train_ts).isdisjoint(set(test_ts))
        assert set(val_ts).isdisjoint(set(test_ts))


def test_validation_split_counts_and_full_train():
    split = time_based_split(
        _multi_user_frame(),
        SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=0.1),
    )
    # User 1: n=20 → test=4, full_train=16 → val=max(1,floor(1.6))=1, fit=15
    assert len(split.test[split.test["user_id"] == 1]) == 4
    assert len(split.val[split.val["user_id"] == 1]) == 1
    assert len(split.train[split.train["user_id"] == 1]) == 15
    assert len(split.full_train[split.full_train["user_id"] == 1]) == 16
    # full_train == train ∪ val and equals the pre-test pool.
    full = pd.concat([split.train, split.val], ignore_index=True)
    pd.testing.assert_frame_equal(
        full.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True),
        split.full_train.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(
            drop=True
        ),
    )


def test_validation_ordering_is_chronological():
    split = time_based_split(
        _multi_user_frame(),
        SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=0.1),
    )
    for uid in (1, 2, 3):
        train = split.train[split.train["user_id"] == uid].sort_values(
            "timestamp", kind="mergesort"
        )
        val = split.val[split.val["user_id"] == uid].sort_values("timestamp", kind="mergesort")
        test = split.test[split.test["user_id"] == uid].sort_values(
            "timestamp", kind="mergesort"
        )
        assert list(train["timestamp"]) == sorted(train["timestamp"])
        assert train["timestamp"].iloc[-1] <= val["timestamp"].iloc[0]
        assert val["timestamp"].iloc[-1] <= test["timestamp"].iloc[0]


def test_val_fraction_zero_keeps_s2_behaviour():
    split = time_based_split(
        _multi_user_frame(),
        SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=0.0),
    )
    assert split.val is None
    assert len(split.train[split.train["user_id"] == 1]) == 16
    assert len(split.test[split.test["user_id"] == 1]) == 4
    pd.testing.assert_frame_equal(split.full_train, split.train)


def test_global_cutoff_train_before_test_after():
    # Shared timeline: early ratings train, late ratings test.
    rows = []
    for uid in range(1, 6):
        for t in range(1, 11):
            rows.append(
                {
                    "user_id": uid,
                    "item_id": t,
                    "rating": 5.0,
                    "timestamp": t + uid * 0,  # timestamps 1..10 for all
                }
            )
    # Add late interactions so quantile 0.8 falls inside the range.
    for uid in range(1, 6):
        for t in range(11, 16):
            rows.append(
                {
                    "user_id": uid,
                    "item_id": t,
                    "rating": 4.0,
                    "timestamp": t,
                }
            )
    ratings = pd.DataFrame(rows)
    gc = global_time_cutoff_split(
        ratings, GlobalCutoffConfig(timestamp_quantile=0.8, min_train_ratings=5)
    )
    assert gc.train["timestamp"].max() < gc.cutoff_timestamp
    assert gc.test["timestamp"].min() >= gc.cutoff_timestamp
    assert set(gc.test["user_id"]).issubset(set(gc.train["user_id"]))
    assert gc.n_test_interactions_kept == len(gc.test)
    assert gc.n_users_kept == gc.train["user_id"].nunique()


def test_global_cutoff_drops_users_below_min_train():
    rows = [
        {"user_id": 1, "item_id": i, "rating": 5.0, "timestamp": i} for i in range(1, 20)
    ] + [
        {"user_id": 2, "item_id": 1, "rating": 5.0, "timestamp": 1},
        {"user_id": 2, "item_id": 99, "rating": 5.0, "timestamp": 100},
    ]
    ratings = pd.DataFrame(rows)
    gc = global_time_cutoff_split(
        ratings, GlobalCutoffConfig(timestamp_quantile=0.8, min_train_ratings=5)
    )
    assert 2 not in set(gc.train["user_id"])
    assert 2 not in set(gc.test["user_id"])


def test_activity_terciles_partition_and_order():
    counts = {i: i for i in range(1, 10)}  # 1..9
    buckets = assign_activity_terciles(counts)
    assert set(buckets) == {"low", "mid", "high"}
    all_users = buckets["low"] + buckets["mid"] + buckets["high"]
    assert sorted(all_users) == list(range(1, 10))
    assert len(all_users) == len(set(all_users))
    # low has smallest counts
    assert max(counts[u] for u in buckets["low"]) <= min(counts[u] for u in buckets["mid"])
    assert max(counts[u] for u in buckets["mid"]) <= min(counts[u] for u in buckets["high"])


def test_head_tail_fraction_and_disjoint():
    rows = []
    # Items 1..10 with decreasing popularity; item 1 most popular.
    for item in range(1, 11):
        for u in range(1, 12 - item):
            rows.append(
                {"user_id": u, "item_id": item, "rating": 4.0, "timestamp": 1}
            )
    train = pd.DataFrame(rows)
    ht = head_tail_items(train, head_fraction=0.2)
    assert ht["head"].isdisjoint(ht["tail"])
    assert ht["head"] | ht["tail"] == set(train["item_id"])
    assert len(ht["head"]) == 2  # top 20% of 10
    assert 1 in ht["head"]


def test_item_segment_metric_restricts_relevant_and_recs():
    recommendations = {1: [10, 20, 30, 40], 2: [20, 99]}
    relevant = {1: {10, 99}, 2: {20, 30}}
    segment = {10, 20, 30}
    # User 1: relevant∩S={10}; recs filtered=[10,20,30]
    # User 2: relevant∩S={20,30}; recs filtered=[20]
    out = ndcg_for_item_segment(
        recommendations, relevant, segment, k=2, n_bootstrap=50, seed=0
    )
    assert out["n_users"] == 2
    assert out["n_segment_items"] == 3
    assert 0.0 <= out["ndcg@2"] <= 1.0
    assert filter_recs_to_segment([10, 99, 20], segment, 2) == [10, 20]


def test_val_fraction_out_of_range_raises():
    with pytest.raises(ValueError, match="val_fraction"):
        time_based_split(
            _multi_user_frame(),
            SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=1.0),
        )
