"""Tests for cleaning rules and checksum pinning metadata."""

from __future__ import annotations

import pandas as pd
import pytest

from movielens_recommender.data import DATASET_SHA256, DATASET_VERSION_LABELS, clean_ratings


def test_checksums_are_pinned():
    assert len(DATASET_SHA256["ml-latest-small"]) == 64
    assert len(DATASET_SHA256["ml-1m"]) == 64
    assert "sha256:" in DATASET_VERSION_LABELS["ml-latest-small"]


def test_clean_drops_nulls_and_invalid_ratings():
    raw = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 1, "rating": 4.0, "timestamp": 10},
            {"user_id": 1, "item_id": 2, "rating": None, "timestamp": 11},
            {"user_id": 1, "item_id": 3, "rating": 6.0, "timestamp": 12},
            {"user_id": 0, "item_id": 4, "rating": 4.0, "timestamp": 13},
            {"user_id": 2, "item_id": 5, "rating": 0.0, "timestamp": 14},
        ]
    )
    clean, stats = clean_ratings(raw)
    assert stats.n_dropped_null == 1
    assert stats.n_dropped_invalid_rating >= 1
    assert stats.n_dropped_invalid_ids >= 1
    assert len(clean) == 1
    assert clean.iloc[0]["item_id"] == 1


def test_clean_dedupes_keeping_latest():
    raw = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 3.0, "timestamp": 1},
            {"user_id": 1, "item_id": 10, "rating": 5.0, "timestamp": 9},
            {"user_id": 1, "item_id": 10, "rating": 4.0, "timestamp": 5},
        ]
    )
    clean, stats = clean_ratings(raw)
    assert len(clean) == 1
    assert stats.n_dropped_duplicates == 2
    assert clean.iloc[0]["rating"] == pytest.approx(5.0)
    assert clean.iloc[0]["timestamp"] == 9
