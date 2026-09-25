"""Unit tests for ranking metrics against hand-computed values."""

from __future__ import annotations

import math

import pytest

from movielens_recommender.metrics import ndcg_at_k, precision_at_k, recall_at_k


def test_precision_at_k_hand_computed():
    # recommended: [1, 2, 3, 4, 5]; relevant: {2, 4, 9}
    # hits in top-3: {2} -> 1/3; top-5: {2,4} -> 2/5
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4, 9}
    assert precision_at_k(recs, rel, 3) == pytest.approx(1 / 3)
    assert precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)


def test_recall_at_k_hand_computed():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4, 9}
    # hits top-3: 1 of 3 relevant -> 1/3; top-5: 2 of 3 -> 2/3
    assert recall_at_k(recs, rel, 3) == pytest.approx(1 / 3)
    assert recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    assert recall_at_k(recs, set(), 5) == 0.0


def test_ndcg_at_k_hand_computed():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4, 9}
    # DCG@5: rank2 (item2) + rank4 (item4) = 1/log2(3) + 1/log2(5)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    # IDCG@5 with 3 relevant: 1/log2(2) + 1/log2(3) + 1/log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(recs, rel, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(recs, set(), 5) == 0.0


def test_perfect_ranking_ndcg_is_one():
    recs = [10, 20, 30]
    rel = {10, 20, 30}
    assert ndcg_at_k(recs, rel, 3) == pytest.approx(1.0)
