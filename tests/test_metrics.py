"""Unit tests for ranking + diagnostic metrics (hand-computed where possible)."""

from __future__ import annotations

import math

import pytest

from movielens_recommender.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    mean_popularity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k_hand_computed():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4, 9}
    assert precision_at_k(recs, rel, 3) == pytest.approx(1 / 3)
    assert precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)


def test_recall_at_k_hand_computed():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4, 9}
    assert recall_at_k(recs, rel, 3) == pytest.approx(1 / 3)
    assert recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    assert recall_at_k(recs, set(), 5) == 0.0


def test_ndcg_at_k_hand_computed():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4, 9}
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(recs, rel, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(recs, set(), 5) == 0.0


def test_perfect_ranking_ndcg_is_one():
    assert ndcg_at_k([10, 20, 30], {10, 20, 30}, 3) == pytest.approx(1.0)


def test_catalog_coverage_hand_computed():
    recs = {1: [10, 20, 30], 2: [20, 40, 50]}
    catalog = {10, 20, 30, 40, 50, 60}
    # unique recommended in top-2: {10,20,40} → 3/6
    assert catalog_coverage(recs, catalog, 2) == pytest.approx(0.5)
    assert catalog_coverage(recs, catalog, 3) == pytest.approx(5 / 6)


def test_mean_popularity_hand_computed():
    recs = {1: [10, 20], 2: [20, 30]}
    pop = {10: 100.0, 20: 50.0, 30: 10.0}
    # user1: (100+50)/2=75; user2: (50+10)/2=30; mean=(75+30)/2=52.5
    assert mean_popularity(recs, pop, 2) == pytest.approx(52.5)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    mean, low, high = bootstrap_mean_ci(values, n_bootstrap=500, seed=0)
    assert low <= mean <= high
    mean2, low2, high2 = bootstrap_mean_ci(values, n_bootstrap=500, seed=0)
    assert (mean, low, high) == (mean2, low2, high2)
    # Different seed should usually change bounds (sanity, not a hard guarantee
    # on tiny n — check at least the API runs).
    mean3, low3, high3 = bootstrap_mean_ci(values, n_bootstrap=500, seed=1)
    assert low3 <= mean3 <= high3


def test_bootstrap_single_value():
    mean, low, high = bootstrap_mean_ci([0.42], n_bootstrap=100, seed=0)
    assert mean == low == high == pytest.approx(0.42)


def test_seed_summary_keeps_primary_intervals_separate_from_mean():
    from movielens_recommender.reporting import summarize_seed_metrics

    rows = [
        {"seed": 42, "metrics": {
            "ndcg@10": 0.05, "recall@100": 0.2, "n_eval_users": 10,
            "confidence_intervals": {"ndcg@10": {"mean": 0.05, "low": 0.04, "high": 0.06}},
            "segments": {"primary_only": True},
        }},
        {"seed": 43, "metrics": {
            "ndcg@10": 0.15, "recall@100": 0.4, "n_eval_users": 10,
            "confidence_intervals": {"ndcg@10": {"mean": 0.15, "low": 0.14, "high": 0.16}},
        }},
    ]
    summary = summarize_seed_metrics(rows, primary_seed=42)
    assert summary["seed_mean_metrics"]["ndcg@10"] == pytest.approx(0.1)
    assert summary["recall@100"] == pytest.approx(0.3)
    assert summary["confidence_intervals"] == {}
    assert "segments" not in summary
    assert summary["primary_seed_metrics"]["ndcg@10"] == 0.05
    assert summary["primary_seed_metrics"]["confidence_intervals"]["ndcg@10"]["high"] == 0.06
    assert summary["per_seed_metrics"]["43"]["ndcg@10"] == 0.15
    assert rows[0]["metrics"]["ndcg@10"] == 0.05


def test_seed_summary_rejects_different_evaluation_populations():
    from movielens_recommender.reporting import summarize_seed_metrics

    rows = [
        {"seed": 42, "metrics": {"ndcg@10": 0.1, "n_eval_users": 10}},
        {"seed": 43, "metrics": {"ndcg@10": 0.2, "n_eval_users": 9}},
    ]
    with pytest.raises(ValueError, match="evaluation populations"):
        summarize_seed_metrics(rows, primary_seed=42)
