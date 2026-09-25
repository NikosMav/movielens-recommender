"""Tests for coverage CI policy and bootstrap validity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.evaluate import COVERAGE_CI_POLICY, evaluate_recommender
from movielens_recommender.metrics import bootstrap_mean_ci, catalog_coverage
from movielens_recommender.split import SplitConfig, SplitResult


def test_user_bootstrap_coverage_is_biased_low():
    """Demonstrate why user-bootstrap CIs are invalid for catalog coverage.

    With disjoint per-user recommendations, dropping unique users (as with
    replacement sampling) shrinks the union and underestimates coverage. The
    full-sample point estimate then lies above the bootstrap percentile CI.
    """
    catalog = set(range(20))
    # 10 users, each exclusively recommending 2 distinct items → coverage=1.0
    recommendations = {u: [2 * u, 2 * u + 1] for u in range(10)}
    point = catalog_coverage(recommendations, catalog, k=2)
    assert point == pytest.approx(1.0)

    rng = np.random.default_rng(0)
    user_ids = list(recommendations.keys())
    samples: list[float] = []
    for _ in range(500):
        sample_ids = rng.choice(user_ids, size=len(user_ids), replace=True)
        uniq = {int(uid): recommendations[int(uid)] for uid in set(int(x) for x in sample_ids)}
        samples.append(catalog_coverage(uniq, catalog, k=2))

    low = float(np.quantile(samples, 0.025))
    high = float(np.quantile(samples, 0.975))
    # The bug condition: point estimate outside the naive user-bootstrap CI.
    assert point > high
    assert low < high < point


def test_evaluate_omits_coverage_ci_and_keeps_valid_cis():
    train = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 5.0, "timestamp": 1},
            {"user_id": 1, "item_id": 20, "rating": 4.0, "timestamp": 2},
            {"user_id": 2, "item_id": 10, "rating": 5.0, "timestamp": 1},
            {"user_id": 2, "item_id": 30, "rating": 4.0, "timestamp": 2},
            {"user_id": 3, "item_id": 20, "rating": 5.0, "timestamp": 1},
            {"user_id": 3, "item_id": 30, "rating": 4.0, "timestamp": 2},
        ]
    )
    test = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 30, "rating": 5.0, "timestamp": 3},
            {"user_id": 2, "item_id": 20, "rating": 5.0, "timestamp": 3},
            {"user_id": 3, "item_id": 10, "rating": 5.0, "timestamp": 3},
        ]
    )
    split = SplitResult(
        train=train,
        test=test,
        config=SplitConfig(min_ratings=2, test_fraction=0.5, relevance_threshold=4.0),
        n_users_kept=3,
        n_users_dropped=0,
    )

    def recommend(user_id: int, n: int) -> list[int]:
        # Deterministic global ranking; harness filters seen items.
        del user_id
        return [10, 20, 30, 40, 50][:n]

    metrics = evaluate_recommender(
        recommend,
        train,
        test,
        ks=(2,),
        n_bootstrap=200,
        seed=0,
        split=split,
    )

    assert COVERAGE_CI_POLICY == "point_estimate_only"
    assert "coverage@2" in metrics
    assert "coverage@2" not in metrics["confidence_intervals"]
    assert metrics["uncertainty_policy"]["coverage_ci_policy"] == "point_estimate_only"

    for name, bounds in metrics["confidence_intervals"].items():
        assert bounds["low"] <= bounds["mean"] <= bounds["high"], name
        # Point estimate stored on metrics must match CI mean when both exist.
        if name in metrics:
            assert metrics[name] == pytest.approx(bounds["mean"])


def test_bootstrap_mean_ci_contains_mean():
    values = [0.01 * i for i in range(50)]
    mean, low, high = bootstrap_mean_ci(values, n_bootstrap=1000, seed=42)
    assert low <= mean <= high
