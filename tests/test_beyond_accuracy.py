"""Hand-computed checks for the beyond-accuracy metrics (ADR-0017)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.beyond_accuracy import (
    genre_item_vectors,
    long_tail_items,
    score_recommendation_lists,
)
from movielens_recommender.metrics import (
    intra_list_diversity,
    long_tail_share,
    mean_popularity,
    novelty_at_k,
    per_user_intra_list_diversity,
    per_user_novelty,
)
from movielens_recommender.movies import GENRES

# counts 8, 4, 2, 2. total 16.
# self-information: 1 → 1, 2 → 2, 3 → 3, 4 → 3.
POP = {1: 8.0, 2: 4.0, 3: 2.0, 4: 2.0}
VECTORS = {
    1: np.array([1.0, 0.0]),
    2: np.array([1.0, 0.0]),
    3: np.array([0.0, 1.0]),
    4: np.array([1.0, 1.0]),
}
RECS = {10: [1, 2], 20: [3, 4]}
TAIL = {3, 4}


def test_novelty_hand_computed():
    # user 10: (1 + 2) / 2 = 1.5; user 20: (3 + 3) / 2 = 3; mean = 2.25
    assert per_user_novelty(RECS, POP, 2) == pytest.approx([1.5, 3.0])
    assert novelty_at_k(RECS, POP, 2) == pytest.approx(2.25)


def test_novelty_skips_items_with_no_train_count():
    # Only item 1 is known, p = 1, self-information 0. Item 9 is omitted.
    recs = {1: [1, 9]}
    assert novelty_at_k(recs, {1: 4.0}, 2) == pytest.approx(0.0)
    # A list of only unknown items scores 0, not an error.
    assert novelty_at_k({1: [9]}, {1: 4.0}, 1) == pytest.approx(0.0)


def test_intra_list_diversity_hand_computed():
    # user 10: cosine(1, 2) = 1, diversity 0.
    # user 20: cosine([0, 1], [1, 1]) = 1/sqrt(2), diversity 1 - 1/sqrt(2).
    expected_user = [0.0, 1.0 - 1.0 / math.sqrt(2.0)]
    assert per_user_intra_list_diversity(RECS, VECTORS, 2) == pytest.approx(expected_user)
    assert intra_list_diversity(RECS, VECTORS, 2) == pytest.approx(sum(expected_user) / 2.0)


def test_intra_list_diversity_edges():
    identical = {1: [1, 2]}
    assert intra_list_diversity(identical, VECTORS, 2) == pytest.approx(0.0)
    orthogonal = {1: [1, 3]}
    assert intra_list_diversity(orthogonal, VECTORS, 2) == pytest.approx(1.0)
    # One item has no pair.
    assert intra_list_diversity({1: [1]}, VECTORS, 2) == pytest.approx(0.0)
    # A zero vector contributes cosine 0.
    zero = {1: [1, 5]}
    vecs = {1: np.array([1.0, 0.0]), 5: np.array([0.0, 0.0])}
    assert intra_list_diversity(zero, vecs, 2) == pytest.approx(1.0)


def test_long_tail_share_hand_computed():
    # user 10: neither item is in the tail → 0. user 20: both are → 1. mean 0.5.
    assert long_tail_share(RECS, TAIL, 2) == pytest.approx(0.5)
    assert long_tail_share({1: [1, 3, 3, 4]}, {3, 4}, 4) == pytest.approx(0.75)


def test_long_tail_items_bottom_80_percent_and_ties():
    # 10 items, counts 10..1. Head is the top 2 (20%). Tail is the bottom 8.
    rows = []
    for item, count in enumerate(range(10, 0, -1), start=1):
        rows.extend((1, item, 5.0, 1) for _ in range(count))
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    tail = long_tail_items(frame)
    assert tail == set(range(3, 11))

    # n_head = max(1, int(3 * 0.2)) = 1. Equal counts: the smaller item id is head.
    tied = pd.DataFrame(
        [(1, 1, 5.0, 1)] * 5 + [(1, 2, 5.0, 1)] * 5 + [(1, 3, 5.0, 1)],
        columns=["user_id", "item_id", "rating", "timestamp"],
    )
    assert long_tail_items(tied) == {2, 3}


def test_genre_vector_maps_childrens_and_missing_titles():
    movies = pd.DataFrame(
        [
            {"item_id": 7, "title": "Toy (1995)", "genres": "Children's|Comedy", "year": 1995.0},
            {"item_id": 8, "title": "Bare (1995)", "genres": "", "year": 1995.0},
        ]
    )
    vecs = genre_item_vectors(movies, [7, 8, 99])
    children = GENRES.index("Children")
    comedy = GENRES.index("Comedy")
    none = GENRES.index("(no genres listed)")
    assert vecs[7][children] == pytest.approx(1.0)
    assert vecs[7][comedy] == pytest.approx(1.0)
    assert float(vecs[7].sum()) == pytest.approx(2.0)
    assert vecs[8][none] == pytest.approx(1.0)
    assert vecs[99][none] == pytest.approx(1.0)


def test_each_metric_rejects_non_positive_k():
    with pytest.raises(ValueError):
        novelty_at_k(RECS, POP, 0)
    with pytest.raises(ValueError):
        intra_list_diversity(RECS, VECTORS, 0)
    with pytest.raises(ValueError):
        long_tail_share(RECS, TAIL, 0)


def test_scorer_matches_hand_values_and_bootstraps_user_means_only():
    relevant = {10: {2}, 20: {3, 4}}
    scored = score_recommendation_lists(
        RECS,
        relevant=relevant,
        catalog=[1, 2, 3, 4, 5],
        item_popularity=POP,
        item_vectors=VECTORS,
        tail_items=TAIL,
        k=2,
        n_bootstrap=200,
        seed=0,
    )
    assert scored["novelty@2"] == pytest.approx(2.25)
    expected_div = (1.0 - 1.0 / math.sqrt(2.0)) / 2.0
    assert scored["intra_list_diversity@2"] == pytest.approx(expected_div)
    assert scored["long_tail_share@2"] == pytest.approx(0.5)
    assert scored["mean_popularity@2"] == pytest.approx(mean_popularity(RECS, POP, 2))
    assert scored["mean_popularity@2"] == pytest.approx(4.0)
    assert scored["coverage@2"] == pytest.approx(0.8)
    # user 10 hit at rank 2; user 20 is perfect. Mean of those two NDCGs.
    user10 = (1.0 / math.log2(3.0))
    assert scored["ndcg@2"] == pytest.approx((user10 + 1.0) / 2.0)

    cis = scored["confidence_intervals"]
    assert "coverage@2" not in cis
    for name in (
        "novelty@2",
        "intra_list_diversity@2",
        "mean_popularity@2",
        "long_tail_share@2",
        "ndcg@2",
    ):
        bounds = cis[name]
        assert bounds["low"] <= bounds["mean"] <= bounds["high"]
        assert bounds["mean"] == pytest.approx(scored[name])
    again = score_recommendation_lists(
        RECS,
        relevant=relevant,
        catalog=[1, 2, 3, 4, 5],
        item_popularity=POP,
        item_vectors=VECTORS,
        tail_items=TAIL,
        k=2,
        n_bootstrap=200,
        seed=0,
    )
    assert again["confidence_intervals"] == scored["confidence_intervals"]


def _panel_payload() -> dict:
    def ci(value: float) -> dict[str, float]:
        return {"mean": value, "low": value - 0.01, "high": value + 0.01}

    def model(
        ndcg: float,
        novelty: float,
        coverage: float,
        diversity: float,
        popularity: float,
        tail: float,
        *,
        committed: float | None = None,
        headline: float | None = None,
    ) -> dict:
        committed_value = ndcg if committed is None else committed
        headline_value = ndcg if headline is None else headline
        return {
            "ndcg@10": ndcg,
            "coverage@10": coverage,
            "novelty@10": novelty,
            "intra_list_diversity@10": diversity,
            "mean_popularity@10": popularity,
            "long_tail_share@10": tail,
            "confidence_intervals": {
                "ndcg@10": ci(ndcg),
                "novelty@10": ci(novelty),
                "intra_list_diversity@10": ci(diversity),
                "mean_popularity@10": ci(popularity),
                "long_tail_share@10": ci(tail),
            },
            "ndcg@10_committed": committed_value,
            "ndcg@10_headline": headline_value,
            "ndcg@10_matches_committed_at_4_decimals": f"{ndcg:.4f}" == f"{committed_value:.4f}",
            "ndcg@10_matches_headline_at_4_decimals": f"{ndcg:.4f}" == f"{headline_value:.4f}",
        }

    return {
        "dataset": "ml-1m",
        "adr": "docs/adr/0017-beyond-accuracy.md",
        "k": 10,
        "n_eval_users": 4,
        "bootstrap": {"n_bootstrap": 1000, "alpha": 0.05, "seed": 42},
        "model_order": ["most_popular", "item_item_cosine", "als_tuned"],
        "ndcg_comparison_note": "Lists use the committed configs. Two-tower is seed 42.",
        "future_work": "ml-32M is future work. This panel is ml-1m only.",
        "reproduction_command": "python scripts/run_beyond_accuracy.py",
        "runtime_sec": 12.5,
        "host": {
            "os": "Linux",
            "os_release": "6.12",
            "machine": "x86_64",
            "cpu_count": 4,
            "python": "3.12.0",
            "numpy_simd": ["AVX2"],
        },
        "models": {
            "most_popular": model(0.10, 2.0, 0.05, 0.10, 100.0, 0.0),
            "item_item_cosine": model(0.50, 4.0, 0.20, 0.30, 40.0, 0.25),
            "als_tuned": model(
                0.20,
                8.0,
                0.60,
                0.70,
                10.0,
                0.80,
                committed=0.22,
                headline=0.19,
            ),
        },
    }


def test_readme_panel_is_generated_from_the_payload():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from make_results_table import render_beyond_accuracy, render_beyond_accuracy_payload

    assert render_beyond_accuracy(Path("/tmp/does-not-exist-beyond-accuracy.json")) == ""
    payload = _panel_payload()
    text = render_beyond_accuracy_payload(payload)
    assert text == render_beyond_accuracy_payload(payload)
    # Three-sentence reading. It names the NDCG, novelty, and popularity leaders.
    reading = next(block for block in text.split("\n\n") if "leads NDCG@10" in block)
    sentences = [part for part in reading.split(". ") if part]
    assert len(sentences) == 3
    assert "`item_item_cosine`" in reading
    assert "0.5000" in reading
    assert "`als_tuned`" in reading
    assert "8.0000" in reading
    assert "`most_popular`" in reading
    assert "100.0000" in reading
    # Coverage is a bare point estimate in the als row, not a bootstrap interval.
    als_row = next(line for line in text.splitlines() if line.startswith("| als_tuned |"))
    assert "0.6000 [" not in als_row
    assert "0.6000 |" in als_row
    assert "| no |" in als_row
    assert "ml-32M is future work" in text
    assert "does not match" in text
    assert "0.2000 versus committed 0.2200" in text
    assert "versus headline 0.1900" in text
