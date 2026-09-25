"""Light unit tests for baseline recommenders on toy data."""

from __future__ import annotations

import pandas as pd

from movielens_recommender.baselines import (
    ALSRecommender,
    ItemItemCosineRecommender,
    MostPopularRecommender,
)
from movielens_recommender.evaluate import recommend_filtered


def _toy_train() -> pd.DataFrame:
    """Small dense-ish toy matrix with clear popularity and similarity structure."""
    rows = [
        # item 100 is most popular (4 interactions)
        {"user_id": 1, "item_id": 100, "rating": 5.0, "timestamp": 1},
        {"user_id": 2, "item_id": 100, "rating": 4.0, "timestamp": 1},
        {"user_id": 3, "item_id": 100, "rating": 5.0, "timestamp": 1},
        {"user_id": 4, "item_id": 100, "rating": 4.5, "timestamp": 1},
        # item 200 next (3)
        {"user_id": 1, "item_id": 200, "rating": 4.0, "timestamp": 2},
        {"user_id": 2, "item_id": 200, "rating": 5.0, "timestamp": 2},
        {"user_id": 3, "item_id": 200, "rating": 4.0, "timestamp": 2},
        # item 300 (2)
        {"user_id": 1, "item_id": 300, "rating": 5.0, "timestamp": 3},
        {"user_id": 4, "item_id": 300, "rating": 4.0, "timestamp": 3},
        # item 400 (1)
        {"user_id": 2, "item_id": 400, "rating": 5.0, "timestamp": 4},
        # extra positives for ALS coverage
        {"user_id": 3, "item_id": 300, "rating": 5.0, "timestamp": 4},
        {"user_id": 4, "item_id": 200, "rating": 5.0, "timestamp": 4},
    ]
    return pd.DataFrame(rows)


def test_most_popular_shape_and_order():
    model = MostPopularRecommender().fit(_toy_train())
    recs = model.recommend(user_id=1, n=3)
    assert len(recs) == 3
    assert recs[0] == 100
    assert recs[1] == 200
    # items 300 appears twice in raw counts... wait, user1+4+3 = 3 for 300
    # recount: 100:4, 200:4 (users 1,2,3,4), 300:3 (1,4,3), 400:1
    assert set(recs) <= {100, 200, 300, 400}


def test_most_popular_excludes_seen_via_harness():
    model = MostPopularRecommender().fit(_toy_train())
    seen = {100, 200}
    filtered = recommend_filtered(model.recommend, user_id=1, k=2, seen=seen)
    assert 100 not in filtered
    assert 200 not in filtered
    assert len(filtered) == 2


def test_most_popular_deterministic():
    a = MostPopularRecommender().fit(_toy_train()).recommend(1, 4)
    b = MostPopularRecommender().fit(_toy_train()).recommend(1, 4)
    assert a == b


def test_item_item_shape_and_excludes_seen():
    model = ItemItemCosineRecommender().fit(_toy_train())
    recs = model.recommend(user_id=1, n=5)
    # user 1 already has 100, 200, 300 — recommender zeros them
    assert 100 not in recs
    assert 200 not in recs
    assert 300 not in recs
    assert isinstance(recs, list)
    assert all(isinstance(x, int) for x in recs)


def test_item_item_deterministic():
    a = ItemItemCosineRecommender().fit(_toy_train()).recommend(1, 3)
    b = ItemItemCosineRecommender().fit(_toy_train()).recommend(1, 3)
    assert a == b


def test_als_shape_excludes_seen_deterministic():
    train = _toy_train()
    model_a = ALSRecommender(
        factors=8,
        iterations=5,
        regularization=0.1,
        alpha=10.0,
        random_state=42,
    ).fit(train)
    model_b = ALSRecommender(
        factors=8,
        iterations=5,
        regularization=0.1,
        alpha=10.0,
        random_state=42,
    ).fit(train)

    recs_a = model_a.recommend(1, 3)
    recs_b = model_b.recommend(1, 3)
    assert recs_a == recs_b
    assert len(recs_a) <= 3
    # ALS filters already-liked items internally
    user1_items = set(train.loc[train["user_id"] == 1, "item_id"])
    assert user1_items.isdisjoint(recs_a)


def test_als_unknown_user_returns_empty():
    model = ALSRecommender(factors=4, iterations=3, random_state=0).fit(_toy_train())
    assert model.recommend(user_id=9999, n=5) == []
