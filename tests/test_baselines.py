"""Light unit tests for baseline recommenders on toy data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movielens_recommender.baselines import (
    ALSRecommender,
    EASERecommender,
    ItemItemCosineRecommender,
    MostPopularRecommender,
    RP3betaRecommender,
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


def test_item_item_batch_topk_matches_recommend():
    model = ItemItemCosineRecommender().fit(_toy_train())
    batch = model.topk_with_scores(3)
    for uid in (1, 2, 3, 4):
        assert [item for item, _score in batch[uid]] == model.recommend(uid, 3)


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


def _closed_form_train() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 5.0, "timestamp": 1},
            {"user_id": 1, "item_id": 20, "rating": 4.0, "timestamp": 2},
            {"user_id": 2, "item_id": 10, "rating": 3.0, "timestamp": 1},
            {"user_id": 2, "item_id": 30, "rating": 5.0, "timestamp": 2},
            {"user_id": 3, "item_id": 20, "rating": 4.0, "timestamp": 1},
            {"user_id": 3, "item_id": 30, "rating": 4.0, "timestamp": 2},
        ]
    )


def _binary_x(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    items = np.sort(train["item_id"].unique())
    users = np.sort(train["user_id"].unique())
    x = np.zeros((len(users), len(items)), dtype=np.float64)
    item_index = {int(it): i for i, it in enumerate(items)}
    user_index = {int(u): i for i, u in enumerate(users)}
    for row in train.itertuples(index=False):
        x[user_index[int(row.user_id)], item_index[int(row.item_id)]] = 1.0
    return x, items


def test_ease_diag_zero_shape_and_closed_form():
    train = _closed_form_train()
    model = EASERecommender(l2=1.0).fit(train)
    sim = model.similarity
    n_items = train["item_id"].nunique()
    assert sim.shape == (n_items, n_items)
    assert np.all(np.diag(sim) == 0.0)

    x, _items = _binary_x(train)
    gram = x.T @ x
    gram.flat[:: n_items + 1] += 1.0
    precision = np.linalg.inv(gram)
    expected = np.eye(n_items) - precision / np.diag(precision)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(sim, expected, atol=1e-10)


def test_ease_excludes_seen_and_unknown_user():
    train = _toy_train()
    model = EASERecommender(l2=10.0).fit(train)
    recs = model.recommend(user_id=1, n=5)
    seen = set(train.loc[train["user_id"] == 1, "item_id"].astype(int))
    assert seen.isdisjoint(recs)
    assert recs  # the one unseen catalog item is still eligible
    assert model.recommend(user_id=9999, n=5) == []


def test_rp3beta_nonnegative_topk_and_closed_form():
    train = _closed_form_train()
    top_k = 1
    model = RP3betaRecommender(alpha=0.8, beta=0.4, top_k=top_k).fit(train)
    sim = model.similarity
    n_items = train["item_id"].nunique()
    assert sim.shape == (n_items, n_items)
    assert np.all(sim >= 0.0)
    assert np.all(np.diag(sim) == 0.0)
    assert np.all((sim > 0.0).sum(axis=1) <= top_k)

    # alpha=1, beta=0.5, top_k larger than the catalog: full walk, no pruning.
    full = RP3betaRecommender(alpha=1.0, beta=0.5, top_k=10).fit(train)
    x, _items = _binary_x(train)
    user_degree = x.sum(axis=1, keepdims=True)
    item_degree = x.sum(axis=0)
    p_ui = x / user_degree
    p_iu = (x / item_degree).T
    expected = p_iu @ p_ui
    expected *= np.power(item_degree, -0.5)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(full.similarity, expected, atol=1e-10)
    assert np.all(full.similarity >= 0.0)


def test_rp3beta_alpha_zero_keeps_nonnegative_topk():
    model = RP3betaRecommender(alpha=0.0, beta=0.6, top_k=1).fit(_closed_form_train())
    sim = model.similarity
    assert sim.shape == (3, 3)
    assert np.all(sim >= 0.0)
    assert np.all(np.diag(sim) == 0.0)
    assert np.all((sim > 0.0).sum(axis=1) <= 1)


def test_rp3beta_excludes_seen_and_unknown_user():
    train = _toy_train()
    model = RP3betaRecommender(alpha=1.0, beta=0.3, top_k=2).fit(train)
    recs = model.recommend(user_id=1, n=5)
    seen = set(train.loc[train["user_id"] == 1, "item_id"].astype(int))
    assert seen.isdisjoint(recs)
    assert recs
    assert model.recommend(user_id=9999, n=5) == []
    pruned = model.similarity
    assert np.all(pruned >= 0.0)
    assert np.all((pruned > 0.0).sum(axis=1) <= 2)
