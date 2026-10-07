"""ml-32M loader fixture, eval-user sampling, and EASE head-catalog restriction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.baselines import EASERecommender, ItemItemCosineRecommender
from movielens_recommender.baselines.rp3beta import RP3betaRecommender
from movielens_recommender.data import DATASET_SHA256, download_dataset, load_ratings
from movielens_recommender.movies import load_movies
from movielens_recommender.scale import apply_eval_user_sample, sample_user_ids
from movielens_recommender.split import SplitConfig, time_based_split


def test_ml32m_checksum_is_pinned():
    digest = DATASET_SHA256["ml-32m"]
    assert len(digest) == 64
    assert digest == "e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee"


def test_download_refuses_when_disabled(monkeypatch):
    monkeypatch.setenv("MOVIELENS_ALLOW_DOWNLOAD", "0")
    with pytest.raises(RuntimeError, match="MOVIELENS_ALLOW_DOWNLOAD"):
        download_dataset("ml-32m")


def test_ml32m_csv_fixture_parses(tmp_path):
    root = tmp_path / "ml-32m" / "ml-32m"
    root.mkdir(parents=True)
    (root / "ratings.csv").write_text(
        "userId,movieId,rating,timestamp\n"
        "1,10,4.0,100\n"
        "1,10,5.0,200\n"
        "2,11,3.5,150\n"
        "2,12,0.0,160\n",
        encoding="utf-8",
    )
    (root / "movies.csv").write_text(
        "movieId,title,genres\n"
        '10,"Toy Story (1995)",Adventure|Animation|Children|Comedy|Fantasy\n'
        '11,"Jumanji (1995)",Adventure|Children|Fantasy\n',
        encoding="utf-8",
    )
    ratings, stats = load_ratings("ml-32m", tmp_path, clean=True)
    assert stats.n_dropped_duplicates == 1
    assert stats.n_dropped_invalid_rating == 1
    assert len(ratings) == 2
    kept = ratings.set_index(["user_id", "item_id"])
    assert kept.loc[(1, 10), "rating"] == pytest.approx(5.0)
    assert int(kept.loc[(1, 10), "timestamp"]) == 200
    movies = load_movies("ml-32m", tmp_path)
    toy = movies.loc[movies["item_id"] == 10].iloc[0]
    assert toy["year"] == pytest.approx(1995.0)
    assert "Animation" in toy["genres"]


def test_sample_user_ids_is_deterministic_and_sorted():
    ids = np.array([5, 1, 9, 3, 7, 1], dtype=np.int64)
    first = sample_user_ids(ids, 3, seed=42)
    second = sample_user_ids(ids, 3, seed=42)
    other = sample_user_ids(ids, 3, seed=7)
    assert list(first) == list(second)
    assert list(first) == sorted(first)
    assert set(first.tolist()).issubset({1, 3, 5, 7, 9})
    assert len(first) == 3
    assert list(first) != list(other)
    assert list(sample_user_ids(ids, 0, seed=1)) == [1, 3, 5, 7, 9]


def _ratings_for_sample() -> pd.DataFrame:
    """Six users, shifted catalogs so each user's last item is in someone else's train."""
    rows = []
    for user_id in range(1, 7):
        for step in range(6):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": 100 + ((user_id - 1 + step) % 8),
                    "rating": 5.0 if step >= 4 else 3.0,
                    "timestamp": user_id * 100 + step,
                }
            )
    return pd.DataFrame(rows)


def test_eval_sample_keeps_full_histories_and_is_stable():
    split = time_based_split(
        _ratings_for_sample(),
        SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=0.1),
    )
    full_before = split.full_train.copy()
    first = apply_eval_user_sample(
        split, n_users=2, seed=42, relevance_threshold=4.0
    )
    second = apply_eval_user_sample(
        split, n_users=2, seed=42, relevance_threshold=4.0
    )
    assert first.eval_user_sample["user_ids_sha256"] == second.eval_user_sample[
        "user_ids_sha256"
    ]
    assert first.eval_user_sample["n_sampled_users"] == 2
    sampled_users = set(first.test["user_id"].astype(int))
    assert sampled_users == set(second.test["user_id"].astype(int))
    assert set(first.full_train["user_id"].astype(int)) == set(
        full_before["user_id"].astype(int)
    )
    uid = next(iter(sampled_users))
    before_items = set(full_before.loc[full_before["user_id"] == uid, "item_id"].astype(int))
    after_items = set(
        first.full_train.loc[first.full_train["user_id"] == uid, "item_id"].astype(int)
    )
    assert before_items == after_items
    assert len(after_items) > len(
        set(first.test.loc[first.test["user_id"] == uid, "item_id"].astype(int))
    )


def test_ease_top_n_matches_closed_form_and_drops_the_tail():
    rows = []
    for user_id in range(1, 5):
        rows.append({"user_id": user_id, "item_id": 30, "rating": 5.0, "timestamp": 1})
    for user_id in range(1, 4):
        rows.append({"user_id": user_id, "item_id": 10, "rating": 4.0, "timestamp": 2})
    rows.append({"user_id": 1, "item_id": 20, "rating": 5.0, "timestamp": 3})
    train = pd.DataFrame(rows)

    restricted = EASERecommender(l2=1.0, max_items=2).fit(train)
    assert restricted.n_items_before_restriction == 3
    assert restricted.n_items_fit == 2
    assert set(int(i) for i in restricted._item_ids) == {10, 30}

    head = train.loc[train["item_id"].isin([10, 30])]
    reference = EASERecommender(l2=1.0).fit(head)
    np.testing.assert_allclose(restricted.similarity, reference.similarity, atol=1e-8)

    uncached = EASERecommender(l2=1.0, max_items=2, cache_user_scores=False).fit(train)
    for user_id in train["user_id"].unique():
        uid = int(user_id)
        assert 20 not in restricted.recommend(uid, 5)
        assert restricted.recommend(uid, 5) == uncached.recommend(uid, 5)


def test_ease_top_n_tie_breaks_toward_smaller_item_id():
    rows = []
    for user_id, item_id in ((1, 8), (2, 8), (1, 3), (3, 3), (2, 9)):
        rows.append(
            {"user_id": user_id, "item_id": item_id, "rating": 4.0, "timestamp": 1}
        )
    train = pd.DataFrame(rows)
    model = EASERecommender(l2=5.0, max_items=1).fit(train)
    assert [int(i) for i in model._item_ids] == [3]


def test_sparse_item_item_matches_dense_topk():
    rows = [
        {"user_id": 1, "item_id": 100, "rating": 5.0, "timestamp": 1},
        {"user_id": 2, "item_id": 100, "rating": 4.0, "timestamp": 1},
        {"user_id": 3, "item_id": 100, "rating": 5.0, "timestamp": 1},
        {"user_id": 1, "item_id": 200, "rating": 4.0, "timestamp": 2},
        {"user_id": 2, "item_id": 200, "rating": 5.0, "timestamp": 2},
        {"user_id": 1, "item_id": 300, "rating": 5.0, "timestamp": 3},
        {"user_id": 3, "item_id": 300, "rating": 4.0, "timestamp": 3},
        {"user_id": 2, "item_id": 400, "rating": 5.0, "timestamp": 4},
    ]
    train = pd.DataFrame(rows)
    dense = ItemItemCosineRecommender(k_neighbors=2, shrinkage=10.0).fit(train)
    sparse = ItemItemCosineRecommender(
        k_neighbors=2, shrinkage=10.0, sparse_topk=True
    ).fit(train)
    for user_id in train["user_id"].unique():
        uid = int(user_id)
        assert dense.recommend(uid, 3) == sparse.recommend(uid, 3)
    batched = sparse.topk_with_scores(3, user_ids=[int(u) for u in train["user_id"].unique()])
    for user_id in train["user_id"].unique():
        uid = int(user_id)
        dense_ids = [item_id for item_id, _score in dense.topk_for_user(uid, 3)]
        sparse_ids = [item_id for item_id, _score in batched[uid]]
        assert dense_ids == sparse_ids


def test_rp3beta_block_matches_dense_similarity():
    train = pd.DataFrame(
        [
            {"user_id": 1, "item_id": 10, "rating": 5.0, "timestamp": 1},
            {"user_id": 1, "item_id": 20, "rating": 4.0, "timestamp": 2},
            {"user_id": 2, "item_id": 10, "rating": 3.0, "timestamp": 1},
            {"user_id": 2, "item_id": 30, "rating": 5.0, "timestamp": 2},
            {"user_id": 3, "item_id": 20, "rating": 4.0, "timestamp": 1},
            {"user_id": 3, "item_id": 30, "rating": 4.0, "timestamp": 2},
        ]
    )
    dense = RP3betaRecommender(alpha=0.8, beta=0.4, top_k=2).fit(train)
    blocked = RP3betaRecommender(
        alpha=0.8, beta=0.4, top_k=2, block_topk=True, cache_user_scores=False
    ).fit(train)
    np.testing.assert_allclose(dense.similarity, blocked.similarity, atol=1e-8)
    for user_id in (1, 2, 3):
        assert dense.recommend(user_id, 2) == blocked.recommend(user_id, 2)
