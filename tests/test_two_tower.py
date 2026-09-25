"""Unit tests for the two-tower model on tiny synthetic data (no MovieLens download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from movielens_recommender.movies import GENRES, parse_year_from_title  # noqa: E402
from movielens_recommender.split import SplitConfig, time_based_split  # noqa: E402
from movielens_recommender.two_tower.features import (  # noqa: E402
    build_features,
    pad_histories,
)
from movielens_recommender.two_tower.model import TwoTowerModel  # noqa: E402
from movielens_recommender.two_tower.train import (  # noqa: E402
    fit_two_tower_recommender,
    sampled_softmax_loss,
    tune_two_tower,
)


def _toy_movies() -> pd.DataFrame:
    rows = []
    catalog = [
        (100, "Alpha (1995)", "Action|Comedy"),
        (200, "Beta (2001)", "Drama"),
        (300, "Gamma (1999)", "Comedy|Romance"),
        (400, "Delta (2010)", "Sci-Fi|Action"),
        (500, "Epsilon", "Horror"),
        (600, "Zeta (1998)", "Thriller"),
        (700, "Eta (2005)", "Adventure|Children"),
        (800, "Theta (2012)", "Documentary"),
    ]
    for iid, title, genres in catalog:
        rows.append({"item_id": iid, "title": title, "genres": genres, "year": None})
    df = pd.DataFrame(rows)
    df["year"] = df["title"].map(parse_year_from_title)
    return df


def _toy_ratings() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    items = [100, 200, 300, 400, 500, 600, 700, 800]
    t = 1
    for uid in range(1, 31):
        n = 6
        chosen = rng.choice(items, size=n, replace=False)
        for item in chosen:
            rows.append(
                {
                    "user_id": uid,
                    "item_id": int(item),
                    "rating": float(rng.choice([3.0, 4.0, 5.0])),
                    "timestamp": t,
                }
            )
            t += 1
    return pd.DataFrame(rows)


def test_parse_year_from_title():
    assert parse_year_from_title("Toy Story (1995)") == 1995.0
    assert parse_year_from_title("No Year Here") is None


def test_build_features_and_pad_excludes_positive():
    train = _toy_ratings()
    movies = _toy_movies()
    feat = build_features(
        train,
        dataset="synthetic",
        movies=movies,
        relevance_threshold=4.0,
        max_history=10,
    )
    assert feat.n_users == train["user_id"].nunique()
    assert feat.n_items == train["item_id"].nunique()
    assert feat.genres.shape == (feat.n_items, len(GENRES))
    assert feat.item_q.shape == (feat.n_items,)
    assert abs(feat.item_q.sum() - 1.0) < 1e-6

    u = feat.pos_user_idx[:4]
    i = feat.pos_item_idx[:4]
    hist, mask = pad_histories(u, feat, exclude_item_idx=i, max_history=10)
    for row in range(len(u)):
        used = hist[row][mask[row] > 0]
        assert int(i[row]) not in set(int(x) for x in used)


def test_sampled_softmax_loss_shapes():
    logits = torch.randn(8, 8)
    log_q = torch.log(torch.full((8,), 1.0 / 8.0))
    loss = sampled_softmax_loss(logits, log_q)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_model_encode_and_recommend_deterministic():
    train = _toy_ratings()
    movies = _toy_movies()
    hp = {
        "embedding_dim": 16,
        "learning_rate": 1e-2,
        "temperature": 0.1,
        "batch_size": 32,
        "weight_decay": 0.0,
        "max_epochs": 2,
        "patience": 5,
        "max_history": 10,
    }
    rec_a, _, _ = fit_two_tower_recommender(
        train,
        dataset="synthetic",
        movies=movies,
        hyperparams=hp,
        seed=0,
        relevance_threshold=4.0,
        n_epochs=2,
    )
    rec_b, _, _ = fit_two_tower_recommender(
        train,
        dataset="synthetic",
        movies=movies,
        hyperparams=hp,
        seed=0,
        relevance_threshold=4.0,
        n_epochs=2,
    )
    assert rec_a.recommend(1, 3) == rec_b.recommend(1, 3)
    recs = rec_a.recommend(1, 3)
    assert 1 <= len(recs) <= 3
    assert all(isinstance(x, int) for x in recs)


def test_tune_two_tower_uses_validation_only():
    ratings = _toy_ratings()
    # Duplicate interactions over time so the split keeps enough rows.
    extra = ratings.copy()
    extra["timestamp"] = ratings["timestamp"] + 1000
    extra["rating"] = 5.0
    ratings = pd.concat([ratings, extra], ignore_index=True)

    split = time_based_split(
        ratings,
        SplitConfig(min_ratings=5, test_fraction=0.2, val_fraction=0.2),
    )
    assert split.val is not None and not split.val.empty
    movies = _toy_movies()
    tiny_grid = [
        {
            "embedding_dim": 8,
            "learning_rate": 1e-2,
            "temperature": 0.1,
            "batch_size": 16,
            "weight_decay": 0.0,
            "max_epochs": 2,
            "patience": 2,
            "max_history": 8,
        }
    ]
    result = tune_two_tower(
        split,
        dataset="synthetic",
        movies=movies,
        relevance_threshold=4.0,
        seed=0,
        grid=tiny_grid,
        show_progress=False,
    )
    assert result.best_val_score >= 0.0
    assert result.best_epoch >= 1
    assert len(result.trials) == 1


def test_two_tower_model_forward_shapes():
    model = TwoTowerModel(n_users=5, n_items=7, n_genres=len(GENRES), embedding_dim=8)
    user_idx = torch.tensor([0, 1])
    hist = torch.tensor([[1, 2, 0], [3, 0, 0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    u = model.encode_users(user_idx, hist, mask)
    assert u.shape == (2, 8)
    assert torch.allclose(u.norm(dim=-1), torch.ones(2), atol=1e-5)

    item_idx = torch.tensor([0, 1, 2])
    genres = torch.zeros(3, len(GENRES))
    years = torch.zeros(3)
    v = model.encode_items(item_idx, genres, years)
    assert v.shape == (3, 8)
