"""Unit tests for the two-tower model on tiny synthetic data (no MovieLens download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from movielens_recommender.config import TWO_TOWER_LOSS_NAMES, load_config  # noqa: E402
from movielens_recommender.movies import GENRES, parse_year_from_title  # noqa: E402
from movielens_recommender.split import SplitConfig, time_based_split  # noqa: E402
from movielens_recommender.two_tower.features import (  # noqa: E402
    build_features,
    pad_histories,
)
from movielens_recommender.two_tower.model import TwoTowerModel  # noqa: E402
from movielens_recommender.two_tower.train import (  # noqa: E402
    TWO_TOWER_LOSSES,
    fit_two_tower_recommender,
    full_softmax_loss,
    resolve_two_tower_loss,
    sampled_softmax_logq_loss,
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


def _toy_hyperparams(**overrides):
    hp = {
        "embedding_dim": 16,
        "learning_rate": 1e-2,
        "temperature": 0.1,
        "batch_size": 32,
        "weight_decay": 0.0,
        "max_epochs": 1,
        "patience": 5,
        "max_history": 10,
    }
    hp.update(overrides)
    return hp


def test_loss_names_match_config_and_default_is_in_batch():
    assert tuple(TWO_TOWER_LOSS_NAMES) == TWO_TOWER_LOSSES
    assert resolve_two_tower_loss(None) == "in_batch"
    assert resolve_two_tower_loss({}) == "in_batch"
    assert resolve_two_tower_loss({"loss": "in_batch"}) == "in_batch"
    root = Path(__file__).resolve().parents[1]
    for name in ("default.yaml", "ml-1m.yaml", "ml-32m.yaml"):
        cfg = load_config(root / "configs" / name)
        assert cfg.models.two_tower.loss == "in_batch"
        assert cfg.models.two_tower.n_negatives == 256


def test_loader_rejects_unknown_loss(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("models:\n  two_tower:\n    loss: bpr\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two-tower loss"):
        load_config(path)


def test_default_loss_matches_explicit_in_batch():
    train = _toy_ratings()
    movies = _toy_movies()
    omitted = _toy_hyperparams()
    explicit = _toy_hyperparams(loss="in_batch")
    rec_a, _, _ = fit_two_tower_recommender(
        train,
        dataset="synthetic",
        movies=movies,
        hyperparams=omitted,
        seed=0,
        relevance_threshold=4.0,
        n_epochs=1,
    )
    rec_b, _, _ = fit_two_tower_recommender(
        train,
        dataset="synthetic",
        movies=movies,
        hyperparams=explicit,
        seed=0,
        relevance_threshold=4.0,
        n_epochs=1,
    )
    assert rec_a.recommend(1, 5) == rec_b.recommend(1, 5)


def test_unknown_loss_raises_during_fit():
    with pytest.raises(ValueError, match="two-tower loss"):
        fit_two_tower_recommender(
            _toy_ratings(),
            dataset="synthetic",
            movies=_toy_movies(),
            hyperparams=_toy_hyperparams(loss="bpr"),
            seed=0,
            relevance_threshold=4.0,
            n_epochs=1,
        )


def test_full_softmax_matches_logsumexp():
    logits = torch.tensor([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])
    targets = torch.tensor([1, 2])
    got = full_softmax_loss(logits, targets)
    manual = torch.stack(
        [-(row[t] - torch.logsumexp(row, dim=0)) for row, t in zip(logits, targets, strict=True)]
    ).mean()
    assert torch.allclose(got, manual)


def test_logq_uniform_is_constant_shift():
    pos = torch.tensor([1.0, 0.2])
    neg = torch.tensor([[0.3, -0.4], [0.5, 0.1]])
    log_q = torch.log(torch.tensor(0.25))
    corrected = sampled_softmax_logq_loss(
        pos,
        neg,
        log_q.expand(2),
        log_q.expand(2),
    )
    plain = sampled_softmax_logq_loss(pos, neg, torch.zeros(2), torch.zeros(2))
    assert torch.allclose(corrected, plain, atol=1e-6)


def test_logq_downweights_popular_classes():
    """A more popular class has a smaller corrected logit, at equal scores."""
    pos = torch.zeros(1)
    neg = torch.zeros(1, 2)
    popular_negative = sampled_softmax_logq_loss(
        pos,
        neg,
        torch.tensor([torch.log(torch.tensor(0.1))]),
        torch.tensor([torch.log(torch.tensor(0.8)), torch.log(torch.tensor(0.05))]),
    )
    rare_negative = sampled_softmax_logq_loss(
        pos,
        neg,
        torch.tensor([torch.log(torch.tensor(0.1))]),
        torch.tensor([torch.log(torch.tensor(0.01)), torch.log(torch.tensor(0.05))]),
    )
    assert popular_negative < rare_negative

    popular_positive = sampled_softmax_logq_loss(
        pos,
        neg,
        torch.tensor([torch.log(torch.tensor(0.8))]),
        torch.log(torch.tensor([0.05, 0.05])),
    )
    rare_positive = sampled_softmax_logq_loss(
        pos,
        neg,
        torch.tensor([torch.log(torch.tensor(0.01))]),
        torch.log(torch.tensor([0.05, 0.05])),
    )
    assert popular_positive > rare_positive


def test_logq_masks_negative_that_copies_the_positive():
    loss = sampled_softmax_logq_loss(
        torch.tensor([1.0]),
        torch.tensor([[5.0]]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        positive_index=torch.tensor([3]),
        negative_index=torch.tensor([3]),
    )
    assert torch.allclose(loss, torch.zeros(()))


def test_lock_plan_uses_timings_not_metrics():
    from movielens_recommender.two_tower.plan import lock_plan

    def block(epoch: dict[str, float]) -> dict:
        return {
            "losses": {
                name: {"extrapolated_epoch_train_sec": epoch[name]}
                for name in ("in_batch", "full_softmax", "sampled_softmax")
            },
            "val_score_sec_per_epoch": 10.0,
        }

    cheap = lock_plan(
        {
            "datasets": {
                "ml-1m": block({"in_batch": 10.0, "full_softmax": 15.0, "sampled_softmax": 12.0})
            }
        },
        {"ml-1m": 6},
    )
    ml1m = cheap["datasets"]["ml-1m"]
    assert cheap["ranking_metrics_used"] is False
    assert ml1m["n_trials"] == 2 * 2 * 3 * 3
    assert ml1m["test"]["seeds"] == [42, 43, 44]
    assert ml1m["test"]["test_both_losses"] is True
    assert ml1m["cap_within_limit"] is True

    slow = lock_plan(
        {
            "datasets": {
                "ml-32m": block(
                    {"in_batch": 500.0, "full_softmax": 4000.0, "sampled_softmax": 600.0}
                )
            }
        },
        {"ml-32m": 5},
    )
    ml32 = slow["datasets"]["ml-32m"]
    assert ml32["n_trials"] < 24
    assert ml32["test"]["seeds"] == [42]
    assert ml32["test"]["test_both_losses"] is False
    assert any(row["loss"] == "full_softmax" for row in ml32["grid"])
    assert any(row["learning_rate"] == 0.0003 for row in ml32["grid"])

    measured = lock_plan(
        {
            "datasets": {
                "ml-32m": block(
                    {"in_batch": 1312.0, "full_softmax": 6852.6, "sampled_softmax": 1182.6}
                )
            }
        },
        {"ml-32m": 5},
    )
    chosen = measured["datasets"]["ml-32m"]
    assert chosen["cap_within_limit"] is True
    assert chosen["test"]["seeds"] == [42]
    assert chosen["test"]["test_both_losses"] is False
    full = [row for row in chosen["grid"] if row["loss"] == "full_softmax"]
    sampled = [row for row in chosen["grid"] if row["loss"] == "sampled_softmax"]
    assert {row["learning_rate"] for row in full} == {0.0003, 0.001}
    assert {row["max_epochs"] for row in full} == {4}
    assert {row["temperature"] for row in sampled} == {0.05, 0.1}
    assert {row["embedding_dim"] for row in sampled} == {32, 64}
    assert {row["max_epochs"] for row in sampled} == {3}


def test_full_and_sampled_softmax_train_on_toy_data():
    train = _toy_ratings()
    movies = _toy_movies()
    for loss_name in ("full_softmax", "sampled_softmax"):
        rec, _, result = fit_two_tower_recommender(
            train,
            dataset="synthetic",
            movies=movies,
            hyperparams=_toy_hyperparams(loss=loss_name, n_negatives=4),
            seed=0,
            relevance_threshold=4.0,
            n_epochs=1,
        )
        assert result.epochs_trained == 1
        recs = rec.recommend(1, 3)
        assert 1 <= len(recs) <= 3


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
    batch = rec_a.topk_with_scores(3)
    assert [item for item, _score in batch[1]] == rec_a.recommend(1, 3)
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
