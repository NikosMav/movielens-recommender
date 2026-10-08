"""Shared synthetic serving snapshots. No MovieLens download.

Heavy imports (torch, LightGBM) stay inside the fixtures so modules that do
not use them still collect without the optional extras.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.movies import GENRES

SERVING_SHA256 = "synthetic-sha256"
SERVING_GIT_SHA = "0123456789abcdef0123456789abcdef01234567"


def _ratings_movies_users() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from movielens_recommender.data import parse_users

    # Rotate each user's chronology so a validation item is in some other
    # user's fit-train. Otherwise that item is cold and retrieval recall has
    # no warm relevant users.
    catalog = list(range(1, 13))
    rows = []
    for user_id in range(1, 9):
        shift = user_id % len(catalog)
        order = catalog[shift:] + catalog[:shift]
        for step, item_id in enumerate(order, start=1):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "rating": 5.0 if step % 2 == 0 else 4.0,
                    "timestamp": user_id * 100 + step,
                }
            )
    ratings = pd.DataFrame(rows)
    movies = pd.DataFrame(
        [
            {
                "item_id": item_id,
                "title": f"Movie {item_id} (199{item_id % 10})",
                "genres": GENRES[(item_id - 1) % 8],
                "year": float(1990 + (item_id % 10)),
            }
            for item_id in range(1, 13)
        ]
    )
    users = parse_users(
        pd.DataFrame(
            [
                {
                    "user_id": user_id,
                    "gender": "F" if user_id % 2 == 0 else "M",
                    "age": 25,
                    "occupation": user_id % 5,
                    "zip_code": "02139",
                }
                for user_id in range(1, 9)
            ]
        )
    )
    return ratings, movies, users


def _config():
    from movielens_recommender.config import (
        EvalYAML,
        ItemKNNYAML,
        ModelsYAML,
        RankerYAML,
        RunConfig,
        SplitYAML,
        TwoTowerYAML,
    )

    return RunConfig(
        seed=42,
        dataset="synthetic",
        tune=True,
        split=SplitYAML(min_ratings=5, test_fraction=0.2, val_fraction=0.1),
        eval=EvalYAML(ks=[10], relevance_threshold=4.0, n_bootstrap=10),
        models=ModelsYAML(
            item_item_cosine=ItemKNNYAML(min_common=1, k_neighbors=5, shrinkage=0.0),
            two_tower=TwoTowerYAML(enabled=True, embedding_dim=8, seeds=[42]),
            ranker=RankerYAML(
                enabled=True,
                candidate_k=20,
                early_stop_fraction=0.25,
                num_boost_round=8,
                early_stopping_rounds=2,
                learning_rate=0.1,
                num_leaves=4,
                min_data_in_leaf=1,
                feature_fraction=1.0,
                bagging_fraction=1.0,
                seeds=[42],
                demographics="both",
            ),
        ),
    )


@pytest.fixture(scope="session")
def artifact_dir(tmp_path_factory):
    """A production serving snapshot for 8 users and 12 movies."""
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    from movielens_recommender.serving.build import fit_serving_bundle
    from movielens_recommender.serving.bundle import (
        load_bundle,
        recommend_for_user,
        save_bundle,
    )
    from movielens_recommender.split import SplitConfig, time_based_split

    ratings, movies, users = _ratings_movies_users()
    config = _config()
    split = time_based_split(
        ratings,
        SplitConfig(
            min_ratings=config.split.min_ratings,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            relevance_threshold=config.eval.relevance_threshold,
        ),
    )
    bundle = fit_serving_bundle(
        config=config,
        split=split,
        movies=movies,
        users=users,
        item_item_hp={"min_common": 1, "k_neighbors": 5, "shrinkage": 0.0},
        two_tower_hp={
            "embedding_dim": 8,
            "learning_rate": 1e-2,
            "temperature": 0.1,
            "batch_size": 16,
            "weight_decay": 1e-4,
            "max_epochs": 1,
            "patience": 5,
            "max_history": 10,
        },
        two_tower_epochs=1,
        dataset_sha256=SERVING_SHA256,
        git_sha_value=SERVING_GIT_SHA,
        show_progress=False,
    )
    before = [row["item_id"] for row in recommend_for_user(bundle, 1, n=5)["production"]]
    directory = tmp_path_factory.mktemp("serving")
    save_bundle(bundle, directory)
    loaded_ids = [
        row["item_id"] for row in recommend_for_user(load_bundle(directory), 1, n=5)["production"]
    ]
    assert loaded_ids == before
    assert before
    return directory


@pytest.fixture(scope="session")
def cold_artifact_dir(tmp_path_factory):
    """A single-source new-user snapshot over movies 1–8."""
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    from movielens_recommender.baselines.ease import EASERecommender
    from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
    from movielens_recommender.ranker.features import (
        build_feature_context,
        cold_start_feature_names,
    )
    from movielens_recommender.ranker.train import RankerMatrices, fit_lambdarank, lgb_params
    from movielens_recommender.serving.cold import save_cold_start_bundle
    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    rows = []
    for user_id in range(1, 9):
        for step in range(8):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": step + 1,
                    "rating": float((user_id + step) % 5 + 1),
                    "timestamp": step + 1,
                }
            )
    ratings = pd.DataFrame(rows)
    movies = pd.DataFrame(
        [
            {
                "item_id": item_id,
                "title": f"Movie {item_id} (199{item_id})",
                "genres": GENRES[item_id % len(GENRES)],
                "year": 1990 + item_id,
            }
            for item_id in range(1, 9)
        ]
    )
    item_item = ItemItemCosineRecommender(min_common=1, k_neighbors=0, shrinkage=0.0).fit(
        ratings
    )
    ease = EASERecommender(l2=10.0).fit(ratings)
    tower, _features, _result = fit_two_tower_recommender(
        ratings,
        dataset="synthetic",
        movies=movies,
        hyperparams={
            "embedding_dim": 8,
            "learning_rate": 1e-2,
            "temperature": 0.1,
            "batch_size": 16,
            "weight_decay": 0.0,
            "max_epochs": 1,
            "patience": 2,
            "max_history": 5,
            "history_only": True,
            "user_id_dropout": 0.0,
            "score_without_user_id": True,
        },
        seed=0,
        relevance_threshold=4.0,
        n_epochs=1,
    )
    ctx = build_feature_context(ratings, movies, mode="off", relevance_threshold=4.0)
    ctx.names = cold_start_feature_names("off")
    rng = np.random.default_rng(0)
    matrices = RankerMatrices(
        x=rng.normal(size=(12, len(ctx.names))),
        y=np.array([1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=np.float64),
        group=np.array([4, 4, 4], dtype=np.int64),
        user_ids=np.array([1, 2, 3], dtype=np.int64),
        feature_names=list(ctx.names),
    )
    booster, info = fit_lambdarank(
        matrices,
        params=lgb_params(
            seed=0,
            learning_rate=0.1,
            num_leaves=4,
            min_data_in_leaf=1,
            feature_fraction=1.0,
            bagging_fraction=1.0,
        ),
        num_boost_round=3,
    )
    return save_cold_start_bundle(
        tmp_path_factory.mktemp("serving_cold") / "cold_start",
        movies=movies,
        item_item=item_item,
        ease=ease,
        two_tower=tower,
        booster=booster,
        context=ctx,
        manifest={
            "dataset": "synthetic",
            "candidate_source": "item_item_fold_in",
            "candidate_k": 5,
            "ranker_mode": "demographics_off",
            "demographics_missing": False,
            "history_only": True,
            "user_id_dropout": 0.0,
            "best_iteration": int(info["best_iteration"]),
            "feature_names": list(ctx.names),
            "relevance_threshold": 4.0,
        },
    )
