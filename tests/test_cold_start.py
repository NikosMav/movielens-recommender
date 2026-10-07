"""S5c cold-start checks. Synthetic data only; no MovieLens download."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.cold_start import (
    assert_held_out_absent,
    eligible_user_ids,
    exclude_held_out_users,
    select_held_out_user_ids,
)
from movielens_recommender.movies import GENRES
from movielens_recommender.serving.reasons import HistoryItem, explain_recommendation


def _ratings(n_users: int = 40, n_each: int = 6) -> pd.DataFrame:
    rows = []
    for user_id in range(1, n_users + 1):
        for step in range(n_each):
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": step + 1,
                    "rating": float((user_id + step) % 5 + 1),
                    "timestamp": step + 1,
                }
            )
    return pd.DataFrame(rows)


def test_held_out_users_are_absent_from_training_and_ignore_rating_values():
    ratings = _ratings()
    flipped = ratings.copy()
    flipped["rating"] = 5.0 - flipped["rating"] + 1.0
    first = eligible_user_ids(ratings, min_ratings=5)
    second = eligible_user_ids(flipped, min_ratings=5)
    assert np.array_equal(first, second)
    held_a = select_held_out_user_ids(first, fraction=0.1, seed=42)
    held_b = select_held_out_user_ids(second, fraction=0.1, seed=42)
    assert np.array_equal(held_a, held_b)
    assert len(held_a) == int(round(0.1 * len(first)))
    remain = exclude_held_out_users(ratings, held_a)
    assert_held_out_absent(held_a, remain)
    assert set(held_a.tolist()).isdisjoint(set(remain["user_id"].tolist()))
    # Shuffling the id array before selection does not change the held-out set.
    shuffled = np.array(first.tolist()[::-1], dtype=np.int64)
    assert np.array_equal(held_a, select_held_out_user_ids(shuffled, fraction=0.1, seed=42))


def test_fold_in_excludes_rated_items():
    ratings = _ratings(n_users=12, n_each=8)
    profile = [(1, 5.0), (2, 4.0), (3, 1.0)]
    item_item = ItemItemCosineRecommender(min_common=1, k_neighbors=0, shrinkage=0.0).fit(
        ratings
    )
    ease = EASERecommender(l2=10.0).fit(ratings)
    for model in (item_item, ease):
        picked = model.topk_for_profiles({99: profile}, n=20)[99]
        ids = [item for item, _score in picked]
        assert 1 not in ids
        assert 2 not in ids
        assert 3 not in ids
        assert len(ids) >= 1


def test_new_user_reasons_omit_demographic_names():
    history = [HistoryItem(7, "Mask, The (1994)", "Comedy", 5.0, timestamp=1)]
    explanation = explain_recommendation(
        {
            "demo_gender": 3.0,
            "group_age_pos": 2.0,
            "item_item_score": 0.5,
            "ease_score": 0.4,
        },
        history=history,
        neighbor_similarity={7: 0.8},
        include_demographics=False,
        because_rated=True,
    )
    texts = [reason.text for reason in explanation.reasons]
    joined = " ".join(texts)
    assert "Popular with viewers similar to you" not in joined
    assert "demo_" not in joined
    assert "group_" not in joined
    assert "Because you rated The Mask highly" in texts
    for name, _value in explanation.details:
        assert not name.startswith("demo_")
        assert not name.startswith("group_")

    silent = explain_recommendation(
        {"item_popularity": 0.2},
        history=history,
        neighbor_similarity={7: 0.0},
        include_demographics=False,
        because_rated=True,
    )
    assert all("Because you rated" not in reason.text for reason in silent.reasons)


def test_history_only_encoding_ignores_user_index():
    torch = pytest.importorskip("torch")
    from movielens_recommender.two_tower.model import TwoTowerModel

    model = TwoTowerModel(
        n_users=6,
        n_items=4,
        n_genres=len(GENRES),
        embedding_dim=8,
        history_only=True,
    )
    history = torch.tensor([[1, 2, 0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    first = model.encode_users(torch.tensor([0]), history, mask)
    second = model.encode_users(torch.tensor([5]), history, mask)
    assert torch.allclose(first, second)


def test_cold_start_bundle_loads_and_excludes_rated(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    from movielens_recommender.ranker.features import (
        build_feature_context,
        cold_start_feature_names,
    )
    from movielens_recommender.ranker.train import RankerMatrices, fit_lambdarank, lgb_params
    from movielens_recommender.serving.cold import (
        load_cold_start_bundle,
        recommend_new_user,
        save_cold_start_bundle,
    )
    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    ratings = _ratings(n_users=8, n_each=8)
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
    n_features = len(ctx.names)
    rng = np.random.default_rng(0)
    matrices = RankerMatrices(
        x=rng.normal(size=(12, n_features)),
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
    directory = save_cold_start_bundle(
        tmp_path / "cold_start",
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
    loaded = load_cold_start_bundle(directory)
    profile = [(1, 5.0), (2, 5.0), (3, 4.0)]
    result = recommend_new_user(loaded, profile, n=10)
    recommended = [card["item_id"] for card in result["recommendations"]]
    assert recommended
    assert set(recommended).isdisjoint({1, 2, 3})
    for card in result["recommendations"]:
        blob = " ".join(card["reasons"])
        assert "demo_" not in blob
        assert "group_" not in blob
        assert "Popular with viewers similar to you" not in blob
        for row in card["details"]:
            assert not str(row["feature"]).startswith("demo_")
            assert not str(row["feature"]).startswith("group_")
    assert result["latency_sec"] >= 0.0
    assert "method_sentence" not in result


def test_dropout_grid_includes_the_low_edge():
    from movielens_recommender.cold_start import DROPOUT_GRID

    assert DROPOUT_GRID[0] == 0.0
    assert 0.1 in DROPOUT_GRID
    assert 0.25 in DROPOUT_GRID


def test_profile_bucket_and_serving_ties():
    from movielens_recommender.cold_start_rank import (
        SERVING_TIE_ORDER,
        choose_candidate_k,
        choose_serving_method,
        profile_bucket,
    )

    assert profile_bucket(1) == 1
    assert profile_bucket(2) == 1
    assert profile_bucket(3) == 3
    assert profile_bucket(4) == 3
    assert profile_bucket(9) == 5
    assert profile_bucket(10) == 10
    assert profile_bucket(40) == 10
    tied = {name: 0.2 for name in SERVING_TIE_ORDER}
    assert choose_serving_method(tied) == "most_popular"
    assert choose_candidate_k({50: 0.1, 100: 0.2, 200: 0.2}) == 100


def test_ranker_validation_users_are_disjoint_from_the_held_out_set():
    import inspect

    from movielens_recommender.cold_start_rank import (
        carve_ranker_validation_users,
        select_cold_start_ranker,
    )

    signature = inspect.signature(select_cold_start_ranker)
    assert "held_ratings" not in signature.parameters
    assert "round1" not in signature.parameters
    ratings = _ratings()
    eligible = eligible_user_ids(ratings, min_ratings=5)
    held = select_held_out_user_ids(eligible, fraction=0.1, seed=42)
    remain = exclude_held_out_users(ratings, held)
    pool = np.sort(remain["user_id"].unique().astype(np.int64))
    fit_ids, val_ids = carve_ranker_validation_users(pool, held, fraction=0.2, seed=42)
    assert set(val_ids.tolist()).isdisjoint(held.tolist())
    assert set(fit_ids.tolist()).isdisjoint(held.tolist())
    assert set(fit_ids.tolist()).isdisjoint(val_ids.tolist())
    assert set(fit_ids.tolist()) | set(val_ids.tolist()) == set(pool.tolist())


def test_union_and_ranker_features_have_no_demographics():
    from movielens_recommender.cold_start_rank import union_candidate_ids
    from movielens_recommender.ranker.features import (
        annotate_cold_start_ranker_rows,
        cold_start_ranker_feature_names,
    )

    items, ease_ids, popular_ids, tower_ids = union_candidate_ids(
        [1, 2],
        [2, 3],
        [3, 4],
        banned={1},
    )
    assert items == [2, 3, 4]
    assert ease_ids == {2}
    assert popular_ids == {2, 3}
    assert tower_ids == {3, 4}
    names = cold_start_ranker_feature_names()
    assert "profile_n" in names
    assert "from_ease" in names
    assert "from_most_popular" in names
    assert "from_history_tower" in names
    assert "item_popularity" in names
    assert not any(name.startswith(("demo_", "group_")) for name in names)
    matrix = np.zeros((2, len(names)), dtype=np.float64)
    out = annotate_cold_start_ranker_rows(
        matrix,
        names,
        [2, 4],
        profile_n=3,
        ease_ids=ease_ids,
        popular_ids=popular_ids,
        tower_ids=tower_ids,
    )
    assert out[0, names.index("profile_n")] == 3.0
    assert out[0, names.index("from_ease")] == 1.0
    assert out[0, names.index("from_most_popular")] == 1.0
    assert out[1, names.index("from_history_tower")] == 1.0
    assert out[1, names.index("from_ease")] == 0.0


def test_popularity_reason_for_a_new_user_is_not_demographic():
    from movielens_recommender.serving.reasons import POPULAR_WITH_MANY_VIEWERS

    history = [HistoryItem(7, "Mask, The (1994)", "Comedy", 5.0, timestamp=1)]
    explanation = explain_recommendation(
        {"item_popularity": 1.0, "demo_gender": 3.0},
        history=history,
        include_demographics=False,
        because_rated=False,
        popularity_text=POPULAR_WITH_MANY_VIEWERS,
    )
    texts = [reason.text for reason in explanation.reasons]
    assert texts == ["Popular with many viewers"]
    assert "Popular with viewers similar to you" not in texts


def test_harness_tail_is_the_last_slice_after_the_profile():
    from movielens_recommender.cold_start import _held_out_targets

    rows = []
    for step in range(10):
        rows.append(
            {
                "user_id": 1,
                "item_id": step + 1,
                "rating": 5.0,
                "timestamp": step + 1,
            }
        )
    ratings = pd.DataFrame(rows)
    catalog = set(range(1, 11))
    _profiles, _hist, _frames, relevant, _counts = _held_out_targets(ratings, 1, catalog, 4.0)
    assert relevant[1] == set(range(2, 11))
    _p, _h, _f, tail, _c = _held_out_targets(
        ratings, 1, catalog, 4.0, target="harness_tail", test_fraction=0.2
    )
    assert tail[1] == {9, 10}


def test_served_popularity_states_the_method(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    from movielens_recommender.baselines.popular import MostPopularRecommender
    from movielens_recommender.ranker.features import (
        build_feature_context,
        cold_start_ranker_feature_names,
    )
    from movielens_recommender.ranker.train import RankerMatrices, fit_lambdarank, lgb_params
    from movielens_recommender.serving.cold import (
        load_cold_start_bundle,
        recommend_new_user,
        save_cold_start_bundle,
    )
    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    ratings = _ratings(n_users=8, n_each=8)
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
    item_item = ItemItemCosineRecommender(min_common=1, k_neighbors=0, shrinkage=0.0).fit(ratings)
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
    ctx.names = cold_start_ranker_feature_names()
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
    popular = MostPopularRecommender().fit(ratings)
    directory = save_cold_start_bundle(
        tmp_path / "cold_start",
        movies=movies,
        item_item=item_item,
        ease=ease,
        two_tower=tower,
        booster=booster,
        context=ctx,
        popular_item_ids=popular._ranked_items,
        manifest={
            "dataset": "synthetic",
            "candidate_source": "cold_start_union",
            "candidate_k": 5,
            "ranker_mode": "cold_start_ranker",
            "demographics_missing": False,
            "history_only": True,
            "user_id_dropout": 0.0,
            "best_iteration": int(info["best_iteration"]),
            "feature_names": list(ctx.names),
            "relevance_threshold": 4.0,
            "serving_by_n": {
                "1": "most_popular",
                "3": "most_popular",
                "5": "most_popular",
                "10": "most_popular",
            },
            "profile_buckets": [1, 3, 5, 10],
        },
    )
    loaded = load_cold_start_bundle(directory)
    result = recommend_new_user(loaded, [(1, 5.0), (2, 4.0), (3, 5.0)], n=5)
    assert result["method"] == "most_popular"
    assert result["profile_bucket"] == 3
    assert "titles most people have rated" in result["method_sentence"]
    assert "N=3" in result["method_sentence"]
    assert result["recommendations"]
    assert {card["item_id"] for card in result["recommendations"]}.isdisjoint({1, 2, 3})
    blob = " ".join(result["recommendations"][0]["reasons"])
    assert "Popular with many viewers" in blob
    assert "Popular with viewers similar to you" not in blob
    assert "demo_" not in blob
