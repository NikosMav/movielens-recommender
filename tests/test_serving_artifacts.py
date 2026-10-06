"""Serving snapshot round-trip and Streamlit smoke test. Synthetic data only."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from movielens_recommender.config import (
    EvalYAML,
    ItemKNNYAML,
    ModelsYAML,
    RankerYAML,
    RunConfig,
    SplitYAML,
    TwoTowerYAML,
)
from movielens_recommender.data import parse_users
from movielens_recommender.movies import GENRES
from movielens_recommender.ranker.features import feature_names
from movielens_recommender.serving.build import fit_serving_bundle
from movielens_recommender.serving.bundle import load_bundle, recommend_for_user, save_bundle
from movielens_recommender.split import SplitConfig, time_based_split

_APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
_SHA = "synthetic-sha256"
_GIT = "0123456789abcdef0123456789abcdef01234567"


def _ratings_movies_users() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _config() -> RunConfig:
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


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory):
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
        dataset_sha256=_SHA,
        git_sha_value=_GIT,
        show_progress=False,
    )
    before = [
        row["item_id"] for row in recommend_for_user(bundle, 1, n=5)["production"]
    ]
    directory = tmp_path_factory.mktemp("serving")
    save_bundle(bundle, directory)
    loaded_ids = [
        row["item_id"] for row in recommend_for_user(load_bundle(directory), 1, n=5)["production"]
    ]
    assert loaded_ids == before
    assert before
    return directory


def test_manifest_round_trip(artifact_dir):
    loaded = load_bundle(artifact_dir)
    manifest = loaded.manifest
    assert manifest["schema_version"] == 1
    assert manifest["dataset"] == "synthetic"
    assert manifest["dataset_sha256"] == _SHA
    assert manifest["git_sha"] == _GIT
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", manifest["created_at"])
    assert manifest["config"]["seed"] == 42
    assert manifest["config"]["models"]["ranker"]["demographics"] == "both"
    assert manifest["cold_start"] == "existing_users_only"
    assert manifest["candidate_set"] in {
        "item_item",
        "two_tower",
        "union_balanced",
        "union_unbalanced",
    }
    assert manifest["best_iteration"] >= 1
    again = load_bundle(artifact_dir)
    assert again.manifest == manifest


def test_recommend_round_trip_and_plain_reasons(artifact_dir):
    loaded = load_bundle(artifact_dir)
    first = recommend_for_user(loaded, 1, n=5)
    second = recommend_for_user(load_bundle(artifact_dir), 1, n=5)
    assert [row["item_id"] for row in first["production"]] == [
        row["item_id"] for row in second["production"]
    ]
    assert first["production"]
    assert len(first["item_item"]) <= 5
    assert len(first["two_tower"]) <= 5
    forbidden = ("shap", "gender", "occupation", "region", "zip", "male", "female")
    age = re.compile(r"\bage\b", re.IGNORECASE)
    names = feature_names("both")
    for card in first["production"]:
        main = "\n".join(card["reasons"])
        lowered = main.lower()
        for word in forbidden:
            assert word not in lowered
        assert age.search(main) is None
        for name in names:
            assert name not in main
        assert len(card["reasons"]) <= 3
        if card["details"]:
            assert "feature" in card["details"][0]


def test_unknown_user_is_rejected(artifact_dir):
    loaded = load_bundle(artifact_dir)
    with pytest.raises(KeyError):
        recommend_for_user(loaded, 999, n=5)


def test_streamlit_missing_snapshot_shows_build_command(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MOVIELENS_ARTIFACTS", str(tmp_path / "missing"))
    app = AppTest.from_file(str(_APP))
    app.run()
    assert not app.exception
    text = "\n".join(block.value for block in list(app.markdown) + list(app.code))
    assert "build-artifacts" in text


def test_streamlit_smoke_shows_history_and_reasons(artifact_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MOVIELENS_ARTIFACTS", str(artifact_dir))
    app = AppTest.from_file(str(_APP))
    app.run()
    assert not app.exception
    titles = app.dataframe[0].value["title"].astype(str).tolist()
    assert any(title.startswith("Movie") for title in titles)
    labels = [expander.label for expander in app.expander]
    assert "Why this result?" in labels
    why = next(expander for expander in app.expander if expander.label == "Why this result?")
    reason_text = " ".join(block.value for block in why.markdown)
    assert "SHAP" not in reason_text
    assert "shap" not in reason_text.lower()
    detail_labels = [child.label for child in why.expander]
    assert "Details" in detail_labels
    app.checkbox[0].set_value(True).run()
    assert not app.exception
    headers = [block.value for block in app.subheader]
    assert "Production ranker" in headers
    assert "Item–item cosine" in headers
    assert "Two-tower" in headers
