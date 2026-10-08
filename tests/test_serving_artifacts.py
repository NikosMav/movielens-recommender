"""Serving snapshot round-trip and Streamlit smoke test. Synthetic data only."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from movielens_recommender.ranker.features import feature_names
from movielens_recommender.serving.bundle import load_bundle, recommend_for_user

_APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
# Must match the values the ``artifact_dir`` fixture in conftest.py writes.
_SHA = "synthetic-sha256"
_GIT = "0123456789abcdef0123456789abcdef01234567"


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


def test_indexed_lookups_match_full_scans(artifact_dir):
    from movielens_recommender.serving.bundle import (
        known_user_ids,
        movie_row,
        user_history_items,
    )

    loaded = load_bundle(artifact_dir)
    frame = loaded.histories
    assert known_user_ids(loaded) == sorted(int(uid) for uid in frame["user_id"].unique())
    for uid in known_user_ids(loaded):
        rows = frame.loc[frame["user_id"] == uid].sort_values(
            ["timestamp", "item_id"], kind="mergesort"
        )
        got = user_history_items(loaded, uid)
        assert [item.item_id for item in got] == rows["item_id"].astype(int).tolist()
        assert [item.rating for item in got] == rows["rating"].astype(float).tolist()
        assert [item.timestamp for item in got] == rows["timestamp"].astype(int).tolist()
    assert user_history_items(loaded, 999) == []
    for record in loaded.movies.itertuples(index=False):
        row = movie_row(loaded, int(record.item_id))
        assert row["title"] == str(record.title)
        assert row["genres"] == str(record.genres)
    assert movie_row(loaded, 999)["title"] == "Item 999"


def test_rank_for_user_matches_the_explained_list(artifact_dir):
    from movielens_recommender.serving.bundle import rank_for_user, user_history_items

    loaded = load_bundle(artifact_dir)
    for uid in (1, 4, 8):
        ranked = rank_for_user(loaded, uid, n=5)
        explained = recommend_for_user(loaded, uid, n=5)["production"]
        assert [item for item, _score in ranked] == [card["item_id"] for card in explained]
        scores = [score for _item, score in ranked]
        assert scores == sorted(scores, reverse=True)
        seen = {item.item_id for item in user_history_items(loaded, uid)}
        assert seen.isdisjoint(item for item, _score in ranked)
    with pytest.raises(KeyError):
        rank_for_user(loaded, 999, n=5)
