"""Serving snapshot round-trip and Streamlit smoke test. Synthetic data only."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
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


def test_saved_snapshot_identity_detects_modified_model_files(artifact_dir, tmp_path):
    directory = tmp_path / "copy"
    shutil.copytree(artifact_dir, directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["snapshot_id"]) == 64
    assert "movies.csv" in manifest["artifact_sha256"]
    assert manifest["library_versions"]["numpy"]
    assert manifest["host"]["python"]
    with (directory / "movies.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(directory)


def test_legacy_snapshot_loads_without_inventing_provenance(artifact_dir, tmp_path):
    directory = tmp_path / "legacy"
    shutil.copytree(artifact_dir, directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    for key in ("snapshot_id", "artifact_sha256", "provenance_version"):
        manifest.pop(key, None)
    path.write_text(json.dumps(manifest))
    loaded = load_bundle(directory)
    assert loaded.manifest.get("snapshot_id") is None
    assert recommend_for_user(loaded, 1, n=3)["production"]


@pytest.mark.parametrize("change", ["staged", "untracked"])
def test_git_provenance_marks_all_uncommitted_code(tmp_path, change):
    from movielens_recommender.serving.build import git_sha

    if shutil.which("git") is None:
        pytest.skip("git is unavailable")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "source.py"
    source.write_text("original")
    subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    subprocess.run([
        "git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.test",
        "commit", "-qm", "fixture",
    ], check=True)
    clean = git_sha(tmp_path)
    if change == "staged":
        source.write_text("changed")
        subprocess.run(["git", "-C", str(tmp_path), "add", "source.py"], check=True)
    else:
        (tmp_path / "new.py").write_text("new code")
    assert git_sha(tmp_path) == clean + "-dirty"


def test_container_build_revision_is_used_without_git_checkout(tmp_path, monkeypatch):
    from movielens_recommender.serving.provenance import git_sha

    monkeypatch.setenv("MOVIELENS_BUILD_SHA", "0123456789abcdef-build")
    assert git_sha(tmp_path) == "0123456789abcdef-build"


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
