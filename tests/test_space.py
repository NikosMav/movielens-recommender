"""Space folder: weights round-trip and an isolated app start. No MovieLens download."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1] / "space"
_PIN = "23ab9515a3cbc0a18471d0fd36d03ae56b96a641"
_DEMO_REASON = (
    "demo_",
    "group_",
    "gender",
    "occupation",
    "zip",
    "region",
    "male",
    "female",
)


def _load_runtime():
    path = SPACE / "runtime.py"
    spec = importlib.util.spec_from_file_location("space_runtime_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _boom(*_args, **_kwargs):
    raise AssertionError("the Space path must not train")


def test_space_readme_and_requirements_pin():
    readme = (SPACE / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    front, _body = readme.split("\n---\n", 1)
    assert "sdk: streamlit" in front
    assert "app_file: app.py" in front
    assert 'python_version: "3.12"' in front
    assert "sdk_version: 1.65.0" in front
    description = next(
        line.split(":", 1)[1].strip()
        for line in front.splitlines()
        if line.startswith("short_description:")
    )
    assert len(description) <= 60
    requirements = (SPACE / "requirements.txt").read_text(encoding="utf-8")
    assert _PIN in requirements
    assert "torch==2.6.0+cpu" in requirements
    assert "lightgbm==4.6.0" in requirements
    assert "streamlit==1.65.0" in requirements
    assert "HF_TOKEN" not in readme
    assert "HF_TOKEN" not in requirements
    assert (SPACE / "app.py").is_file()
    assert (SPACE / "runtime.py").is_file()
    assert (SPACE / "packages.txt").read_text(encoding="utf-8").strip() == "libgomp1"
    runtime_source = (SPACE / "runtime.py").read_text(encoding="utf-8")
    assert "fit_two_tower_recommender" not in runtime_source
    assert "fit_lambdarank" not in runtime_source
    for path in SPACE.rglob("*"):
        if path.is_file() and path.suffix in {".dat", ".csv"} and "README" not in path.name:
            raise AssertionError(f"MovieLens-like file committed under space/: {path.name}")


def test_load_round2_choices_is_read_only():
    pytest.importorskip("torch")
    path = Path(__file__).resolve().parents[1] / "results" / "cold-start" / "ml-1m.json"
    before = path.read_bytes()
    choices = _load_runtime().load_round2_choices(path)
    assert path.read_bytes() == before
    payload = json.loads(before)
    assert choices["best_iteration"] == payload["round2"]["best_iteration"]
    assert choices["best_iteration"] == payload["ranker"]["best_iteration"]
    assert choices["candidate_k"] == payload["round2"]["candidate_k"]
    assert choices["feature_names"] == payload["round2"]["feature_names"]
    assert choices["feature_names"] == payload["ranker"]["feature_names"]
    assert choices["representation_winner"] == payload["representation"]["winner"]
    assert choices["representation_winner"] == payload["round2"]["representation_winner"]
    assert choices["history_only"] is payload["round2"]["history_only"]
    assert choices["history_only"] is payload["representation"]["history_only"]
    assert choices["representation_best_epoch"] == payload["round2"]["representation_best_epoch"]
    assert choices["user_id_dropout"] == payload["representation"]["user_id_dropout"]
    assert choices["fixed_hyperparams"] == payload["representation"]["fixed_hyperparams"]
    assert choices["serving_by_n"] == {
        str(key): value for key, value in payload["round2"]["serving_by_n"].items()
    }
    assert set(choices["serving_by_n"].values()) == {"cold_start_ranker"}


def test_visible_reasons_drop_demographic_text_only():
    pytest.importorskip("torch")
    runtime = _load_runtime()
    assert runtime.visible_reasons(
        ["Popular with viewers similar to you", "Because of gender"]
    ) == ["Popular with viewers similar to you"]
    details = runtime.visible_details(
        [
            {"feature": "demo_gender", "contribution": 1.0},
            {"feature": "group_age_pos_rate", "contribution": 0.2},
            {"feature": "item_popularity", "contribution": 0.4},
        ]
    )
    assert details == [{"feature": "item_popularity", "contribution": 0.4}]


def test_weights_round_trip_does_not_train(
    artifact_dir,
    cold_artifact_dir,
    serving_tables,
    cold_tables,
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    import movielens_recommender.cold_start as cold_start
    import movielens_recommender.ranker.train as ranker_train
    import movielens_recommender.two_tower.train as tower_train
    from movielens_recommender.baselines.ease import EASERecommender
    from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
    from movielens_recommender.serving.bundle import load_bundle, recommend_for_user
    from movielens_recommender.serving.cold import load_cold_start_bundle, recommend_new_user

    monkeypatch.setattr(ItemItemCosineRecommender, "fit", _boom)
    monkeypatch.setattr(EASERecommender, "fit", _boom)
    monkeypatch.setattr(tower_train, "fit_two_tower_recommender", _boom)
    monkeypatch.setattr(cold_start, "fit_two_tower_recommender", _boom)
    monkeypatch.setattr(ranker_train, "fit_lambdarank", _boom)

    runtime = _load_runtime()
    weights = tmp_path / "weights"
    runtime.export_space_weights(
        artifact_dir,
        cold_artifact_dir,
        weights,
        cold_train_protocol="caller",
    )
    runtime.assert_weights_only(weights)
    blob = b"".join(path.read_bytes() for path in weights.rglob("*") if path.is_file())
    assert b"Movie 1" not in blob
    for name in ("movies.csv", "histories.npz", "item_item_user_item.npz", "popular_items.npy"):
        assert list(weights.rglob(name)) == []

    ratings, movies, users = serving_tables
    cold_ratings, cold_movies = cold_tables
    runtime_dir = tmp_path / "runtime"
    production, cold = runtime.materialize_space(
        weights,
        runtime_dir,
        ratings=ratings,
        movies=movies,
        users=users,
        cold_ratings=cold_ratings,
        cold_movies=cold_movies,
    )
    original_cards = recommend_for_user(load_bundle(artifact_dir), 1, n=5)["production"]
    original = [row["item_id"] for row in original_cards]
    rebuilt = [
        row["item_id"] for row in recommend_for_user(load_bundle(production), 1, n=5)["production"]
    ]
    assert rebuilt == original
    assert original
    profile = [(1, 5.0), (2, 4.0), (3, 5.0)]
    cold_before = [
        row["item_id"]
        for row in recommend_new_user(load_cold_start_bundle(cold_artifact_dir), profile, n=5)[
            "recommendations"
        ]
    ]
    cold_after = [
        row["item_id"]
        for row in recommend_new_user(load_cold_start_bundle(cold), profile, n=5)["recommendations"]
    ]
    assert cold_after == cold_before
    assert cold_after


def _copy_space(dest: Path) -> Path:
    shutil.copytree(
        SPACE,
        dest,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "weights", "data", ".runtime"),
    )
    return dest


def _reason_text(app) -> str:
    chunks: list[str] = []
    for expander in app.expander:
        if expander.label != "Why this result?":
            continue
        chunks.extend(block.value for block in expander.markdown)
        chunks.extend(block.value for block in expander.text)
        for child in expander.expander:
            chunks.extend(block.value for block in child.markdown)
            chunks.extend(block.value for block in child.text)
            chunks.extend(block.value for block in child.caption)
    return "\n".join(chunks)


def test_copied_space_app_starts_from_synthetic_snapshots(
    artifact_dir,
    cold_artifact_dir,
    serving_tables,
    cold_tables,
    tmp_path,
    monkeypatch,
):
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import movielens_recommender.data as data

    runtime = _load_runtime()
    weights = tmp_path / "weights"
    runtime.export_space_weights(
        artifact_dir,
        cold_artifact_dir,
        weights,
        cold_train_protocol="caller",
    )
    ratings, movies, users = serving_tables
    cold_ratings, cold_movies = cold_tables
    production, cold = runtime.materialize_space(
        weights,
        tmp_path / "runtime",
        ratings=ratings,
        movies=movies,
        users=users,
        cold_ratings=cold_ratings,
        cold_movies=cold_movies,
    )
    calls = {"n": 0}

    def _refuse_network(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("the isolated app must not download MovieLens")

    monkeypatch.setattr(data, "urlopen", _refuse_network)
    monkeypatch.setenv("MOVIELENS_ALLOW_DOWNLOAD", "0")
    monkeypatch.setenv("MOVIELENS_ARTIFACTS", str(production))
    monkeypatch.setenv("MOVIELENS_COLD_ARTIFACTS", str(cold))

    copy = _copy_space(tmp_path / "space")
    app = AppTest.from_file(str(copy / "app.py"))
    app.run()
    assert not app.exception
    assert calls["n"] == 0
    labels = [radio.label for radio in app.radio]
    assert "Who is this for?" in labels
    text = "\n".join(block.value for block in list(app.markdown) + list(app.caption))
    assert "Existing user" in text or app.radio
    assert "10.1145/2827872" in text
    assert "GroupLens" in text
    reasons = _reason_text(app).lower()
    for word in _DEMO_REASON:
        assert word not in reasons

    app.radio[0].set_value("New user").run()
    assert not app.exception
    assert calls["n"] == 0
    new_user = "\n".join(block.value for block in list(app.markdown) + list(app.caption))
    assert "New-user model" in new_user


def test_copied_space_app_without_weights_does_not_download(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import movielens_recommender.data as data

    calls = {"n": 0}

    def _refuse_network(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("missing weights must not download MovieLens")

    monkeypatch.setattr(data, "urlopen", _refuse_network)
    monkeypatch.setenv("MOVIELENS_ALLOW_DOWNLOAD", "0")
    monkeypatch.delenv("MOVIELENS_ARTIFACTS", raising=False)
    monkeypatch.delenv("MOVIELENS_COLD_ARTIFACTS", raising=False)

    copy = _copy_space(tmp_path / "space")
    app = AppTest.from_file(str(copy / "app.py"))
    app.run()
    assert not app.exception
    assert calls["n"] == 0
    blocks = list(app.markdown) + list(app.code) + list(app.caption)
    text = "\n".join(block.value for block in blocks)
    assert "make space-snapshot" in text
    assert "10.1145/2827872" in text
