"""Historical configs and selected settings are isolated from mutable defaults."""

import json

import pytest


def _record(tmp_path):
    source = tmp_path / "published.json"
    source.write_text(json.dumps({
        "dataset": "ml-1m",
        "dataset_sha256": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
        "config": {
            "dataset": "ml-1m", "data_dir": str(tmp_path / "data"), "tune": True,
            "models": {"ranker": {"demographics": "off"}},
        },
        "hyperparameters": {
            "als_tuned": {"factors": 32}, "item_item_cosine_tuned": {"k_neighbors": 50},
            "ease": {"l2": 5000}, "rp3beta": {"alpha": 0.4, "beta": 0.6, "top_k": 2000},
        },
        "tuning_summary": {
            "als_best_val_ndcg@10": 0.1, "item_item_cosine_best_val_ndcg@10": 0.2,
            "ease_best_val_ndcg@10": 0.3, "rp3beta_best_val_ndcg@10": 0.4,
        },
        "two_tower": {
            "best_hyperparams": {"embedding_dim": 64}, "best_epoch": 6,
            "best_val_ndcg@10": 0.25,
        },
    }))
    return source


def test_reproduction_freezes_record_config_and_tuning_in_unique_directory(tmp_path):
    from movielens_recommender.experiments import prepare_reproduction

    source = _record(tmp_path)
    before = source.read_bytes()
    config, directory = prepare_reproduction(source, tmp_path / "runs", run_id="check-1")
    assert config.models.ranker.demographics == "off"
    assert config.results_dir == str(directory / "results")
    assert config.models_dir == str(directory / "models")
    tuning = json.loads((directory / "results/tuning/ml-1m.json").read_text())
    assert tuning["item_item_cosine"]["best_hyperparams"]["k_neighbors"] == 50
    tower = json.loads((directory / "results/tuning/two_tower_ml-1m.json").read_text())
    assert tower["two_tower"]["best_epoch"] == 6
    assert len(json.loads((directory / "run.json").read_text())["source_record_sha256"]) == 64
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        prepare_reproduction(source, tmp_path / "runs", run_id="check-1")


def test_reproduction_cli_can_prepare_without_training(tmp_path):
    from movielens_recommender.cli import main

    source = _record(tmp_path)
    assert main([
        "reproduce", "--from-results", str(source), "--output-root", str(tmp_path / "runs"),
        "--run-id", "preview", "--prepare-only",
    ]) == 0
    directory = tmp_path / "runs/preview"
    assert (directory / "config.yaml").is_file()
    assert not (directory / "results/ml-1m.json").exists()


def test_reproduction_rejects_path_traversal_before_creating_run(tmp_path):
    from movielens_recommender.experiments import prepare_reproduction

    with pytest.raises(ValueError, match="run_id"):
        prepare_reproduction(_record(tmp_path), tmp_path / "runs", run_id="../escape")
    assert not (tmp_path / "runs").exists()


def test_reproduction_runs_real_pipeline_without_touching_source(tmp_path):
    import pandas as pd

    from movielens_recommender.cli import main

    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "ml-latest-small" / "ml-latest-small"
    dataset_dir.mkdir(parents=True)
    rows = []
    for user in range(1, 9):
        for step in range(12):
            rows.append({
                "userId": user, "movieId": (user + step) % 12 + 1,
                "rating": 4.0 + step % 2, "timestamp": user * 100 + step + 1,
            })
    pd.DataFrame(rows).to_csv(dataset_dir / "ratings.csv", index=False)
    pd.DataFrame([
        {"movieId": item, "title": f"Movie {item} (2000)", "genres": "Action"}
        for item in range(1, 13)
    ]).to_csv(dataset_dir / "movies.csv", index=False)
    source = tmp_path / "small.json"
    source.write_text(json.dumps({
        "dataset": "ml-latest-small",
        "dataset_sha256": "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436",
        "config": {
            "dataset": "ml-latest-small", "data_dir": str(data_dir), "tune": False,
            "eval": {"n_bootstrap": 10},
            "models": {
                "two_tower": {"enabled": False}, "ranker": {"enabled": False},
                "als": {"factors": 4, "iterations": 1},
            },
        },
    }))
    before = source.read_bytes()
    assert main([
        "reproduce", "--from-results", str(source), "--output-root", str(tmp_path / "runs"),
        "--run-id", "pipeline",
    ]) == 0
    directory = tmp_path / "runs/pipeline"
    manifest = json.loads((directory / "run.json").read_text())
    result = json.loads((directory / "results/ml-latest-small.json").read_text())
    assert manifest["status"] == result["reproduction"]["status"] == "complete"
    assert result["reproduction"]["run_id"] == "pipeline"
    assert set(result["metrics"]) == {"most_popular", "item_item_cosine", "als"}
    assert source.read_bytes() == before


def test_failed_reproduction_is_recorded_in_its_own_run(tmp_path):
    from movielens_recommender.cli import main

    with pytest.raises(FileNotFoundError):
        main([
            "reproduce", "--from-results", str(_record(tmp_path)),
            "--output-root", str(tmp_path / "runs"), "--run-id", "missing-data",
        ])
    manifest = json.loads((tmp_path / "runs/missing-data/run.json").read_text())
    assert manifest["status"] == "failed"
