"""S5b batch scoring. Synthetic snapshot only; no MovieLens download."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from movielens_recommender.cli import main


def test_batch_writes_every_user_in_rank_order(artifact_dir, tmp_path):
    from movielens_recommender.serving.batch import batch_recommend
    from movielens_recommender.serving.bundle import known_user_ids, load_bundle, rank_for_user

    out = tmp_path / "batch"
    manifest = batch_recommend(artifact_dir, out, n=3)
    frame = pd.read_csv(out / "recommendations.csv.gz")
    assert list(frame.columns) == ["user_id", "rank", "item_id", "score", "title", "year"]
    bundle = load_bundle(artifact_dir)
    users = known_user_ids(bundle)
    assert sorted(frame["user_id"].unique().tolist()) == users
    for uid in users:
        rows = frame.loc[frame["user_id"] == uid].sort_values("rank")
        assert rows["rank"].tolist() == list(range(1, len(rows) + 1))
        expected = [item for item, _score in rank_for_user(bundle, uid, n=3)]
        assert rows["item_id"].tolist() == expected
    on_disk = json.loads((out / "batch_manifest.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["n"] == 3
    assert manifest["n_users"] == len(users)
    assert manifest["n_rows"] == len(frame)
    assert manifest["dataset_sha256"] == bundle.manifest["dataset_sha256"]
    assert manifest["git_sha"] == bundle.manifest["git_sha"]
    assert manifest["snapshot_created_at"] == bundle.manifest["created_at"]
    assert manifest["wall_sec"] >= 0.0


def test_batch_user_subset_and_unknown_user(artifact_dir, tmp_path):
    from movielens_recommender.serving.batch import batch_recommend

    manifest = batch_recommend(artifact_dir, tmp_path / "some", n=2, user_ids=[2, 5])
    frame = pd.read_csv(tmp_path / "some" / "recommendations.csv.gz")
    assert sorted(frame["user_id"].unique().tolist()) == [2, 5]
    assert manifest["n_users"] == 2
    with pytest.raises(KeyError):
        batch_recommend(artifact_dir, tmp_path / "bad", n=2, user_ids=[2, 999])
    assert not (tmp_path / "bad" / "recommendations.csv.gz").exists()


def test_batch_cli(artifact_dir, tmp_path, capsys):
    out = tmp_path / "cli"
    code = main(
        [
            "batch-recommend",
            "--artifacts",
            str(artifact_dir),
            "--out-dir",
            str(out),
            "--n",
            "2",
            "--users",
            "1,3",
        ]
    )
    assert code == 0
    assert (out / "recommendations.csv.gz").is_file()
    assert "2 users" in capsys.readouterr().out
