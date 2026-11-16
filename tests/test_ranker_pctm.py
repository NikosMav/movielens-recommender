"""PCTM as a retriever and ranker feature, and the ADR-0033 serving rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.ranker.features import (
    build_feature_context,
    build_feature_matrix,
    feature_names,
    pctm_ranker_feature_names,
)
from movielens_recommender.sasrec.pctm import PCTMRecommender

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _sequences() -> pd.DataFrame:
    rows = []
    for uid in range(1, 20):
        for step, iid in enumerate([1, 2, 3, 4, 5, (uid % 5) + 6], start=1):
            rows.append(
                {"user_id": uid, "item_id": iid, "rating": 5.0, "timestamp": uid * 100 + step}
            )
    for step, iid in enumerate([1, 2, 3], start=1):
        rows.append({"user_id": 99, "item_id": iid, "rating": 5.0, "timestamp": 9900 + step})
    return pd.DataFrame(rows)


def _script():
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "run_ranker_pctm", SCRIPTS / "run_ranker_pctm.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_topk_with_scores_matches_recommend_and_hides_seen_items():
    model = PCTMRecommender(max_distance=5, smoothing_tau=10.0, pop_beta=0.0).fit(_sequences())
    rows = model.topk_with_scores(4)
    assert set(rows) == set(range(1, 20)) | {99}
    assert [item for item, _score in rows[99]] == model.recommend(99, 4)
    assert not {1, 2, 3} & {item for item, _score in rows[99]}
    scores = [score for _item, score in rows[99]]
    assert scores == sorted(scores, reverse=True)
    assert model.topk_with_scores(4, user_ids=[99, 12345]).keys() == {99}
    assert model.topk_with_scores(0) == {}


def test_evidence_names_the_history_films_behind_a_score():
    model = PCTMRecommender(max_distance=5, smoothing_tau=10.0, pop_beta=0.0).fit(_sequences())
    parts = model.evidence(99, 4, n=3)
    assert [item for item, _value in parts][0] == 3
    assert {item for item, _value in parts} <= {1, 2, 3}
    values = [value for _item, value in parts]
    assert values == sorted(values, reverse=True) and values[-1] > 0.0
    assert model.evidence(99, 4, n=1) == parts[:1]
    assert model.evidence(12345, 4) == []
    assert model.evidence(99, 777) == []


def test_pctm_feature_columns_sit_beside_the_other_retrievers():
    names = pctm_ranker_feature_names("off")
    base = feature_names("off")
    assert [name for name in names if name not in base] == ["pctm_score", "pctm_rank"]
    assert names.index("pctm_score") == names.index("two_tower_rank") + 1
    assert [name for name in names if not name.startswith("pctm_")] == base
    both = pctm_ranker_feature_names("both")
    assert both[: len(names)] == names
    assert both[len(names) :] == feature_names("both")[len(base) :]

    train = _sequences()
    movies = pd.DataFrame(
        [
            {"item_id": i, "title": f"Film {i}", "genres": "Action|Drama", "year": 1999}
            for i in range(1, 12)
        ]
    )
    ctx = build_feature_context(train, movies, mode="off")
    ctx.names = names
    matrix = build_feature_matrix(ctx, 99, [4, 5], {4: (0.8, 1)}, {}, pctm={4: (2.5, 1)})
    col = {name: index for index, name in enumerate(names)}
    assert matrix[0, col["pctm_score"]] == 2.5
    assert matrix[0, col["pctm_rank"]] == 1.0
    assert np.isnan(matrix[1, col["pctm_score"]])
    assert np.isnan(matrix[1, col["pctm_rank"]])


def test_candidate_sets_keep_the_budget_and_the_tie_order():
    script = _script()
    ii = {1: [(10, 0.9), (11, 0.8), (12, 0.7)]}
    tt = {1: [(20, 0.9), (10, 0.8), (21, 0.7)]}
    pc = {1: [(30, 0.9), (20, 0.8), (31, 0.7)]}
    assert script.build_lists("pctm", ii, tt, pc, 2)[1] == [30, 20]
    assert script.build_lists("two_tower", ii, tt, pc, 2)[1] == [20, 10]
    for name in ("pctm_two_tower", "pctm_tri"):
        union = script.build_lists(name, ii, tt, pc, 4)[1]
        assert len(union) == len(set(union)) == 4
        assert 30 in union and 20 in union
    with pytest.raises(ValueError):
        script.build_lists("unknown", ii, tt, pc, 2)

    tied = {name: {"recall@100": 0.4, "recall@200": 0.6} for name in script.SEQUENCE_SETS}
    assert script.choose_sequence_set(tied) == "pctm"
    tied["pctm_tri"] = {"recall@100": 0.4, "recall@200": 0.61}
    assert script.choose_sequence_set(tied) == "pctm_tri"


def test_serving_rule_follows_the_four_registered_steps():
    script = _script()
    wins = {f"{right}_minus_{left}": False for right, left in script.PAIRS}
    assert script.serving_choice(wins)["serve"] == "bi_both"
    wins["pctm_only_minus_bi_both"] = True
    assert script.serving_choice(wins) == {
        "ranker_candidate": "seq_off",
        "serve": "pctm_only",
        "serving_rule": "3",
    }
    wins["seq_off_minus_pctm_only"] = True
    assert script.serving_choice(wins)["serve"] == "seq_off"
    # seq_both is the candidate only when it beats seq_off; it must then beat PCTM itself.
    wins["seq_both_minus_seq_off"] = True
    assert script.serving_choice(wins)["serve"] == "pctm_only"
    wins["seq_both_minus_pctm_only"] = True
    assert script.serving_choice(wins) == {
        "ranker_candidate": "seq_both",
        "serve": "seq_both",
        "serving_rule": "2",
    }


def test_chunked_fit_gives_the_same_lists_and_can_drop_its_counts():
    frame = _sequences()
    whole = PCTMRecommender(max_distance=5, smoothing_tau=10.0, pop_beta=-0.25).fit(frame)
    chunked = PCTMRecommender(max_distance=5, smoothing_tau=10.0, pop_beta=-0.25).fit(
        frame, chunk_users=3
    )
    np.testing.assert_allclose(chunked._counts.toarray(), whole._counts.toarray(), rtol=1e-12)
    assert chunked.topk_with_scores(4).keys() == whole.topk_with_scores(4).keys()
    for uid in (1, 7, 99):
        assert chunked.recommend(uid, 4) == whole.recommend(uid, 4)
    before = chunked.recommend(99, 4)
    assert chunked.release_counts().recommend(99, 4) == before
    assert chunked._counts is None
    with pytest.raises(ValueError):
        PCTMRecommender().fit(frame, chunk_users=0)


def _pctm_snapshot(artifact_dir, tmp_path, *, serve: str):
    from movielens_recommender.serving.bundle import (
        fit_pctm,
        load_bundle,
        pctm_manifest_block,
        save_bundle,
    )

    bundle = load_bundle(artifact_dir)
    manifest = {
        key: value
        for key, value in bundle.manifest.items()
        if key not in {"snapshot_id", "artifact_sha256", "provenance_version"}
    }
    manifest["pctm"] = pctm_manifest_block(
        {"max_distance": 5, "smoothing_tau": 10.0, "pop_beta": -0.25},
        relevance_threshold=4.0,
        source="test",
    )
    manifest["serving_list"] = serve
    bundle.manifest = manifest
    bundle.pctm = fit_pctm(bundle.histories, manifest["pctm"])
    return load_bundle(save_bundle(bundle, tmp_path / f"snapshot-{serve}"))


def test_snapshot_serves_the_transition_list_with_named_evidence(artifact_dir, tmp_path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    from movielens_recommender.serving.bundle import (
        known_user_ids,
        rank_for_user,
        recommend_for_user,
        user_history_items,
    )

    bundle = _pctm_snapshot(artifact_dir, tmp_path, serve="pctm")
    assert bundle.pctm is not None and bundle.serving_list == "pctm"
    assert not any(path.name.startswith("pctm") for path in (tmp_path / "snapshot-pctm").iterdir())
    uid = known_user_ids(bundle)[0]
    result = recommend_for_user(bundle, uid, n=5)
    ids = [card["item_id"] for card in result["production"]]
    assert ids == bundle.pctm.recommend(uid, 5)
    assert ids == [card["item_id"] for card in result["pctm"]]
    assert ids == [item for item, _score in rank_for_user(bundle, uid, n=5)]
    assert result["serving_list"] == "pctm"
    assert result["pipeline"]["candidate_source"] == "pctm"
    assert result["pipeline"]["scored_by_ranker"] is False
    assert result["pipeline"]["ranked_ids"][:5] == ids
    seen = {item.item_id for item in user_history_items(bundle, uid)}
    assert not seen & set(ids)
    titles = {item.title for item in user_history_items(bundle, uid)}
    for card in result["production"]:
        assert card["reasons"]
        assert card["bias"] == 0.0
        for row in card["details"]:
            assert row["feature"].startswith("after ")
            assert any(title.startswith(row["feature"][6:].split(" (")[0]) for title in titles)


def test_snapshot_without_a_pctm_block_serves_the_ranker(artifact_dir, tmp_path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    from movielens_recommender.serving.bundle import (
        known_user_ids,
        load_bundle,
        recommend_for_user,
    )

    plain = load_bundle(artifact_dir)
    assert plain.pctm is None and plain.serving_list == "ranker"
    uid = known_user_ids(plain)[0]
    base = recommend_for_user(plain, uid, n=5)
    assert "pctm" not in base and base["serving_list"] == "ranker"

    with_model = _pctm_snapshot(artifact_dir, tmp_path, serve="ranker")
    again = recommend_for_user(with_model, uid, n=5)
    assert [c["item_id"] for c in again["production"]] == [
        c["item_id"] for c in base["production"]
    ]
    assert 1 <= len(again["pctm"]) <= 5


def test_served_model_panels_read_the_committed_record():
    import json

    root = Path(__file__).resolve().parents[1]
    path = root / "results" / "ranker-pctm" / "ml-1m.json"
    before = path.read_bytes()
    record = json.loads(before.decode("utf-8"))
    script = _script()
    wins = record["decision"]["wins"]
    assert wins == {name: block["low"] > 0.0 for name, block in record["paired"].items()}
    choice = script.serving_choice(wins)
    assert {key: record["decision"][key] for key in choice} == choice
    assert record["selected_sequence_candidate_set"] == script.choose_sequence_set(
        record["validation_candidates"]
    )

    def _load(file: Path, name: str):
        spec = importlib.util.spec_from_file_location(name, file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    table = _load(root / "scripts" / "make_results_table.py", "served_panel_table")
    assert table.render_ranker_pctm(root / "missing.json") == ""
    panel = table.render_ranker_pctm(path)
    assert "### Pairwise transitions in the ranker, and the served model (`ml-1m`)" in panel
    served = record["variants"][record["decision"]["serve"]]
    assert table._fmt(served["ndcg@10_mean"]) in panel
    assert f"`serve={record['decision']['serve']}`" in panel

    quality = _load(root / "space" / "quality.py", "served_panel_quality")
    view = quality.load_quality(root / "results")
    block = view["served"]
    assert block["serve"] == record["decision"]["serve"]
    assert [row["model"] for row in block["models"]] == [
        "served_ranker",
        "pctm_only",
        "previous_ranker",
    ]
    assert block["models"][0]["ndcg@10"] == served["ndcg@10_mean"]
    ranker = block["ranker_candidate"]
    pair = record["paired"][f"{ranker}_minus_bi_both"]
    lead = quality.served_lead(block)
    assert quality.signed(pair["mean"]) in lead
    assert quality.signed(pair["low"]) in lead
    assert quality.fmt(pair["mean"]) in quality.served_caption(view)
    rows = quality.chart_rows(block["models"])
    assert rows[0]["best"] and rows[0]["label"] == "Full model, served"
    assert path.read_bytes() == before


def test_build_reads_the_recorded_serving_choice(tmp_path):
    import json

    from movielens_recommender.config import load_config
    from movielens_recommender.serving.build import load_sequence_serving

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "ml-1m.yaml")
    config.results_dir = str(root / "results")
    record = json.loads((root / "results" / "ranker-pctm" / "ml-1m.json").read_text("utf-8"))
    choice = load_sequence_serving(config)
    assert choice is not None
    assert choice["serve"] == record["decision"]["serve"]
    assert choice["candidate_set"] == record["selected_sequence_candidate_set"]
    assert choice["pctm_hyperparams"] == record["pctm_hyperparams"]
    assert choice["demographics"] == {"seq_off": "off", "seq_both": "both"}.get(choice["serve"])

    config.results_dir = str(tmp_path)
    assert load_sequence_serving(config) is None
    target = tmp_path / "ranker-pctm" / "ml-1m.json"
    target.parent.mkdir(parents=True)
    record["decision"]["serve"] = "bi_both"
    target.write_text(json.dumps(record), encoding="utf-8")
    assert load_sequence_serving(config) is None
    record["decision"]["serve"] = "something_else"
    target.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        load_sequence_serving(config)


@pytest.mark.parametrize("serve", ["seq_off", "pctm_only"])
def test_build_applies_the_sequence_choice_and_survives_the_space_rebuild(
    serve, serving_tables, tmp_path, monkeypatch
):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    from conftest import SERVING_GIT_SHA, SERVING_SHA256, _config
    from movielens_recommender.ranker.features import feature_names as plain_names
    from movielens_recommender.serving.build import fit_serving_bundle
    from movielens_recommender.serving.bundle import (
        known_user_ids,
        load_bundle,
        recommend_for_user,
        save_bundle,
    )
    from movielens_recommender.split import SplitConfig, time_based_split

    ratings, movies, users = serving_tables
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
        sequence={
            "serve": serve,
            "candidate_set": "pctm_two_tower",
            "demographics": "off" if serve == "seq_off" else None,
            "pctm_hyperparams": {"max_distance": 5, "smoothing_tau": 10.0, "pop_beta": -0.25},
            "source": "test",
        },
    )
    manifest = bundle.manifest
    assert manifest["pctm"]["hyperparams"]["max_distance"] == 5
    if serve == "seq_off":
        assert manifest["serving_list"] == "ranker"
        assert manifest["candidate_set"] == "pctm_two_tower"
        assert manifest["demographics"] == "off"
        assert list(bundle.context.names) == pctm_ranker_feature_names("off")
        assert list(bundle.booster.feature_name()) == pctm_ranker_feature_names("off")
    else:
        assert manifest["serving_list"] == "pctm"
        assert manifest["demographics"] == "both"
        assert list(bundle.context.names) == plain_names("both")

    directory = save_bundle(bundle, tmp_path / "full")
    loaded = load_bundle(directory)
    uid = known_user_ids(loaded)[0]
    full = recommend_for_user(loaded, uid, n=5)
    ids = [card["item_id"] for card in full["production"]]
    assert ids and ids == [c["item_id"] for c in recommend_for_user(bundle, uid, n=5)["production"]]
    if serve == "seq_off":
        # No demographic columns, so no "viewers similar to you" sentence.
        reasons = [reason for card in full["production"] for reason in card["reasons"]]
        assert not any("similar to you" in reason for reason in reasons)

    # The demo ships parameter files and rebuilds the snapshot from ratings at startup.
    monkeypatch.syspath_prepend(str(SCRIPTS.parent / "space"))
    monkeypatch.delitem(sys.modules, "runtime", raising=False)
    import runtime

    weights = tmp_path / "weights"
    runtime._export_one(directory, weights / "production", extra={})
    runtime.assert_weights_only(weights)
    assert not any(path.name.startswith("pctm") for path in (weights / "production").iterdir())
    production, _cold = runtime.materialize_space(
        weights, tmp_path / "runtime", ratings=ratings, movies=movies, users=users
    )
    rebuilt = load_bundle(production)
    assert rebuilt.pctm is not None and rebuilt.serving_list == loaded.serving_list
    again = [card["item_id"] for card in recommend_for_user(rebuilt, uid, n=5)["production"]]
    assert again == ids
