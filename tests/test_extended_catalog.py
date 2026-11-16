"""The larger Discover catalogue (ADR-0034): script rules, snapshot, and startup rebuild."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script():
    return _load(ROOT / "scripts" / "run_extended_catalog.py", "run_extended_catalog_test")


def _ratings() -> pd.DataFrame:
    """Forty users. Films 1-3 go together, films 4-6 go together, 7 is rare, 8 is unlisted."""
    rows = []
    for uid in range(1, 41):
        group = [1, 2, 3] if uid % 2 else [4, 5, 6]
        other = [4, 5, 6] if uid % 2 else [1, 2, 3]
        films = group + other[: uid % 3] + ([7] if uid == 1 else []) + [8]
        for step, item in enumerate(films, start=1):
            rating = 5.0 if item in group else 3.0
            rows.append(
                {"user_id": uid, "item_id": item, "rating": rating, "timestamp": uid * 100 + step}
            )
    return pd.DataFrame(rows)


def _movies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item_id": i, "title": f"Film {i} (20{i:02d})", "genres": "Drama", "year": 2000 + i}
            for i in range(1, 8)
        ]
    )


def _fit(script):
    ratings = _ratings()
    listed = _movies()
    catalog = script.catalog_item_ids(ratings, listed, floor=2)
    train = ratings[ratings["item_id"].isin(set(catalog.tolist()))]
    model = script._fit(train, {"k_neighbors": 3, "shrinkage": 0.0, "min_common": 1})
    counts = train.groupby("item_id").size().reindex(catalog).fillna(0).to_numpy(dtype=np.int64)
    return model, counts, catalog


def _manifest(script, serving: dict[str, str]) -> dict:
    return {
        "adr": "0034",
        "candidate_k": 3,
        "candidate_source": "item_item_fold_in",
        "catalog_weights_version": script.WEIGHTS_VERSION,
        "dataset": "synthetic",
        "kind": "catalog",
        "serving_by_n": serving,
        "titles_dataset": "ml-latest-small",
        "weights_only": True,
    }


def test_catalogue_keeps_listed_films_with_enough_ratings():
    script = _script()
    catalog = script.catalog_item_ids(_ratings(), _movies(), floor=2)
    # Film 7 has one rating. Film 8 is rated by everyone and is not listed.
    assert catalog.tolist() == [1, 2, 3, 4, 5, 6]


def test_held_out_users_are_seeded_disjoint_halves():
    script = _script()
    ratings = _ratings()
    val, test, eligible = script.held_out_users(ratings, min_ratings=4, n_held_out=10, seed=42)
    again = script.held_out_users(ratings, min_ratings=4, n_held_out=10, seed=42)
    assert val.tolist() == again[0].tolist() and test.tolist() == again[1].tolist()
    assert len(val) == len(test) == 5 and not set(val.tolist()) & set(test.tolist())
    per_user = ratings.groupby("user_id").size()
    assert eligible == int((per_user >= 4).sum())
    assert all(per_user[uid] >= 4 for uid in [*val.tolist(), *test.tolist()])
    with pytest.raises(ValueError):
        script.held_out_users(ratings, min_ratings=4, n_held_out=9, seed=42)


def test_selection_and_ship_rules_follow_the_record():
    script = _script()
    first, second = (config["name"] for config in script.CONFIGS)
    assert script.choose_config({first: 0.2, second: 0.2}) == first
    assert script.choose_config({first: 0.2, second: 0.2000001}) == first
    assert script.choose_config({first: 0.2, second: 0.21}) == second
    assert script.choose_method(0.1, 0.1) == "most_popular"
    assert script.choose_method(0.11, 0.1) == "item_item_fold_in"
    paired = {n: {"low": 0.01} for n in script.PROFILE_NS}
    assert script.ship_decision(paired) is True
    paired[1] = {"low": -0.5}
    assert script.ship_decision(paired) is True
    paired[5] = {"low": 0.0}
    assert script.ship_decision(paired) is False


def test_fit_drops_user_rows_and_scores_a_profile():
    script = _script()
    model, _counts, _catalog = _fit(script)
    assert model._user_item.shape[0] == 0 and model._user_index == {}
    lists = script._fold_in_lists(model, {900: [(1, 5.0), (2, 5.0)]})
    assert lists[900][0] == 3
    assert not {1, 2} & set(lists[900])
    assert script._popular_lists([3, 1, 2, 4], {900: [(1, 5.0)]})[900] == [3, 2, 4]


def test_catalog_snapshot_serves_fold_in_and_popular(tmp_path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    from movielens_recommender.serving.cold import (
        load_cold_start_bundle,
        recommend_new_user,
        save_catalog_bundle,
    )

    script = _script()
    model, counts, catalog = _fit(script)
    serving = {"1": "most_popular", "3": "item_item_fold_in", "5": "item_item_fold_in",
               "10": "item_item_fold_in"}
    directory = save_catalog_bundle(
        tmp_path / "catalog",
        movies=_movies(),
        item_ids=catalog,
        similarity=model._similarity,
        item_counts=counts,
        hyperparams=model.hyperparams(),
        manifest={key: value for key, value in _manifest(script, serving).items()
                  if key not in {"weights_only", "kind", "catalog_weights_version"}},
    )
    names = {path.name for path in directory.iterdir()}
    assert "item_item_user_item.npz" not in names and "histories.npz" not in names
    bundle = load_cold_start_bundle(directory)
    assert bundle.ease is None and bundle.two_tower is None and bundle.booster is None
    assert bundle.manifest["kind"] == "catalog"
    assert list(bundle.context.item_ids) == catalog.tolist()
    assert bundle.movies["item_id"].tolist() == catalog.tolist()
    order = sorted(zip((-counts).tolist(), catalog.tolist(), strict=True))
    assert bundle.popular_item_ids.tolist() == [item for _count, item in order]

    one = recommend_new_user(bundle, [(1, 5.0)], n=3)
    assert one["method"] == "most_popular"
    assert [card["item_id"] for card in one["recommendations"]] == [
        item for item in bundle.popular_item_ids.tolist() if item != 1
    ][:3]
    three = recommend_new_user(bundle, [(1, 5.0), (2, 5.0), (4, 2.0)], n=3)
    assert three["method"] == "item_item_fold_in"
    ids = [card["item_id"] for card in three["recommendations"]]
    assert ids[0] == 3 and not {1, 2, 4} & set(ids)
    assert three["pipeline"]["scored_by_ranker"] is False
    assert three["pipeline"]["n_catalog"] == len(catalog)
    assert all(card["title"].startswith("Film ") for card in three["recommendations"])

    with pytest.raises(ValueError, match="serves only"):
        save_catalog_bundle(
            tmp_path / "bad",
            movies=_movies(),
            item_ids=catalog,
            similarity=model._similarity,
            item_counts=counts,
            hyperparams=model.hyperparams(),
            manifest={"serving_by_n": {"1": "cold_start_ranker"}},
        )
    with pytest.raises(ValueError, match="one catalogue"):
        save_catalog_bundle(
            tmp_path / "bad2",
            movies=_movies(),
            item_ids=catalog[:-1],
            similarity=model._similarity,
            item_counts=counts,
            hyperparams=model.hyperparams(),
            manifest={"serving_by_n": serving},
        )


def test_startup_rebuild_reads_parameter_files_and_a_title_frame(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    from movielens_recommender.serving.cold import load_cold_start_bundle, recommend_new_user

    monkeypatch.syspath_prepend(str(SPACE))
    monkeypatch.delitem(sys.modules, "runtime", raising=False)
    import runtime

    script = _script()
    model, counts, _catalog = _fit(script)
    serving = {str(n): "item_item_fold_in" for n in script.PROFILE_NS}
    weights = tmp_path / "weights" / runtime.EXTENDED_DIR
    sizes = script.export_weights(model, counts, _manifest(script, serving), weights)
    assert set(sizes) == {
        "item_counts.npy",
        "item_ids.npy",
        "item_item_meta.json",
        "item_item_similarity.npz",
        "manifest.json",
    }
    runtime.assert_weights_only(tmp_path / "weights")
    stored = sparse.load_npz(weights / "item_item_similarity.npz")
    assert stored.dtype == np.float32

    rebuilt = runtime.materialize_extended(weights, tmp_path / "runtime", movies=_movies())
    bundle = load_cold_start_bundle(rebuilt)
    assert "weights_only" not in bundle.manifest
    result = recommend_new_user(bundle, [(4, 5.0), (5, 5.0), (1, 1.0)], n=2)
    assert [card["item_id"] for card in result["recommendations"]][0] == 6

    # A title file for another catalogue is refused, and so is a foreign manifest.
    with pytest.raises(ValueError, match="does not list"):
        runtime.materialize_extended(weights, tmp_path / "r2", movies=_movies().head(2))
    manifest = json.loads((weights / "manifest.json").read_text(encoding="utf-8"))
    manifest["kind"] = "cold_start"
    (weights / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="catalog snapshot"):
        runtime.materialize_extended(weights, tmp_path / "r3", movies=_movies())


def test_share_link_names_the_larger_catalogue():
    value = _load(SPACE / "value.py", "extended_value")
    base = "https://example.test/~/+/?view=explore"
    plain = value.share_url(base, [(1, 5), (2, 4)], mix=1)
    extended = value.share_url(base, [(1, 5), (2, 4)], mix=1, extended=True)
    assert "catalog=" not in plain
    assert extended == plain + "&catalog=extended"
    assert "/~/" not in extended


def _blend_script():
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    return _load(ROOT / "scripts" / "run_extended_blend.py", "run_extended_blend_test")


def test_blend_score_is_scaled_similarity_plus_weighted_log_popularity():
    from movielens_recommender.serving.cold import blend_scores

    fold = np.array([[4.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    counts = np.array([0, 9, 99])
    prior = np.log1p(counts) / np.log1p(99)
    np.testing.assert_allclose(blend_scores(fold, counts, 0.0), [[1.0, 0.5, 0.0], [0, 0, 0]])
    np.testing.assert_allclose(blend_scores(fold, counts, 2.0)[0], [1.0, 0.5, 0.0] + 2.0 * prior)
    # No neighbour evidence: the row is the popularity term alone.
    np.testing.assert_allclose(blend_scores(fold, counts, 2.0)[1], 2.0 * prior)
    assert blend_scores(fold, np.zeros(3), 2.0)[0].tolist() == [1.0, 0.5, 0.0]


def test_blend_ranking_moves_from_neighbours_to_popular_as_the_weight_grows():
    from movielens_recommender.serving.cold import blend_rankings

    script = _script()
    model, _counts, catalog = _fit(script)
    counts = np.array([1, 1, 1, 500, 400, 300])
    profile = {7: [(1, 5.0), (2, 5.0)]}
    near = [item for item, _score in blend_rankings(model, counts, profile, 0.0, 4)[7]]
    far = [item for item, _score in blend_rankings(model, counts, profile, 50.0, 4)[7]]
    assert near[0] == 3
    assert far == [4, 5, 6, 3]
    assert not {1, 2} & set(near) and not {1, 2} & set(far)
    scores = [score for _item, score in blend_rankings(model, counts, profile, 1.0, 4)[7]]
    assert scores == sorted(scores, reverse=True)
    assert blend_rankings(model, counts, {}, 1.0, 4) == {}
    assert catalog.tolist() == model._item_ids.tolist()


def test_catalog_snapshot_serves_the_blend(tmp_path):
    pytest.importorskip("lightgbm")
    pytest.importorskip("torch")
    from movielens_recommender.serving.cold import (
        BLEND_METHOD,
        blend_rankings,
        load_cold_start_bundle,
        recommend_new_user,
        save_catalog_bundle,
    )

    script = _script()
    model, counts, catalog = _fit(script)
    serving = {"1": "most_popular", "3": BLEND_METHOD, "5": BLEND_METHOD, "10": BLEND_METHOD}
    directory = save_catalog_bundle(
        tmp_path / "blend",
        movies=_movies(),
        item_ids=catalog,
        similarity=model._similarity,
        item_counts=counts,
        hyperparams=model.hyperparams(),
        manifest={"serving_by_n": serving, "blend_lambda": 0.5, "candidate_k": 3,
                  "candidate_source": BLEND_METHOD, "dataset": "synthetic"},
    )
    bundle = load_cold_start_bundle(directory)
    profile = [(1, 5.0), (2, 5.0), (4, 2.0)]
    result = recommend_new_user(bundle, profile, n=3)
    assert result["method"] == BLEND_METHOD
    assert "how widely it is rated" in result["method_sentence"]
    expected = blend_rankings(bundle.item_item, counts, {0: profile}, 0.5, 3)[0]
    assert [card["item_id"] for card in result["recommendations"]] == [
        item for item, _score in expected
    ]
    assert result["pipeline"]["scored_by_ranker"] is False
    assert all(card["reasons"] for card in result["recommendations"])


def test_blend_script_rules_and_fresh_test_users():
    script = _blend_script()
    assert script.choose_lambda({0.25: 0.3, 0.5: 0.3, 1.0: 0.3}) == 0.25
    assert script.choose_lambda({0.25: 0.3, 0.5: 0.31, 1.0: 0.31}) == 0.5
    assert script.choose_method(0.3, 0.3) == "most_popular"
    assert script.choose_method(0.31, 0.3) == "item_item_popularity_blend"
    assert script.ship_decision({n: {"low": 0.001} for n in script.PROFILE_NS}) is True
    lows = {1: {"low": 1}, 3: {"low": 1}, 5: {"low": 1}, 10: {"low": 0}}
    assert script.ship_decision(lows) is False

    ratings = _ratings()
    used = {1, 2, 3, 4, 5}
    fresh = script.new_test_users(ratings, used, min_ratings=4, n=6, seed=43)
    again = script.new_test_users(ratings, used, min_ratings=4, n=6, seed=43)
    assert fresh.tolist() == again.tolist() and len(set(fresh.tolist())) == 6
    assert not used & set(fresh.tolist())
    with pytest.raises(ValueError):
        script.new_test_users(ratings, used, min_ratings=4, n=1000, seed=43)


def test_readme_panel_reads_both_committed_records(tmp_path):
    base = ROOT / "results" / "extended-catalog"
    first = json.loads((base / "ml-32m.json").read_text(encoding="utf-8"))
    blend = json.loads((base / "ml-32m-blend.json").read_text(encoding="utf-8"))
    script = _script()
    blend_script = _blend_script()
    kept = first["decision"]["kept_config"]
    paired = {
        int(n): first["test"][n]["item_item"][kept]["paired_minus_most_popular"]
        for n in first["test"]
    }
    assert first["decision"]["ships"] is script.ship_decision(paired) is False
    lam = str(blend["decision"]["kept_lambda"])
    blend_paired = {
        int(n): blend["test"][n]["blend"][lam]["paired_minus_most_popular"] for n in blend["test"]
    }
    assert blend["decision"]["ships"] is blend_script.ship_decision(blend_paired) is False
    assert blend["protocol"]["validation_user_ids_sha256"] == (
        first["protocol"]["validation_user_ids_sha256"]
    )
    assert blend["protocol"]["test_user_ids_sha256"] != first["protocol"]["test_user_ids_sha256"]
    assert not (ROOT / "space" / "weights" / "extended").exists()

    table = _load(ROOT / "scripts" / "make_results_table.py", "extended_panel_table")
    assert table.render_extended_catalog(tmp_path / "missing.json") == ""
    panel = table.render_extended_catalog()
    assert "### A larger Discover catalogue (`ml-32m`)" in panel
    assert panel.count("**Negative result (`ships=False`).**") == 2
    assert table._fmt(first["test"]["5"]["item_item"][kept]["ndcg@10"]) in panel
    assert table._fmt(blend["test"]["5"]["blend"][lam]["ndcg@10"]) in panel
    only_first = table.render_extended_catalog(blend_path=tmp_path / "missing.json")
    assert only_first.count("ships=") == 1
