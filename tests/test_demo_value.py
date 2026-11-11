"""Re-ranking, why-not, and share links. No model is trained here."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"


def _load(name: str):
    path = SPACE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"space_{name}_value_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_slider_keeps_the_measured_order_and_a_move_is_deterministic():
    value = _load("value")
    ids = [10, 20, 30, 40]
    scores = [0.9, 0.8, 0.7, 0.6]
    popularity = [90.0, 80.0, 10.0, 70.0]
    vectors = np.eye(4)
    assert value.rerank_candidates(
        ids, scores, adventurous=0.0, embeddings=vectors, popularity=popularity, n=3
    ) == [10, 20, 30]
    moved = value.rerank_candidates(
        ids, scores, adventurous=1.0, embeddings=vectors, popularity=popularity, n=3
    )
    again = value.rerank_candidates(
        ids, scores, adventurous=1.0, embeddings=vectors, popularity=popularity, n=3
    )
    assert moved == again
    assert moved != [10, 20, 30]
    assert moved[0] == 30
    assert value.unevaluated_note(0.0) is None
    assert "not been evaluated offline" in (value.unevaluated_note(0.2) or "")


def test_why_not_uses_only_the_list_this_request_scored():
    value = _load("value")
    ranked = [5, 6, 7, 8]
    assert "already in your ratings" in value.why_not(
        title="Toy Story", item_id=5, rated_ids={5}, ranked_ids=ranked
    )
    assert "top 10" in value.why_not(
        title="Fargo", item_id=6, rated_ids=set(), ranked_ids=ranked
    )
    assert "below the top 2" in value.why_not(
        title="Heat", item_id=8, rated_ids=set(), ranked_ids=ranked, top_n=2
    )
    assert "not among the candidates" in value.why_not(
        title="Clerks", item_id=99, rated_ids=set(), ranked_ids=ranked
    )


def test_share_link_round_trip_drops_bad_tokens_and_keeps_the_last_stars():
    value = _load("value")
    encoded = value.encode_ratings([(260, 5), (1, 4)])
    assert encoded == "260-5.1-4"
    assert value.decode_ratings(encoded) == [(260, 5), (1, 4)]
    assert value.decode_ratings("260-5.nope.260-3.1-9") == [(260, 3)]
    assert value.decode_ratings("") == []


def test_stale_reason_hides_a_film_that_is_no_longer_rated():
    value = _load("value")
    kept = {"Toy Story"}
    assert value.cites_unrated("Because you liked Fargo", kept)
    assert value.cites_unrated("Because you liked Toy Story and Fargo", kept)
    assert not value.cites_unrated("Because you liked Toy Story", kept)
    assert not value.cites_unrated("A widely loved film that fits your mix", kept)
    assert value.cites_unrated("Shares Drama with Fargo", kept)
    assert not value.cites_unrated("Shares Comedy with Toy Story", kept)
    steps = value.funnel_steps(
        {
            "n_catalog": 3883,
            "n_candidates": 325,
            "n_scored": 325,
            "scored_by_ranker": True,
        }
    )
    assert [step["name"] for step in steps] == [
        "Catalog",
        "Candidates",
        "Ranked",
        "Your top 10",
    ]
    assert steps[0]["detail"] == "3,883 films"
    assert steps[1]["detail"] == "325"
    visible, tip = value.readout_parts(
        {"genre_count": 10, "mean_popularity_percentile": 100}
    )
    assert "Mostly very popular films" in visible
    assert "100" not in visible
    assert "100" in tip


def test_cold_start_nudge_cites_the_file_only_when_the_flags_match():
    value = _load("value")
    payload = json.loads((ROOT / "results/cold-start/ml-1m.json").read_text(encoding="utf-8"))
    text = value.cold_start_nudge(payload)
    assert "results/cold-start/ml-1m.json" in text
    assert "3 ratings" in text
    assert value.cold_start_nudge({"by_n": []}) == "Rate at least 3 films."


def test_beyond_takeaway_names_the_stored_leader_of_each_column():
    quality = _load("quality")
    payload = json.loads(
        (ROOT / "results/beyond-accuracy/ml-1m.json").read_text(encoding="utf-8")
    )
    rows = quality.beyond_rows(payload)
    sentence = quality.beyond_takeaway(rows)
    for key, label in quality.BEYOND_FIELDS:
        best = max(rows, key=lambda row: float(row[key]))
        assert quality.friendly_name(str(best["model"])) in sentence
        assert label in sentence
        assert quality.rounded(best[key]) in sentence
    assert quality.plain_negative_result(True) == "Did not beat the baseline"
    assert quality.plain_excludes_zero(True) == "Interval excludes zero"
    serving = json.loads((ROOT / "results/serving/ml-1m.json").read_text(encoding="utf-8"))
    notes = quality.serving_latency_notes(serving)
    assert notes
    assert notes[0][0] == "About 137 ms for a new user"
    assert all("median" in tip and "95th percentile" in tip for _visible, tip in notes)
    assert "136.66" in notes[0][1]


def test_failed_poster_image_does_not_retry():
    cards = _load("cards")
    html = cards.poster_html("Toy Story", "https://image.tmdb.org/t/p/w185/toy.jpg")
    assert "this.onerror=null" in html
    assert "removeAttribute('src')" in html
    assert 'referrerpolicy="no-referrer"' in html


def test_missing_evidence_poster_keeps_its_thumbnail_size_class():
    cards = _load("cards")
    html = cards.poster_html("Toy Story", None, extra_class="evidence-thumb")
    assert 'class="poster evidence-thumb placeholder"' in html


def test_results_table_script_does_not_rewrite_the_engineering_section():
    spec = importlib.util.spec_from_file_location(
        "make_results_table_prefix", ROOT / "scripts" / "make_results_table.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    updated = module.replace_section(readme, module.BEGIN + "\nkept\n" + module.END)
    prefix = updated.split(module.BEGIN, 1)[0]
    assert "Engineering process" in prefix
    assert "How this was built" not in prefix
    assert "AI coding" not in prefix
    assert "coding agents" not in prefix
