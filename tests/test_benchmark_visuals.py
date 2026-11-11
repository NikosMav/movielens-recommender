"""The simplified visual comparison preserves scores, uncertainty, and provenance."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPACE = Path(__file__).resolve().parents[1] / "space"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"visual_test_{name}", SPACE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_has_a_zero_origin_and_positions_points_and_ranges_on_the_same_scale():
    cards = _load("cards")
    rendered = cards.metric_chart_html([
        {"model": "example", "label": "Example", "ndcg": 0.1, "low": 0.08, "high": 0.12},
    ], caption="An example.")
    assert "0.000" in rendered and "0.150" in rendered
    assert 'class="bar-fill" style="width:66.67%"' in rendered
    assert 'class="bar-point" style="left:66.67%"' in rendered
    assert 'class="bar-ci" style="left:53.33%;width:26.67%"' in rendered
    assert "range 0.080 to 0.120" in rendered
    assert 'role="listitem"' in rendered
    assert 'class="bar-id"' not in rendered


def test_overview_keeps_the_leader_and_every_selected_value_unchanged():
    quality = _load("quality")
    view = quality.load_quality(SPACE.parent / "results")
    models = view["ml1m"]["models"]
    subset = quality.overview_models(models)
    assert {row["model"] for row in subset} == {
        "lambdarank", "item_item_cosine", "two_tower", "most_popular",
    }
    assert subset[0] == quality.rank_models(models)["best"]
    assert all(row in models for row in subset)
    newcomer = {**models[0], "model": "new_leader", "ndcg@10": 0.9}
    assert quality.overview_models([*models, newcomer])[0] == newcomer
    assert len(quality.chart_rows(models)) == len(models)


def test_generic_diagram_explains_stages_without_placeholder_counts():
    value, cards = _load("value"), _load("cards")
    steps = value.funnel_steps(None)
    assert all(step["description"] for step in steps)
    assert all(not step["detail"] for step in steps)
    rendered = cards.funnel_html(steps)
    assert "funnel-count" not in rendered
    assert "shortlist" in rendered and "explanations" in rendered
    live = value.funnel_steps({
        "n_catalog": 1234, "n_candidates": 27, "n_scored": 27, "scored_by_ranker": True,
    })
    assert "1,234 films" in cards.funnel_html(live)
    assert "27 candidates" in cards.funnel_html(live)
    unscored = value.funnel_steps({"n_catalog": 12, "n_candidates": 3})
    assert "retrieval method" in unscored[2]["description"]
    assert unscored[2]["detail"] == "Not scored"
