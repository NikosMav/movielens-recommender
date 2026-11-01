"""Result rendering preserves the identity of points and intervals."""

import runpy
from pathlib import Path

import pytest

from movielens_recommender.reporting import summarize_seed_metrics


@pytest.mark.parametrize("legacy", [True, False])
def test_headline_renders_mean_and_primary_run_as_separate_values(legacy):
    renderer = runpy.run_path(str(Path(__file__).parents[1] / "scripts/make_results_table.py"))
    first = {key: 0.05 for key in renderer["METRIC_COLS"]}
    first["confidence_intervals"] = {
        "ndcg@10": {"mean": 0.05, "low": 0.04, "high": 0.06},
    }
    second = {**first, "ndcg@10": 0.15}
    metrics = summarize_seed_metrics(
        [{"seed": 42, "metrics": first}, {"seed": 43, "metrics": second}], primary_seed=42,
    )
    if legacy:
        metrics = {**first, "ndcg@10": 0.1, "seed_summary": metrics["seed_summary"]}
    text = renderer["render_headline"]("synthetic.json", {
        "dataset": "synthetic", "seed": 42, "metrics": {"two_tower": metrics},
    })
    assert "0.1000±0.0500" in text
    assert "42 | 0.0500 [0.0400, 0.0600]" in text


def test_sasrec_reference_reader_accepts_explicit_seed_summaries():
    pytest.importorskip("torch")
    reader = runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_sasrec.py"))
    summary = summarize_seed_metrics([
        {"seed": 42, "metrics": {
            "ndcg@10": 0.05,
            "confidence_intervals": {"ndcg@10": {"mean": 0.05, "low": 0.04, "high": 0.06}},
        }},
        {"seed": 43, "metrics": {
            "ndcg@10": 0.15,
            "confidence_intervals": {"ndcg@10": {"mean": 0.15, "low": 0.14, "high": 0.16}},
        }},
    ], primary_seed=42)
    published = reader["_published"]({"metrics": {
        "two_tower": summary, "lambdarank": summary,
    }})
    assert published["two_tower"]["ndcg@10"] == pytest.approx(0.1)
    assert published["two_tower"]["primary_seed_ndcg@10"] == 0.05
    assert published["lambdarank"]["primary_seed_ci"]["high"] == 0.06
