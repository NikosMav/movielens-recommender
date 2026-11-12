"""README and quality panels for ADR-0030 and ADR-0032 read stored JSON only."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sequence_and_slider_panels_match_committed_json(tmp_path):
    sequence_path = ROOT / "results" / "sequence-modern" / "ml-1m.json"
    slider_path = ROOT / "results" / "slider-eval" / "ml-1m.json"
    before = (sequence_path.read_bytes(), slider_path.read_bytes())
    sequence = json.loads(before[0].decode("utf-8"))
    slider = json.loads(before[1].decode("utf-8"))
    assert sequence["decision"]["pctm_beats_sasrec"] is True
    assert sequence["decision"]["esasrec_beats_sasrec"] is False
    assert sequence["decision"]["higher_order_beats_pairwise_pctm"] is False
    assert sequence["paired"]["pctm_minus_sasrec"]["low"] > 0.0

    table = _load(ROOT / "scripts" / "make_results_table.py", "modern_panels_table")
    assert table.render_sequence_modern(tmp_path / "missing.json") == ""
    assert table.render_slider_eval(tmp_path / "missing.json") == ""
    assert table.render_ranker_ease_dan(tmp_path / "missing.json") == ""
    sequence_panel = table.render_sequence_modern(sequence_path)
    slider_panel = table.render_slider_eval(slider_path)
    assert "### Sequential models: eSASRec and PCTM (`ml-1m`)" in sequence_panel
    assert "### Discovery slider (`ml-1m` cold start)" in slider_panel
    pctm_ndcg = sequence["models"]["pctm"]["ndcg@10"]
    assert table._fmt(pctm_ndcg) in sequence_panel
    n5 = next(block for block in slider["by_n"] if block["n"] == 5)
    familiar = next(row for row in n5["positions"] if row["label"] == "Familiar")
    assert table._fmt(familiar["ndcg@10"]) in slider_panel

    quality = _load(ROOT / "space" / "quality.py", "modern_panels_quality")
    view = quality.load_quality(ROOT / "results")
    block = view["sequence_modern"]
    pair = block["paired"]["pctm_minus_sasrec"]
    stored = sequence["paired"]["pctm_minus_sasrec"]
    assert pair["mean"] == stored["mean"]
    lead = quality.sequence_modern_lead(block)
    caption = quality.sequence_modern_caption(view)
    assert quality.rounded(pair["mean"]) in lead
    assert quality.fmt(pair["mean"]) in caption
    slider_block = view["slider"]
    assert slider_block["n"] == 5
    assert slider_block["positions"][0]["ndcg@10"] == familiar["ndcg@10"]
    assert quality.rounded(familiar["ndcg@10"]) in quality.slider_lead(slider_block)
    assert quality.fmt(familiar["ndcg@10"]) in quality.slider_caption(view)

    ranker_path = ROOT / "results" / "ranker-ease-dan" / "ml-1m.json"
    ranker_before = ranker_path.read_bytes()
    ranker = json.loads(ranker_before.decode("utf-8"))
    assert ranker["decision"]["tri_both_lifts_bi_both"] is False
    assert ranker["decision"]["tri_off_lifts_bi_off"] is False
    ranker_panel = table.render_ranker_ease_dan(ranker_path)
    assert "### EASE-DAN as a third retriever (`ml-1m`)" in ranker_panel
    both_mean = ranker["variants"]["tri_both"]["ndcg@10_mean"]
    assert table._fmt(both_mean) in ranker_panel
    ranker_block = view["ranker_ease_dan"]
    paired = ranker_block["paired"]["tri_both_minus_bi_both"]
    assert paired["mean"] == ranker["paired"]["tri_both_minus_bi_both"]["mean"]
    assert quality.rounded(paired["mean"]) in quality.ranker_ease_dan_lead(ranker_block)
    assert quality.fmt(paired["mean"]) in quality.ranker_ease_dan_caption(view)
    assert ranker_path.read_bytes() == ranker_before
    assert (sequence_path.read_bytes(), slider_path.read_bytes()) == before
