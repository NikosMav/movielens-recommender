"""The How good is it page repeats committed JSON values and does not write them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
SPACE = Path(__file__).resolve().parents[1] / "space"


def _quality():
    path = SPACE / "quality.py"
    spec = importlib.util.spec_from_file_location("space_quality_panel_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_panel_values_equal_the_json_files():
    quality = _quality()
    paths = [
        RESULTS / "ml-1m.json",
        RESULTS / "ml-32m.json",
        RESULTS / "sasrec" / "ml-1m.json",
    ]
    before = [path.read_bytes() for path in paths]
    view = quality.load_quality(RESULTS)
    assert [path.read_bytes() for path in paths] == before

    ml1 = _load("ml-1m.json")
    for row in view["ml1m"]["models"]:
        raw = ml1["metrics"][row["model"]]
        interval = raw["confidence_intervals"]["ndcg@10"]
        assert row["ndcg@10"] == raw["ndcg@10"]
        assert row["ci_low"] == interval["low"]
        assert row["ci_high"] == interval["high"]
        assert row["ci_mean"] == interval["mean"]
        if "n_seeds" in row:
            assert row["n_seeds"] == raw["n_seeds"]
        else:
            assert "n_seeds" not in raw
    assert view["ml1m"]["ranker_gate"] == ml1["ranker"]["gate"]
    assert view["ml1m"]["two_tower_negative_result"] == ml1["two_tower"]["gate"]["negative_result"]
    assert (
        view["ml1m"]["two_tower_beats_both_item_item_bars"]
        == ml1["two_tower"]["gate"]["beats_both_item_item_bars"]
    )

    ml32 = _load("ml-32m.json")
    for row in view["ml32m"]["models"]:
        raw = ml32["metrics"][row["model"]]
        interval = raw["confidence_intervals"]["ndcg@10"]
        assert row["ndcg@10"] == raw["ndcg@10"]
        assert row["ci_low"] == interval["low"]
        assert row["ci_high"] == interval["high"]
        assert row["ci_mean"] == interval["mean"]
    assert view["ml32m"]["ranker_gate"] == ml32["ranker"]["gate"]
    assert (
        view["ml32m"]["two_tower_negative_result"] == ml32["two_tower"]["gate"]["negative_result"]
    )
    assert (
        view["ml32m"]["two_tower_beats_both_item_item_bars"]
        == ml32["two_tower"]["gate"]["beats_both_item_item_bars"]
    )
    sample = ml32["eval_user_sample"]
    assert view["ml32m"]["n_sampled_users"] == sample["n_sampled_users"]
    assert view["ml32m"]["n_eligible_users"] == sample["n_eligible_users"]
    assert view["ml32m"]["selection"] == sample["selection"]

    sas = _load("sasrec/ml-1m.json")
    paired = sas["paired"]["real_minus_shuffled"]
    block = view["sasrec"]
    assert block["order_helps"] == sas["decision"]["order_helps"]
    assert block["negative_result"] == sas["decision"]["negative_result"]
    assert block["rule"] == sas["decision"]["rule"]
    assert block["real_minus_shuffled_mean"] == paired["mean"]
    assert block["real_minus_shuffled_low"] == paired["low"]
    assert block["real_minus_shuffled_high"] == paired["high"]
    assert block["real_minus_shuffled_excludes_zero"] == paired["excludes_zero"]
    assert block["real_minus_shuffled_n_users"] == paired["n_users"]
    assert block["ndcg@10_mean"] == sas["models"]["sasrec"]["ndcg@10_mean"]
    primary = sas["models"]["sasrec"]["primary_ndcg@10_ci"]
    assert block["primary_low"] == primary["low"]
    assert block["primary_high"] == primary["high"]
    assert block["primary_mean"] == primary["mean"]
    shuffled = sas["models"]["sasrec_shuffled"]
    assert block["shuffled_ndcg@10"] == shuffled["ndcg@10"]
    assert block["shuffled_low"] == shuffled["ndcg@10_ci"]["low"]
    assert block["shuffled_high"] == shuffled["ndcg@10_ci"]["high"]
    assert block["shuffled_mean"] == shuffled["ndcg@10_ci"]["mean"]

    caption = "\n".join(
        [
            quality.ml1_caption(view),
            quality.ml32_caption(view),
            quality.sasrec_caption(view),
        ]
    )
    assert quality.fmt(view["ml1m"]["ranker_gate"]["ranker_mean_ndcg@10"]) in caption
    assert quality.fmt(view["ml32m"]["n_sampled_users"]) in caption
    assert quality.fmt(view["ml32m"]["n_eligible_users"]) in caption
    assert quality.fmt(block["real_minus_shuffled_mean"]) in caption
    assert quality.fmt(block["ndcg@10_mean"]) in caption
    assert quality.fmt(block["shuffled_ndcg@10"]) in caption
    for table in (view["ml1m"]["models"], view["ml32m"]["models"]):
        rendered = quality.display_rows(table)
        for source, shown in zip(table, rendered, strict=True):
            assert shown["NDCG@10"] == quality.fmt(source["ndcg@10"])
            assert shown["interval low"] == quality.fmt(source["ci_low"])
            assert shown["interval high"] == quality.fmt(source["ci_high"])
            assert shown["interval mean"] == quality.fmt(source["ci_mean"])

    lead = quality.lead_paragraph(view["ml1m"]["models"], dataset="MovieLens 1M")
    info = quality.rank_models(view["ml1m"]["models"])
    best = info["best"]
    assert quality.rounded(best["ndcg@10"]) in lead
    assert quality.rounded(best["ci_low"]) in lead
    assert quality.rounded(best["ci_high"]) in lead
    assert quality.friendly_name(best["model"]) in lead
    assert best["model"] in lead
    assert quality.rounded(info["lead"]) in lead
    chart = quality.chart_rows(view["ml1m"]["models"])
    assert chart[0]["ndcg"] == view["ml1m"]["models"][0]["ndcg@10"]
    assert any(row["best"] and row["model"] == best["model"] for row in chart)
    assert quality.rounded(view["ml1m"]["models"][0]["ndcg@10"]) == format(
        float(view["ml1m"]["models"][0]["ndcg@10"]), ".3f"
    )
