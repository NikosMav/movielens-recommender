"""Headline metrics for the demo, read from the committed results JSON.

Every number in the returned view is a value already stored in those files.
Captions format those values. They do not compute a new metric.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The models a reader needs in order to see the ranker against the baselines.
HEADLINE_MODELS: tuple[str, ...] = (
    "most_popular",
    "als",
    "als_tuned",
    "ease",
    "rp3beta",
    "item_item_cosine",
    "item_item_cosine_tuned",
    "two_tower",
    "no_ranker",
    "lambdarank",
)

ML32_MODELS: tuple[str, ...] = ("two_tower", "lambdarank", "item_item_cosine")


def fmt(value: Any) -> str:
    """Render a stored JSON number without adding a new measurement."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def load_quality(results_dir: Path | str) -> dict[str, Any]:
    """Load the view the page renders. Raises ``FileNotFoundError`` if a file is missing."""
    root = Path(results_dir)
    ml1 = _read_json(root / "ml-1m.json")
    ml32 = _read_json(root / "ml-32m.json")
    sas = _read_json(root / "sasrec" / "ml-1m.json")
    sample = ml32["eval_user_sample"]
    shuffled = sas["paired"]["real_minus_shuffled"]
    return {
        "sources": {
            "ml-1m": "results/ml-1m.json",
            "ml-32m": "results/ml-32m.json",
            "sasrec": "results/sasrec/ml-1m.json",
        },
        "ml1m": {
            "models": [_model_row(ml1["metrics"], name) for name in HEADLINE_MODELS],
            "ranker_gate": ml1["ranker"]["gate"],
            "two_tower_negative_result": ml1["two_tower"]["gate"]["negative_result"],
            "two_tower_beats_both_item_item_bars": ml1["two_tower"]["gate"][
                "beats_both_item_item_bars"
            ],
        },
        "ml32m": {
            "models": [_model_row(ml32["metrics"], name) for name in ML32_MODELS],
            "two_tower_negative_result": ml32["two_tower"]["gate"]["negative_result"],
            "two_tower_beats_both_item_item_bars": ml32["two_tower"]["gate"][
                "beats_both_item_item_bars"
            ],
            "ranker_gate": ml32["ranker"]["gate"],
            "n_sampled_users": sample["n_sampled_users"],
            "n_eligible_users": sample["n_eligible_users"],
            "selection": sample["selection"],
        },
        "sasrec": {
            "order_helps": sas["decision"]["order_helps"],
            "negative_result": sas["decision"]["negative_result"],
            "rule": sas["decision"]["rule"],
            "real_minus_shuffled_mean": shuffled["mean"],
            "real_minus_shuffled_low": shuffled["low"],
            "real_minus_shuffled_high": shuffled["high"],
            "real_minus_shuffled_excludes_zero": shuffled["excludes_zero"],
            "real_minus_shuffled_n_users": shuffled["n_users"],
            "ndcg@10_mean": sas["models"]["sasrec"]["ndcg@10_mean"],
            "primary_low": sas["models"]["sasrec"]["primary_ndcg@10_ci"]["low"],
            "primary_high": sas["models"]["sasrec"]["primary_ndcg@10_ci"]["high"],
            "primary_mean": sas["models"]["sasrec"]["primary_ndcg@10_ci"]["mean"],
            "shuffled_ndcg@10": sas["models"]["sasrec_shuffled"]["ndcg@10"],
            "shuffled_low": sas["models"]["sasrec_shuffled"]["ndcg@10_ci"]["low"],
            "shuffled_high": sas["models"]["sasrec_shuffled"]["ndcg@10_ci"]["high"],
            "shuffled_mean": sas["models"]["sasrec_shuffled"]["ndcg@10_ci"]["mean"],
        },
    }


def display_rows(models: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for model in models:
        rows.append(
            {
                "model": str(model["model"]),
                "NDCG@10": fmt(model["ndcg@10"]),
                "interval low": fmt(model["ci_low"]),
                "interval high": fmt(model["ci_high"]),
                "interval mean": fmt(model["ci_mean"]),
                "n_seeds": fmt(model["n_seeds"]) if "n_seeds" in model else "",
            }
        )
    return rows


def ml1_caption(view: dict[str, Any]) -> str:
    gate = view["ml1m"]["ranker_gate"]
    return (
        "NDCG@10 is the committed point for that model. "
        "The interval columns are the stored low, high, and mean of "
        "confidence_intervals.ndcg@10. Where the point and the interval mean differ, "
        "both numbers are in the file and both are shown. "
        f"n_seeds is blank when the file has no such field. "
        f"The ranker gate compares its mean {fmt(gate['ranker_mean_ndcg@10'])} "
        f"with {gate['bar_model']} at {fmt(gate['bar_ndcg@10'])} "
        f"(interval {fmt(gate['bar_ci']['low'])} to {fmt(gate['bar_ci']['high'])}). "
        f"The mean is above that bar's point: {fmt(gate['mean_exceeds_bar_point'])}. "
        "Every seed interval sits above the bar's interval: "
        f"{fmt(gate['all_seed_ci_low_above_bar_ci_high'])}. "
        f"The file marks this gate as a negative result: {fmt(gate['negative_result'])}. "
        "The two-tower gate marks a negative result: "
        f"{fmt(view['ml1m']['two_tower_negative_result'])}. "
        "It beats both item–item bars: "
        f"{fmt(view['ml1m']['two_tower_beats_both_item_item_bars'])}."
    )


def ml32_caption(view: dict[str, Any]) -> str:
    block = view["ml32m"]
    by_name = {row["model"]: row for row in block["models"]}
    tower = by_name["two_tower"]
    ranker = by_name["lambdarank"]
    item = by_name["item_item_cosine"]
    gate = block["ranker_gate"]
    return (
        "On ml-32M the stored NDCG@10 points are "
        f"two-tower {fmt(tower['ndcg@10'])} "
        f"(interval {fmt(tower['ci_low'])} to {fmt(tower['ci_high'])}), "
        f"ranker {fmt(ranker['ndcg@10'])} "
        f"(interval {fmt(ranker['ci_low'])} to {fmt(ranker['ci_high'])}), "
        f"and item–item cosine {fmt(item['ndcg@10'])} "
        f"(interval {fmt(item['ci_low'])} to {fmt(item['ci_high'])}). "
        "The two-tower gate's negative_result is "
        f"{fmt(block['two_tower_negative_result'])}, "
        "and beats_both_item_item_bars is "
        f"{fmt(block['two_tower_beats_both_item_item_bars'])}. "
        "The ranker gate's negative_result is "
        f"{fmt(gate['negative_result'])}. "
        f"Its mean is {fmt(gate['ranker_mean_ndcg@10'])} against {gate['bar_model']} "
        f"at {fmt(gate['bar_ndcg@10'])}. "
        f"These scores use {fmt(block['n_sampled_users'])} of "
        f"{fmt(block['n_eligible_users'])} users. "
        f"{block['selection']}"
    )


def sasrec_caption(view: dict[str, Any]) -> str:
    block = view["sasrec"]
    return (
        "The order check compares the real sequence with a shuffled copy of the same model. "
        f"order_helps is {fmt(block['order_helps'])}. "
        f"negative_result is {fmt(block['negative_result'])}. "
        f"The stored rule is: {block['rule']}. "
        "Real minus shuffled NDCG@10 is "
        f"{fmt(block['real_minus_shuffled_mean'])} "
        f"(from {fmt(block['real_minus_shuffled_low'])} to "
        f"{fmt(block['real_minus_shuffled_high'])}) "
        f"on {fmt(block['real_minus_shuffled_n_users'])} users. "
        f"excludes_zero is {fmt(block['real_minus_shuffled_excludes_zero'])}. "
        f"SASRec's mean NDCG@10 is {fmt(block['ndcg@10_mean'])}. "
        "The primary interval is "
        f"{fmt(block['primary_low'])} to {fmt(block['primary_high'])}, "
        f"and that interval's mean is {fmt(block['primary_mean'])}. "
        f"The shuffled run's NDCG@10 is {fmt(block['shuffled_ndcg@10'])} "
        f"(interval {fmt(block['shuffled_low'])} to {fmt(block['shuffled_high'])}, "
        f"mean {fmt(block['shuffled_mean'])})."
    )


def _model_row(metrics: dict[str, Any], name: str) -> dict[str, Any]:
    block = metrics[name]
    interval = block["confidence_intervals"]["ndcg@10"]
    row: dict[str, Any] = {
        "model": name,
        "ndcg@10": block["ndcg@10"],
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "ci_mean": interval["mean"],
    }
    if "n_seeds" in block:
        row["n_seeds"] = block["n_seeds"]
    return row


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload
