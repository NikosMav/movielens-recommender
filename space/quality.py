"""Headline metrics for the demo, read from the committed results JSON.

Every number in the returned view is a value already stored in those files.
Captions format those values. They do not compute a new metric.
"""

from __future__ import annotations

import html
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

MODEL_GROUPS: dict[str, str] = {
    "most_popular": "Baselines",
    "als": "Baselines",
    "als_tuned": "Baselines",
    "ease": "Baselines",
    "rp3beta": "Baselines",
    "item_item_cosine": "Baselines",
    "item_item_cosine_tuned": "Baselines",
    "two_tower": "Retrieval",
    "no_ranker": "Retrieval",
    "lambdarank": "Full pipeline",
}

BEYOND_FIELDS: tuple[tuple[str, str], ...] = (
    ("coverage@10", "catalog coverage"),
    ("novelty@10", "novelty"),
    ("intra_list_diversity@10", "diversity"),
    ("long_tail_share@10", "long-tail share"),
)

_SERVING_LATENCIES: tuple[tuple[str, str], ...] = (
    ("new_user_five_ratings", "New user with five ratings"),
    ("known_user_ranked", "Known user, ranked"),
    ("known_user_explained", "Known user, with explanations"),
)


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
    ease_dan_path = root / "ease-dan" / "ml-1m.json"
    ease_dan = _read_json(ease_dan_path) if ease_dan_path.is_file() else None
    sample = ml32["eval_user_sample"]
    shuffled = sas["paired"]["real_minus_shuffled"]
    sources = {
        "ml-1m": "results/ml-1m.json",
        "ml-32m": "results/ml-32m.json",
        "sasrec": "results/sasrec/ml-1m.json",
    }
    if ease_dan is not None:
        sources["ease-dan"] = "results/ease-dan/ml-1m.json"
    out: dict[str, Any] = {
        "sources": sources,
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
    if ease_dan is not None:
        dan_m = ease_dan["models"]["ease_dan"]["metrics"]
        ease_m = ease_dan["models"]["ease"]["metrics"]
        ii_m = ease_dan["models"]["item_item_cosine"]["metrics"]
        p_ease = ease_dan["paired"]["ease_dan_minus_ease"]
        p_ii = ease_dan["paired"]["ease_dan_minus_item_item"]
        out["ease_dan"] = {
            "beats_ease": ease_dan["decision"]["beats_ease"],
            "beats_item_item": ease_dan["decision"]["beats_item_item"],
            "negative_result": ease_dan["decision"]["negative_result"],
            "rule": ease_dan["decision"]["rule"],
            "selected_hyperparams": ease_dan["selected"]["hyperparams"],
            "models": [
                _model_row({"ease_dan": dan_m, "ease": ease_m, "item_item_cosine": ii_m}, name)
                for name in ("item_item_cosine", "ease_dan", "ease")
            ],
            "paired_ease_mean": p_ease["mean"],
            "paired_ease_low": p_ease["low"],
            "paired_ease_high": p_ease["high"],
            "paired_ease_excludes_zero": p_ease["excludes_zero"],
            "paired_ii_mean": p_ii["mean"],
            "paired_ii_low": p_ii["low"],
            "paired_ii_high": p_ii["high"],
            "paired_ii_excludes_zero": p_ii["excludes_zero"],
        }
    return out


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
        f"{plain_negative_result(bool(gate['negative_result']))}. "
        "The two-tower gate: "
        f"{plain_negative_result(bool(view['ml1m']['two_tower_negative_result']))}. "
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
        "The two-tower gate: "
        f"{plain_negative_result(bool(block['two_tower_negative_result']))}. "
        "It beats both item–item bars: "
        f"{fmt(block['two_tower_beats_both_item_item_bars'])}. "
        "The ranker gate: "
        f"{plain_negative_result(bool(gate['negative_result']))}. "
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
        f"{plain_negative_result(bool(block['negative_result']))}. "
        f"The stored rule is: {block['rule']}. "
        "Real minus shuffled NDCG@10 is "
        f"{fmt(block['real_minus_shuffled_mean'])} "
        f"(from {fmt(block['real_minus_shuffled_low'])} to "
        f"{fmt(block['real_minus_shuffled_high'])}) "
        f"on {fmt(block['real_minus_shuffled_n_users'])} users. "
        f"{plain_excludes_zero(bool(block['real_minus_shuffled_excludes_zero']))}. "
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


# Display labels. Stored ids remain in the detailed scores.
MODEL_NAMES: dict[str, str] = {
    "most_popular": "Most popular",
    "als": "Matrix factorisation",
    "als_tuned": "Matrix factorisation, tuned",
    "ease": "EASE",
    "ease_dan": "EASE-DAN (denoised)",
    "rp3beta": "Graph similarity",
    "item_item_cosine": "Similar films",
    "item_item_cosine_tuned": "Similar films, tuned",
    "two_tower": "Two-tower match",
    "no_ranker": "Two-tower match alone",
    "lambdarank": "Full model",
    "sasrec": "Sequence model",
    "sasrec_shuffled": "Shuffled sequence",
}

NDCG_SENTENCE = (
    "NDCG@10 is a ranking score from 0 to 1. "
    "Higher means the films a viewer actually liked sit closer to the top of the list."
)


def friendly_name(model_id: str) -> str:
    """A short label for a stored model id. Unknown ids stay readable."""
    return MODEL_NAMES.get(model_id, model_id.replace("_", " "))


def rounded(value: Any) -> str:
    """Three decimal places for a stored JSON number. Not a new measurement."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"not a stored number: {value!r}")
    return format(float(value), ".3f")


def rank_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the leader from stored NDCG@10 points. Ties break on the model id."""
    ordered = sorted(models, key=lambda row: (-float(row["ndcg@10"]), str(row["model"])))
    best = ordered[0]
    second = ordered[1] if len(ordered) > 1 else None
    lead = None if second is None else float(best["ndcg@10"]) - float(second["ndcg@10"])
    overlap = False
    if second is not None:
        overlap = float(best["ci_low"]) <= float(second["ci_high"]) and float(
            second["ci_low"]
        ) <= float(best["ci_high"])
    return {"best": best, "second": second, "lead": lead, "intervals_overlap": overlap}


def lead_paragraph(models: list[dict[str, Any]], *, dataset: str) -> str:
    """One plain sentence built only from the stored points and intervals."""
    info = rank_models(models)
    best = info["best"]
    name = friendly_name(str(best["model"]))
    sentence = (
        f"On {dataset}, {name} has the highest score in this comparison: "
        f"{rounded(best['ndcg@10'])} "
        f"(range {rounded(best['ci_low'])}–{rounded(best['ci_high'])})."
    )
    if info["second"] is None or info["lead"] is None:
        return sentence
    other = info["second"]
    gap = (
        f" That is {rounded(info['lead'])} above {friendly_name(str(other['model']))}."
    )
    if info["intervals_overlap"]:
        tail = " Their uncertainty ranges overlap; the score gap alone is not a clear verdict."
    else:
        tail = " The leader's uncertainty range sits above the next model's range."
    return sentence + gap + tail


def overview_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A small comparison; always include the leader, even if a new model wins."""
    reference = {"lambdarank", "item_item_cosine", "two_tower", "most_popular"}
    if len(models) > 4:
        reference.add(str(rank_models(models)["best"]["model"]))
        selected = [row for row in models if str(row["model"]) in reference]
    else:
        selected = list(models)
    return sorted(selected, key=lambda row: (-float(row["ndcg@10"]), str(row["model"])))


def chart_rows(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows for the bar chart. Widths use the stored floats; labels are rounded later."""
    winner = str(rank_models(models)["best"]["model"])
    rows = []
    for row in models:
        model_id = str(row["model"])
        rows.append(
            {
                "model": model_id,
                "label": friendly_name(model_id),
                "ndcg": float(row["ndcg@10"]),
                "low": float(row["ci_low"]),
                "high": float(row["ci_high"]),
                "best": model_id == winner,
                "group": MODEL_GROUPS.get(model_id, "Baselines"),
            }
        )
    return rows


def sasrec_lead(block: dict[str, Any]) -> str:
    """Plain reading of the stored order check. The numbers are the JSON fields."""
    if block["order_helps"]:
        verdict = "Keeping the real watch order scores higher than a shuffled order."
    else:
        verdict = (
            "Keeping the real watch order does not score higher than a shuffled order "
            "on the stored check."
        )
    return (
        f"{verdict} The stored difference in NDCG@10 is "
        f"{rounded(block['real_minus_shuffled_mean'])} "
        f"(uncertainty range {rounded(block['real_minus_shuffled_low'])} to "
        f"{rounded(block['real_minus_shuffled_high'])})."
    )


def ease_dan_lead(block: dict[str, Any]) -> str:
    """Plain reading of the stored EASE-DAN vs EASE check (ADR-0028)."""
    if block["beats_ease"]:
        verdict = (
            "Adding item-degree dropout regularization (EASE-DAN) beats standard EASE "
            "and closes the statistical gap to item–item similarity."
        )
    else:
        verdict = "EASE-DAN did not beat standard EASE on the stored check."
    return (
        f"{verdict} Paired EASE-DAN minus EASE NDCG@10 is "
        f"{rounded(block['paired_ease_mean'])} "
        f"(range {rounded(block['paired_ease_low'])} to {rounded(block['paired_ease_high'])})."
    )


def ease_dan_caption(view: dict[str, Any]) -> str:
    block = view["ease_dan"]
    hp = block["selected_hyperparams"]
    return (
        f"Selected on validation: l2={fmt(hp['l2'])}, dropout_p={fmt(hp['dropout_p'])}, "
        f"alpha={fmt(hp['alpha'])}, beta={fmt(hp['beta'])}. "
        f"beats_ease is {fmt(block['beats_ease'])}; "
        f"beats_item_item is {fmt(block['beats_item_item'])}. "
        f"Paired ease_dan minus ease: {fmt(block['paired_ease_mean'])} "
        f"({fmt(block['paired_ease_low'])} to {fmt(block['paired_ease_high'])}; "
        f"{plain_excludes_zero(bool(block['paired_ease_excludes_zero']))}). "
        f"Paired ease_dan minus item_item_cosine: {fmt(block['paired_ii_mean'])} "
        f"({fmt(block['paired_ii_low'])} to {fmt(block['paired_ii_high'])}; "
        f"{plain_excludes_zero(bool(block['paired_ii_excludes_zero']))})."
    )


def sample_sentence(block: dict[str, Any]) -> str:
    """The stored ml-32M sample size, without treating the selection note as a score."""
    return (
        f"These scores use {int(block['n_sampled_users']):,} of "
        f"{int(block['n_eligible_users']):,} eligible viewers."
    )


def detail_html(models: list[dict[str, Any]]) -> str:
    """The stored figures, one block per model, using ``fmt`` so nothing is remeasured."""
    blocks = []
    for row in models:
        seeds = ""
        if "n_seeds" in row:
            seeds = f"<p>Seeds recorded: {html.escape(fmt(row['n_seeds']))}</p>"
        blocks.append(
            "<section class='score-detail'>"
            f"<h3>{html.escape(friendly_name(str(row['model'])))} "
            f"<span class='bar-id'>{html.escape(str(row['model']))}</span></h3>"
            f"<p>NDCG@10 {html.escape(fmt(row['ndcg@10']))}</p>"
            "<p>Uncertainty range "
            f"{html.escape(fmt(row['ci_low']))} to {html.escape(fmt(row['ci_high']))}"
            f" (mean {html.escape(fmt(row['ci_mean']))})</p>"
            f"{seeds}</section>"
        )
    return "".join(blocks)


def plain_negative_result(value: bool) -> str:
    """The stored ``negative_result`` flag, in words."""
    if value:
        return "Did not beat the baseline"
    return "Beat the baseline"


def plain_excludes_zero(value: bool) -> str:
    """The stored ``excludes_zero`` flag, in words."""
    if value:
        return "Interval excludes zero"
    return "Interval includes zero"


def beyond_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Point values from ``results/beyond-accuracy/ml-1m.json``, in file order."""
    models = payload["models"]
    rows = []
    for name in payload["model_order"]:
        block = models[name]
        row: dict[str, Any] = {"model": name}
        for key, _label in BEYOND_FIELDS:
            row[key] = block[key]
        rows.append(row)
    return rows


def beyond_takeaway(rows: list[dict[str, Any]]) -> str:
    """One sentence. Each clause is the model with the highest stored point."""
    if not rows:
        return ""
    parts: list[str] = []
    for key, label in BEYOND_FIELDS:
        best = max(
            enumerate(rows),
            key=lambda pair: (float(pair[1][key]), -pair[0]),
        )[1]
        parts.append(
            f"{friendly_name(str(best['model']))} has the highest {label} "
            f"({rounded(best[key])})"
        )
    if len(parts) == 1:
        body = parts[0]
    else:
        body = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"On the stored ml-1m lists, {body}."


def beyond_definitions(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Stored definition strings, in panel order. Missing keys are skipped."""
    definitions = payload.get("definitions") or {}
    wanted = (
        ("coverage", "Coverage"),
        ("novelty", "Novelty"),
        ("intra_list_diversity", "Diversity"),
        ("long_tail_share", "Long-tail share"),
    )
    rows = []
    for key, label in wanted:
        text = definitions.get(key)
        if isinstance(text, str) and text.strip():
            rows.append((label, text.strip()))
    return rows


_LATENCY_PHRASE = {
    "new_user_five_ratings": "for a new user",
    "known_user_ranked": "for a known user",
    "known_user_explained": "for a known user with explanations",
}


def serving_latency_notes(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Rounded sentence, and the stored median and 95th percentile for a tooltip."""
    latency = payload.get("latency")
    if not isinstance(latency, dict):
        return []
    notes = []
    for key, label in _SERVING_LATENCIES:
        block = latency.get(key)
        if not isinstance(block, dict):
            continue
        if "p50_ms" not in block or "p95_ms" not in block:
            continue
        phrase = _LATENCY_PHRASE.get(key, label[:1].lower() + label[1:])
        visible = f"About {round(float(block['p50_ms']))} ms {phrase}"
        tip = (
            f"median {fmt(block['p50_ms'])} ms, "
            f"95th percentile {fmt(block['p95_ms'])} ms"
        )
        notes.append((visible, tip))
    return notes


def serving_latency_lines(payload: dict[str, Any]) -> list[str]:
    """Rounded sentences. The precise stored values are the second half of each note."""
    return [visible for visible, _tip in serving_latency_notes(payload)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload
