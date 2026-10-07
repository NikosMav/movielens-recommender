#!/usr/bin/env python3
"""Regenerate the Results table section in README.md from committed results/*.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS_DIR = ROOT / "results"

BEGIN = "<!-- BEGIN RESULTS TABLE -->"
END = "<!-- END RESULTS TABLE -->"

METRIC_COLS = [
    "ndcg@10",
    "precision@10",
    "recall@10",
    "ndcg@20",
    "precision@20",
    "recall@20",
    "coverage@10",
    "mean_popularity@10",
]

RETRIEVAL_COLS = ["recall@100", "recall@200"]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt6(value: float) -> str:
    """Six decimals so a sub-1e-4 validation margin stays visible."""
    return f"{value:.6f}"


def _fmt_ci(cis: dict, key: str) -> str:
    bounds = cis.get(key)
    if not bounds:
        return ""
    return f"[{_fmt(bounds['low'])}, {_fmt(bounds['high'])}]"


def _ci_pair(cis: dict, key: str) -> tuple[float, float] | None:
    bounds = cis.get(key)
    if not bounds:
        return None
    return float(bounds["low"]), float(bounds["high"])


def _seed_count(summary: dict | None) -> int | None:
    if not summary:
        return None
    values = summary.get("values")
    if isinstance(values, list):
        return len(values)
    return None


def _ndcg_seed_spread(summary: dict) -> str:
    """One seed has no spread. Printing ±0.0000 would look like a measured std."""
    mean = _fmt(float(summary["mean"]))
    if _seed_count(summary) == 1:
        return f"1 seed {mean}"
    return f"mean±std over seeds {mean}±{_fmt(float(summary['std']))}"


def _recall_seed_cell(point: float, summary: dict | None) -> str:
    cell = _fmt(float(point))
    if not summary:
        return cell
    if _seed_count(summary) == 1:
        return f"{cell} (1 seed)"
    if "std" in summary:
        return f"{cell} ±{_fmt(float(summary['std']))}"
    return cell


def _segment_ndcg(model: dict, which: str) -> float | None:
    block = (model.get("segments") or {}).get("item_head_tail") or {}
    cell = block.get(which) or {}
    value = cell.get("ndcg@10")
    if value is None:
        return None
    return float(value)


def _threshold_text(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def _bracket(pair: tuple[float, float]) -> str:
    return f"[{_fmt(pair[0])}, {_fmt(pair[1])}]"


def _model_label(name: str, tuned_flags: dict | None) -> str:
    is_tuned = bool(tuned_flags and tuned_flags.get(name)) or name.endswith("_tuned")
    if name.endswith("_tuned"):
        return name
    if name == "two_tower":
        return "two_tower (tuned)" if is_tuned else "two_tower"
    if is_tuned:
        return f"{name} (tuned)"
    return name


def render_ranker(payload: dict) -> str:
    """S4 block. Every number is read from the ranker object in the results JSON."""
    ranker = payload.get("ranker")
    if not ranker:
        return ""
    lines: list[str] = []
    lines.append("#### Ranker candidates, seeds, and gate (ADR-0007)")
    lines.append("")
    lines.append(
        f"Candidate budget K={ranker.get('candidate_k')}. "
        f"Selection metric: validation {ranker.get('selection_metric')} "
        f"(split: {ranker.get('selection_split')}). "
        f"Tie-break: {ranker.get('tie_break')}. "
        f"Winner: `{ranker.get('winner')}`."
    )
    lines.append("")
    lines.append("| candidate set | recall@100 | recall@200 | mean size |")
    lines.append("| --- | --- | --- | --- |")
    candidates = ranker.get("candidates") or {}
    for name in sorted(candidates):
        block = candidates[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt6(float(block["recall@100"])),
                    _fmt6(float(block["recall@200"])),
                    _fmt6(float(block["mean_size"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        f"Early-stop user fraction={ranker.get('early_stop_fraction')} "
        f"(split seed={ranker.get('early_stop_user_split_seed')}). "
        "Ranker seeds share candidate sets and retriever models."
    )
    lines.append("")
    lines.append(
        "| seed | best_iteration | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in ranker.get("per_seed", []):
        ci = row.get("ndcg@10_ci") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["seed"]),
                    str(row["best_iteration"]),
                    _fmt(float(row["ndcg@10"])),
                    f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "",
                    _fmt(float(row["recall@10"])) if row.get("recall@10") is not None else "",
                    _fmt(float(row["coverage@10"]))
                    if row.get("coverage@10") is not None
                    else "",
                ]
            )
            + " |"
        )
    across = (ranker.get("across_seeds") or {}).get("ndcg@10") or {}
    if across:
        lines.append("")
        lines.append(
            f"Across seeds: NDCG@10 mean={_fmt(across['mean'])}, "
            f"std={_fmt(across['std'])}, "
            f"min={_fmt(across['min'])}, max={_fmt(across['max'])}."
        )
    gate = ranker.get("gate") or {}
    if gate:
        lines.append("")
        bar = gate.get("bar_ndcg@10")
        mean = gate.get("ranker_mean_ndcg@10")
        bar_ci = gate.get("bar_ci") or {}
        comparison = (
            f"LambdaRank mean NDCG@10={_fmt(float(mean))} vs "
            f"{gate.get('bar_model')} {_fmt(float(bar))} "
            f"[{_fmt(float(bar_ci['low']))}, {_fmt(float(bar_ci['high']))}] "
            f"(read from `{gate.get('bar_source')}`)."
        )
        if gate.get("negative_result"):
            lines.append(
                f"**Gate: negative result.** {comparison} "
                "The ranker does not beat the S3 item–item cosine point estimate."
            )
        else:
            lines.append(f"**Gate: win on the point estimate.** {comparison}")
        if gate.get("all_seed_ci_low_above_bar_ci_high"):
            lines.append("")
            lines.append(
                "Every ranker-seed NDCG@10 CI low sits above the bar CI high."
            )
        else:
            lines.append("")
            lines.append(
                "Seed NDCG@10 intervals are not all above the bar CI high "
                f"(`all_seed_ci_low_above_bar_ci_high="
                f"{gate.get('all_seed_ci_low_above_bar_ci_high')}`)."
            )
    lines.append("")
    ablations = ranker.get("ablations") or {}
    if ablations:
        lines.append(
            "Ablations are the primary seed, except `ndcg@10`, `recall@10`, "
            "and `coverage@10` on the `lambdarank` row, which are means over "
            "the three ranker seeds. The `ndcg@10` CI and tail NDCG@10 on that "
            "row stay the primary seed; that CI is the primary-seed user "
            "bootstrap, not a confidence interval for the 3-seed mean. "
            "`no_ranker` keeps the winning candidate order. "
            "`lambdarank_drop_retriever_features` drops retriever score and rank."
        )
        lines.append("")
        lines.append(
            "| ablation | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 | tail ndcg@10 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name in sorted(ablations):
            block = ablations[name]
            ci = block.get("ndcg@10_ci") or {}
            tail_ci = block.get("tail_ndcg@10_ci") or {}
            tail = block.get("tail_ndcg@10")
            tail_cell = ""
            if tail is not None:
                tail_cell = _fmt(float(tail))
                if tail_ci:
                    tail_cell += f" [{_fmt(float(tail_ci['low']))}, {_fmt(float(tail_ci['high']))}]"
            ci_cell = (
                f"[{_fmt(float(ci['low']))}, {_fmt(float(ci['high']))}]" if ci else ""
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(float(block["ndcg@10"])) if block.get("ndcg@10") is not None else "",
                        ci_cell,
                        _fmt(float(block["recall@10"]))
                        if block.get("recall@10") is not None
                        else "",
                        _fmt(float(block["coverage@10"]))
                        if block.get("coverage@10") is not None
                        else "",
                        tail_cell,
                    ]
                )
                + " |"
            )
        lines.append("")
    gains = ranker.get("feature_importance_gain") or []
    if gains:
        lines.append("Top feature gains (refit ranker, primary seed):")
        lines.append("")
        lines.append("| feature | gain |")
        lines.append("| --- | --- |")
        for row in gains[:10]:
            lines.append(f"| {row['feature']} | {_fmt(float(row['gain']))} |")
        lines.append("")
    artifact = ranker.get("model_artifact") or {}
    if artifact.get("recreate_command"):
        lines.append(
            f"Refit ranker artifact (gitignored): `{artifact.get('path')}` "
            f"schema_version={artifact.get('schema_version')}. "
            f"Recreate with `{artifact.get('recreate_command')}`."
        )
        lines.append("")
    if ranker.get("runtime_sec") is not None:
        lines.append(f"Ranker stage runtime: {_fmt(float(ranker['runtime_sec']))}s.")
        lines.append("")
    return "\n".join(lines)


def _sample_paragraph(payload: dict, sample: dict, split: dict) -> str:
    """Sample line. Counts come from the results JSON."""
    text = (
        "Evaluation users: seeded sample of "
        f"{sample.get('n_sampled_users')} out of "
        f"{sample.get('n_eligible_users')} warm-relevant users "
        f"(seed={sample.get('seed')}, requested={sample.get('requested_users')}, "
        f"user_ids_sha256=`{sample.get('user_ids_sha256')}`). "
    )
    n_eligible = sample.get("n_eligible_users")
    n_train = split.get("n_train_users")
    if n_eligible is not None and n_train:
        pct = 100.0 * float(n_eligible) / float(n_train)
        text += (
            f"{int(n_eligible):,} of {int(n_train):,} train users ({pct:.1f}%) "
            "are eligible, the same eligibility rule the harness uses for test "
            "metrics"
        )
        threshold = payload.get("relevance_threshold")
        if threshold is not None:
            text += (
                f" (at least one test rating >= {_threshold_text(float(threshold))} "
                "on an item in the train catalog)"
            )
        text += ". "
    text += (
        "The sample is partly test-informed: those users are chosen using their "
        "test ratings, and validation rows are restricted to the sample, so "
        "tuning and ranker training see users chosen partly by their test "
        "ratings. The expected effect is small. Drawing the sample without "
        "test ratings is the cleaner alternative and is a known limitation. "
        "Training uses every training interaction. Sampled users keep full "
        "histories. Tuning selection and ranker labels use this same sample."
    )
    return text


def render_scale_ranking(payload: dict) -> str:
    """Answer whether the ml-1m order holds. Figures are read from this JSON."""
    if payload.get("dataset") != "ml-32m":
        return ""
    metrics = payload.get("metrics") or {}
    two_tower = metrics.get("two_tower")
    lambdarank = metrics.get("lambdarank")
    item_item = metrics.get("item_item_cosine")
    no_ranker = metrics.get("no_ranker")
    if not (two_tower and lambdarank and item_item and no_ranker):
        return ""
    tt_ci = _ci_pair(two_tower.get("confidence_intervals") or {}, "ndcg@10")
    lr_ci = _ci_pair(lambdarank.get("confidence_intervals") or {}, "ndcg@10")
    ii_ci = _ci_pair(item_item.get("confidence_intervals") or {}, "ndcg@10")
    lr_seeds = (lambdarank.get("seed_summary") or {}).get("ndcg@10") or {}
    if not (tt_ci and lr_ci and ii_ci and "mean" in lr_seeds):
        return ""
    n_seeds = _seed_count(lr_seeds)
    seed_label = f"{n_seeds}-seed mean" if n_seeds else "mean"
    tt_ndcg = float(two_tower["ndcg@10"])
    ii_ndcg = float(item_item["ndcg@10"])
    nr_ndcg = float(no_ranker["ndcg@10"])
    lr_mean = float(lr_seeds["mean"])
    tied = tt_ci[0] <= lr_ci[1] and lr_ci[0] <= tt_ci[1]
    separated = tt_ci[0] > ii_ci[1] and lr_ci[0] > ii_ci[1]
    tt_head = _segment_ndcg(two_tower, "head")
    tt_tail = _segment_ndcg(two_tower, "tail")
    ii_head = _segment_ndcg(item_item, "head")
    ii_tail = _segment_ndcg(item_item, "tail")
    relation = "are statistically tied at the top" if tied else "are the top two"
    top = (
        f"Two-tower {_fmt(tt_ndcg)} {_bracket(tt_ci)} and LambdaRank "
        f"{seed_label} {_fmt(lr_mean)} (primary-seed CI {_bracket(lr_ci)}) "
        f"{relation}"
    )
    if separated:
        how = (
            "with separated intervals, so the neural approach overtakes "
            "item–item at this scale"
        )
    else:
        how = "on the point estimates"
    beat = f"Both beat item–item cosine {_fmt(ii_ndcg)} {_bracket(ii_ci)} {how}"
    if _ml1m_two_tower_trails_item_item() and tt_ndcg > ii_ndcg:
        beat += ", reversing the ml-1m result"
    if _fmt(nr_ndcg) == _fmt(tt_ndcg):
        no_lift = (
            "LambdaRank adds no lift over its two-tower candidate list "
            f"(`no_ranker` is also {_fmt(nr_ndcg)})"
        )
    else:
        no_lift = (
            "LambdaRank versus the two-tower candidate list (`no_ranker`) is "
            f"{_fmt(lr_mean)} versus {_fmt(nr_ndcg)}"
        )
    head_tail = "The two-tower's lead is on head items"
    have_segments = None not in (tt_head, ii_head, tt_tail, ii_tail)
    if have_segments and tt_head > ii_head and tt_tail < ii_tail:
        head_tail = (
            "The two-tower's lead is on head items "
            f"(head NDCG@10 {_fmt(tt_head)} versus item–item {_fmt(ii_head)}); "
            f"its tail NDCG@10 ({_fmt(tt_tail)}) is below item–item's "
            f"({_fmt(ii_tail)})"
        )
    lines = [
        "#### Does the ranking hold at scale?",
        "",
        f"{top}. {beat}. {no_lift}. {head_tail}.",
    ]
    return "\n".join(lines)


def _ml1m_two_tower_trails_item_item() -> bool:
    path = RESULTS_DIR / "ml-1m.json"
    if not path.is_file():
        return False
    prior = json.loads(path.read_text(encoding="utf-8"))
    metrics = prior.get("metrics") or {}
    two_tower = metrics.get("two_tower") or {}
    item_item = metrics.get("item_item_cosine") or {}
    if "ndcg@10" not in two_tower or "ndcg@10" not in item_item:
        return False
    return float(two_tower["ndcg@10"]) < float(item_item["ndcg@10"])


def load_headline_results() -> list[tuple[str, dict]]:
    files = sorted(p for p in RESULTS_DIR.glob("*.json") if p.is_file())
    if not files:
        raise SystemExit(f"No JSON files found in {RESULTS_DIR}")
    out = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out.append((path.name, payload))
    return out


def render_headline(filename: str, payload: dict) -> str:
    lines: list[str] = []
    dataset = payload["dataset"]
    version = payload.get("dataset_version", dataset)
    tuned_flags = payload.get("tuned") or {}
    lines.append(f"### `{dataset}` (from `results/{filename}`)")
    lines.append("")
    lines.append(f"Pinned version: `{version}`.")
    lines.append("")
    split = payload.get("split", {})
    cfg = split.get("config", {})
    boot = payload.get("bootstrap", {})
    val_f = cfg.get("val_fraction", 0.0)
    lines.append(
        f"Split: min_ratings={cfg.get('min_ratings')}, "
        f"test_fraction={cfg.get('test_fraction')}, "
        f"val_fraction={val_f}, "
        f"relevance_threshold={payload.get('relevance_threshold')}, "
        f"seed={payload.get('seed')}, ks={payload.get('ks')}, "
        f"retrieval_ks={payload.get('retrieval_ks', [])}, "
        f"bootstrap={boot.get('n_bootstrap')} @ alpha={boot.get('alpha')}. "
        f"Primary metric: **{payload.get('primary_metric', 'ndcg@10')}**. "
        "Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003). "
        "Names marked **(tuned)** used validation-selected hyperparameters "
        "(ADR-0005 / ADR-0006"
        + (" / ADR-0007" if payload.get("ranker") else "")
        + "); others are S2 YAML defaults."
    )
    lines.append("")
    sample = payload.get("eval_user_sample") or {}
    if sample.get("enabled"):
        lines.append(_sample_paragraph(payload, sample, split))
        lines.append("")
    ease_hp = (payload.get("hyperparameters") or {}).get("ease") or {}
    if ease_hp.get("max_items"):
        lines.append(
            "EASE is restricted to the top "
            f"{ease_hp.get('max_items')} items by train-interaction count "
            f"(n_items_before={ease_hp.get('n_items_before_restriction')}, "
            f"n_items_fit={ease_hp.get('n_items_fit')}; ties: smaller item id). "
            "Items outside that head are not scored."
        )
        lines.append("")
    header = ["model", *METRIC_COLS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    metrics = payload["metrics"]
    for model in sorted(metrics):
        row = [_model_label(model, tuned_flags)]
        for col in METRIC_COLS:
            row.append(_fmt(float(metrics[model][col])))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("95% bootstrap CIs (NDCG@10):")
    lines.append("")
    lines.append("| model | ndcg@10 CI |")
    lines.append("| --- | --- |")
    for model in sorted(metrics):
        cis = metrics[model].get("confidence_intervals", {})
        label = _model_label(model, tuned_flags)
        if metrics[model].get("seed_summary"):
            ss = metrics[model]["seed_summary"]["ndcg@10"]
            lines.append(
                f"| {label} | {_ndcg_seed_spread(ss)}; "
                f"primary-seed CI {_fmt_ci(cis, 'ndcg@10')} |"
            )
        else:
            lines.append(f"| {label} | {_fmt_ci(cis, 'ndcg@10')} |")
    lines.append("")
    scale = render_scale_ranking(payload)
    if scale:
        lines.append(scale)
        lines.append("")

    # Retrieval recall for models that report it.
    if any(all(c in metrics[m] for c in RETRIEVAL_COLS) for m in metrics):
        lines.append("#### Retrieval recall (candidate generation)")
        lines.append("")
        lines.append(
            "Recall@100 / Recall@200 for models that report them, "
            "evaluated at the same cutoffs. This is test-list recall, "
            "not the validation candidate-set recall in the ranker section."
        )
        lines.append("")
        lines.append("| model | recall@100 | recall@200 |")
        lines.append("| --- | --- | --- |")
        for model in sorted(metrics):
            m = metrics[model]
            if not all(c in m for c in RETRIEVAL_COLS):
                continue
            label = _model_label(model, tuned_flags)
            r100 = m["recall@100"]
            r200 = m["recall@200"]
            if m.get("seed_summary"):
                s100 = m["seed_summary"].get("recall@100") or {}
                s200 = m["seed_summary"].get("recall@200") or {}
                cell100 = _recall_seed_cell(float(r100), s100)
                cell200 = _recall_seed_cell(float(r200), s200)
            else:
                cell100 = _fmt(float(r100))
                cell200 = _fmt(float(r200))
            lines.append(f"| {label} | {cell100} | {cell200} |")
        lines.append("")

    # Two-tower seed table + gate.
    tt = payload.get("two_tower")
    if tt:
        lines.append("#### Two-tower seeds and gate (ADR-0006)")
        lines.append("")
        lines.append(
            f"Chosen hyperparams: `{tt.get('best_hyperparams')}` "
            f"(val NDCG@10={_fmt(float(tt['best_val_ndcg@10']))}; "
            f"early-stopping best_epoch={tt.get('best_epoch')}; "
            f"{tt.get('early_stopping')}). "
            f"Tuning log: [`{tt.get('tuning_json')}`]({tt.get('tuning_json')})."
        )
        lines.append("")
        lines.append("| seed | ndcg@10 | ndcg@10 CI | recall@100 | recall@200 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in tt.get("per_seed", []):
            ci = row.get("ndcg@10_ci") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["seed"]),
                        _fmt(float(row["ndcg@10"])),
                        f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}]" if ci else "",
                        _fmt(float(row["recall@100"]))
                        if row.get("recall@100") is not None
                        else "",
                        _fmt(float(row["recall@200"]))
                        if row.get("recall@200") is not None
                        else "",
                    ]
                )
                + " |"
            )
        across = tt.get("across_seeds", {}).get("ndcg@10", {})
        if across:
            lines.append("")
            lines.append(
                f"Across seeds: NDCG@10 mean={_fmt(across['mean'])}, "
                f"std={_fmt(across['std'])}, "
                f"min={_fmt(across['min'])}, max={_fmt(across['max'])}."
            )
        gate = tt.get("gate") or {}
        lines.append("")
        if gate.get("negative_result"):
            lines.append(
                "**Gate: negative result.** Two-tower does **not** beat both "
                "item–item bars on ml-1m-style NDCG@10 with CIs taken into "
                "account (see ADR-0006). The retriever or candidate set will "
                "be chosen in S4 by validation recall."
            )
        elif gate.get("beats_both_item_item_bars"):
            lines.append(
                "**Gate: win.** Two-tower beats both item–item default and tuned "
                "bars on NDCG@10 with CIs taken into account."
            )
        lines.append("")

    ranker_section = render_ranker(payload)
    if ranker_section:
        lines.append(ranker_section)

    # Segment breakdowns (NDCG@10 only).
    if any("segments" in metrics[m] for m in metrics):
        lines.append("#### Segment NDCG@10")
        lines.append("")
        lines.append(
            "User activity = train rating-count terciles (low/mid/high). "
            "Item head = top 20% of train items by popularity; tail = rest. "
            "Item-segment metrics restrict **relevant and recommended** items to "
            "the segment (users with no relevant items in-segment are excluded)."
        )
        lines.append("")
        lines.append(
            "| model | activity low | activity mid | activity high | item head | item tail |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for model in sorted(metrics):
            seg = metrics[model].get("segments") or {}
            act = seg.get("user_activity_terciles") or {}
            items = seg.get("item_head_tail") or {}

            def _cell(block: dict, key: str) -> str:
                if key not in block:
                    return ""
                val = block[key].get("ndcg@10")
                ci = block[key].get("confidence_intervals", {}).get("ndcg@10", {})
                if val is None:
                    return ""
                return f"{_fmt(float(val))} {_fmt_ci({'ndcg@10': ci}, 'ndcg@10')}"

            lines.append(
                "| "
                + " | ".join(
                    [
                        _model_label(model, tuned_flags),
                        _cell(act, "low"),
                        _cell(act, "mid"),
                        _cell(act, "high"),
                        _cell(items, "head"),
                        _cell(items, "tail"),
                    ]
                )
                + " |"
            )
        lines.append("")

    tuning = payload.get("tuning_summary")
    if tuning:
        bits = [
            f"Tuning log: [`{tuning.get('tuning_json')}`]({tuning.get('tuning_json')})."
        ]
        if tuning.get("als_best") is not None:
            bits.append(
                f"Chosen ALS={tuning.get('als_best')} "
                f"(val NDCG@10={_fmt(float(tuning['als_best_val_ndcg@10']))}); "
                f"item–item={tuning.get('item_item_cosine_best')} "
                f"(val NDCG@10="
                f"{_fmt(float(tuning['item_item_cosine_best_val_ndcg@10']))})."
            )
        if tuning.get("ease_best") is not None:
            bits.append(
                f"EASE={tuning.get('ease_best')} "
                f"(val NDCG@10={_fmt(float(tuning['ease_best_val_ndcg@10']))}); "
                f"RP3beta={tuning.get('rp3beta_best')} "
                f"(val NDCG@10={_fmt(float(tuning['rp3beta_best_val_ndcg@10']))})."
            )
        if tuning.get("two_tower_best") is not None:
            bits.append(
                f"Two-tower={tuning.get('two_tower_best')} "
                f"(val NDCG@10={_fmt(float(tuning['two_tower_best_val_ndcg@10']))}, "
                f"best_epoch={tuning.get('two_tower_best_epoch')}; "
                f"log [`{tuning.get('two_tower_tuning_json')}`]"
                f"({tuning.get('two_tower_tuning_json')}))."
            )
        lines.append(" ".join(bits))
        lines.append("")
    if payload.get("runtime_sec") is not None:
        lines.append(f"Pipeline runtime: {_fmt(float(payload['runtime_sec']))}s.")
        lines.append("")
    if payload.get("s3c_runtime_sec") is not None:
        lines.append(
            "EASE/RP3beta tune and test runtime "
            f"(same harness, not included in the pipeline runtime above): "
            f"{_fmt(float(payload['s3c_runtime_sec']))}s."
        )
        lines.append("")
    return "\n".join(lines)


def render_global_cutoff() -> str:
    path = RESULTS_DIR / "global_cutoff" / "ml-1m.json"
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("### Global-time-cutoff sanity check (`ml-1m`, secondary)")
    lines.append("")
    lines.append(
        f"From `results/global_cutoff/ml-1m.json`. "
        f"{payload.get('notes', '')} "
        f"Hyperparams: {payload.get('hyperparams_source', '')} "
        f"**Not re-tuned** for this protocol (`retuned={payload.get('retuned')}`)."
    )
    lines.append("")
    split = payload.get("split", {})
    surv = payload.get("surviving", {})
    lines.append(
        f"Cutoff: timestamp quantile={split.get('timestamp_quantile')} "
        f"(cutoff_timestamp={split.get('cutoff_timestamp')}). "
        f"Surviving: train users={surv.get('n_train_users')}, "
        f"train items={surv.get('n_train_items')}, "
        f"test interactions (after user filter)="
        f"{surv.get('n_test_interactions_after_user_filter')}, "
        f"eval users (warm relevant)={surv.get('n_eval_users')}."
    )
    lines.append("")
    lines.append(
        "| model | ndcg@10 | ndcg@10 CI | precision@10 | recall@10 | "
        "recall@100 | recall@200 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    metrics = payload["metrics"]
    for model in sorted(metrics):
        m = metrics[model]
        cis = m.get("confidence_intervals", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    _fmt(float(m["ndcg@10"])),
                    _fmt_ci(cis, "ndcg@10"),
                    _fmt(float(m["precision@10"])),
                    _fmt(float(m["recall@10"])),
                    _fmt(float(m["recall@100"])) if "recall@100" in m else "",
                    _fmt(float(m["recall@200"])) if "recall@200" in m else "",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_ci_cell(ci: dict | None) -> str:
    if not ci:
        return ""
    return f"[{_fmt(float(ci['low']))}, {_fmt(float(ci['high']))}]"


def render_demographics() -> str:
    """S4b block. Every number is read from results/demographics/ml-1m.json."""
    path = RESULTS_DIR / "demographics" / "ml-1m.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("### User demographic features (`ml-1m`, S4b)")
    lines.append("")
    lines.append(
        "From `results/demographics/ml-1m.json`. "
        f"Candidate set `{payload.get('candidate_set')}` "
        f"(K={payload.get('candidate_k')}; {payload.get('candidate_set_rule')}). "
        "Group-affinity statistics use fit-train only. "
        "NDCG@10 mean and std are over the three ranker seeds. "
        "The NDCG@10 CI, tail NDCG@10, activity slices, fairness slices, "
        "and paired difference are the primary seed."
    )
    lines.append("")
    repro = payload.get("reproduction") or {}
    if repro:
        lines.append(
            "S4 LambdaRank reference "
            f"({repro.get('s4_source')}): "
            f"NDCG@10 mean={_fmt(float(repro['s4_ndcg@10_mean']))}, "
            f"std={_fmt(float(repro['s4_ndcg@10_std']))}, "
            f"winner=`{repro.get('s4_winner')}`. "
            "This run's baseline: "
            f"mean={_fmt(float(repro['baseline_ndcg@10_mean']))}, "
            f"std={_fmt(float(repro['baseline_ndcg@10_std']))}, "
            f"abs diff={_fmt(float(repro['abs_diff_mean']))}, "
            f"matches at 4 decimals={repro.get('matches_s4_at_4_decimals')}, "
            f"candidate set matches={repro.get('candidate_set_matches_s4')}."
        )
        lines.append("")
    lines.append(
        "| variant | ndcg@10 mean | std | ndcg@10 CI | recall@10 | coverage@10 | tail ndcg@10 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    variants = payload.get("variants") or {}
    for name in ("baseline", "raw", "affinity", "both"):
        block = variants.get(name)
        if not block:
            continue
        summary = block.get("summary") or {}
        tail = summary.get("tail_ndcg@10")
        tail_ci = summary.get("tail_ndcg@10_ci") or {}
        tail_cell = ""
        if tail is not None:
            tail_cell = _fmt(float(tail))
            if tail_ci:
                tail_cell += f" {_fmt_ci_cell(tail_ci)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(float(summary["ndcg@10_mean"])),
                    _fmt(float(summary["ndcg@10_std"])),
                    _fmt_ci_cell(summary.get("ndcg@10_ci")),
                    _fmt(float(summary["recall@10_mean"])),
                    _fmt(float(summary["coverage@10_mean"])),
                    tail_cell,
                ]
            )
            + " |"
        )
    lines.append("")
    paired = payload.get("paired_bootstrap") or {}
    if paired:
        lines.append(
            "Paired bootstrap of NDCG@10 "
            f"({paired.get('comparison')}, seed {paired.get('seed')}, "
            f"n={paired.get('n_users')}): "
            f"mean={_fmt(float(paired['mean']))} "
            f"[{_fmt(float(paired['low']))}, {_fmt(float(paired['high']))}], "
            f"excludes_zero={paired.get('excludes_zero')}."
        )
        lines.append("")
    activity = payload.get("activity_segments") or {}
    if activity:
        lines.append(
            "Primary-seed NDCG@10 by the existing user-activity terciles "
            "(train rating count)."
        )
        lines.append("")
        lines.append("| variant | activity low | activity mid | activity high |")
        lines.append("| --- | --- | --- | --- |")
        for name in ("baseline", "raw", "affinity", "both"):
            block = activity.get(name) or {}
            cells = []
            for label in ("low", "mid", "high"):
                cell = block.get(label) or {}
                if cell.get("ndcg@10") is None:
                    cells.append("")
                else:
                    cells.append(
                        f"{_fmt(float(cell['ndcg@10']))} "
                        f"{_fmt_ci_cell(cell.get('ndcg@10_ci'))}"
                    )
            lines.append("| " + " | ".join([name, *cells]) + " |")
        lines.append("")
    cold = payload.get("cold_start") or {}
    cold_rows = cold.get("rows") or []
    if cold_rows:
        lines.append(
            "Simulated cold start: earliest N full-train ratings as the query "
            "profile. Test targets are unchanged. "
            "`most_popular` and `group_most_popular` do not use that profile "
            f"(matrices: {cold.get('most_popular_matrix')}; "
            f"{cold.get('group_popularity_matrix')})."
        )
        lines.append("")
        lines.append(
            "| N | model | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in cold_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["n"]),
                        str(row["model"]),
                        _fmt(float(row["ndcg@10"])),
                        _fmt_ci_cell(row.get("ndcg@10_ci")),
                        _fmt(float(row["recall@10"])),
                        _fmt(float(row["coverage@10"])),
                    ]
                )
                + " |"
            )
        lines.append("")
    fairness = payload.get("fairness") or {}
    by_gender = fairness.get("by_gender") or {}
    if by_gender:
        lines.append(
            "Primary-seed NDCG@10 by gender and by age bucket, baseline ranker "
            "versus +both. Delta is both minus baseline."
        )
        lines.append("")
        lines.append(
            "| group | n | baseline ndcg@10 | both ndcg@10 | delta | delta CI |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for label in sorted(by_gender):
            block = by_gender[label]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"gender {label}",
                        str(block["n_users"]),
                        _fmt(float(block["baseline_ndcg@10"])),
                        _fmt(float(block["both_ndcg@10"])),
                        _fmt(float(block["delta_ndcg@10"])),
                        _fmt_ci_cell(block.get("delta_ndcg@10_ci")),
                    ]
                )
                + " |"
            )
        by_age = fairness.get("by_age") or {}
        for label in sorted(by_age, key=lambda value: int(value) if str(value).isdigit() else 999):
            block = by_age[label]
            pretty = block.get("label") or label
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"age {label} ({pretty})",
                        str(block["n_users"]),
                        _fmt(float(block["baseline_ndcg@10"])),
                        _fmt(float(block["both_ndcg@10"])),
                        _fmt(float(block["delta_ndcg@10"])),
                        _fmt_ci_cell(block.get("delta_ndcg@10_ci")),
                    ]
                )
                + " |"
            )
        lines.append("")
    gains = payload.get("demographic_feature_importance") or []
    if gains:
        lines.append(
            "Demographic feature gains from the +both primary-seed refit booster. "
            "Rank is among every feature of that booster (1 = highest gain)."
        )
        lines.append("")
        lines.append("| feature | gain | rank |")
        lines.append("| --- | --- | --- |")
        for row in gains:
            lines.append(
                f"| {row['feature']} | {_fmt(float(row['gain']))} | {row['rank']} |"
            )
        lines.append("")
    decision = payload.get("decision") or {}
    if decision:
        if decision.get("keep_as_default"):
            lines.append(
                "**Decision: keep demographic features as the ranker default.** "
                f"+both mean NDCG@10={_fmt(float(decision['both_mean_ndcg@10']))} "
                f"versus baseline {_fmt(float(decision['baseline_mean_ndcg@10']))}. "
                "The paired CI low is above 0."
            )
        else:
            lines.append(
                "**Decision: negative result for adoption.** "
                f"+both mean NDCG@10={_fmt(float(decision['both_mean_ndcg@10']))}, "
                f"baseline {_fmt(float(decision['baseline_mean_ndcg@10']))}, "
                f"mean_beats_baseline={decision.get('mean_beats_baseline')}, "
                f"paired CI [{_fmt(float(decision['paired_ci_low']))}, "
                f"{_fmt(float(decision['paired_ci_high']))}], "
                f"excludes_zero={decision.get('paired_ci_excludes_zero')}. "
                "The S4 feature set stays the default "
                "(`models.ranker.demographics: off`)."
            )
        lines.append("")
    if payload.get("runtime_sec") is not None:
        lines.append(f"Demographic experiment runtime: {_fmt(float(payload['runtime_sec']))}s.")
        lines.append("")
    return "\n".join(lines)


def _cold_metric_row(n: int, name: str, metrics: dict, n_users: int) -> str:
    return (
        "| "
        + " | ".join(
            [
                str(n),
                name,
                _fmt(float(metrics["ndcg@10"])),
                _fmt_ci_cell(metrics.get("ndcg@10_ci")),
                _fmt(float(metrics["recall@10"])),
                _fmt(float(metrics["coverage@10"])),
                str(n_users),
            ]
        )
        + " |"
    )


def _beats_popularity_sentence(block: dict) -> str:
    n = int(block["n"])
    method = str(block.get("served_method"))
    paired = block.get("paired_vs_most_popular") or {}
    mean = float(paired["mean"])
    low = float(paired["low"])
    high = float(paired["high"])
    excludes = "excludes 0" if paired.get("excludes_zero") else "includes 0"
    gap = f"{_fmt(mean)} [{_fmt(low)}, {_fmt(high)}] ({excludes})"
    if method == "most_popular":
        return (
            f"N={n}: the served method is most-popular, so it does not beat "
            f"popularity. Serving popularity at this profile size is the "
            f"validation choice. Difference {gap}."
        )
    if paired.get("beats_baseline"):
        return (
            f"N={n}: the served method `{method}` beats most-popular. "
            f"Difference {gap}."
        )
    if mean > 0.0 and not paired.get("excludes_zero"):
        return (
            f"N={n}: `{method}` is above most-popular by {_fmt(mean)}, and the "
            f"interval [{_fmt(low)}, {_fmt(high)}] includes 0."
        )
    return f"N={n}: `{method}` does not beat most-popular. Difference {gap}."


def _coverage_tradeoff(by_n: list) -> str:
    """Served Coverage@10 against popularity and the fold-in lists."""

    def _cov(block: dict, name: str) -> str:
        return _fmt(float(block["models"][name]["coverage@10"]))

    ns = ", ".join(str(int(block["n"])) for block in by_n)
    served = ", ".join(_cov(block, "served") for block in by_n)
    popular = ", ".join(_cov(block, "most_popular") for block in by_n)
    item_item = ", ".join(_cov(block, "item_item_fold_in") for block in by_n)
    ease = ", ".join(_cov(block, "ease_fold_in") for block in by_n)
    return (
        f"Coverage trade-off: the served ranker's Coverage@10 is {served} at "
        f"N={ns}. Most-popular is {popular}. Item-item fold-in is {item_item} "
        f"and EASE fold-in is {ease}. The served list stays close to popularity "
        "and far below those fold-in methods, so it leans on popular titles."
    )


def _sensitivity_n10_note(block: dict) -> str:
    """Point comparison at N=10 on the harness tail, from the JSON."""
    if int(block["n"]) != 10:
        return ""
    models = block.get("models") or {}
    served = float(models["served"]["ndcg@10"])
    item_item = float(models["item_item_fold_in"]["ndcg@10"])
    tower = float(models["history_two_tower"]["ndcg@10"])
    if served >= item_item or served >= tower:
        return ""
    return (
        f" The served ranker ({_fmt(served)}) is below item-item fold-in "
        f"({_fmt(item_item)}) and the history two-tower ({_fmt(tower)})."
    )


def _render_cold_start_round2(payload: dict) -> str:
    """Panel once a round-2 measurement is in the JSON. Round 1 stays visible."""
    lines: list[str] = []
    lines.append("### New-user cold start (`ml-1m`, S5c)")
    lines.append("")
    lines.append(str(payload.get("difference_from_s4b", "")).strip())
    lines.append("")
    lines.append(
        "The first pipeline is a negative result. It was trained on long "
        "histories and re-ranked one candidate source. Those numbers are copied "
        "below as `pipeline_v1` and were not recomputed. Round 2 was redesigned "
        "after those results on the same 604 held-out users: dropout p=0.0 and "
        "p=0.1, the short-profile ranker, and the per-N rule. The round-2 "
        "held-out numbers are not a fully fresh test. Every choice was frozen "
        "on validation before that second score."
    )
    lines.append("")
    round2 = payload.get("round2") or {}
    representation = payload.get("representation") or {}
    serving = round2.get("serving_by_n") or {}
    serving_text = ", ".join(
        f"N={n} `{serving[n]}`" for n in ("1", "3", "5", "10") if n in serving
    )
    lines.append(
        f"Round 2 representation: `{representation.get('winner')}` "
        f"(validation NDCG@10 {_fmt(float(representation.get('val_ndcg@10', 0.0)))}). "
        f"Dropout grid edge: best p={representation.get('dropout_best_p')} "
        f"(at edge: {representation.get('dropout_best_at_edge')}); "
        f"selected model at edge: {representation.get('selected_at_grid_edge')}."
    )
    lines.append("")
    lines.append(
        f"Cold-start ranker K={round2.get('candidate_k')} "
        f"({round2.get('best_iteration')} trees, demographics off). "
        f"Served method by profile size: {serving_text}."
    )
    lines.append("")
    why = round2.get("popularity_vs_known_users") or payload.get("popularity_vs_known_users") or ""
    lines.append(str(why))
    lines.append("")
    lines.append(
        "Known users below are copied from `results/ml-1m.json`. "
        "They were in the training matrix. The new-user rows were not."
    )
    lines.append("")
    lines.append("| known-user model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 |")
    lines.append("| --- | --- | --- | --- | --- |")
    known = (payload.get("known_users") or {}).get("models") or {}
    for name in ("most_popular", "item_item_cosine", "two_tower", "lambdarank"):
        block = known.get(name) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(float(block["ndcg@10"])),
                    _fmt_ci_cell(block.get("ndcg@10_ci")),
                    _fmt(float(block["recall@10"])),
                    _fmt(float(block["coverage@10"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Primary protocol, fixed before this run: the model sees the first N "
        "chronological ratings, and the targets are all later ratings "
        "(relevance at least 4). Coverage has no interval."
    )
    lines.append("")
    lines.append(
        "| N | model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 | eval users |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    order = (
        "pipeline_v1",
        "cold_start_ranker",
        "served",
        "most_popular",
        "item_item_fold_in",
        "history_two_tower",
        "ease_fold_in",
    )
    for block in payload.get("by_n") or []:
        n = int(block["n"])
        models = block.get("models") or {}
        n_users = int(block.get("n_eval_users", 0))
        for name in order:
            metrics = models.get(name) or {}
            lines.append(_cold_metric_row(n, name, metrics, n_users))
    lines.append("")
    primary = list(payload.get("by_n") or [])
    if primary:
        lines.append(_coverage_tradeoff(primary))
        lines.append("")
    for block in primary:
        lines.append(_beats_popularity_sentence(block))
        paired = block.get("paired_bootstrap") or {}
        # Same gap as the popularity sentence when most-popular is the best simple baseline.
        if paired.get("baseline") == "most_popular":
            continue
        n = int(block["n"])
        excludes = "excludes 0" if paired.get("excludes_zero") else "includes 0"
        lines.append(
            f"N={n}: served minus `{paired.get('baseline')}` "
            f"NDCG@10 {_fmt(float(paired['mean']))} "
            f"[{_fmt(float(paired['low']))}, {_fmt(float(paired['high']))}] "
            f"({excludes})."
        )
    lines.append("")
    round1 = payload.get("round1") or {}
    lines.append(
        "Round 1 paired gaps (pipeline minus the best simple baseline on that "
        "table) stay the recorded miss:"
    )
    lines.append("")
    for block in round1.get("by_n") or []:
        paired = block.get("paired_bootstrap") or {}
        n = int(block["n"])
        excludes = "excludes 0" if paired.get("excludes_zero") else "includes 0"
        lines.append(
            f"N={n}: pipeline_v1 minus `{paired.get('baseline')}` "
            f"NDCG@10 {_fmt(float(paired['mean']))} "
            f"[{_fmt(float(paired['low']))}, {_fmt(float(paired['high']))}] "
            f"({excludes})."
        )
    lines.append("")
    sens = payload.get("sensitivity_last_20_percent") or {}
    lines.append(
        "Sensitivity view, added after the first results. Targets are only each "
        "held-out user's last 20% of ratings (the harness tail), and only where "
        "that tail is after the first N. The serving rule was not re-chosen here. "
        "Pipeline v1 is not scored on this target."
    )
    lines.append("")
    lines.append(
        "| N | model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 | eval users |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    sens_order = (
        "cold_start_ranker",
        "served",
        "most_popular",
        "item_item_fold_in",
        "history_two_tower",
        "ease_fold_in",
    )
    for block in sens.get("by_n") or []:
        n = int(block["n"])
        models = block.get("models") or {}
        n_users = int(block.get("n_eval_users", 0))
        for name in sens_order:
            metrics = models.get(name) or {}
            lines.append(_cold_metric_row(n, name, metrics, n_users))
    lines.append("")
    for block in sens.get("by_n") or []:
        lines.append(
            "Sensitivity " + _beats_popularity_sentence(block) + _sensitivity_n10_note(block)
        )
    lines.append("")
    serving_latency = payload.get("serving") or {}
    if serving_latency.get("latency_sec") is not None:
        method = serving_latency.get("method")
        extra = f" Method `{method}`." if method else ""
        lines.append(
            "Warmed new-user top-10 latency "
            f"(five popular titles rated 5): {_fmt(float(serving_latency['latency_sec']))}s."
            f"{extra}"
        )
        lines.append("")
    if payload.get("runtime_sec") is not None:
        lines.append(f"Cold-start experiment runtime: {_fmt(float(payload['runtime_sec']))}s.")
        lines.append("")
    return "\n".join(lines)


def render_cold_start() -> str:
    """S5c panel. Empty when results/cold-start/ml-1m.json is absent.

    An empty return leaves the earlier README panels byte-identical.
    A file without ``round2`` keeps the original panel text.
    """
    path = RESULTS_DIR / "cold-start" / "ml-1m.json"
    if not path.is_file():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "round2" in payload:
        return _render_cold_start_round2(payload)
    lines: list[str] = []
    lines.append("### New-user cold start (`ml-1m`, S5c)")
    lines.append("")
    lines.append(str(payload.get("difference_from_s4b", "")).strip())
    lines.append("")
    representation = payload.get("representation") or {}
    candidates = payload.get("candidates") or {}
    ranker = payload.get("ranker") or {}
    lines.append(
        "Choices were fixed on the validation window of the users who stayed "
        "in training, before the held-out ratings were scored."
    )
    lines.append("")
    lines.append(
        f"User representation: `{representation.get('winner')}` "
        f"(validation NDCG@10 {_fmt(float(representation.get('val_ndcg@10', 0.0)))}). "
        f"Dropout grid edge: best p={representation.get('dropout_best_p')} "
        f"(at edge: {representation.get('dropout_best_at_edge')}); "
        f"selected model at edge: {representation.get('selected_at_grid_edge')}."
    )
    lines.append("")
    lines.append(
        f"Candidate source: `{candidates.get('winner')}` "
        f"(K={candidates.get('k')}). "
        f"Ranker: `{ranker.get('winner')}` "
        f"({ranker.get('best_iteration')} trees)."
    )
    lines.append("")
    lines.append(
        "Known users below are copied from `results/ml-1m.json`. "
        "They were in the training matrix. The new-user rows were not."
    )
    lines.append("")
    lines.append("| known-user model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 |")
    lines.append("| --- | --- | --- | --- | --- |")
    known = (payload.get("known_users") or {}).get("models") or {}
    for name in ("most_popular", "item_item_cosine", "two_tower", "lambdarank"):
        block = known.get(name) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(float(block["ndcg@10"])),
                    _fmt_ci_cell(block.get("ndcg@10_ci")),
                    _fmt(float(block["recall@10"])),
                    _fmt(float(block["coverage@10"])),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "For a brand-new user the model sees only the first N chronological "
        "ratings. NDCG@10, Recall@10, and Coverage@10 are on the later ratings "
        "(relevance at least 4). Coverage has no interval."
    )
    lines.append("")
    lines.append(
        "| N | model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 | eval users |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    order = (
        "most_popular",
        "item_item_fold_in",
        "ease_fold_in",
        "history_two_tower",
        "pipeline",
    )
    for block in payload.get("by_n") or []:
        n = int(block["n"])
        models = block.get("models") or {}
        n_users = int(block.get("n_eval_users", 0))
        for name in order:
            metrics = models.get(name) or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(n),
                        name,
                        _fmt(float(metrics["ndcg@10"])),
                        _fmt_ci_cell(metrics.get("ndcg@10_ci")),
                        _fmt(float(metrics["recall@10"])),
                        _fmt(float(metrics["coverage@10"])),
                        str(n_users),
                    ]
                )
                + " |"
            )
    lines.append("")
    for block in payload.get("by_n") or []:
        paired = block.get("paired_bootstrap") or {}
        n = int(block["n"])
        excludes = "excludes 0" if paired.get("excludes_zero") else "includes 0"
        lines.append(
            f"N={n}: pipeline minus `{paired.get('baseline')}` "
            f"NDCG@10 {_fmt(float(paired['mean']))} "
            f"[{_fmt(float(paired['low']))}, {_fmt(float(paired['high']))}] "
            f"({excludes})."
        )
    lines.append("")
    serving = payload.get("serving") or {}
    if serving.get("latency_sec") is not None:
        lines.append(
            "Warmed new-user top-10 latency "
            f"(five popular titles rated 5): {_fmt(float(serving['latency_sec']))}s."
        )
        lines.append("")
    if payload.get("runtime_sec") is not None:
        lines.append(f"Cold-start experiment runtime: {_fmt(float(payload['runtime_sec']))}s.")
        lines.append("")
    return "\n".join(lines)


def _v2_edges(edges: list[dict]) -> str:
    if not edges:
        return "none"
    parts = []
    for edge in edges:
        parts.append(
            f"{edge['parameter']}={edge['value']} ({edge['edge']} of {edge['grid_values']})"
        )
    return "; ".join(parts)


def _v2_pair_sentence(label: str, paired: dict) -> str:
    excludes = "excludes 0" if paired.get("excludes_zero") else "includes 0"
    return (
        f"{label} {_fmt(float(paired['mean']))} "
        f"[{_fmt(float(paired['low']))}, {_fmt(float(paired['high']))}] ({excludes})"
    )


def _answer_full_softmax(payload: dict) -> str:
    tuning = payload.get("tuning") or {}
    best = (tuning.get("best_by_loss") or {}).get("full_softmax") or {}
    ref_val = float(tuning.get("reference_val_ndcg@10"))
    val = best.get("val_ndcg@10")
    model = (payload.get("models") or {}).get("full_softmax")
    if val is None:
        return "Full softmax has no validation score in this file."
    val_f = float(val)
    if model is None:
        relation = "above" if val_f > ref_val else "not above"
        return (
            f"No test score. Full softmax validation NDCG@10 was {_fmt(val_f)}, "
            f"{relation} the reference {_fmt(ref_val)}. "
            f"The validation winner was `{tuning.get('selected_loss')}`."
        )
    paired = (payload.get("paired_vs_reference") or {}).get("full_softmax", {}).get("ndcg@10") or {}
    point = float(model["metrics"]["ndcg@10"])
    ref_point = float(payload["models"]["in_batch"]["metrics"]["ndcg@10"])
    if float(paired.get("mean", 0.0)) > 0.0 and paired.get("excludes_zero"):
        verdict = "Yes"
    elif float(paired.get("mean", 0.0)) > 0.0:
        verdict = "No clear gain"
    else:
        verdict = "No"
    return (
        f"{verdict}. Full softmax test NDCG@10 is {_fmt(point)} against the "
        f"reference rerun {_fmt(ref_point)}. "
        + _v2_pair_sentence("Paired difference", paired)
        + f". Validation was {_fmt(val_f)} against reference {_fmt(ref_val)}."
    )


def _answer_tail(payload: dict) -> str:
    """Compare the scored new tower's tail with the reference and with item-item."""
    published = payload.get("reference_published") or {}
    item_tail = published.get("item_item_tail_ndcg@10")
    ref_tail = published.get("tail_ndcg@10")
    selected = str((payload.get("tuning") or {}).get("selected_loss"))
    models = payload.get("models") or {}
    name = "full_softmax" if "full_softmax" in models else selected
    model = models.get(name)
    if model is None or item_tail is None or ref_tail is None:
        return "Tail NDCG@10 was not measured for a new loss."
    tail = _segment_value_metrics(model["metrics"], "tail")
    head = _segment_value_metrics(model["metrics"], "head")
    paired = ((payload.get("paired_vs_reference") or {}).get(name) or {}).get("tail_ndcg@10") or {}
    if tail is None or head is None:
        return "Tail NDCG@10 is missing from this file."
    if float(tail) >= float(item_tail) and float(tail) > float(ref_tail):
        verdict = "Yes"
    elif float(tail) > float(ref_tail):
        verdict = "It beats the reference two-tower and is still below item-item"
    else:
        verdict = "No"
    text = (
        f"{verdict}. `{name}` tail NDCG@10 is {_fmt(float(tail))} "
        f"(head {_fmt(float(head))}). Published reference tail is {_fmt(float(ref_tail))}; "
        f"item-item tail is {_fmt(float(item_tail))}."
    )
    if paired:
        text += (
            " "
            + _v2_pair_sentence("Paired tail difference versus the reference rerun", paired)
            + "."
        )
    return text


def _segment_value_metrics(metrics: dict, which: str) -> float | None:
    cell = ((metrics.get("segments") or {}).get("item_head_tail") or {}).get(which) or {}
    value = cell.get("ndcg@10")
    if value is None:
        return None
    return float(value)


def _answer_ranker(payload: dict) -> str:
    tuning = payload.get("tuning") or {}
    ranker = payload.get("ranker")
    if not ranker:
        return (
            "No. The best new two-tower did not beat the reference on validation "
            f"({_fmt(float(tuning.get('selected_val_ndcg@10')))} versus "
            f"{_fmt(float(tuning.get('reference_val_ndcg@10')))}), so LambdaRank was not re-run."
        )
    new = ranker["new"]
    no_ranker = float(new["no_ranker_metrics"]["ndcg@10"])
    new_point = float(new["metrics"]["ndcg@10"])
    lift = ranker["paired_vs_no_ranker"]["ndcg@10"]
    versus = ranker["paired_vs_reference_ranker"]["ndcg@10"]
    published = payload["reference_published"]
    if new_point > no_ranker and lift.get("excludes_zero") and float(lift["mean"]) > 0:
        over_list = "Yes, it adds lift over its candidate list"
    elif new_point > no_ranker:
        over_list = (
            "The point estimate is above the candidate list and the paired interval includes 0"
        )
    else:
        over_list = "No lift over its candidate list"
    if float(versus["mean"]) > 0 and versus.get("excludes_zero"):
        over_old = "and it beats the reference-tower ranker"
    elif float(versus["mean"]) > 0:
        over_old = "and the gain versus the reference-tower ranker includes 0"
    else:
        over_old = "and it does not beat the reference-tower ranker"
    return (
        f"{over_list} {over_old}. New ranker NDCG@10 {_fmt(new_point)} versus "
        f"`no_ranker` {_fmt(no_ranker)}. "
        + _v2_pair_sentence("Paired versus no_ranker", lift)
        + ". "
        + _v2_pair_sentence("Paired versus the reference-tower ranker", versus)
        + f". Published LambdaRank in `{published.get('lambdarank_source')}` is "
        f"{_fmt(float(published['lambdarank_ndcg@10']))} "
        f"(demographics {published.get('lambdarank_demographics')})."
    )


def render_two_tower_v2() -> str:
    """S3e panel. Empty when neither results file exists, so earlier panels stay put."""
    paths = [
        RESULTS_DIR / "two-tower-v2" / "ml-1m.json",
        RESULTS_DIR / "two-tower-v2" / "ml-32m.json",
    ]
    present = [path for path in paths if path.is_file()]
    if not present:
        return ""
    lines: list[str] = []
    lines.append("### Full-softmax two-tower (S3e)")
    lines.append("")
    lines.append(
        "From `results/two-tower-v2/`. The reference loss is the in-batch "
        "sampled softmax. Temperature, learning rate, and embedding dim were "
        "chosen on validation only. Paired intervals are the primary seed, "
        "candidate minus reference."
    )
    lines.append("")
    for path in present:
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        lines.append(f"#### `{dataset}`")
        lines.append("")
        published = payload["reference_published"]
        lines.append(
            f"Published reference two-tower (`{published['source']}`): "
            f"NDCG@10 {_fmt(float(published['ndcg@10']))} "
            f"{_fmt_ci_cell(published.get('ndcg@10_ci'))}, "
            f"Recall@10 {_fmt(float(published['recall@10']))}, "
            f"Recall@100 {_fmt(float(published['recall@100']))}, "
            f"Recall@200 {_fmt(float(published['recall@200']))}, "
            f"Coverage@10 {_fmt(float(published['coverage@10']))}, "
            f"head {_fmt(float(published['head_ndcg@10']))}, "
            f"tail {_fmt(float(published['tail_ndcg@10']))}. "
            f"Validation NDCG@10 {_fmt(float(published['val_ndcg@10']))}."
        )
        lines.append("")
        tuning = payload["tuning"]
        lines.append("| loss | val NDCG@10 | best epoch | grid edges |")
        lines.append("| --- | --- | --- | --- |")
        lines.append(
            "| reference in_batch | "
            f"{_fmt(float(tuning['reference_val_ndcg@10']))} | "
            f"{published['best_epoch']} | published config |"
        )
        for name, block in sorted((tuning.get("best_by_loss") or {}).items()):
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        _fmt(float(block["val_ndcg@10"])),
                        str(block["best_epoch"]),
                        _v2_edges(block.get("grid_edges") or []),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append(
            f"Validation winner: `{tuning['selected_loss']}` "
            f"({_fmt(float(tuning['selected_val_ndcg@10']))}). "
            f"Beats the reference on validation: {tuning['beats_reference_validation']}."
        )
        lines.append("")
        lines.append(
            "| model | NDCG@10 | 95% CI | Recall@10 | Recall@100 | Recall@200 | "
            "Coverage@10 | head | tail |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for name in ("in_batch", "full_softmax", "sampled_softmax"):
            model = (payload.get("models") or {}).get(name)
            if not model:
                continue
            metrics = model["metrics"]
            ci = (metrics.get("confidence_intervals") or {}).get("ndcg@10") or {}
            n_seeds = len(model.get("per_seed") or [])
            label = name if n_seeds != 1 else f"{name} (1 seed)"
            if n_seeds > 1:
                label = f"{name} ({n_seeds}-seed mean)"
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        _fmt(float(metrics["ndcg@10"])),
                        _fmt_ci_cell(ci),
                        _fmt(float(metrics["recall@10"])),
                        _fmt(float(metrics["recall@100"])),
                        _fmt(float(metrics["recall@200"])),
                        _fmt(float(metrics["coverage@10"])),
                        _fmt(float(_segment_value_metrics(metrics, "head"))),
                        _fmt(float(_segment_value_metrics(metrics, "tail"))),
                    ]
                )
                + " |"
            )
        lines.append("")
        for name, paired in sorted((payload.get("paired_vs_reference") or {}).items()):
            lines.append(
                _v2_pair_sentence(f"Paired NDCG@10, {name} minus reference", paired["ndcg@10"])
                + "."
            )
            if paired.get("head_ndcg@10"):
                lines.append(
                    _v2_pair_sentence(f"Paired head NDCG@10, {name}", paired["head_ndcg@10"]) + "."
                )
            if paired.get("tail_ndcg@10"):
                lines.append(
                    _v2_pair_sentence(f"Paired tail NDCG@10, {name}", paired["tail_ndcg@10"]) + "."
                )
        lines.append("")
        lines.append(f"Did full softmax help? {_answer_full_softmax(payload)}")
        lines.append("")
        lines.append(f"Does it fix the tail weakness? {_answer_tail(payload)}")
        lines.append("")
        lines.append(f"Does the ranker gain? {_answer_ranker(payload)}")
        lines.append("")
        compute = payload.get("compute") or {}
        if compute:
            lines.append(
                f"Compute: tune {_fmt(float(compute['tune_wall_sec']))}s, "
                f"test training {_fmt(float(compute['test_train_sec']))}s, "
                f"ranker "
                + (
                    f"{_fmt(float(compute['ranker_wall_sec']))}s"
                    if compute.get("ranker_wall_sec") is not None
                    else "not run"
                )
                + f", seeds {compute.get('seeds')}."
            )
            lines.append("")
    return "\n".join(lines)


def build_section(results: list[tuple[str, dict]]) -> str:
    parts = [BEGIN, ""]
    for filename, payload in results:
        parts.append(render_headline(filename, payload))
    gc = render_global_cutoff()
    if gc:
        parts.append(gc)
    demo = render_demographics()
    if demo:
        parts.append(demo)
    cold = render_cold_start()
    if cold:
        parts.append(cold)
    v2 = render_two_tower_v2()
    if v2:
        parts.append(v2)
    parts.append(END)
    return "\n".join(parts) + "\n"


def replace_section(readme_text: str, section: str) -> str:
    if BEGIN not in readme_text or END not in readme_text:
        raise SystemExit(f"README.md must contain markers {BEGIN!r} and {END!r}")
    before, rest = readme_text.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    return before + section.rstrip("\n") + "\n" + after.lstrip("\n")


def main() -> int:
    results = load_headline_results()
    section = build_section(results)
    readme = README.read_text(encoding="utf-8")
    updated = replace_section(readme, section)
    README.write_text(updated, encoding="utf-8")
    print(f"Updated {README} from {len(results)} results file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
