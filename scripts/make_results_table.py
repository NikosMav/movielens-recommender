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


def _fmt_ci(cis: dict, key: str) -> str:
    bounds = cis.get(key)
    if not bounds:
        return ""
    return f"[{_fmt(bounds['low'])}, {_fmt(bounds['high'])}]"


def _model_label(name: str, tuned_flags: dict | None) -> str:
    is_tuned = bool(tuned_flags and tuned_flags.get(name)) or name.endswith("_tuned")
    if name.endswith("_tuned"):
        return name
    if name == "two_tower":
        return "two_tower (tuned)" if is_tuned else "two_tower"
    if is_tuned:
        return f"{name} (tuned)"
    return name


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
        "(ADR-0005 / ADR-0006); others are S2 YAML defaults."
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
        if model == "two_tower" and metrics[model].get("seed_summary"):
            ss = metrics[model]["seed_summary"]["ndcg@10"]
            lines.append(
                f"| {label} | mean±std over seeds "
                f"{_fmt(ss['mean'])}±{_fmt(ss['std'])}; "
                f"primary-seed CI {_fmt_ci(cis, 'ndcg@10')} |"
            )
        else:
            lines.append(f"| {label} | {_fmt_ci(cis, 'ndcg@10')} |")
    lines.append("")

    # Retrieval recall for models that report it.
    if any(all(c in metrics[m] for c in RETRIEVAL_COLS) for m in metrics):
        lines.append("#### Retrieval recall (candidate generation)")
        lines.append("")
        lines.append(
            "Recall@100 / Recall@200 for models that report them "
            "(two-tower and baselines evaluated at the same cutoffs)."
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
            if model == "two_tower" and m.get("seed_summary"):
                s100 = m["seed_summary"].get("recall@100", {})
                s200 = m["seed_summary"].get("recall@200", {})
                cell100 = (
                    f"{_fmt(float(r100))}"
                    + (f" ±{_fmt(s100['std'])}" if s100 else "")
                )
                cell200 = (
                    f"{_fmt(float(r200))}"
                    + (f" ±{_fmt(s200['std'])}" if s200 else "")
                )
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
                "account (see ADR-0006). S4 should use item–item cosine as the "
                "retriever unless a later stage reverses this."
            )
        elif gate.get("beats_both_item_item_bars"):
            lines.append(
                "**Gate: win.** Two-tower beats both item–item default and tuned "
                "bars on NDCG@10 with CIs taken into account."
            )
        lines.append("")

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


def build_section(results: list[tuple[str, dict]]) -> str:
    parts = [BEGIN, ""]
    for filename, payload in results:
        parts.append(render_headline(filename, payload))
    gc = render_global_cutoff()
    if gc:
        parts.append(gc)
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
