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


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_ci(cis: dict, key: str) -> str:
    bounds = cis.get(key)
    if not bounds:
        return ""
    return f"[{_fmt(bounds['low'])}, {_fmt(bounds['high'])}]"


def _model_label(name: str, tuned_flags: dict | None) -> str:
    if tuned_flags and tuned_flags.get(name):
        return f"{name} (tuned)"
    if name.endswith("_tuned"):
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
        f"bootstrap={boot.get('n_bootstrap')} @ alpha={boot.get('alpha')}. "
        f"Primary metric: **{payload.get('primary_metric', 'ndcg@10')}**. "
        "Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003). "
        "Names marked **(tuned)** used validation-selected hyperparameters "
        "(ADR-0005); others are S2 YAML defaults."
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
        lines.append(
            f"| {_model_label(model, tuned_flags)} | {_fmt_ci(cis, 'ndcg@10')} |"
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
        lines.append(
            f"Tuning log: [`{tuning.get('tuning_json')}`]"
            f"({tuning.get('tuning_json')}). "
            f"Chosen ALS={tuning.get('als_best')} "
            f"(val NDCG@10={_fmt(float(tuning['als_best_val_ndcg@10']))}); "
            f"item–item={tuning.get('item_item_cosine_best')} "
            f"(val NDCG@10={_fmt(float(tuning['item_item_cosine_best_val_ndcg@10']))})."
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
    lines.append("| model | ndcg@10 | ndcg@10 CI | precision@10 | recall@10 |")
    lines.append("| --- | --- | --- | --- | --- |")
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
