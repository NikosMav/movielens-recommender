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


def load_results() -> list[tuple[str, dict]]:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {RESULTS_DIR}")
    out = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out.append((path.name, payload))
    return out


def render_table(filename: str, payload: dict) -> str:
    lines: list[str] = []
    dataset = payload["dataset"]
    version = payload.get("dataset_version", dataset)
    lines.append(f"### `{dataset}` (from `results/{filename}`)")
    lines.append("")
    lines.append(f"Pinned version: `{version}`.")
    lines.append("")
    split = payload.get("split", {})
    cfg = split.get("config", {})
    boot = payload.get("bootstrap", {})
    lines.append(
        f"Split: min_ratings={cfg.get('min_ratings')}, "
        f"test_fraction={cfg.get('test_fraction')}, "
        f"relevance_threshold={payload.get('relevance_threshold')}, "
        f"seed={payload.get('seed')}, ks={payload.get('ks')}, "
        f"bootstrap={boot.get('n_bootstrap')} @ alpha={boot.get('alpha')}. "
        f"Primary metric: **{payload.get('primary_metric', 'ndcg@10')}**."
    )
    lines.append("")
    header = ["model", *METRIC_COLS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    metrics = payload["metrics"]
    for model in sorted(metrics):
        row = [model]
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
        lines.append(f"| {model} | {_fmt_ci(cis, 'ndcg@10')} |")
    lines.append("")
    return "\n".join(lines)


def build_section(results: list[tuple[str, dict]]) -> str:
    parts = [BEGIN, ""]
    for filename, payload in results:
        parts.append(render_table(filename, payload))
    parts.append(END)
    return "\n".join(parts) + "\n"


def replace_section(readme_text: str, section: str) -> str:
    if BEGIN not in readme_text or END not in readme_text:
        raise SystemExit(f"README.md must contain markers {BEGIN!r} and {END!r}")
    before, rest = readme_text.split(BEGIN, 1)
    _, after = rest.split(END, 1)
    return before + section.rstrip("\n") + "\n" + after.lstrip("\n")


def main() -> int:
    results = load_results()
    section = build_section(results)
    readme = README.read_text(encoding="utf-8")
    updated = replace_section(readme, section)
    README.write_text(updated, encoding="utf-8")
    print(f"Updated {README} from {len(results)} results file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
