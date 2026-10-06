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
        lines.append(
            "Evaluation users: seeded sample of "
            f"{sample.get('n_sampled_users')} out of "
            f"{sample.get('n_eligible_users')} warm-relevant users "
            f"(seed={sample.get('seed')}, requested={sample.get('requested_users')}, "
            f"user_ids_sha256=`{sample.get('user_ids_sha256')}`). "
            "Training uses every training interaction. Sampled users keep full "
            "histories. Tuning selection and ranker labels use this same sample."
        )
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
