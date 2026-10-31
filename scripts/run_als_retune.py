#!/usr/bin/env python3
"""Retune implicit ALS on ml-1m (ADR-0018).

Stages: probe (wall time only), tune (validation NDCG@10), test (one shot).
Does not write results/ml-1m.json.
"""

from __future__ import annotations

import os

# Set before NumPy is imported. implicit ALS uses its own threads, and OpenBLAS
# on top of that oversubscribes this host (the warning in implicit's fit).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from movielens_recommender.als_retune import (
    BUDGET_SEC,
    EASE_CONFIGS,
    ITEM_ITEM_HYPERPARAMS,
    OLD_ALS_HYPERPARAMS,
    ORIGINAL_ALS_CONFIGS,
    PROBE_LARGE,
    PROBE_SMALL,
    RESERVE_SEC,
    RP3BETA_CONFIGS,
    SEARCH_WALL_SEC,
    als_retune_grid,
    batch_ndcg_point_estimate,
    config_key,
    edge_extension_configs,
    find_grid_edges,
    is_old_als,
    lock_grid,
    projected_seconds,
    select_best,
    trails_item_item,
)
from movielens_recommender.baselines.als import als_kwargs
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.cli import host_info, library_versions
from movielens_recommender.config import load_config
from movielens_recommender.data import DATASET_SHA256, load_ratings
from movielens_recommender.evaluate import (
    _prepare_eval,
    collect_recommendations,
    evaluate_recommender,
    format_metrics,
    ndcg_point_estimate,
)
from movielens_recommender.metrics import ndcg_at_k, paired_bootstrap_delta_ci
from movielens_recommender.scale import release_memory
from movielens_recommender.split import SplitConfig, SplitResult, time_based_split

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "results" / "budget" / "als-retune-probe.json"
TUNING_PATH = ROOT / "results" / "tuning" / "als_retune_ml-1m.json"
RESULTS_PATH = ROOT / "results" / "als-retune" / "ml-1m.json"
HEADLINE_PATH = ROOT / "results" / "ml-1m.json"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split() -> tuple[Any, pd.DataFrame, SplitResult]:
    config = load_config(ROOT / "configs" / "ml-1m.yaml")
    ratings, _stats = load_ratings("ml-1m", ROOT / "data", clean=True)
    split = time_based_split(
        ratings,
        SplitConfig(
            min_ratings=config.split.min_ratings,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            relevance_threshold=config.eval.relevance_threshold,
        ),
    )
    if split.val is None or split.val.empty:
        raise RuntimeError("ml-1m split has no validation rows")
    return config, ratings, split


def _val_view(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def _fit_als(train: pd.DataFrame, hyperparams: dict[str, Any], *, threshold: float, seed: int):
    from movielens_recommender.baselines.als import ALSRecommender

    kwargs = als_kwargs(hyperparams)
    kwargs["confidence_threshold"] = threshold
    kwargs["random_state"] = seed
    return ALSRecommender(**kwargs).fit(train)


def _trial_record(
    hyperparams: dict[str, Any], score: float, fit_sec: float, score_sec: float
) -> dict:
    return {
        "fit_sec": round(float(fit_sec), 3),
        "hyperparams": hyperparams,
        "score_sec": round(float(score_sec), 3),
        "val_ndcg@10": round(float(score), 6),
    }


def _score_val(split: SplitResult, hyperparams: dict[str, Any], *, threshold: float, seed: int):
    started = time.perf_counter()
    model = _fit_als(split.train, hyperparams, threshold=threshold, seed=seed)
    fit_sec = time.perf_counter() - started
    started = time.perf_counter()
    score = batch_ndcg_point_estimate(
        model,
        split.train,
        split.val,
        relevance_threshold=threshold,
        k=10,
        split=_val_view(split),
    )
    score_sec = time.perf_counter() - started
    del model
    release_memory()
    return score, fit_sec, score_sec


def _probe_sample(
    split: SplitResult, hyperparams: dict[str, Any], *, threshold: float, seed: int
) -> dict[str, Any]:
    """Time fit and a validation-sized recommend pass. No NDCG."""
    started = time.perf_counter()
    model = _fit_als(split.train, hyperparams, threshold=threshold, seed=seed)
    fit_sec = time.perf_counter() - started
    _prepared, _seen, relevant = _prepare_eval(
        split.train,
        split.val,
        relevance_threshold=threshold,
        split=_val_view(split),
    )
    started = time.perf_counter()
    lists = model.recommend_many(list(relevant), 10)
    score_sec = time.perf_counter() - started
    n_users = len(relevant)
    del lists
    del model
    release_memory()
    return {
        "elapsed_sec": round(fit_sec + score_sec, 3),
        "fit_sec": round(fit_sec, 3),
        "hyperparams": hyperparams,
        "n_eval_users": n_users,
        "score_sec": round(score_sec, 3),
    }


def run_probe(split: SplitResult, *, threshold: float, seed: int) -> dict[str, Any]:
    if PROBE_PATH.is_file():
        return _read(PROBE_PATH)
    warmup = {
        "alpha": 20.0,
        "confidence": "linear",
        "factors": 8,
        "iterations": 1,
        "regularization": 0.1,
    }
    _fit_als(split.train, warmup, threshold=threshold, seed=seed)
    release_memory()
    small_hp = {
        "alpha": 20.0,
        "confidence": "linear",
        "factors": PROBE_SMALL["factors"],
        "iterations": PROBE_SMALL["iterations"],
        "regularization": 0.1,
    }
    large_hp = {
        "alpha": 20.0,
        "confidence": "linear",
        "factors": PROBE_LARGE["factors"],
        "iterations": PROBE_LARGE["iterations"],
        "regularization": 0.1,
    }
    small = _probe_sample(split, small_hp, threshold=threshold, seed=seed)
    large = _probe_sample(split, large_hp, threshold=threshold, seed=seed)
    _grid, cut, projected = lock_grid(small["elapsed_sec"], large["elapsed_sec"])
    full_projection = projected_seconds(
        als_retune_grid(), small["elapsed_sec"], large["elapsed_sec"]
    )
    payload = {
        "adr": "0018",
        "budget_sec": BUDGET_SEC,
        "cut": cut,
        "host": host_info(),
        "locked_configs": len(_grid),
        "locked_projection_sec": round(projected, 3),
        "projection_sec_full_grid": round(full_projection, 3),
        "ranking_metrics_recorded": False,
        "reserve_sec": RESERVE_SEC,
        "samples": {"large": large, "small": small},
    }
    _write(PROBE_PATH, payload)
    print(
        f"probe: small {small['elapsed_sec']}s, large {large['elapsed_sec']}s, "
        f"full projection {payload['projection_sec_full_grid']}s, "
        f"locked {payload['locked_configs']} ({cut})",
        flush=True,
    )
    return payload


def _spent(trials: list[dict[str, Any]]) -> float:
    return float(sum(float(t["fit_sec"]) + float(t["score_sec"]) for t in trials))


def _tuning_payload(
    *,
    grid: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    extensions: list[dict[str, Any]],
    probe: dict[str, Any],
    status: str,
    truncated: bool,
    split: SplitResult,
) -> dict[str, Any]:
    finished = trials + extensions
    best = select_best(finished) if finished else None
    best_hp = None if best is None else best["hyperparams"]
    # Edges are against the locked grid plus any extension that was scored,
    # so an unfinished high factor is still the high end of the search.
    space = list(grid) + [row["hyperparams"] for row in extensions]
    edges = [] if best_hp is None else find_grid_edges(best_hp, space)
    base_edges = [] if best_hp is None else find_grid_edges(best_hp, grid)
    old = next((t for t in trials if is_old_als(t["hyperparams"])), None)
    return {
        "adr": "0018",
        "base_grid_edges": base_edges,
        "best_hyperparams": best_hp,
        "best_val_score": None if best is None else best["val_ndcg@10"],
        "budget": {
            "cut": probe.get("cut"),
            "ease_configs": EASE_CONFIGS,
            "n_configs": len(grid),
            "original_als_configs": ORIGINAL_ALS_CONFIGS,
            "projection_sec": probe.get("locked_projection_sec"),
            "rp3beta_configs": RP3BETA_CONFIGS,
            "truncated": truncated,
        },
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "edge_extension_trials": extensions,
        "grid": grid,
        "grid_edges": edges,
        "model": "als",
        "old_config": OLD_ALS_HYPERPARAMS,
        "old_config_val_ndcg@10": None if old is None else old["val_ndcg@10"],
        "primary_metric": "ndcg@10",
        "protocol": "fit on fit-train, select by validation NDCG@10, test scored once",
        "split": split.summary(),
        "status": status,
        "trials": trials,
    }


def run_tune(split: SplitResult, *, threshold: float, seed: int) -> dict[str, Any]:
    probe = run_probe(split, threshold=threshold, seed=seed)
    if probe.get("ranking_metrics_recorded"):
        raise RuntimeError("probe recorded a ranking metric; refusing to tune")
    small = probe["samples"]["small"]["elapsed_sec"]
    large = probe["samples"]["large"]["elapsed_sec"]
    grid, cut, _projected = lock_grid(small, large)
    if cut != probe.get("cut") or len(grid) != int(probe["locked_configs"]):
        raise RuntimeError("probe cut does not match lock_grid; the budget rule drifted")

    trials: list[dict[str, Any]] = []
    extensions: list[dict[str, Any]] = []
    if TUNING_PATH.is_file():
        prior = _read(TUNING_PATH)
        if prior.get("status") == "complete":
            return prior
        if [config_key(row) for row in prior.get("grid", [])] != [config_key(row) for row in grid]:
            raise RuntimeError("existing tuning file grid does not match the locked grid")
        trials = list(prior.get("trials", []))
        extensions = list(prior.get("edge_extension_trials", []))

    done = {config_key(t["hyperparams"]) for t in trials}
    truncated = False
    clock = time.perf_counter()
    prior_spent = _spent(trials + extensions)
    print(f"tune: {len(grid)} configs, {len(done)} already scored", flush=True)

    def _over_budget() -> bool:
        return prior_spent + (time.perf_counter() - clock) >= SEARCH_WALL_SEC

    for i, hyperparams in enumerate(grid, start=1):
        if config_key(hyperparams) in done:
            continue
        if _over_budget():
            truncated = True
            print(f"tune: stopping at {SEARCH_WALL_SEC}s after {len(trials)} trials", flush=True)
            break
        score, fit_sec, score_sec = _score_val(
            split, hyperparams, threshold=threshold, seed=seed
        )
        trials.append(_trial_record(hyperparams, score, fit_sec, score_sec))
        print(
            f"  trial {i}/{len(grid)} val_ndcg@10={score:.6f} "
            f"fit={fit_sec:.1f}s score={score_sec:.1f}s {hyperparams}",
            flush=True,
        )
        _write(
            TUNING_PATH,
            _tuning_payload(
                grid=grid,
                trials=trials,
                extensions=extensions,
                probe=probe,
                status="running",
                truncated=False,
                split=split,
            ),
        )

    if not truncated:
        base_best = select_best(trials)
        planned = edge_extension_configs(base_best["hyperparams"], grid)
        done_ext = {config_key(t["hyperparams"]) for t in extensions}
        for hyperparams in planned:
            if config_key(hyperparams) in done_ext:
                continue
            if _over_budget():
                truncated = True
                print("tune: stopping before a remaining edge step", flush=True)
                break
            score, fit_sec, score_sec = _score_val(
                split, hyperparams, threshold=threshold, seed=seed
            )
            extensions.append(_trial_record(hyperparams, score, fit_sec, score_sec))
            print(
                f"  edge val_ndcg@10={score:.6f} fit={fit_sec:.1f}s {hyperparams}",
                flush=True,
            )
            _write(
                TUNING_PATH,
                _tuning_payload(
                    grid=grid,
                    trials=trials,
                    extensions=extensions,
                    probe=probe,
                    status="running",
                    truncated=False,
                    split=split,
                ),
            )

    winner = select_best(trials + extensions)
    batch_now, harness = _both_val_scores(
        split, winner["hyperparams"], threshold=threshold, seed=seed
    )
    # A one-user rank change moves the mean by well more than 1e-6. Agreement
    # at 6 decimals is the recorded precision (ADR-0005 rounds validation there).
    if round(batch_now, 6) != round(harness, 6):
        raise RuntimeError(
            f"winner batch NDCG@10 {batch_now:.8f} != harness {harness:.8f}; not scoring test"
        )
    if round(harness, 6) != float(winner["val_ndcg@10"]):
        raise RuntimeError(
            f"stored trial NDCG@10 {winner['val_ndcg@10']} != fresh harness {harness:.8f}"
        )
    winner["val_ndcg@10"] = round(harness, 6)
    payload = _tuning_payload(
        grid=grid,
        trials=trials,
        extensions=extensions,
        probe=probe,
        status="complete",
        truncated=truncated,
        split=split,
    )
    payload["best_val_score"] = round(harness, 6)
    payload["winner_harness_val_ndcg@10"] = round(harness, 6)
    _write(TUNING_PATH, payload)
    print(
        f"tune complete: {payload['best_hyperparams']} val={payload['best_val_score']}",
        flush=True,
    )
    return payload


def _both_val_scores(
    split: SplitResult, hyperparams: dict[str, Any], *, threshold: float, seed: int
) -> tuple[float, float]:
    """Batch score and harness score on one fit. The harness value is official."""
    model = _fit_als(split.train, hyperparams, threshold=threshold, seed=seed)
    batch = batch_ndcg_point_estimate(
        model,
        split.train,
        split.val,
        relevance_threshold=threshold,
        k=10,
        split=_val_view(split),
    )
    harness = ndcg_point_estimate(
        model.recommend,
        split.train,
        split.val,
        relevance_threshold=threshold,
        k=10,
        split=_val_view(split),
    )
    del model
    release_memory()
    return float(batch), float(harness)


def _per_user_ndcg(model, train, test, split: SplitResult, *, threshold: float) -> list[float]:
    view = SplitResult(
        train=train,
        test=test,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    _prepared, seen, relevant = _prepare_eval(
        train, test, relevance_threshold=threshold, split=view
    )
    del _prepared
    recs = collect_recommendations(model.recommend, seen, relevant, max_k=10)
    return [ndcg_at_k(recs[uid], rel, 10) for uid, rel in relevant.items()]


def _round_paired(paired: dict[str, Any]) -> dict[str, Any]:
    return {
        "excludes_zero": bool(paired["excludes_zero"]),
        "high": round(float(paired["high"]), 6),
        "low": round(float(paired["low"]), 6),
        "mean": round(float(paired["mean"]), 6),
        "n_users": int(paired["n_users"]),
    }


def _eval_model(
    model, train, test, split: SplitResult, config
) -> tuple[dict[str, Any], list[float]]:
    view = SplitResult(
        train=train,
        test=test,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    metrics = evaluate_recommender(
        model.recommend,
        train,
        test,
        relevance_threshold=config.eval.relevance_threshold,
        ks=tuple(config.eval.ks),
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
        split=view,
    )
    per_user = _per_user_ndcg(
        model, train, test, split, threshold=config.eval.relevance_threshold
    )
    if abs(float(sum(per_user) / len(per_user)) - float(metrics["ndcg@10"])) > 1e-9:
        raise RuntimeError("per-user NDCG@10 mean does not match the harness")
    return metrics, per_user


def run_test(split: SplitResult, config) -> dict[str, Any]:
    if not TUNING_PATH.is_file():
        raise RuntimeError("tuning file missing; run tune first")
    tuning = _read(TUNING_PATH)
    if tuning.get("status") != "complete":
        raise RuntimeError("tuning is not complete")
    headline = _read(HEADLINE_PATH)
    threshold = config.eval.relevance_threshold
    seed = config.seed
    full_train = split.full_train

    winner_hp = tuning["best_hyperparams"]
    print(f"test: refit winner {winner_hp}", flush=True)
    winner = _fit_als(full_train, winner_hp, threshold=threshold, seed=seed)
    winner_metrics, winner_users = _eval_model(winner, full_train, split.test, split, config)
    del winner
    release_memory()

    print("test: refit old ALS config", flush=True)
    old = _fit_als(full_train, OLD_ALS_HYPERPARAMS, threshold=threshold, seed=seed)
    old_hp = old.hyperparams()
    old_metrics, old_users = _eval_model(old, full_train, split.test, split, config)
    del old
    release_memory()

    print("test: fit item-item default", flush=True)
    item_model = ItemItemCosineRecommender(
        min_common=int(ITEM_ITEM_HYPERPARAMS["min_common"]),
        k_neighbors=int(ITEM_ITEM_HYPERPARAMS["k_neighbors"]),
        shrinkage=float(ITEM_ITEM_HYPERPARAMS["shrinkage"]),
    ).fit(full_train)
    item_metrics, item_users = _eval_model(item_model, full_train, split.test, split, config)
    del item_model
    release_memory()

    if not (len(winner_users) == len(old_users) == len(item_users)):
        raise RuntimeError("paired comparison users are not aligned")
    paired_old = paired_bootstrap_delta_ci(
        old_users,
        winner_users,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
        seed=seed,
    )
    paired_item = paired_bootstrap_delta_ci(
        item_users,
        winner_users,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
        seed=seed,
    )
    winner_fmt = format_metrics(winner_metrics)
    old_fmt = format_metrics(old_metrics)
    item_fmt = format_metrics(item_metrics)
    negative = trails_item_item(winner_fmt["ndcg@10"], item_fmt["ndcg@10"])
    space = list(tuning["grid"])
    space += [t["hyperparams"] for t in tuning["edge_extension_trials"]]
    edges = find_grid_edges(winner_hp, space)
    base_edges = find_grid_edges(winner_hp, tuning["grid"])
    published_als = float(headline["metrics"]["als_tuned"]["ndcg@10"])
    published_item = float(headline["metrics"]["item_item_cosine"]["ndcg@10"])
    budget = dict(tuning["budget"])
    budget["n_configs_planned"] = len(tuning["grid"])
    budget["n_edge_extensions"] = len(tuning["edge_extension_trials"])
    budget["n_trials_scored"] = len(tuning["trials"]) + len(tuning["edge_extension_trials"])
    payload = {
        "adr": "0018",
        "budget": budget,
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "decision": {
            "item_item_test_ndcg@10": item_fmt["ndcg@10"],
            "negative_result": negative,
            "rule": (
                "negative_result if selected test ndcg@10 is below item-item "
                "test ndcg@10 on this code, at the stored 6-decimal values"
            ),
            "selected_test_ndcg@10": winner_fmt["ndcg@10"],
            "trails_item_item": negative,
        },
        "host": host_info(),
        "models": {
            "als_old": {
                "hyperparams": old_hp,
                "metrics": old_fmt,
                "published_source": "results/ml-1m.json metrics.als_tuned.ndcg@10",
                "published_test_ndcg@10": published_als,
                "val_ndcg@10": tuning["old_config_val_ndcg@10"],
            },
            "als_retuned": {
                "hyperparams": winner_hp,
                "metrics": winner_fmt,
                "val_ndcg@10": tuning["best_val_score"],
            },
            "item_item_cosine": {
                "hyperparams": ITEM_ITEM_HYPERPARAMS,
                "metrics": item_fmt,
                "published_source": "results/ml-1m.json metrics.item_item_cosine.ndcg@10",
                "published_test_ndcg@10": published_item,
                "val_ndcg@10": None,
            },
        },
        "paired": {
            "retuned_minus_item_item": _round_paired(paired_item),
            "retuned_minus_old_als": _round_paired(paired_old),
        },
        "protocol": {
            "bootstrap_alpha": config.eval.bootstrap_alpha,
            "decision_split": "test",
            "n_bootstrap": config.eval.n_bootstrap,
            "paired_definition": "retuned minus reference, per-user test NDCG@10",
            "relevance_threshold": threshold,
            "seed": seed,
            "selection_metric": "ndcg@10",
            "selection_split": "validation",
            "test_scored_once": True,
            "tie_break": "highest validation NDCG@10; earliest config on a tie",
        },
        "selected": {
            "base_grid_edges": base_edges,
            "grid_edges": edges,
            "hyperparams": winner_hp,
            "on_grid_edge": bool(edges),
            "val_ndcg@10": tuning["best_val_score"],
        },
        "split": split.summary(),
        "versions": library_versions(),
    }
    _write(RESULTS_PATH, payload)
    print(
        f"test NDCG@10 retuned={winner_fmt['ndcg@10']} "
        f"old={old_fmt['ndcg@10']} item-item={item_fmt['ndcg@10']} "
        f"negative={negative} edges={edges}",
        flush=True,
    )
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Retune implicit ALS on ml-1m (ADR-0018).")
    parser.add_argument("--stage", choices=("probe", "tune", "test", "all"), default="all")
    args = parser.parse_args()
    config, _ratings, split = load_split()
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)
    if args.stage in {"probe", "all"} and not PROBE_PATH.is_file():
        run_probe(split, threshold=threshold, seed=seed)
    elif args.stage == "probe":
        run_probe(split, threshold=threshold, seed=seed)
    if args.stage in {"tune", "all"}:
        run_tune(split, threshold=threshold, seed=seed)
    if args.stage in {"test", "all"}:
        run_test(split, config)


if __name__ == "__main__":
    main()
