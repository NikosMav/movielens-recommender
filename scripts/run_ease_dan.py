#!/usr/bin/env python3
"""Train and score EASE-DAN on ml-1m (ADR-0028).

Stages: probe (wall time only), tune (validation NDCG@10), test (one shot).
Does not write results/ml-1m.json or any other existing results file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.ease_dan import (
    BUDGET_SEC,
    RESERVE_SEC,
    SEARCH_WALL_SEC,
    EASEDANRecommender,
    config_key,
    ease_dan_grid,
    edge_extension_configs,
    find_grid_edges,
    lock_grid,
    select_best,
)
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.beyond_accuracy import (
    genre_item_vectors,
    long_tail_items,
    score_recommendation_lists,
)
from movielens_recommender.cli import host_info, library_versions
from movielens_recommender.config import load_config
from movielens_recommender.data import DATASET_SHA256, download_dataset, load_ratings
from movielens_recommender.evaluate import (
    _prepare_eval,
    collect_recommendations,
    evaluate_recommender,
    format_metrics,
    ndcg_point_estimate,
)
from movielens_recommender.metrics import ndcg_at_k, paired_bootstrap_delta_ci
from movielens_recommender.movies import load_movies, movies_path
from movielens_recommender.scale import release_memory
from movielens_recommender.split import SplitConfig, SplitResult, time_based_split

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "results" / "budget" / "ease-dan-probe.json"
TUNING_PATH = ROOT / "results" / "tuning" / "ease_dan_ml-1m.json"
RESULTS_PATH = ROOT / "results" / "ease-dan" / "ml-1m.json"
HEADLINE_PATH = ROOT / "results" / "ml-1m.json"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    default_dir = ROOT / "data"
    if movies_path("ml-1m", default_dir).is_file():
        return default_dir
    space_dir = ROOT / "space" / ".runtime" / "data"
    if movies_path("ml-1m", space_dir).is_file():
        return space_dir
    download_dataset("ml-1m", default_dir)
    return default_dir


def load_split(data_dir: Path) -> tuple[Any, pd.DataFrame, pd.DataFrame, SplitResult]:
    config = load_config(ROOT / "configs" / "ml-1m.yaml")
    ratings, _stats = load_ratings("ml-1m", data_dir, clean=True)
    movies = load_movies("ml-1m", data_dir)
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
    return config, ratings, movies, split


def _val_view(split: SplitResult) -> SplitResult:
    assert split.val is not None
    return SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def _fit_ease_dan(train: pd.DataFrame, hyperparams: dict[str, Any]) -> EASEDANRecommender:
    return EASEDANRecommender(
        l2=float(hyperparams["l2"]),
        dropout_p=float(hyperparams["dropout_p"]),
        alpha=float(hyperparams["alpha"]),
        beta=float(hyperparams["beta"]),
    ).fit(train)


def run_probe(split: SplitResult, *, threshold: float) -> dict[str, Any]:
    """Time one fit + validation-sized recommend pass. No ranking metric."""
    if PROBE_PATH.is_file():
        payload = _read(PROBE_PATH)
        if payload.get("ranking_metrics_recorded") is not False:
            raise RuntimeError("existing probe file recorded a ranking metric")
        return payload

    sample_hp = {"l2": 100.0, "dropout_p": 0.3, "alpha": 0.2, "beta": 0.3}
    started = time.perf_counter()
    model = _fit_ease_dan(split.train, sample_hp)
    fit_sec = time.perf_counter() - started

    _prepared, _seen, relevant = _prepare_eval(
        split.train,
        split.val,
        relevance_threshold=threshold,
        split=_val_view(split),
    )
    started = time.perf_counter()
    lists = {uid: model.recommend(uid, 10) for uid in relevant}
    score_sec = time.perf_counter() - started
    n_users = len(relevant)
    del lists, model
    release_memory()

    elapsed = fit_sec + score_sec
    grid, cut, projected = lock_grid(elapsed)
    full_projection = len(ease_dan_grid()) * elapsed
    payload = {
        "adr": "0028",
        "budget_sec": BUDGET_SEC,
        "cut": cut,
        "host": host_info(),
        "locked_configs": len(grid),
        "locked_projection_sec": round(projected, 3),
        "projection_sec_full_grid": round(full_projection, 3),
        "ranking_metrics_recorded": False,
        "reserve_sec": RESERVE_SEC,
        "sample": {
            "elapsed_sec": round(elapsed, 3),
            "fit_sec": round(fit_sec, 3),
            "hyperparams": sample_hp,
            "n_eval_users": n_users,
            "score_sec": round(score_sec, 3),
        },
    }
    blob = json.dumps(payload).lower()
    if "ndcg" in blob or "recall" in blob:
        raise RuntimeError("probe payload contains a ranking metric name")
    _write(PROBE_PATH, payload)
    print(
        f"probe: elapsed={payload['sample']['elapsed_sec']}s "
        f"projected={payload['locked_projection_sec']}s "
        f"locked={payload['locked_configs']} ({cut})",
        flush=True,
    )
    return payload


def _score_val(split: SplitResult, hyperparams: dict[str, Any], *, threshold: float):
    started = time.perf_counter()
    model = _fit_ease_dan(split.train, hyperparams)
    fit_sec = time.perf_counter() - started
    started = time.perf_counter()
    score = ndcg_point_estimate(
        model.recommend,
        split.train,
        split.val,
        relevance_threshold=threshold,
        k=10,
        split=_val_view(split),
    )
    score_sec = time.perf_counter() - started
    del model
    release_memory()
    return float(score), fit_sec, score_sec


def _trial_record(
    hyperparams: dict[str, Any], score: float, fit_sec: float, score_sec: float
) -> dict[str, Any]:
    return {
        "fit_sec": round(float(fit_sec), 3),
        "hyperparams": {k: float(v) for k, v in hyperparams.items()},
        "score_sec": round(float(score_sec), 3),
        "val_ndcg@10": round(float(score), 6),
    }


def _spent(trials: list[dict[str, Any]]) -> float:
    return float(sum(float(t["fit_sec"]) + float(t["score_sec"]) for t in trials))


def _tuning_payload(
    *,
    grid: list[dict[str, float]],
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
    base_best = select_best(trials) if trials else None
    base_hp = None if base_best is None else base_best["hyperparams"]
    space = list(grid) + [row["hyperparams"] for row in extensions]
    edges = [] if best_hp is None else find_grid_edges(best_hp, space)
    base_edges = [] if base_hp is None else find_grid_edges(base_hp, grid)
    return {
        "adr": "0028",
        "base_grid_edges": base_edges,
        "best_hyperparams": best_hp,
        "best_val_score": None if best is None else best["val_ndcg@10"],
        "budget": {
            "cut": probe.get("cut"),
            "n_configs": len(grid),
            "projection_sec": probe.get("locked_projection_sec"),
            "truncated": truncated,
        },
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "edge_extension_trials": extensions,
        "grid": grid,
        "grid_edges": edges,
        "model": "ease_dan",
        "primary_metric": "ndcg@10",
        "protocol": "fit on fit-train, select by validation NDCG@10, test scored once",
        "split": split.summary(),
        "status": status,
        "trials": trials,
    }


def run_tune(split: SplitResult, *, threshold: float) -> dict[str, Any]:
    probe = run_probe(split, threshold=threshold)
    if probe.get("ranking_metrics_recorded") is not False:
        raise RuntimeError("probe recorded a ranking metric; refusing to tune")
    grid, cut, _projected = lock_grid(float(probe["sample"]["elapsed_sec"]))
    if cut != probe.get("cut") or len(grid) != int(probe["locked_configs"]):
        raise RuntimeError("probe cut does not match lock_grid")

    trials: list[dict[str, Any]] = []
    extensions: list[dict[str, Any]] = []
    if TUNING_PATH.is_file():
        prior = _read(TUNING_PATH)
        if prior.get("status") == "complete":
            return prior
        if [config_key(r) for r in prior.get("grid", [])] != [config_key(r) for r in grid]:
            raise RuntimeError("existing tuning file grid does not match locked grid")
        trials = list(prior.get("trials", []))
        extensions = list(prior.get("edge_extension_trials", []))

    done = {config_key(t["hyperparams"]) for t in trials}
    truncated = False
    clock = time.perf_counter()
    prior_spent = _spent(trials + extensions)
    print(f"tune: {len(grid)} configs, {len(done)} already scored", flush=True)

    def _over_budget() -> bool:
        return prior_spent + (time.perf_counter() - clock) >= SEARCH_WALL_SEC

    for i, hp in enumerate(grid, start=1):
        if config_key(hp) in done:
            continue
        if _over_budget():
            truncated = True
            print(f"tune: stopping at {SEARCH_WALL_SEC}s after {len(trials)} trials", flush=True)
            break
        score, fit_sec, score_sec = _score_val(split, hp, threshold=threshold)
        trials.append(_trial_record(hp, score, fit_sec, score_sec))
        if i % 10 == 0 or i == 1 or i == len(grid):
            print(
                f"  trial {i}/{len(grid)} val_ndcg@10={score:.6f} "
                f"fit={fit_sec:.2f}s {hp}",
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
        for hp in planned:
            if config_key(hp) in done_ext:
                continue
            if _over_budget():
                truncated = True
                print("tune: stopping before remaining edge step", flush=True)
                break
            score, fit_sec, score_sec = _score_val(split, hp, threshold=threshold)
            extensions.append(_trial_record(hp, score, fit_sec, score_sec))
            print(f"  edge val_ndcg@10={score:.6f} fit={fit_sec:.2f}s {hp}", flush=True)

    payload = _tuning_payload(
        grid=grid,
        trials=trials,
        extensions=extensions,
        probe=probe,
        status="complete",
        truncated=truncated,
        split=split,
    )
    _write(TUNING_PATH, payload)
    print(
        f"tune complete: {payload['best_hyperparams']} val={payload['best_val_score']}",
        flush=True,
    )
    return payload


def _eval_with_lists(
    model: Any,
    train: pd.DataFrame,
    test: pd.DataFrame,
    split: SplitResult,
    config: Any,
) -> tuple[dict[str, Any], list[int], list[float], dict[int, list[int]]]:
    view = SplitResult(
        train=train,
        test=test,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    ks = tuple(sorted(set(int(k) for k in list(config.eval.ks) + list(config.eval.retrieval_ks))))
    metrics = evaluate_recommender(
        model.recommend,
        train,
        test,
        relevance_threshold=config.eval.relevance_threshold,
        ks=ks,
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
        split=view,
        include_segments=True,
    )
    _prepared, seen, relevant = _prepare_eval(
        train, test, relevance_threshold=config.eval.relevance_threshold, split=view
    )
    recs_10 = collect_recommendations(model.recommend, seen, relevant, max_k=10)
    user_ids = [int(uid) for uid in relevant]
    per_user = [float(ndcg_at_k(recs_10[uid], relevant[uid], 10)) for uid in user_ids]
    if abs(float(sum(per_user) / len(per_user)) - float(metrics["ndcg@10"])) > 1e-9:
        raise RuntimeError("per-user NDCG@10 mean does not match harness")
    return format_metrics(metrics), user_ids, per_user, recs_10


def _round_paired(paired: dict[str, Any]) -> dict[str, Any]:
    low = round(float(paired["low"]), 6)
    high = round(float(paired["high"]), 6)
    return {
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "high": high,
        "low": low,
        "mean": round(float(paired["mean"]), 6),
        "n_users": int(paired["n_users"]),
    }


def _round_beyond(scored: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "coverage@10",
        "novelty@10",
        "intra_list_diversity@10",
        "mean_popularity@10",
        "long_tail_share@10",
        "ndcg@10",
    )
    out = {k: round(float(scored[k]), 6) for k in keys}
    out["confidence_intervals"] = {
        k: {b: round(float(v), 6) for b, v in bounds.items()}
        for k, bounds in scored["confidence_intervals"].items()
    }
    return out


def run_test(split: SplitResult, movies: pd.DataFrame, config: Any) -> dict[str, Any]:
    if not TUNING_PATH.is_file():
        raise RuntimeError("tuning file missing; run tune first")
    tuning = _read(TUNING_PATH)
    if tuning.get("status") != "complete":
        raise RuntimeError("tuning is not complete")

    headline_sha = _sha256(HEADLINE_PATH)
    headline = _read(HEADLINE_PATH)
    full_train = split.full_train
    seed = int(config.seed)

    winner_hp = dict(tuning["best_hyperparams"])
    print(f"test: refit EASE-DAN winner {winner_hp}", flush=True)
    t0 = time.perf_counter()
    dan_model = _fit_ease_dan(full_train, winner_hp)
    dan_fit_sec = round(time.perf_counter() - t0, 3)
    dan_rec_hp = dan_model.hyperparams()
    dan_metrics, dan_uids, dan_users, dan_lists = _eval_with_lists(
        dan_model, full_train, split.test, split, config
    )
    del dan_model
    release_memory()

    ease_hp = dict(headline["hyperparameters"]["ease"])
    print(f"test: refit committed EASE l2={ease_hp['l2']}", flush=True)
    t0 = time.perf_counter()
    ease_model = EASERecommender(l2=float(ease_hp["l2"])).fit(full_train)
    ease_fit_sec = round(time.perf_counter() - t0, 3)
    ease_metrics, ease_uids, ease_users, ease_lists = _eval_with_lists(
        ease_model, full_train, split.test, split, config
    )
    del ease_model
    release_memory()

    ii_hp = dict(headline["hyperparameters"]["item_item_cosine"])
    print(f"test: refit committed item-item cosine {ii_hp}", flush=True)
    t0 = time.perf_counter()
    ii_model = ItemItemCosineRecommender(
        min_common=int(ii_hp["min_common"]),
        k_neighbors=int(ii_hp["k_neighbors"]),
        shrinkage=float(ii_hp["shrinkage"]),
    ).fit(full_train)
    ii_fit_sec = round(time.perf_counter() - t0, 3)
    ii_metrics, ii_uids, ii_users, ii_lists = _eval_with_lists(
        ii_model, full_train, split.test, split, config
    )
    del ii_model
    release_memory()

    if not (dan_uids == ease_uids == ii_uids):
        raise RuntimeError("paired comparison users are not aligned")

    paired_ease = _round_paired(
        paired_bootstrap_delta_ci(
            ease_users,
            dan_users,
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=seed,
        )
    )
    paired_ii = _round_paired(
        paired_bootstrap_delta_ci(
            ii_users,
            dan_users,
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=seed,
        )
    )

    view = SplitResult(
        train=full_train,
        test=split.test,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    _prepared, _seen, relevant = _prepare_eval(
        full_train,
        split.test,
        relevance_threshold=config.eval.relevance_threshold,
        split=view,
    )
    catalog = [int(i) for i in full_train["item_id"].unique()]
    popularity = {
        int(item): float(count)
        for item, count in full_train.groupby("item_id").size().items()
    }
    tail = long_tail_items(full_train)
    vectors = genre_item_vectors(movies, sorted(catalog))
    beyond = {}
    for name, recs in (
        ("ease_dan", dan_lists),
        ("ease", ease_lists),
        ("item_item_cosine", ii_lists),
    ):
        scored = score_recommendation_lists(
            recs,
            relevant=relevant,
            catalog=catalog,
            item_popularity=popularity,
            item_vectors=vectors,
            tail_items=tail,
            k=10,
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=seed,
        )
        beyond[name] = _round_beyond(scored)

    beats_ease = bool(paired_ease["low"] > 0.0)
    beats_item_item = bool(paired_ii["low"] > 0.0)
    ease_val = float(headline["tuning_summary"]["ease_best_val_ndcg@10"])

    budget = dict(tuning["budget"])
    budget["n_configs_planned"] = len(tuning["grid"])
    budget["n_edge_extensions"] = len(tuning["edge_extension_trials"])
    budget["n_trials_scored"] = len(tuning["trials"]) + len(tuning["edge_extension_trials"])

    tt_pub = headline["metrics"]["two_tower"]
    lr_pub = headline["metrics"]["lambdarank"]
    lr_demo_path = ROOT / "results" / "demographics" / "ml-1m.json"
    lr_demo_ndcg = None
    if lr_demo_path.is_file():
        demo_doc = _read(lr_demo_path)
        lr_demo_ndcg = demo_doc["variants"]["both"]["across_seeds"]["ndcg@10"]["mean"]

    payload = {
        "adr": "0028",
        "beyond_accuracy": beyond,
        "budget": budget,
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "decision": {
            "beats_ease": beats_ease,
            "beats_item_item": beats_item_item,
            "negative_result": not beats_ease,
            "rule": (
                "beats_ease when the paired ease_dan minus ease 95% bootstrap "
                "interval low end is strictly above 0"
            ),
        },
        "host": host_info(),
        "library_versions": library_versions(),
        "models": {
            "ease": {
                "fit_sec": ease_fit_sec,
                "hyperparams": {"l2": float(ease_hp["l2"])},
                "metrics": ease_metrics,
                "published_test_ndcg@10": float(headline["metrics"]["ease"]["ndcg@10"]),
                "val_ndcg@10": ease_val,
            },
            "ease_dan": {
                "fit_sec": dan_fit_sec,
                "hyperparams": dan_rec_hp,
                "metrics": dan_metrics,
                "val_ndcg@10": tuning["best_val_score"],
            },
            "item_item_cosine": {
                "fit_sec": ii_fit_sec,
                "hyperparams": ii_hp,
                "metrics": ii_metrics,
                "published_test_ndcg@10": float(
                    headline["metrics"]["item_item_cosine"]["ndcg@10"]
                ),
                "val_ndcg@10": None,
            },
        },
        "paired": {
            "ease_dan_minus_ease": paired_ease,
            "ease_dan_minus_item_item": paired_ii,
        },
        "protocol": {
            "bootstrap_alpha": config.eval.bootstrap_alpha,
            "decision_split": "test",
            "n_bootstrap": config.eval.n_bootstrap,
            "paired_definition": "ease_dan minus reference, per-user test NDCG@10",
            "relevance_threshold": float(config.eval.relevance_threshold),
            "seed": seed,
            "selection_metric": "ndcg@10",
            "selection_split": "validation",
            "test_scored_once": True,
            "tie_break": "highest validation NDCG@10; earliest config on a tie",
        },
        "published_headlines": {
            "lambdarank_demographics_both_ndcg@10": lr_demo_ndcg,
            "lambdarank_s4_ndcg@10": float(lr_pub["ndcg@10"]),
            "lambdarank_s4_primary_ci": lr_pub["per_seed_confidence_intervals"]["42"]["ndcg@10"],
            "source": "results/ml-1m.json",
            "two_tower_ndcg@10": float(tt_pub["ndcg@10"]),
            "two_tower_primary_ci": tt_pub["per_seed_confidence_intervals"]["42"]["ndcg@10"],
        },
        "selected": {
            "base_grid_edges": tuning["base_grid_edges"],
            "grid_edges": tuning["grid_edges"],
            "hyperparams": winner_hp,
            "on_grid_edge": bool(tuning["grid_edges"]),
            "val_ndcg@10": tuning["best_val_score"],
        },
        "split": split.summary(),
    }
    _write(RESULTS_PATH, payload)
    if _sha256(HEADLINE_PATH) != headline_sha:
        raise RuntimeError("results/ml-1m.json changed during the EASE-DAN run")
    print(
        f"test NDCG@10 ease_dan={dan_metrics['ndcg@10']} "
        f"ease={ease_metrics['ndcg@10']} "
        f"item_item={ii_metrics['ndcg@10']} "
        f"beats_ease={beats_ease} beats_item_item={beats_item_item}",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0028 EASE-DAN on ml-1m.")
    parser.add_argument("--stage", choices=("probe", "tune", "test", "all"), default="all")
    parser.add_argument("--data-dir", default=None, help="Optional MovieLens data directory")
    args = parser.parse_args()

    headline_sha = _sha256(HEADLINE_PATH)
    data_dir = _resolve_data_dir(args.data_dir)
    config, _ratings, movies, split = load_split(data_dir)
    threshold = float(config.eval.relevance_threshold)

    if args.stage in {"probe", "all"}:
        run_probe(split, threshold=threshold)
    if args.stage in {"tune", "all"}:
        run_tune(split, threshold=threshold)
    if args.stage in {"test", "all"}:
        run_test(split, movies, config)
    if _sha256(HEADLINE_PATH) != headline_sha:
        raise RuntimeError("results/ml-1m.json changed")


if __name__ == "__main__":
    main()
