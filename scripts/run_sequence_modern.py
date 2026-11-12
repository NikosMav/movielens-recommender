#!/usr/bin/env python3
"""Compare eSASRec (RecSys 2025) and PCTM (RecSys 2026) against SASRec on ml-1m (ADR-0030).

Does not write results/ml-1m.json or results/sasrec/ml-1m.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    ndcg_point_estimate,
)
from movielens_recommender.metrics import paired_bootstrap_delta_ci
from movielens_recommender.movies import load_movies, movies_path
from movielens_recommender.sasrec.pctm import (
    PCTMRecommender,
    esasrec_grid,
    pctm_grid,
    select_best_trial,
)
from movielens_recommender.sasrec.train import evaluate_split, per_user_ndcg, train_sasrec
from movielens_recommender.scale import release_memory
from movielens_recommender.split import SplitConfig, SplitResult, time_based_split

ROOT = Path(__file__).resolve().parents[1]
TUNING_PATH = ROOT / "results" / "tuning" / "sequence_modern_ml-1m.json"
RESULTS_PATH = ROOT / "results" / "sequence-modern" / "ml-1m.json"
HEADLINE_PATH = ROOT / "results" / "ml-1m.json"
SASREC_PATH = ROOT / "results" / "sasrec" / "ml-1m.json"

TEST_SEEDS: tuple[int, ...] = (42, 43, 44)
PRIMARY_SEED = 42


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


def load_split(data_dir: Path) -> tuple[Any, pd.DataFrame, SplitResult]:
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
    return config, movies, split


def _val_view(split: SplitResult) -> SplitResult:
    assert split.val is not None
    return SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def _test_view(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.full_train,
        test=split.test,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def tune_pctm(split: SplitResult, *, threshold: float) -> dict[str, Any]:
    grid = pctm_grid()
    val_view = _val_view(split)
    trials: list[dict[str, Any]] = []
    # Cache transition matrices per (max_distance, decay) so we count each graph once.
    by_graph: dict[tuple[int, str], PCTMRecommender] = {}
    for idx, hp in enumerate(grid, start=1):
        g_key = (int(hp["max_distance"]), str(hp["decay"]))
        t0 = time.perf_counter()
        if g_key not in by_graph:
            by_graph[g_key] = PCTMRecommender(
                max_distance=g_key[0],
                decay=g_key[1],
                smoothing_tau=float(hp["smoothing_tau"]),
                max_history=int(hp["max_history"]),
                pop_beta=float(hp["pop_beta"]),
                relevance_threshold=threshold,
            ).fit(split.train)
        model = by_graph[g_key].set_scoring_params(
            smoothing_tau=float(hp["smoothing_tau"]),
            pop_beta=float(hp["pop_beta"]),
        )
        score = ndcg_point_estimate(
            model.recommend,
            split.train,
            split.val,
            relevance_threshold=threshold,
            k=10,
            split=val_view,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        trials.append(
            {
                "hyperparams": dict(hp),
                "val_ndcg@10": round(float(score), 6),
                "wall_sec": elapsed,
            }
        )
        if idx % 9 == 0 or idx == 1 or idx == len(grid):
            print(f"  [pctm] {idx}/{len(grid)} val_ndcg@10={score:.6f} {hp}", flush=True)
    by_graph.clear()
    release_memory()
    best = select_best_trial(trials)
    return {
        "best_hyperparams": best["hyperparams"],
        "best_val_score": best["val_ndcg@10"],
        "grid": grid,
        "n_configs": len(grid),
        "trials": trials,
    }


def tune_esasrec(split: SplitResult, *, threshold: float) -> dict[str, Any]:
    grid = esasrec_grid()
    trials: list[dict[str, Any]] = []
    for idx, hp in enumerate(grid, start=1):
        print(
            f"  [esasrec] {idx}/{len(grid)} seq={hp['max_seq_len']} "
            f"layers={hp['n_layers']} drop={hp['dropout']}",
            flush=True,
        )
        rec, _data, result = train_sasrec(
            split.train,
            hyperparams=hp,
            seed=PRIMARY_SEED,
            relevance_threshold=threshold,
            val_frame=split.val,
            val_template=split,
            show_progress=True,
        )
        del rec, _data
        release_memory()
        if result.best_val_ndcg10 is None:
            raise RuntimeError("eSASRec trial produced no validation score")
        trials.append(
            {
                "best_epoch": int(result.best_epoch),
                "epochs_trained": int(result.epochs_trained),
                "history": result.history,
                "hyperparams": dict(hp),
                "val_ndcg@10": round(float(result.best_val_ndcg10), 6),
                "wall_sec": float(result.wall_time_sec),
            }
        )
    best = select_best_trial(trials)
    return {
        "best_epoch": int(best["best_epoch"]),
        "best_hyperparams": best["hyperparams"],
        "best_val_score": best["val_ndcg@10"],
        "grid": grid,
        "n_configs": len(grid),
        "trials": trials,
    }


def run_tune(split: SplitResult, *, threshold: float) -> dict[str, Any]:
    if TUNING_PATH.is_file():
        existing = _read(TUNING_PATH)
        if existing.get("status") == "complete":
            print(f"Reusing completed tuning from {TUNING_PATH}", flush=True)
            return existing

    print("Tuning PCTM on validation...", flush=True)
    pctm_block = tune_pctm(split, threshold=threshold)
    print(
        f"PCTM winner: {pctm_block['best_hyperparams']} val={pctm_block['best_val_score']}",
        flush=True,
    )

    print("Tuning eSASRec (LiGR + SwiGLU) on validation...", flush=True)
    esasrec_block = tune_esasrec(split, threshold=threshold)
    print(
        f"eSASRec winner: {esasrec_block['best_hyperparams']} "
        f"epoch={esasrec_block['best_epoch']} val={esasrec_block['best_val_score']}",
        flush=True,
    )

    payload = {
        "adr": "0030",
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "esasrec": esasrec_block,
        "pctm": pctm_block,
        "primary_metric": "ndcg@10",
        "protocol": "fit on fit-train, select by validation NDCG@10, test scored once",
        "split": split.summary(),
        "status": "complete",
    }
    _write(TUNING_PATH, payload)
    return payload


def _store_pair(left: list[float], right: list[float], config: Any) -> dict[str, Any]:
    raw = paired_bootstrap_delta_ci(
        left,
        right,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
        seed=int(config.seed),
    )
    low = round(float(raw["low"]), 6)
    high = round(float(raw["high"]), 6)
    return {
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "high": high,
        "low": low,
        "mean": round(float(raw["mean"]), 6),
        "n_users": int(raw["n_users"]),
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
    return {k: round(float(scored[k]), 6) for k in keys}


def run_test(split: SplitResult, movies: pd.DataFrame, config: Any) -> dict[str, Any]:
    tuning = _read(TUNING_PATH)
    if tuning.get("status") != "complete":
        raise RuntimeError("tuning is not complete")

    headline_sha = _sha256(HEADLINE_PATH)
    sasrec_sha = _sha256(SASREC_PATH)
    sasrec_committed = _read(SASREC_PATH)
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)
    view = _test_view(split)
    full_train = split.full_train
    _prepared, seen, relevant = _prepare_eval(
        full_train, split.test, relevance_threshold=threshold, split=view
    )

    # 1. PCTM test refit
    pctm_hp = dict(tuning["pctm"]["best_hyperparams"])
    print(f"Fitting PCTM winner on full_train: {pctm_hp}", flush=True)
    t0 = time.perf_counter()
    pctm_model = PCTMRecommender(
        max_distance=int(pctm_hp["max_distance"]),
        decay=str(pctm_hp["decay"]),
        smoothing_tau=float(pctm_hp["smoothing_tau"]),
        max_history=int(pctm_hp["max_history"]),
        pop_beta=float(pctm_hp["pop_beta"]),
        relevance_threshold=threshold,
    ).fit(full_train)
    pctm_fit_sec = round(time.perf_counter() - t0, 3)
    pctm_metrics = evaluate_split(
        pctm_model.recommend,
        full_train,
        split.test,
        relevance_threshold=threshold,
        split=view,
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=seed,
    )
    pctm_uids, pctm_users = per_user_ndcg(
        pctm_model.recommend,
        full_train,
        split.test,
        relevance_threshold=threshold,
        split=view,
    )
    pctm_lists = collect_recommendations(pctm_model.recommend, seen, relevant, max_k=10)
    del pctm_model
    release_memory()

    # 2. eSASRec 3-seed test refit
    esas_hp = dict(tuning["esasrec"]["best_hyperparams"])
    esas_epoch = int(tuning["esasrec"]["best_epoch"])
    esas_per_seed: list[dict[str, Any]] = []
    esas_primary_metrics: dict[str, Any] | None = None
    esas_uids: list[int] | None = None
    esas_primary_users: list[float] | None = None
    esas_lists: dict[int, list[int]] = {}

    for s in TEST_SEEDS:
        print(f"Refitting eSASRec seed {s} for {esas_epoch} epochs...", flush=True)
        rec, _data, tr = train_sasrec(
            full_train,
            hyperparams=esas_hp,
            seed=s,
            relevance_threshold=threshold,
            n_epochs=esas_epoch,
            show_progress=True,
        )
        if s == PRIMARY_SEED:
            esas_primary_metrics = evaluate_split(
                rec.recommend,
                full_train,
                split.test,
                relevance_threshold=threshold,
                split=view,
                n_bootstrap=config.eval.n_bootstrap,
                bootstrap_alpha=config.eval.bootstrap_alpha,
                seed=seed,
            )
            esas_uids, esas_primary_users = per_user_ndcg(
                rec.recommend,
                full_train,
                split.test,
                relevance_threshold=threshold,
                split=view,
            )
            esas_lists = collect_recommendations(rec.recommend, seen, relevant, max_k=10)
            pt = float(esas_primary_metrics["ndcg@10"])
        else:
            u_s, v_s = per_user_ndcg(
                rec.recommend,
                full_train,
                split.test,
                relevance_threshold=threshold,
                split=view,
            )
            if esas_uids is not None and u_s != esas_uids:
                raise RuntimeError("user order changed across eSASRec seeds")
            pt = float(np.mean(v_s))
        esas_per_seed.append(
            {
                "epochs_trained": int(tr.epochs_trained),
                "ndcg@10": round(pt, 6),
                "seed": int(s),
                "wall_sec": float(tr.wall_time_sec),
            }
        )
        del rec, _data
        release_memory()

    assert (
        esas_primary_metrics is not None
        and esas_uids is not None
        and esas_primary_users is not None
    )

    # 3. Reference SASRec primary seed refit for exact aligned paired CIs + beyond-accuracy
    sas_hp = dict(sasrec_committed["selected"]["hyperparams"])
    sas_epoch = int(sasrec_committed["selected"]["best_epoch"])
    print(f"Refitting committed SASRec primary seed (epochs={sas_epoch})...", flush=True)
    sas_rec, _sdata, sas_tr = train_sasrec(
        full_train,
        hyperparams=sas_hp,
        seed=PRIMARY_SEED,
        relevance_threshold=threshold,
        n_epochs=sas_epoch,
        show_progress=True,
    )
    sas_metrics = evaluate_split(
        sas_rec.recommend,
        full_train,
        split.test,
        relevance_threshold=threshold,
        split=view,
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=seed,
    )
    sas_uids, sas_users = per_user_ndcg(
        sas_rec.recommend,
        full_train,
        split.test,
        relevance_threshold=threshold,
        split=view,
    )
    sas_lists = collect_recommendations(sas_rec.recommend, seen, relevant, max_k=10)
    del sas_rec, _sdata
    release_memory()

    if not (pctm_uids == esas_uids == sas_uids):
        raise RuntimeError("sequence comparison user ids are not aligned")

    paired_esas_vs_sas = _store_pair(sas_users, esas_primary_users, config)
    paired_pctm_vs_sas = _store_pair(sas_users, pctm_users, config)
    paired_esas_vs_pctm = _store_pair(pctm_users, esas_primary_users, config)

    catalog = [int(i) for i in full_train["item_id"].unique()]
    popularity = {
        int(item): float(count)
        for item, count in full_train.groupby("item_id").size().items()
    }
    tail = long_tail_items(full_train)
    vectors = genre_item_vectors(movies, sorted(catalog))
    beyond = {}
    for name, recs in (
        ("pctm", pctm_lists),
        ("esasrec", esas_lists),
        ("sasrec", sas_lists),
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

    esas_points = [float(r["ndcg@10"]) for r in esas_per_seed]
    esas_beats_sasrec = bool(paired_esas_vs_sas["low"] > 0.0)
    pctm_beats_sasrec = bool(paired_pctm_vs_sas["low"] > 0.0)
    higher_order_beats_pairwise = bool(paired_esas_vs_pctm["low"] > 0.0)

    payload = {
        "adr": "0030",
        "beyond_accuracy": beyond,
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "decision": {
            "esasrec_beats_sasrec": esas_beats_sasrec,
            "higher_order_beats_pairwise_pctm": higher_order_beats_pairwise,
            "pctm_beats_sasrec": pctm_beats_sasrec,
            "rule": (
                "Each comparison is true when the paired 95% bootstrap interval "
                "low end is strictly above 0."
            ),
        },
        "host": host_info(),
        "library_versions": library_versions(),
        "models": {
            "esasrec": {
                "best_epoch": esas_epoch,
                "hyperparams": esas_hp,
                "metrics": esas_primary_metrics,
                "ndcg@10_mean": round(float(np.mean(esas_points)), 6),
                "ndcg@10_std": round(float(np.std(esas_points, ddof=0)), 6),
                "per_seed": esas_per_seed,
                "primary_ndcg@10_ci": esas_primary_metrics["confidence_intervals"]["ndcg@10"],
                "val_ndcg@10": tuning["esasrec"]["best_val_score"],
            },
            "pctm": {
                "fit_sec": pctm_fit_sec,
                "hyperparams": pctm_hp,
                "metrics": pctm_metrics,
                "ndcg@10": pctm_metrics["ndcg@10"],
                "ndcg@10_ci": pctm_metrics["confidence_intervals"]["ndcg@10"],
                "val_ndcg@10": tuning["pctm"]["best_val_score"],
            },
            "sasrec_reference": {
                "best_epoch": sas_epoch,
                "committed_ndcg@10_mean": sasrec_committed["models"]["sasrec"]["ndcg@10_mean"],
                "hyperparams": sas_hp,
                "metrics": sas_metrics,
                "ndcg@10": sas_metrics["ndcg@10"],
                "ndcg@10_ci": sas_metrics["confidence_intervals"]["ndcg@10"],
                "train_wall_sec": float(sas_tr.wall_time_sec),
                "val_ndcg@10": sasrec_committed["selected"]["val_ndcg@10"],
            },
        },
        "paired": {
            "esasrec_minus_pctm": paired_esas_vs_pctm,
            "esasrec_minus_sasrec": paired_esas_vs_sas,
            "pctm_minus_sasrec": paired_pctm_vs_sas,
        },
        "seeds": {
            "bootstrap": seed,
            "primary": PRIMARY_SEED,
            "test": list(TEST_SEEDS),
        },
        "split": split.summary(),
    }
    _write(RESULTS_PATH, payload)
    if _sha256(HEADLINE_PATH) != headline_sha or _sha256(SASREC_PATH) != sasrec_sha:
        raise RuntimeError("existing results files changed during sequence-modern run")
    print(
        f"test NDCG@10: pctm={pctm_metrics['ndcg@10']} "
        f"esasrec_mean={payload['models']['esasrec']['ndcg@10_mean']} "
        f"(primary={esas_primary_metrics['ndcg@10']}) "
        f"sasrec={sas_metrics['ndcg@10']}",
        flush=True,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0030 eSASRec vs PCTM on ml-1m.")
    parser.add_argument("--stage", choices=("tune", "test", "all"), default="all")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    data_dir = _resolve_data_dir(args.data_dir)
    config, movies, split = load_split(data_dir)
    threshold = float(config.eval.relevance_threshold)

    if args.stage in {"tune", "all"}:
        run_tune(split, threshold=threshold)
    if args.stage in {"test", "all"}:
        run_test(split, movies, config)


if __name__ == "__main__":
    main()
