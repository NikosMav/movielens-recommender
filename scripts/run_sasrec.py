#!/usr/bin/env python3
"""Train and score SASRec on ml-1m (ADR-0019).

Stages: probe (wall time only), tune (validation NDCG@10), test (one shot),
shuffled-order check, and paired refits of the committed two-tower and
LambdaRank headlines. Does not write results/ml-1m.json.
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
import torch

from movielens_recommender.cli import host_info, library_versions
from movielens_recommender.config import load_config
from movielens_recommender.data import load_ratings
from movielens_recommender.metrics import paired_bootstrap_delta_ci
from movielens_recommender.sasrec.compare import score_headline_references
from movielens_recommender.sasrec.model import SASRecModel
from movielens_recommender.sasrec.plan import (
    BOOTSTRAP_ALPHA,
    BOOTSTRAP_SEED,
    BUDGET_SEC,
    COMPARISON_RESERVE_SEC,
    EASE_CONFIGS,
    N_BOOTSTRAP,
    PRIMARY_SEED,
    SHUFFLE_SEED,
    TWO_TOWER_GRID_CONFIGS,
    config_key,
    edge_extension_configs,
    find_edges,
    lock_budget,
    projected_trial_seconds,
    select_winner,
)
from movielens_recommender.sasrec.sequences import build_sequence_data
from movielens_recommender.sasrec.train import (
    SASRecRecommender,
    evaluate_split,
    per_user_ndcg,
    time_one_epoch,
    train_sasrec,
)
from movielens_recommender.scale import release_memory
from movielens_recommender.split import SplitConfig, SplitResult, time_based_split

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "results" / "budget" / "sasrec-probe.json"
TUNING_PATH = ROOT / "results" / "tuning" / "sasrec_ml-1m.json"
RESULTS_PATH = ROOT / "results" / "sasrec" / "ml-1m.json"
HEADLINE_PATH = ROOT / "results" / "ml-1m.json"
CHECKPOINT_DIR = ROOT / "models" / "sasrec"


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


def _checkpoint(cfg: dict[str, Any]) -> Path:
    digest = hashlib.sha256(config_key(cfg).encode()).hexdigest()[:16]
    return CHECKPOINT_DIR / f"{digest}.pt"


def _is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or "not enough memory" in text


def load_split() -> tuple[Any, SplitResult]:
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
    return config, split


def _test_view(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.full_train,
        test=split.test,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def _val_view(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.train,
        test=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


class Clock:
    def __init__(self, already: float) -> None:
        self.already = float(already)
        self.t0 = time.perf_counter()

    def elapsed(self) -> float:
        return self.already + (time.perf_counter() - self.t0)


def run_probe(split: SplitResult, relevance: float) -> dict[str, Any]:
    """Two timed epochs. Ranking metrics are not computed and not stored."""
    if PROBE_PATH.is_file():
        payload = _read(PROBE_PATH)
        if payload.get("ranking_metrics_recorded") is not False:
            raise RuntimeError("existing probe file recorded a ranking metric")
        print(
            f"Reusing probe t50={payload.get('t50_sec')} t200={payload.get('t200_sec')}",
            flush=True,
        )
        return payload

    from movielens_recommender.sasrec.plan import fixed_hyperparams

    def _cfg(length: int) -> dict[str, Any]:
        row = fixed_hyperparams()
        row.update({"dropout": 0.5, "max_seq_len": length, "n_layers": 2})
        return row

    t200: float | None = None
    oom_200 = False
    print("Probe: one epoch at max_seq_len=200, n_layers=2...", flush=True)
    try:
        t200 = time_one_epoch(
            split.train,
            split.val,
            hyperparams=_cfg(200),
            seed=PRIMARY_SEED,
            relevance_threshold=relevance,
        )
    except (MemoryError, RuntimeError) as exc:
        if not _is_oom(exc):
            raise
        oom_200 = True
        print(f"Probe length 200 ran out of memory: {exc}", flush=True)
    release_memory()
    print("Probe: one epoch at max_seq_len=50, n_layers=2...", flush=True)
    t50 = time_one_epoch(
        split.train,
        split.val,
        hyperparams=_cfg(50),
        seed=PRIMARY_SEED,
        relevance_threshold=relevance,
    )
    release_memory()
    payload: dict[str, Any] = {
        "adr": "0019",
        "host": host_info(),
        "oom_200": oom_200,
        "ranking_metrics_recorded": False,
        "t50_sec": round(float(t50), 3),
        "t200_sec": None if t200 is None else round(float(t200), 3),
    }
    blob = json.dumps(payload).lower()
    if "ndcg" in blob or "recall" in blob:
        raise RuntimeError("probe payload contains a ranking metric name")
    _write(PROBE_PATH, payload)
    print(
        f"Probe wrote {PROBE_PATH} t50={payload['t50_sec']} t200={payload['t200_sec']}",
        flush=True,
    )
    return payload


def _tail_seconds(decision: dict[str, Any]) -> float:
    return float(
        decision["edge_reserve_sec"]
        + decision["test_reserve_sec"]
        + decision["shuffle_reserve_sec"]
        + decision["comparison_reserve_sec"]
    )


def _train_trial(
    frame: pd.DataFrame,
    val: pd.DataFrame | None,
    template: SplitResult | None,
    cfg: dict[str, Any],
    *,
    relevance: float,
    shuffle_seed: int | None = None,
    n_epochs: int | None = None,
) -> tuple[SASRecRecommender, Any]:
    recommender, _data, result = train_sasrec(
        frame,
        hyperparams=cfg,
        seed=PRIMARY_SEED,
        relevance_threshold=relevance,
        val_frame=val,
        val_template=template,
        n_epochs=n_epochs,
        shuffle_seed=shuffle_seed,
        show_progress=True,
    )
    return recommender, result


def _save_checkpoint(cfg: dict[str, Any], model: torch.nn.Module) -> None:
    path = _checkpoint(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def _load_recommender(
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    relevance: float,
    shuffle_seed: int | None = None,
) -> SASRecRecommender:
    data = build_sequence_data(
        frame,
        max_seq_len=int(cfg["max_seq_len"]),
        relevance_threshold=relevance,
        shuffle_seed=shuffle_seed,
    )
    model = SASRecModel(
        n_items=int(len(data.item_ids)),
        d_model=int(cfg["d_model"]),
        n_layers=int(cfg["n_layers"]),
        n_heads=int(cfg["n_heads"]),
        dropout=float(cfg["dropout"]),
        max_seq_len=int(cfg["max_seq_len"]),
    )
    state = torch.load(_checkpoint(cfg), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return SASRecRecommender(model, data, device="cpu")


def _trial_record(cfg: dict[str, Any], result: Any) -> dict[str, Any]:
    if result.best_val_ndcg10 is None:
        raise RuntimeError("validation trial did not record NDCG@10")
    return {
        "best_epoch": int(result.best_epoch),
        "epochs_trained": int(result.epochs_trained),
        "history": result.history,
        "hyperparams": cfg,
        "val_ndcg@10": round(float(result.best_val_ndcg10), 6),
        "wall_sec": float(result.wall_time_sec),
    }


def run_tune(split: SplitResult, probe: dict[str, Any], relevance: float) -> dict[str, Any]:
    if probe.get("oom_200") or probe.get("t200_sec") is None:
        t200 = float("inf")
    else:
        t200 = float(probe["t200_sec"])
    decision = lock_budget(float(probe["t50_sec"]), t200)
    print(
        f"Locked grid n={decision['n_configs']} cut={decision['cut']!r} "
        f"projected={decision['projected_sec']} fits={decision['fits']}",
        flush=True,
    )
    if not decision["fits"]:
        payload = {
            "adr": "0019",
            "budget": _budget_block(probe, decision, truncated=False),
            "dataset": "ml-1m",
            "edge_extension_trials": [],
            "search_started": False,
            "trials": [],
        }
        _write(TUNING_PATH, payload)
        raise SystemExit(
            "ADR-0019 cut 4 still exceeds 10800s. Amend the ADR before any validation NDCG."
        )

    existing_trials: list[dict[str, Any]] = []
    existing_edges: list[dict[str, Any]] = []
    if TUNING_PATH.is_file():
        prior = _read(TUNING_PATH)
        if prior.get("search_started") and prior.get("budget", {}).get("cut") == decision["cut"]:
            existing_trials = list(prior.get("trials") or [])
            existing_edges = list(prior.get("edge_extension_trials") or [])
            print(
                f"Resuming tune with {len(existing_trials)} base trials "
                f"and {len(existing_edges)} edge trials",
                flush=True,
            )

    done = {config_key(row["hyperparams"]): row for row in existing_trials}
    edge_done = {config_key(row["hyperparams"]): row for row in existing_edges}
    clock = Clock(
        float(probe["t50_sec"])
        + (0.0 if probe.get("t200_sec") is None else float(probe["t200_sec"]))
        + sum(float(row["wall_sec"]) for row in existing_trials)
        + sum(float(row["wall_sec"]) for row in existing_edges)
    )
    tail = _tail_seconds(decision)
    truncated = False
    trials: list[dict[str, Any]] = []
    for cfg in decision["grid"]:
        key = config_key(cfg)
        if key in done:
            trials.append(done[key])
            continue
        proj = projected_trial_seconds(cfg, t50=float(probe["t50_sec"]), t200=t200)
        if clock.elapsed() + proj + tail > BUDGET_SEC:
            truncated = True
            print("Stopping the search to keep the tail reserve.", flush=True)
            break
        print(f"Trial {cfg['max_seq_len']=} {cfg['n_layers']=} {cfg['dropout']=}", flush=True)
        recommender, result = _train_trial(
            split.train, split.val, split, cfg, relevance=relevance
        )
        _save_checkpoint(cfg, recommender.model)
        del recommender
        release_memory()
        record = _trial_record(cfg, result)
        trials.append(record)
        _write(TUNING_PATH, _tuning_payload(probe, decision, trials, [], truncated=False))

    edges: list[dict[str, Any]] = []
    if trials and not truncated:
        base_winner = select_winner(trials)
        extensions = edge_extension_configs(
            base_winner["hyperparams"],
            [row["hyperparams"] for row in trials],
        )
        for cfg in extensions:
            key = config_key(cfg)
            if key in edge_done:
                edges.append(edge_done[key])
                continue
            proj = projected_trial_seconds(cfg, t50=float(probe["t50_sec"]), t200=t200)
            later = sum(
                projected_trial_seconds(other, t50=float(probe["t50_sec"]), t200=t200)
                for other in extensions[len(edges) + 1 :]
            )
            reserve = (
                decision["test_reserve_sec"]
                + decision["shuffle_reserve_sec"]
                + COMPARISON_RESERVE_SEC
                + later
            )
            if clock.elapsed() + proj + reserve > BUDGET_SEC:
                truncated = True
                print("Skipping remaining edge steps to keep the tail reserve.", flush=True)
                break
            print(
                f"Edge trial {cfg['max_seq_len']=} {cfg['n_layers']=} {cfg['dropout']=}",
                flush=True,
            )
            recommender, result = _train_trial(
                split.train, split.val, split, cfg, relevance=relevance
            )
            _save_checkpoint(cfg, recommender.model)
            del recommender
            release_memory()
            edges.append(_trial_record(cfg, result))
            _write(
                TUNING_PATH,
                _tuning_payload(probe, decision, trials, edges, truncated=False),
            )

    if not trials:
        raise RuntimeError("tune finished with no scored trial")
    payload = _tuning_payload(probe, decision, trials, edges, truncated=truncated)
    payload["selected"] = _selected_block(trials, edges)
    _write(TUNING_PATH, payload)
    print(
        f"Selected val NDCG@10={payload['selected']['val_ndcg@10']} "
        f"epoch={payload['selected']['best_epoch']}",
        flush=True,
    )
    return payload


def _budget_block(
    probe: dict[str, Any], decision: dict[str, Any], *, truncated: bool
) -> dict[str, Any]:
    return {
        "cap_sec": BUDGET_SEC,
        "comparison_reserve_sec": decision["comparison_reserve_sec"],
        "cut": decision["cut"],
        "ease_configs": EASE_CONFIGS,
        "edge_reserve_sec": decision["edge_reserve_sec"],
        "fits": bool(decision["fits"]),
        "grid_sec": decision["grid_sec"],
        "max_epochs": int(decision["max_epochs"]),
        "n_configs": int(decision["n_configs"]),
        "oom_200": bool(probe.get("oom_200")),
        "probe_ranking_metrics_recorded": False,
        "projected_sec": decision["projected_sec"],
        "shuffle_reserve_sec": decision["shuffle_reserve_sec"],
        "t200_sec": probe.get("t200_sec"),
        "t50_sec": probe.get("t50_sec"),
        "test_reserve_sec": decision["test_reserve_sec"],
        "test_seeds": list(decision["test_seeds"]),
        "truncated": bool(truncated),
        "two_tower_grid_configs": TWO_TOWER_GRID_CONFIGS,
    }


def _tuning_payload(
    probe: dict[str, Any],
    decision: dict[str, Any],
    trials: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    truncated: bool,
) -> dict[str, Any]:
    budget = _budget_block(probe, decision, truncated=truncated)
    budget["n_edge_extensions"] = len(edges)
    budget["n_trials_scored"] = len(trials)
    return {
        "adr": "0019",
        "budget": budget,
        "dataset": "ml-1m",
        "edge_extension_trials": edges,
        "grid": decision["grid"],
        "search_started": True,
        "trials": trials,
    }


def _selected_block(trials: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    base = select_winner(trials)
    base_edges = find_edges(base["hyperparams"], [row["hyperparams"] for row in trials])
    final = select_winner(list(trials) + list(edges))
    scored = [row["hyperparams"] for row in trials] + [row["hyperparams"] for row in edges]
    grid_edges = find_edges(final["hyperparams"], scored)
    return {
        "base_grid_edges": base_edges,
        "best_epoch": int(final["best_epoch"]),
        "grid_edges": grid_edges,
        "hyperparams": final["hyperparams"],
        "on_grid_edge": bool(grid_edges),
        "val_ndcg@10": float(final["val_ndcg@10"]),
    }


def _store_pair(left: list[float], right: list[float]) -> dict[str, Any]:
    raw = paired_bootstrap_delta_ci(
        left,
        right,
        n_bootstrap=N_BOOTSTRAP,
        alpha=BOOTSTRAP_ALPHA,
        seed=BOOTSTRAP_SEED,
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


def _published(headline: dict[str, Any]) -> dict[str, Any]:
    two_tower = headline["metrics"]["two_tower"]
    lambdarank = headline["metrics"]["lambdarank"]
    tt_ci = two_tower["per_seed_confidence_intervals"]["42"]["ndcg@10"]
    lr_ci = lambdarank["per_seed_confidence_intervals"]["42"]["ndcg@10"]
    return {
        "lambdarank": {
            "ndcg@10": lambdarank["ndcg@10"],
            "primary_seed_ci": lr_ci,
            "primary_seed_ndcg@10": lr_ci["mean"],
        },
        "source": "results/ml-1m.json",
        "two_tower": {
            "ndcg@10": two_tower["ndcg@10"],
            "primary_seed_ci": tt_ci,
            "primary_seed_ndcg@10": tt_ci["mean"],
        },
    }


def _two_tower_hp(headline: dict[str, Any]) -> dict[str, Any]:
    raw = headline["hyperparameters"]["two_tower"]
    return {
        "batch_size": int(raw["batch_size"]),
        "embedding_dim": int(raw["embedding_dim"]),
        "learning_rate": float(raw["learning_rate"]),
        "max_epochs": int(raw["max_epochs"]),
        "max_history": int(raw["max_history"]),
        "patience": int(raw["patience"]),
        "temperature": float(raw["temperature"]),
        "weight_decay": float(raw["weight_decay"]),
    }


def run_test(split: SplitResult, tuning: dict[str, Any], relevance: float) -> dict[str, Any]:
    headline_sha = _sha256(HEADLINE_PATH)
    headline = _read(HEADLINE_PATH)
    selected = tuning["selected"]
    cfg = dict(selected["hyperparams"])
    seeds = [int(s) for s in tuning["budget"]["test_seeds"]]
    if PRIMARY_SEED not in seeds:
        raise RuntimeError("primary seed 42 is not in the locked test seeds")
    view = _test_view(split)
    val_view = _val_view(split)

    print("Validation bootstrap on the early-stopped fit-train checkpoint...", flush=True)
    val_model = _load_recommender(split.train, cfg, relevance=relevance)
    val_metrics = evaluate_split(
        val_model.recommend,
        split.train,
        split.val,
        relevance_threshold=relevance,
        split=val_view,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_alpha=BOOTSTRAP_ALPHA,
        seed=BOOTSTRAP_SEED,
    )
    if round(float(val_metrics["ndcg@10"]), 6) != round(float(selected["val_ndcg@10"]), 6):
        raise RuntimeError(
            "reloaded checkpoint validation NDCG@10 does not match the stored trial: "
            f"{val_metrics['ndcg@10']} vs {selected['val_ndcg@10']}"
        )
    del val_model
    release_memory()

    per_seed: list[dict[str, Any]] = []
    primary_users: list[int] | None = None
    primary_values: list[float] | None = None
    primary_metrics: dict[str, Any] | None = None
    for seed in seeds:
        print(f"Test refit seed {seed} for {selected['best_epoch']} epochs...", flush=True)
        recommender, result = _train_trial(
            split.full_train,
            None,
            None,
            cfg,
            relevance=relevance,
            n_epochs=int(selected["best_epoch"]),
        )
        if seed == PRIMARY_SEED:
            primary_metrics = evaluate_split(
                recommender.recommend,
                split.full_train,
                split.test,
                relevance_threshold=relevance,
                split=view,
                n_bootstrap=N_BOOTSTRAP,
                bootstrap_alpha=BOOTSTRAP_ALPHA,
                seed=BOOTSTRAP_SEED,
            )
            primary_users, primary_values = per_user_ndcg(
                recommender.recommend,
                split.full_train,
                split.test,
                relevance_threshold=relevance,
                split=view,
            )
            point = float(primary_metrics["ndcg@10"])
        else:
            users, values = per_user_ndcg(
                recommender.recommend,
                split.full_train,
                split.test,
                relevance_threshold=relevance,
                split=view,
            )
            if primary_users is not None and users != primary_users:
                raise RuntimeError("test user order changed across seeds")
            point = float(np.mean(values))
        per_seed.append(
            {
                "epochs_trained": int(result.epochs_trained),
                "ndcg@10": round(point, 6),
                "seed": int(seed),
                "wall_sec": float(result.wall_time_sec),
            }
        )
        del recommender
        release_memory()

    assert primary_metrics is not None and primary_users is not None and primary_values is not None
    points = [float(row["ndcg@10"]) for row in per_seed]
    print("Shuffled-order refit...", flush=True)
    shuffled, shuffled_train = _train_trial(
        split.full_train,
        None,
        None,
        cfg,
        relevance=relevance,
        shuffle_seed=SHUFFLE_SEED,
        n_epochs=int(selected["best_epoch"]),
    )
    shuffled_metrics = evaluate_split(
        shuffled.recommend,
        split.full_train,
        split.test,
        relevance_threshold=relevance,
        split=view,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_alpha=BOOTSTRAP_ALPHA,
        seed=BOOTSTRAP_SEED,
    )
    shuffled_users, shuffled_values = per_user_ndcg(
        shuffled.recommend,
        split.full_train,
        split.test,
        relevance_threshold=relevance,
        split=view,
    )
    del shuffled
    release_memory()
    if shuffled_users != primary_users:
        raise RuntimeError("shuffled eval users do not match SASRec")

    print("Paired refits of the two-tower and LambdaRank headlines...", flush=True)
    ranker_hp = headline["hyperparameters"]["lambdarank"]
    refs = score_headline_references(
        split,
        dataset="ml-1m",
        data_dir=str(ROOT / "data"),
        relevance_threshold=relevance,
        two_tower_hp=_two_tower_hp(headline),
        two_tower_epochs=int(headline["hyperparameters"]["two_tower"]["refit_epochs"]),
        item_item_hp=dict(ranker_hp["retriever_item_item"]),
        ranker_hp=ranker_hp,
        n_bootstrap=N_BOOTSTRAP,
        bootstrap_alpha=BOOTSTRAP_ALPHA,
        seed=PRIMARY_SEED,
    )
    same_users = refs["two_tower"]["user_ids"] == primary_users
    same_users = same_users and refs["lambdarank"]["user_ids"] == primary_users
    if not same_users:
        raise RuntimeError("reference eval users do not match SASRec")

    real_minus_shuffled = _store_pair(shuffled_values, primary_values)
    order_helps = bool(real_minus_shuffled["low"] > 0.0)
    payload = {
        "adr": "0019",
        "budget": tuning["budget"],
        "dataset": "ml-1m",
        "decision": {
            "negative_result": not order_helps,
            "order_helps": order_helps,
            "rule": (
                "negative when the real-minus-shuffled test NDCG@10 interval "
                "includes 0 or lies at or below 0"
            ),
        },
        "edge_extension_trials": [
            {"hyperparams": row["hyperparams"], "val_ndcg@10": row["val_ndcg@10"]}
            for row in tuning.get("edge_extension_trials") or []
        ],
        "host": host_info(),
        "library_versions": library_versions(),
        "models": {
            "lambdarank": {
                "best_iteration": refs["lambdarank"]["best_iteration"],
                "candidate_set": "two_tower",
                "demographics": "off",
                "metrics": refs["lambdarank"]["metrics"],
                "ndcg@10": refs["lambdarank"]["metrics"]["ndcg@10"],
                "ndcg@10_ci": refs["lambdarank"]["metrics"]["confidence_intervals"]["ndcg@10"],
                "seed": PRIMARY_SEED,
            },
            "sasrec": {
                "metrics": primary_metrics,
                "ndcg@10_mean": round(float(np.mean(points)), 6),
                "ndcg@10_std": round(float(np.std(points, ddof=0)), 6),
                "per_seed": per_seed,
                "primary_ndcg@10_ci": primary_metrics["confidence_intervals"]["ndcg@10"],
            },
            "sasrec_shuffled": {
                "epochs_trained": int(shuffled_train.epochs_trained),
                "metrics": shuffled_metrics,
                "ndcg@10": shuffled_metrics["ndcg@10"],
                "ndcg@10_ci": shuffled_metrics["confidence_intervals"]["ndcg@10"],
                "seed": PRIMARY_SEED,
                "shuffle_seed": SHUFFLE_SEED,
                "wall_sec": float(shuffled_train.wall_time_sec),
            },
            "two_tower": {
                "epochs": refs["two_tower"]["epochs"],
                "loss": "in_batch",
                "metrics": refs["two_tower"]["metrics"],
                "ndcg@10": refs["two_tower"]["metrics"]["ndcg@10"],
                "ndcg@10_ci": refs["two_tower"]["metrics"]["confidence_intervals"]["ndcg@10"],
                "seed": PRIMARY_SEED,
                "train_wall_sec": refs["two_tower"]["train_wall_sec"],
            },
        },
        "paired": {
            "real_minus_shuffled": real_minus_shuffled,
            "sasrec_minus_lambdarank": _store_pair(refs["lambdarank"]["ndcg"], primary_values),
            "sasrec_minus_two_tower": _store_pair(refs["two_tower"]["ndcg"], primary_values),
        },
        "published": _published(headline),
        "seeds": {
            "bootstrap": BOOTSTRAP_SEED,
            "epoch_shuffle": "seed + epoch * 1000003",
            "primary": PRIMARY_SEED,
            "shuffle": SHUFFLE_SEED,
            "test": seeds,
            "tuning": PRIMARY_SEED,
        },
        "selected": {
            **selected,
            "val_ndcg@10_ci": val_metrics["confidence_intervals"]["ndcg@10"],
        },
        "split": split.summary(),
    }
    _write(RESULTS_PATH, payload)
    if _sha256(HEADLINE_PATH) != headline_sha:
        raise RuntimeError("results/ml-1m.json changed during the SASRec run")
    print(f"Wrote {RESULTS_PATH}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0019 SASRec on ml-1m")
    parser.add_argument("--stage", choices=("probe", "tune", "test", "all"), default="all")
    args = parser.parse_args()
    headline_sha = _sha256(HEADLINE_PATH)
    _config, split = load_split()
    relevance = float(split.config.relevance_threshold)
    if args.stage in {"probe", "tune", "all"}:
        probe = run_probe(split, relevance)
    else:
        probe = _read(PROBE_PATH)
    if args.stage == "probe":
        if _sha256(HEADLINE_PATH) != headline_sha:
            raise RuntimeError("results/ml-1m.json changed during the probe")
        return
    if args.stage in {"tune", "all"}:
        tuning = run_tune(split, probe, relevance)
    else:
        tuning = _read(TUNING_PATH)
    if args.stage in {"test", "all"}:
        if "selected" not in tuning:
            raise RuntimeError("tune did not select a config")
        run_test(split, tuning, relevance)
    if _sha256(HEADLINE_PATH) != headline_sha:
        raise RuntimeError("results/ml-1m.json changed")


if __name__ == "__main__":
    main()
