#!/usr/bin/env python3
"""S3e: full-softmax and sampled-softmax two-tower (ADR-0013).

Stages, in order: probe (timings only), lock (grid from the probe), tune
(validation NDCG@10), test (sealed). The lock file is required before tune
or test. Checkpoints live under data/, which is gitignored.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.config import RunConfig, load_config
from movielens_recommender.data import DATASET_SHA256, load_ratings, load_users
from movielens_recommender.evaluate import (
    collect_recommendations,
    evaluate_recommender,
    format_metrics,
)
from movielens_recommender.metrics import ndcg_at_k, paired_bootstrap_delta_ci
from movielens_recommender.movies import load_movies
from movielens_recommender.scale import apply_eval_user_sample, release_memory
from movielens_recommender.segments import filter_recs_to_segment, head_tail_items
from movielens_recommender.split import SplitConfig, SplitResult, time_based_split
from movielens_recommender.two_tower.plan import grid_edges, lock_plan

ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "results" / "budget" / "two-tower-v2-probe.json"
LOCK_PATH = ROOT / "results" / "budget" / "two-tower-v2.json"
CHECKPOINT_DIR = ROOT / "data" / "two-tower-v2-checkpoints"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rss_mib() -> float:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024.0, 1)
    return 0.0


def _host() -> dict[str, Any]:
    info: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, rest = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "CommitLimit", "SwapTotal"}:
            info[key] = round(int(rest.split()[0]) / 1024.0, 1)
    return {"cpu_count": os.cpu_count(), "mem_mib": info, "rss_mib": _rss_mib()}


def load_split(dataset: str) -> tuple[RunConfig, pd.DataFrame, SplitResult]:
    config = load_config(ROOT / "configs" / f"{dataset}.yaml")
    ratings, _stats = load_ratings(dataset, ROOT / "data", clean=True)
    movies = load_movies(dataset, ROOT / "data")
    split = time_based_split(
        ratings,
        SplitConfig(
            min_ratings=config.split.min_ratings,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            relevance_threshold=config.eval.relevance_threshold,
        ),
    )
    sample_n = int(config.eval.user_sample_size)
    if sample_n > 0:
        sample_seed = (
            config.seed
            if config.eval.user_sample_seed is None
            else int(config.eval.user_sample_seed)
        )
        split = apply_eval_user_sample(
            split,
            n_users=sample_n,
            seed=sample_seed,
            relevance_threshold=config.eval.relevance_threshold,
        )
    return config, movies, split


def _eval_split(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.full_train,
        test=split.test,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def _val_view(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.train,
        test=split.val,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def _round_map(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            out[key] = round(value, 6)
        elif isinstance(value, np.floating):
            out[key] = round(float(value), 6)
        else:
            out[key] = value
    return out


def probe_dataset(dataset: str, *, n_batches: int = 20) -> dict[str, Any]:
    """Time losses. Does not compute NDCG or any other ranking metric."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from movielens_recommender.two_tower.features import build_features
    from movielens_recommender.two_tower.model import TwoTowerModel
    from movielens_recommender.two_tower.recommender import TwoTowerRecommender
    from movielens_recommender.two_tower.train import (
        _batch_loss,
        build_catalog_tensors,
        set_torch_seed,
    )

    torch.set_num_threads(os.cpu_count() or 1)
    config, movies, split = load_split(dataset)
    t0 = time.perf_counter()
    features = build_features(
        split.train,
        dataset=dataset,
        data_dir=str(ROOT / "data"),
        movies=movies,
        relevance_threshold=config.eval.relevance_threshold,
        max_history=50,
    )
    build_sec = time.perf_counter() - t0
    device = torch.device("cpu")
    dataset_t = TensorDataset(
        torch.from_numpy(features.pos_user_idx),
        torch.from_numpy(features.pos_item_idx),
    )
    n_pos = int(len(dataset_t))
    batch_size = 1024
    drop_last = n_pos > batch_size
    if drop_last:
        batches_per_epoch = n_pos // batch_size
    else:
        batches_per_epoch = max(1, int(np.ceil(n_pos / batch_size)))
    catalog = build_catalog_tensors(features, device)

    def time_loss(loss_name: str, feats: Any, catalog_tensors: Any) -> dict[str, Any]:
        set_torch_seed(42)
        model = TwoTowerModel(
            n_users=feats.n_users,
            n_items=feats.n_items,
            n_genres=feats.n_genres,
            embedding_dim=64,
            temperature=0.1,
        ).to(device)
        loader = DataLoader(dataset_t, batch_size=batch_size, shuffle=True, drop_last=drop_last)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        model.train()
        warmup = 5
        seen = 0
        done = 0
        elapsed = 0.0
        t1 = time.perf_counter()
        for user_idx, item_idx in loader:
            user_idx = user_idx.to(device)
            item_idx = item_idx.to(device)
            opt.zero_grad(set_to_none=True)
            kwargs: dict[str, Any] = {}
            if loss_name != "in_batch":
                kwargs = {"loss": loss_name, "n_negatives": 256, "catalog": catalog_tensors}
            loss = _batch_loss(
                model, feats, user_idx, item_idx, device, 50, **kwargs
            )
            loss.backward()
            opt.step()
            seen += 1
            if seen == warmup:
                t1 = time.perf_counter()
            elif seen > warmup:
                done += 1
                if done >= n_batches:
                    break
        elapsed = time.perf_counter() - t1
        per = elapsed / max(done, 1)
        out = {
            "warmup_batches": warmup,
            "batches_timed": done,
            "batch_wall_sec": round(elapsed, 3),
            "sec_per_batch": round(per, 4),
            "extrapolated_epoch_train_sec": round(per * batches_per_epoch, 1),
            "rss_mib": _rss_mib(),
        }
        del model, opt, loader
        release_memory()
        return out

    losses = {
        name: time_loss(name, features, catalog)
        for name in ("in_batch", "full_softmax", "sampled_softmax")
    }
    set_torch_seed(0)
    model = TwoTowerModel(
        n_users=features.n_users,
        n_items=features.n_items,
        n_genres=features.n_genres,
        embedding_dim=64,
        temperature=0.1,
    ).to(device)
    rec = TwoTowerRecommender.from_trained(model, features, max_history=50, device="cpu")
    sample_users = [int(u) for u in list(features.user_ids[:100])]
    t2 = time.perf_counter()
    for uid in sample_users:
        rec.recommend(uid, 10)
    rec_sec = time.perf_counter() - t2
    n_val = int(split.val["user_id"].nunique()) if split.val is not None else 0
    per_user = rec_sec / max(len(sample_users), 1)
    block = {
        "dataset": dataset,
        "dataset_sha256": DATASET_SHA256[dataset],
        "n_items": int(features.n_items),
        "n_users": int(features.n_users),
        "n_positives": n_pos,
        "batch_size": batch_size,
        "batches_per_epoch": int(batches_per_epoch),
        "drop_last": bool(drop_last),
        "feature_build_sec": round(build_sec, 3),
        "logit_mib_full_softmax_batch": round(batch_size * features.n_items * 4 / (1024 * 1024), 1),
        "losses": losses,
        "recommend_users_timed": len(sample_users),
        "recommend_wall_sec": round(rec_sec, 3),
        "n_val_users": n_val,
        "val_score_sec_per_epoch": round(per_user * n_val, 1),
        "eval_user_sample": split.eval_user_sample,
        "rss_mib": _rss_mib(),
    }
    del model, rec, features, catalog
    release_memory()
    return block


def cmd_probe(dataset: str) -> None:
    print(f"Probing {dataset} (no ranking metric)...", flush=True)
    block = probe_dataset(dataset)
    doc = _read(PROBE_PATH) if PROBE_PATH.is_file() else {
        "experiment": "s3e_full_softmax_two_tower",
        "ranking_metrics": None,
        "host": _host(),
        "datasets": {},
    }
    doc["host"] = _host()
    doc["datasets"][dataset] = block
    _write(PROBE_PATH, doc)
    print(f"Wrote {PROBE_PATH}", flush=True)
    for name, row in block["losses"].items():
        print(
            f"  {name}: {row['sec_per_batch']} s/batch, "
            f"epoch {row['extrapolated_epoch_train_sec']} s",
            flush=True,
        )


def _published_reference_epochs() -> dict[str, int]:
    out = {}
    for dataset in ("ml-1m", "ml-32m"):
        path = ROOT / "results" / "tuning" / f"two_tower_{dataset}.json"
        doc = _read(path)
        out[dataset] = int(doc["two_tower"]["best_epoch"])
    return out


def cmd_lock() -> None:
    if not PROBE_PATH.is_file():
        raise SystemExit(f"Missing probe file {PROBE_PATH}")
    probe = _read(PROBE_PATH)
    missing = [name for name in ("ml-1m", "ml-32m") if name not in probe.get("datasets", {})]
    if missing:
        raise SystemExit(f"Probe is missing datasets: {missing}")
    plan = lock_plan(probe, _published_reference_epochs())
    plan["probe_file"] = "results/budget/two-tower-v2-probe.json"
    plan["host"] = probe.get("host")
    plan["probe_datasets"] = {
        name: {
            "n_items": block.get("n_items"),
            "n_positives": block.get("n_positives"),
            "batches_per_epoch": block.get("batches_per_epoch"),
            "logit_mib_full_softmax_batch": block.get("logit_mib_full_softmax_batch"),
            "losses": block.get("losses"),
            "val_score_sec_per_epoch": block.get("val_score_sec_per_epoch"),
            "eval_user_sample": block.get("eval_user_sample"),
        }
        for name, block in probe["datasets"].items()
    }
    _write(LOCK_PATH, plan)
    print(f"Wrote {LOCK_PATH}", flush=True)
    for name, block in plan["datasets"].items():
        test = block["test"]
        print(
            f"  {name}: {block['n_trials']} trials, cap {block['tune_cap_sec']}s, "
            f"seeds {test['seeds']}, test_both={test['test_both_losses']}",
            flush=True,
        )


def _locked(dataset: str) -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise SystemExit(f"Missing locked budget {LOCK_PATH}. Run lock before tune or test.")
    plan = _read(LOCK_PATH)
    if plan.get("status") != "locked":
        raise SystemExit("Budget file is not locked")
    if dataset not in plan["datasets"]:
        raise SystemExit(f"Locked budget has no grid for {dataset}")
    return plan


def _published_tuning(dataset: str) -> dict[str, Any]:
    return _read(ROOT / "results" / "tuning" / f"two_tower_{dataset}.json")["two_tower"]


def _tuning_path(dataset: str) -> Path:
    return ROOT / "results" / "tuning" / f"two_tower_v2_{dataset}.json"


def cmd_tune(dataset: str) -> None:
    import torch

    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    torch.set_num_threads(os.cpu_count() or 1)
    plan = _locked(dataset)
    block = plan["datasets"][dataset]
    config, movies, split = load_split(dataset)
    published = _published_tuning(dataset)
    reference_val = float(published["best_val_score"])
    path = _tuning_path(dataset)
    doc = _read(path) if path.is_file() else {
        "dataset": dataset,
        "dataset_sha256": DATASET_SHA256[dataset],
        "seed": config.seed,
        "protocol": (
            "Fit each trial on fit-train only. Select by validation NDCG@10. "
            "The test split is not read. ADR-0013."
        ),
        "reference_val_ndcg@10": reference_val,
        "reference_source": f"results/tuning/two_tower_{dataset}.json",
        "grid": block["grid"],
        "trials": [],
        "status": "running",
    }
    done = {json.dumps(row["hyperparams"], sort_keys=True) for row in doc["trials"]}
    val_split = _val_view(split)
    t0 = time.perf_counter()
    for i, hp in enumerate(block["grid"], start=1):
        key = json.dumps(hp, sort_keys=True)
        if key in done:
            print(f"  skip trial {i}/{len(block['grid'])}", flush=True)
            continue
        print(f"  trial {i}/{len(block['grid'])}: {hp}", flush=True)
        _rec, _feat, result = fit_two_tower_recommender(
            split.train,
            dataset=dataset,
            data_dir=str(ROOT / "data"),
            movies=movies,
            hyperparams=hp,
            seed=config.seed,
            relevance_threshold=config.eval.relevance_threshold,
            val_split=val_split,
            show_progress=True,
        )
        score = float("-inf") if result.best_val_ndcg10 is None else float(result.best_val_ndcg10)
        doc["trials"].append(
            {
                "hyperparams": hp,
                "val_ndcg@10": round(score, 6),
                "best_epoch": int(result.best_epoch),
                "epochs_trained": int(result.epochs_trained),
                "wall_time_sec": result.wall_time_sec,
            }
        )
        doc["tune_wall_sec"] = round(time.perf_counter() - t0, 3)
        _write(path, doc)
        del _rec, _feat, result
        release_memory()
        done.add(key)
    best_by_loss: dict[str, Any] = {}
    for row in doc["trials"]:
        loss_name = row["hyperparams"]["loss"]
        prev = best_by_loss.get(loss_name)
        if prev is None or float(row["val_ndcg@10"]) > float(prev["val_ndcg@10"]):
            best_by_loss[loss_name] = {
                "hyperparams": row["hyperparams"],
                "val_ndcg@10": row["val_ndcg@10"],
                "best_epoch": row["best_epoch"],
                "grid_edges": grid_edges(row["hyperparams"], block["grid"]),
            }
    selected_loss = max(best_by_loss, key=lambda name: float(best_by_loss[name]["val_ndcg@10"]))
    selected_val = float(best_by_loss[selected_loss]["val_ndcg@10"])
    doc["best_by_loss"] = best_by_loss
    doc["selected_loss"] = selected_loss
    doc["selected_val_ndcg@10"] = selected_val
    doc["beats_reference_validation"] = bool(selected_val > reference_val)
    doc["status"] = "complete"
    doc["tune_wall_sec"] = round(
        sum(float(row["wall_time_sec"]) for row in doc["trials"]), 3
    )
    _write(path, doc)
    print(
        f"Selected {selected_loss} val {selected_val:.6f} "
        f"vs reference {reference_val:.6f} "
        f"(beats={doc['beats_reference_validation']})",
        flush=True,
    )


def _seed_summary(per_seed: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(row["metrics"][key]) for row in per_seed if key in row["metrics"]]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=0)), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "values": [round(float(v), 6) for v in vals],
        }
    return out


SUMMARY_KEYS = [
    "ndcg@10",
    "ndcg@20",
    "precision@10",
    "recall@10",
    "precision@20",
    "recall@20",
    "recall@100",
    "recall@200",
    "coverage@10",
    "mean_popularity@10",
]


def _headline(per_seed: list[dict[str, Any]], primary_seed: int) -> dict[str, Any]:
    across = _seed_summary(per_seed, SUMMARY_KEYS)
    primary = next(row for row in per_seed if int(row["seed"]) == primary_seed)
    headline = dict(primary["metrics"])
    for key, block in across.items():
        headline[key] = block["mean"]
    headline["seed_summary"] = across
    headline["n_seeds"] = len(per_seed)
    return headline


def _segment_value(metrics: dict[str, Any], which: str) -> float | None:
    cell = (
        (metrics.get("segments") or {})
        .get("item_head_tail", {})
        .get(which, {})
    )
    value = cell.get("ndcg@10")
    return None if value is None else float(value)


def _per_user_scores(
    recommend_fn: Any,
    full_train: pd.DataFrame,
    split: SplitResult,
    config: RunConfig,
) -> dict[str, Any]:
    from movielens_recommender.evaluate import _prepare_eval

    prepared = _eval_split(split)
    _split, seen, relevant = _prepare_eval(
        full_train,
        split.test,
        relevance_threshold=config.eval.relevance_threshold,
        split=prepared,
    )
    ks = sorted(set(int(k) for k in list(config.eval.ks) + list(config.eval.retrieval_ks)))
    recs = collect_recommendations(recommend_fn, seen, relevant, max_k=max(ks))
    users = sorted(int(uid) for uid in relevant)
    ndcg = [float(ndcg_at_k(recs[uid], relevant[uid], 10)) for uid in users]
    segments = head_tail_items(full_train)
    out: dict[str, Any] = {"users": users, "ndcg@10": ndcg}
    for which in ("head", "tail"):
        seg_users: list[int] = []
        seg_vals: list[float] = []
        items = segments[which]
        for uid in users:
            rel = relevant[uid] & items
            if not rel:
                continue
            seg_users.append(uid)
            seg_vals.append(
                float(ndcg_at_k(filter_recs_to_segment(recs[uid], items, 10), rel, 10))
            )
        out[which] = {"users": seg_users, "ndcg@10": seg_vals}
    return out


def _pair(
    reference: dict[str, Any],
    new: dict[str, Any],
    *,
    seed: int,
    n_bootstrap: int,
    alpha: float,
) -> dict[str, Any]:
    def one(which: str) -> dict[str, Any]:
        if which == "ndcg@10":
            left_u, left_v = reference["users"], reference["ndcg@10"]
            right_u, right_v = new["users"], new["ndcg@10"]
        else:
            left_u, left_v = reference[which]["users"], reference[which]["ndcg@10"]
            right_u, right_v = new[which]["users"], new[which]["ndcg@10"]
        left = {int(u): float(v) for u, v in zip(left_u, left_v, strict=True)}
        right = {int(u): float(v) for u, v in zip(right_u, right_v, strict=True)}
        common = sorted(set(left) & set(right))
        stats = paired_bootstrap_delta_ci(
            [left[u] for u in common],
            [right[u] for u in common],
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        return _round_map(
            {
                "comparison": "candidate minus reference",
                "mean": float(stats["mean"]),
                "low": float(stats["low"]),
                "high": float(stats["high"]),
                "n_users": int(stats["n_users"]),
                "excludes_zero": bool(stats["excludes_zero"]),
                "n_bootstrap": n_bootstrap,
                "alpha": alpha,
                "seed": seed,
            }
        )

    return {"ndcg@10": one("ndcg@10"), "head_ndcg@10": one("head"), "tail_ndcg@10": one("tail")}


def _fit_and_eval(
    *,
    full_train: pd.DataFrame,
    movies: pd.DataFrame,
    split: SplitResult,
    config: RunConfig,
    hyperparams: dict[str, Any],
    n_epochs: int,
    seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], float]:
    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    rec, _feat, train_result = fit_two_tower_recommender(
        full_train,
        dataset=config.dataset,
        data_dir=str(ROOT / "data"),
        movies=movies,
        hyperparams=hyperparams,
        seed=seed,
        relevance_threshold=config.eval.relevance_threshold,
        n_epochs=n_epochs,
        val_split=None,
        show_progress=True,
    )
    ks = sorted(set(int(k) for k in list(config.eval.ks) + list(config.eval.retrieval_ks)))
    metrics = format_metrics(
        evaluate_recommender(
            rec.recommend,
            full_train,
            split.test,
            relevance_threshold=config.eval.relevance_threshold,
            ks=tuple(ks),
            n_bootstrap=config.eval.n_bootstrap,
            bootstrap_alpha=config.eval.bootstrap_alpha,
            seed=config.seed,
            split=_eval_split(split),
            include_segments=True,
        )
    )
    per_user = _per_user_scores(rec.recommend, full_train, split, config)
    return rec, metrics, per_user, float(train_result.wall_time_sec)


def _published_reference_block(dataset: str) -> dict[str, Any]:
    published = _read(ROOT / "results" / f"{dataset}.json")
    tuning = _published_tuning(dataset)
    metrics = published["metrics"]
    tower = metrics["two_tower"]
    item = metrics["item_item_cosine"]
    ranker = metrics["lambdarank"]
    ci = tower["confidence_intervals"]["ndcg@10"]
    rank_ci = ranker["confidence_intervals"]["ndcg@10"]
    block: dict[str, Any] = {
        "source": f"results/{dataset}.json",
        "tuning_source": f"results/tuning/two_tower_{dataset}.json",
        "loss": "in_batch",
        "val_ndcg@10": float(tuning["best_val_score"]),
        "best_epoch": int(tuning["best_epoch"]),
        "hyperparams": dict(tuning["best_hyperparams"]),
        "ndcg@10": float(tower["ndcg@10"]),
        "ndcg@10_ci": {"low": float(ci["low"]), "high": float(ci["high"])},
        "n_seeds": int(
            len((tower.get("seed_summary") or {}).get("ndcg@10", {}).get("values", [])) or 1
        ),
        "recall@10": float(tower["recall@10"]),
        "recall@100": float(tower["recall@100"]),
        "recall@200": float(tower["recall@200"]),
        "coverage@10": float(tower["coverage@10"]),
        "head_ndcg@10": _segment_value(tower, "head"),
        "tail_ndcg@10": _segment_value(tower, "tail"),
        "item_item_ndcg@10": float(item["ndcg@10"]),
        "item_item_head_ndcg@10": _segment_value(item, "head"),
        "item_item_tail_ndcg@10": _segment_value(item, "tail"),
        "lambdarank_ndcg@10": float(ranker["ndcg@10"]),
        "lambdarank_ndcg@10_ci": {"low": float(rank_ci["low"]), "high": float(rank_ci["high"])},
        "lambdarank_source": f"results/{dataset}.json",
        "lambdarank_demographics": "off",
        "no_ranker_ndcg@10": float(metrics["no_ranker"]["ndcg@10"]),
    }
    summary = (tower.get("seed_summary") or {}).get("ndcg@10") or {}
    if "std" in summary:
        block["seed_std"] = float(summary["std"])
    if dataset == "ml-1m":
        demo = _read(ROOT / "results" / "demographics" / "ml-1m.json")
        block["production_ranker_ndcg@10"] = float(
            demo["variants"]["both"]["summary"]["ndcg@10_mean"]
        )
        block["production_ranker_source"] = "results/demographics/ml-1m.json"
        block["production_ranker_demographics"] = "both"
    return block


def _checkpoint(dataset: str) -> Path:
    return CHECKPOINT_DIR / f"{dataset}.json"


def _model_specs(
    dataset: str, tuning: dict[str, Any], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    published = _published_tuning(dataset)
    specs = [
        {
            "name": "in_batch",
            "role": "reference",
            "hyperparams": dict(published["best_hyperparams"]),
            "n_epochs": int(published["best_epoch"]),
        }
    ]
    test_both = bool(plan["datasets"][dataset]["test"]["test_both_losses"])
    names = list(tuning["best_by_loss"]) if test_both else [tuning["selected_loss"]]
    for name in names:
        chosen = tuning["best_by_loss"][name]
        specs.append(
            {
                "name": name,
                "role": "candidate",
                "hyperparams": dict(chosen["hyperparams"]),
                "n_epochs": int(chosen["best_epoch"]),
            }
        )
    return specs


def _compact_model(
    per_seed: list[dict[str, Any]], primary_seed: int, spec: dict[str, Any]
) -> dict[str, Any]:
    headline = _headline(per_seed, primary_seed)
    return {
        "role": spec["role"],
        "loss": spec["name"],
        "hyperparams": spec["hyperparams"],
        "n_epochs": spec["n_epochs"],
        "metrics": headline,
        "per_seed": [
            {
                "seed": int(row["seed"]),
                "ndcg@10": row["metrics"]["ndcg@10"],
                "ndcg@10_ci": row["metrics"]["confidence_intervals"]["ndcg@10"],
                "recall@10": row["metrics"].get("recall@10"),
                "recall@100": row["metrics"].get("recall@100"),
                "recall@200": row["metrics"].get("recall@200"),
                "coverage@10": row["metrics"].get("coverage@10"),
                "head_ndcg@10": _segment_value(row["metrics"], "head"),
                "tail_ndcg@10": _segment_value(row["metrics"], "tail"),
                "train_wall_time_sec": row["train_wall_sec"],
            }
            for row in per_seed
        ],
    }


def _run_rankers(
    *,
    dataset: str,
    config: RunConfig,
    movies: pd.DataFrame,
    split: SplitResult,
    tuning: dict[str, Any],
    towers: dict[str, Any],
) -> dict[str, Any]:
    """LambdaRank on the selected tower and on the reference tower.

    Candidates are that tower's top 200. Item-item is fit once and reused
    as a feature source. Demographics follow the dataset config.
    """
    from movielens_recommender.ranker.candidates import (
        PrecomputedRecommender,
        materialize_candidate_set,
        score_rank_maps,
    )
    from movielens_recommender.ranker.features import (
        assert_context_uses_only,
        assert_group_affinity_uses_only,
        build_feature_context,
        categorical_feature_names,
        relevance_pairs,
    )
    from movielens_recommender.ranker.pipeline import (
        _blocks_for_users,
        _evaluate_lists,
        _fit_item_item,
        _ranker_kwargs,
        _release_two_tower_pack,
        _rerank,
    )
    from movielens_recommender.ranker.train import (
        assemble_matrices,
        train_with_early_stop_then_refit,
    )
    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    published = _read(ROOT / "results" / f"{dataset}.json")
    item_hp = dict(published["hyperparameters"]["item_item_cosine_tuned"])
    rank_cfg = config.models.ranker
    k = int(rank_cfg.candidate_k)
    demo_mode = str(rank_cfg.demographics)
    users = load_users(dataset, ROOT / "data") if demo_mode != "off" else None
    primary_seed = int(config.seed)
    seeds = [int(s) for s in rank_cfg.seeds]
    full_train = split.full_train
    test_users = [int(uid) for uid in split.test["user_id"].unique()]
    val_users = [int(uid) for uid in split.val["user_id"].unique()]
    t0 = time.perf_counter()

    print("Fitting item-item on fit-train and full train...", flush=True)
    item_fit = _fit_item_item(split.train, item_hp)
    ii_val = item_fit.topk_with_scores(k, user_ids=val_users)
    del item_fit
    release_memory()
    item_full = _fit_item_item(full_train, item_hp)
    ii_full = item_full.topk_with_scores(k, user_ids=test_users)
    del item_full
    release_memory()

    selected = tuning["selected_loss"]
    published_hp = dict(_published_tuning(dataset)["best_hyperparams"])
    published_epochs = int(_published_tuning(dataset)["best_epoch"])
    jobs = [
        ("reference", "in_batch", published_hp, published_epochs),
        (
            "new",
            selected,
            dict(tuning["best_by_loss"][selected]["hyperparams"]),
            int(tuning["best_by_loss"][selected]["best_epoch"]),
        ),
    ]
    results: dict[str, Any] = {}
    for label, _loss_name, hp, epochs in jobs:
        print(f"Ranker fit-train tower ({label})...", flush=True)
        rec, _feat, _tr = fit_two_tower_recommender(
            split.train,
            dataset=dataset,
            data_dir=str(ROOT / "data"),
            movies=movies,
            hyperparams=hp,
            seed=primary_seed,
            relevance_threshold=config.eval.relevance_threshold,
            n_epochs=epochs,
            val_split=None,
            show_progress=True,
        )
        tt_val = rec.topk_with_scores(k, user_ids=val_users)
        _release_two_tower_pack(rec)
        del rec, _feat, _tr
        release_memory()
        tt_full = towers[label]["topk"]
        lists_val = materialize_candidate_set("two_tower", ii_val, tt_val, k)
        lists_full = materialize_candidate_set("two_tower", ii_full, tt_full, k)
        ctx_fit = build_feature_context(
            split.train, movies, users=users, mode=demo_mode, group_frame=split.train
        )
        assert_context_uses_only(ctx_fit, split.train, split.val)
        assert_context_uses_only(ctx_fit, split.train, split.test)
        if demo_mode in {"affinity", "both"}:
            assert users is not None
            assert_group_affinity_uses_only(
                ctx_fit,
                split.train,
                users,
                pd.concat([split.val, split.test], ignore_index=True),
            )
        positives = relevance_pairs(split.val, relevance_threshold=config.eval.relevance_threshold)
        blocks = _blocks_for_users(
            ctx_fit,
            lists_val,
            score_rank_maps(ii_val),
            score_rank_maps(tt_val),
            positives,
            users=set(val_users),
        )
        matrices = assemble_matrices(blocks, ctx_fit.names)
        matrices.categorical_features = categorical_feature_names(demo_mode)
        ctx_full = build_feature_context(
            full_train, movies, users=users, mode=demo_mode, group_frame=split.train
        )
        assert_context_uses_only(ctx_full, full_train, split.test)
        ii_maps = score_rank_maps(ii_full)
        tt_maps = score_rank_maps(tt_full)
        per_seed = []
        primary_per_user = None
        for seed in seeds:
            print(f"  LambdaRank {label} seed {seed}", flush=True)
            trained = train_with_early_stop_then_refit(
                matrices,
                split_seed=primary_seed,
                early_stop_fraction=float(rank_cfg.early_stop_fraction),
                ranker_seed=seed,
                num_boost_round=int(rank_cfg.num_boost_round),
                early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
                **_ranker_kwargs(config),
            )
            ranked = _rerank(
                trained.booster,
                ctx_full,
                lists_full,
                ii_maps,
                tt_maps,
                trained.refit_info["feature_names"],
            )
            metrics = _evaluate_lists(
                ranked,
                full_train,
                split.test,
                config=config,
                split=_eval_split(split),
                include_segments=True,
            )
            per_user = _per_user_scores(
                PrecomputedRecommender(ranked).recommend, full_train, split, config
            )
            if seed == primary_seed:
                primary_per_user = per_user
            per_seed.append(
                {
                    "seed": seed,
                    "best_iteration": int(trained.best_iteration),
                    "metrics": metrics,
                    "per_user": per_user,
                }
            )
        no_metrics = _evaluate_lists(
            lists_full,
            full_train,
            split.test,
            config=config,
            split=_eval_split(split),
            include_segments=True,
        )
        no_per_user = _per_user_scores(
            PrecomputedRecommender(lists_full).recommend, full_train, split, config
        )
        headline = _headline(per_seed, primary_seed)
        assert primary_per_user is not None
        results[label] = {
            "loss": _loss_name,
            "hyperparams": hp,
            "n_epochs": epochs,
            "demographics": demo_mode,
            "candidate_k": k,
            "candidate_set": "two_tower",
            "metrics": headline,
            "no_ranker_metrics": no_metrics,
            "per_seed": [
                {
                    "seed": int(row["seed"]),
                    "best_iteration": int(row["best_iteration"]),
                    "ndcg@10": row["metrics"]["ndcg@10"],
                    "ndcg@10_ci": row["metrics"]["confidence_intervals"]["ndcg@10"],
                    "recall@10": row["metrics"].get("recall@10"),
                    "coverage@10": row["metrics"].get("coverage@10"),
                    "tail_ndcg@10": _segment_value(row["metrics"], "tail"),
                    "head_ndcg@10": _segment_value(row["metrics"], "head"),
                }
                for row in per_seed
            ],
            "per_user": primary_per_user,
            "no_ranker_per_user": no_per_user,
        }
        release_memory()
    paired_ranker = _pair(
        results["reference"]["per_user"],
        results["new"]["per_user"],
        seed=config.seed,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
    )
    paired_lift = _pair(
        results["new"]["no_ranker_per_user"],
        results["new"]["per_user"],
        seed=config.seed,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
    )
    for row in results.values():
        row.pop("per_user", None)
        row.pop("no_ranker_per_user", None)
    return {
        "demographics": demo_mode,
        "candidate_k": k,
        "candidate_set": "two_tower",
        "seeds": seeds,
        "reference": results["reference"],
        "new": results["new"],
        "paired_vs_reference_ranker": paired_ranker,
        "paired_vs_no_ranker": paired_lift,
        "runtime_sec": round(time.perf_counter() - t0, 3),
    }


def cmd_test(dataset: str) -> None:
    import torch

    torch.set_num_threads(os.cpu_count() or 1)
    plan = _locked(dataset)
    tuning_path = _tuning_path(dataset)
    if not tuning_path.is_file():
        raise SystemExit(f"Missing tuning file {tuning_path}")
    tuning = _read(tuning_path)
    if tuning.get("status") != "complete":
        raise SystemExit("Tuning is not complete; refusing to score test")
    config, movies, split = load_split(dataset)
    from movielens_recommender.ranker.pipeline import _release_two_tower_pack

    ckpt_path = _checkpoint(dataset)
    ckpt = _read(ckpt_path) if ckpt_path.is_file() else {"models": {}, "topks": {}}
    specs = _model_specs(dataset, tuning, plan)
    seeds = [int(s) for s in plan["datasets"][dataset]["test"]["seeds"]]
    primary = int(config.seed)
    full_train = split.full_train
    test_users = [int(uid) for uid in split.test["user_id"].unique()]
    keep_names = {"in_batch", tuning["selected_loss"]}
    t0 = time.perf_counter()
    kept_topk = {
        name: _restore_topk(raw) for name, raw in (ckpt.get("topks") or {}).items()
    }
    for spec in specs:
        name = spec["name"]
        ckpt["models"].setdefault(name, {"per_seed": []})
        have = {int(row["seed"]) for row in ckpt["models"][name]["per_seed"]}
        for seed in seeds:
            if seed in have:
                print(f"skip {name} seed {seed}", flush=True)
                continue
            print(f"Refit {name} epochs={spec['n_epochs']} seed={seed}", flush=True)
            rec, metrics, per_user, wall = _fit_and_eval(
                full_train=full_train,
                movies=movies,
                split=split,
                config=config,
                hyperparams=spec["hyperparams"],
                n_epochs=int(spec["n_epochs"]),
                seed=seed,
            )
            row = {
                "seed": seed,
                "metrics": metrics,
                "per_user": per_user,
                "train_wall_sec": wall,
            }
            if seed == primary and name in keep_names:
                kept_topk[name] = rec.topk_with_scores(
                    int(config.models.ranker.candidate_k), user_ids=test_users
                )
            _release_two_tower_pack(rec)
            del rec
            release_memory()
            ckpt["models"][name]["per_seed"].append(row)
            ckpt["models"][name]["spec"] = {
                "role": spec["role"],
                "hyperparams": spec["hyperparams"],
                "n_epochs": spec["n_epochs"],
            }
            ckpt["topks"] = {key: _json_topk(val) for key, val in kept_topk.items()}
            _write(ckpt_path, ckpt)
    models_out: dict[str, Any] = {}
    per_user_primary: dict[str, Any] = {}
    for spec in specs:
        rows = ckpt["models"][spec["name"]]["per_seed"]
        rows = sorted(rows, key=lambda row: int(row["seed"]))
        models_out[spec["name"]] = _compact_model(rows, primary, spec)
        per_user_primary[spec["name"]] = next(
            row["per_user"] for row in rows if int(row["seed"]) == primary
        )
    paired = {
        name: _pair(
            per_user_primary["in_batch"],
            per_user_primary[name],
            seed=config.seed,
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
        )
        for name in models_out
        if name != "in_batch"
    }
    ranker = ckpt.get("ranker")
    if tuning["beats_reference_validation"] and ranker is None:
        towers = {
            "reference": {"topk": _restore_topk(ckpt["topks"]["in_batch"])},
            "new": {"topk": _restore_topk(ckpt["topks"][tuning["selected_loss"]])},
        }
        ranker = _run_rankers(
            dataset=dataset,
            config=config,
            movies=movies,
            split=split,
            tuning=tuning,
            towers=towers,
        )
        ckpt["ranker"] = ranker
        _write(ckpt_path, ckpt)
    payload = {
        "experiment": "s3e_full_softmax_two_tower",
        "adr": "ADR-0013",
        "dataset": dataset,
        "dataset_sha256": DATASET_SHA256[dataset],
        "seed": config.seed,
        "eval_user_sample": split.eval_user_sample,
        "reference_published": _published_reference_block(dataset),
        "budget_source": "results/budget/two-tower-v2.json",
        "tuning_source": f"results/tuning/two_tower_v2_{dataset}.json",
        "tuning": {
            "selected_loss": tuning["selected_loss"],
            "selected_val_ndcg@10": tuning["selected_val_ndcg@10"],
            "reference_val_ndcg@10": tuning["reference_val_ndcg@10"],
            "beats_reference_validation": tuning["beats_reference_validation"],
            "best_by_loss": tuning["best_by_loss"],
            "tune_wall_sec": tuning.get("tune_wall_sec"),
            "n_trials": len(tuning["trials"]),
        },
        "test_seeds": seeds,
        "test_both_losses": bool(plan["datasets"][dataset]["test"]["test_both_losses"]),
        "models": models_out,
        "paired_vs_reference": paired,
        "ranker": ranker,
        "compute": {
            "tune_wall_sec": tuning.get("tune_wall_sec"),
            "test_train_sec": round(
                sum(
                    float(row["train_wall_time_sec"])
                    for model in models_out.values()
                    for row in model["per_seed"]
                ),
                3,
            ),
            "test_wall_sec": round(time.perf_counter() - t0, 3),
            "ranker_wall_sec": None if ranker is None else ranker.get("runtime_sec"),
            "seeds": seeds,
            "cpu_count": os.cpu_count(),
        },
        "runtime_sec": round(time.perf_counter() - t0, 3),
    }
    out = ROOT / "results" / "two-tower-v2" / f"{dataset}.json"
    _write(out, payload)
    print(f"Wrote {out}", flush=True)


def _json_topk(topk: dict[int, list[tuple[int, float]]]) -> dict[str, list[list[float]]]:
    return {
        str(uid): [[int(item), float(score)] for item, score in rows]
        for uid, rows in topk.items()
    }


def _restore_topk(raw: dict[str, list[list[float]]]) -> dict[int, list[tuple[int, float]]]:
    return {
        int(uid): [(int(item), float(score)) for item, score in rows]
        for uid, rows in raw.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="S3e two-tower loss experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--dataset", required=True, choices=["ml-1m", "ml-32m"])
    sub.add_parser("lock")
    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--dataset", required=True, choices=["ml-1m", "ml-32m"])
    p_test = sub.add_parser("test")
    p_test.add_argument("--dataset", required=True, choices=["ml-1m", "ml-32m"])
    args = parser.parse_args()
    if args.cmd == "probe":
        cmd_probe(args.dataset)
    elif args.cmd == "lock":
        cmd_lock()
    elif args.cmd == "tune":
        cmd_tune(args.dataset)
    elif args.cmd == "test":
        cmd_test(args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
