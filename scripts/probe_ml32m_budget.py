#!/usr/bin/env python3
"""Measure ml-32M memory and runtime before any test metrics (ADR-0011).

Writes ``results/budget/ml-32m.json`` after each stage. Ranking metrics are
not computed. Allocation attempts that would exceed the kernel commit
headroom are recorded as skipped rather than issued.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "budget" / "ml-32m.json"

# Square sizes to try for an EASE-style dense item-item matrix.
EASE_NS = (30000, 25000, 20000, 16000, 14000, 12000, 10000, 8000)
# Realistic EASE peak: Gram, inverse, and B. overwrite_a reuses the Gram
# buffer; ascontiguousarray does not copy a C-contiguous B.
EASE_MATRICES = 3


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, rest = line.split(":", 1)
        out[key] = int(rest.strip().split()[0]) * 1024
    return out


def _rss() -> dict[str, float]:
    rss_kib = hwm_kib = None
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            rss_kib = int(line.split()[1])
        elif line.startswith("VmHWM:"):
            hwm_kib = int(line.split()[1])
    info = _meminfo()
    return {
        "rss_mib": round((rss_kib or 0) / 1024, 1),
        "hwm_mib": round((hwm_kib or 0) / 1024, 1),
        "mem_available_mib": round(info.get("MemAvailable", 0) / (1024 * 1024), 1),
        "commit_limit_mib": round(info.get("CommitLimit", 0) / (1024 * 1024), 1),
        "committed_as_mib": round(info.get("Committed_AS", 0) / (1024 * 1024), 1),
        "commit_headroom_mib": round(
            (info.get("CommitLimit", 0) - info.get("Committed_AS", 0)) / (1024 * 1024),
            1,
        ),
    }


def _trim() -> None:
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _load(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "purpose": (
            "Pre-test budget probe for ml-32m. No NDCG, recall, coverage, "
            "or other ranking metric is computed."
        ),
        "stages": {},
    }


def _save(doc: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage(doc: dict, name: str, payload: dict) -> None:
    payload = dict(payload)
    payload["finished_at_unix"] = time.time()
    doc["stages"][name] = payload
    _save(doc)
    print(f"STAGE {name}: {json.dumps(payload, sort_keys=True)[:500]}", flush=True)


def stage_host(doc: dict) -> None:
    info = _meminfo()
    _stage(
        doc,
        "host",
        {
            "mem_total_mib": round(info["MemTotal"] / (1024 * 1024), 1),
            "swap_total_mib": round(info.get("SwapTotal", 0) / (1024 * 1024), 1),
            "commit_limit_mib": round(info["CommitLimit"] / (1024 * 1024), 1),
            "cpu_count": Path("/proc/cpuinfo").read_text(encoding="utf-8").count("processor\t:"),
            "overcommit": Path("/proc/sys/vm/overcommit_memory")
            .read_text(encoding="utf-8")
            .strip(),
            "overcommit_ratio": Path("/proc/sys/vm/overcommit_ratio")
            .read_text(encoding="utf-8")
            .strip(),
            **_rss(),
        },
    )


def stage_load_split(doc: dict) -> tuple[pd.DataFrame, object]:
    from movielens_recommender.data import load_ratings
    from movielens_recommender.movies import load_movies
    from movielens_recommender.scale import warm_relevant_user_ids
    from movielens_recommender.split import SplitConfig, time_based_split

    t0 = time.perf_counter()
    ratings, stats = load_ratings("ml-32m", ROOT / "data", clean=True)
    load_sec = time.perf_counter() - t0
    movies = load_movies("ml-32m", ROOT / "data")
    load_rss = _rss()
    t1 = time.perf_counter()
    split = time_based_split(
        ratings,
        SplitConfig(
            min_ratings=5,
            test_fraction=0.2,
            val_fraction=0.1,
            relevance_threshold=4.0,
        ),
    )
    split_sec = time.perf_counter() - t1
    eligible = warm_relevant_user_ids(split.full_train, split.test, 4.0)
    _stage(
        doc,
        "load_split",
        {
            "load_sec": round(load_sec, 3),
            "split_sec": round(split_sec, 3),
            "n_ratings_clean": int(len(ratings)),
            "n_dropped_duplicates": int(stats.n_dropped_duplicates),
            "n_dropped_invalid_rating": int(stats.n_dropped_invalid_rating),
            "n_movies_rows": int(len(movies)),
            "n_fit_train": int(len(split.train)),
            "n_val": int(len(split.val)) if split.val is not None else 0,
            "n_test": int(len(split.test)),
            "n_full_train": int(len(split.full_train)),
            "n_train_users": int(split.train["user_id"].nunique()),
            "n_train_items": int(split.train["item_id"].nunique()),
            "n_full_train_items": int(split.full_train["item_id"].nunique()),
            "n_eligible_eval_users": int(len(eligible)),
            "rss_after_load": load_rss,
            "rss_after_split": _rss(),
        },
    )
    # Eligible ids are only a count. Drop the array before later stages.
    del eligible
    _trim()
    return ratings, split


def stage_allocations(doc: dict) -> None:
    from scipy import linalg

    trials = []
    for n in EASE_NS:
        nbytes = n * n * 8
        headroom = _meminfo()["CommitLimit"] - _meminfo()["Committed_AS"]
        needed = nbytes * EASE_MATRICES
        trial: dict = {
            "n": n,
            "one_matrix_mib": round(nbytes / (1024 * 1024), 1),
            "assumed_matrices": EASE_MATRICES,
            "needed_mib": round(needed / (1024 * 1024), 1),
            "headroom_before_mib": round(headroom / (1024 * 1024), 1),
        }
        if needed > int(headroom * 0.85):
            trial["status"] = "skipped_over_commit_headroom"
            trials.append(trial)
            _stage(doc, "ease_allocations", {"trials": trials, "partial": True})
            continue
        print(f"  allocating {EASE_MATRICES} x {n}x{n} float64", flush=True)
        t0 = time.perf_counter()
        mats: list[np.ndarray] = []
        inv = None
        try:
            for i in range(EASE_MATRICES):
                mat = np.zeros((n, n), dtype=np.float64)
                if i == 0:
                    np.fill_diagonal(mat, float(n))
                mats.append(mat)
            trial["alloc_sec"] = round(time.perf_counter() - t0, 3)
            trial["rss_after_alloc"] = _rss()
            # Same LAPACK entry the fitter uses, on a diagonally dominant matrix.
            print(f"  inverting {n}x{n}", flush=True)
            t1 = time.perf_counter()
            inv = linalg.inv(mats[0], overwrite_a=True, check_finite=False)
            trial["inv_sec"] = round(time.perf_counter() - t1, 3)
            trial["inv_shape"] = list(inv.shape)
            trial["status"] = "allocated_and_inverted"
        except MemoryError as exc:
            trial["status"] = "memory_error"
            trial["error"] = str(exc)
        except Exception as exc:
            trial["status"] = "error"
            trial["error"] = repr(exc)
        finally:
            del inv
            del mats
            _trim()
        trial["rss_after_free"] = _rss()
        trials.append(trial)
        _stage(doc, "ease_allocations", {"trials": trials, "partial": True})
        if trial["status"] == "allocated_and_inverted":
            # Largest N that fits is enough; smaller N also fit.
            break
    _stage(doc, "ease_allocations", {"trials": trials, "partial": False})


def _binary(train: pd.DataFrame):
    from movielens_recommender.baselines.common import binary_user_item

    t0 = time.perf_counter()
    mat, item_ids, _user_index, _item_index = binary_user_item(train)
    return mat, item_ids, time.perf_counter() - t0


def stage_item_item(doc: dict, train: pd.DataFrame) -> None:
    from scipy import sparse

    mat, _item_ids, build_sec = _binary(train)
    n_items = mat.shape[1]
    binary = mat.copy()
    binary.data = np.ones(binary.data.shape, dtype=np.float64)
    norms = np.sqrt(np.asarray(mat.power(2).sum(axis=0))).ravel()
    norms[norms == 0.0] = 1.0
    mat_norm = (mat @ sparse.diags(1.0 / norms)).tocsc()
    binary_csc = binary.tocsc()
    del mat, binary
    _trim()
    row_bytes = max(n_items * 8 * 2, 1)
    block_rows = max(1, min(n_items, (256 * 1024 * 1024) // row_bytes))
    starts = [0, max(0, n_items // 2), max(0, n_items - block_rows)]
    starts = sorted(set(starts))
    times = []
    for start in starts:
        end = min(start + block_rows, n_items)
        t0 = time.perf_counter()
        sim = (mat_norm[:, start:end].T @ mat_norm).toarray()
        common = (binary_csc[:, start:end].T @ binary_csc).toarray()
        sim[np.arange(end - start), np.arange(start, end)] = 0.0
        _ = common.shape
        elapsed = time.perf_counter() - t0
        times.append({"start": int(start), "end": int(end), "sec": round(elapsed, 3)})
        del sim, common
        _trim()
    n_blocks = int(np.ceil(n_items / block_rows))
    mean_sec = float(np.mean([row["sec"] for row in times]))
    _stage(
        doc,
        "item_item_blocks",
        {
            "matrix": "fit_train",
            "build_binary_sec": round(build_sec, 3),
            "n_items": int(n_items),
            "block_rows": int(block_rows),
            "n_blocks_full": n_blocks,
            "sampled_blocks": times,
            "mean_block_sec": round(mean_sec, 3),
            "extrapolated_full_fit_sec": round(mean_sec * n_blocks, 1),
            "note": (
                "Same block matmul as _sparse_topk_cosine "
                "(column-normalized cosine plus common counts). "
                "Extrapolation is mean of the sampled blocks times n_blocks. "
                "No recommendations or metrics."
            ),
            "rss": _rss(),
        },
    )
    del mat_norm, binary_csc
    _trim()


def stage_rp3(doc: dict, train: pd.DataFrame) -> None:
    from movielens_recommender.baselines.rp3beta import _l1_normalize_rows

    mat, _item_ids, build_sec = _binary(train)
    n_items = mat.shape[1]
    t0 = time.perf_counter()
    p_ui = _l1_normalize_rows(mat)
    x_bool_t = mat.T.tocsr().copy()
    x_bool_t.data = np.ones(x_bool_t.data.shape, dtype=np.float64)
    p_iu = _l1_normalize_rows(x_bool_t)
    prep_sec = time.perf_counter() - t0
    del mat, x_bool_t
    _trim()
    row_bytes = max(n_items * 8, 1)
    block = max(1, min(n_items, (256 * 1024 * 1024) // row_bytes))
    starts = [0, max(0, n_items // 2), max(0, n_items - block)]
    starts = sorted(set(starts))
    times = []
    for start in starts:
        end = min(start + block, n_items)
        t1 = time.perf_counter()
        dense = np.asarray((p_iu[start:end] @ p_ui).toarray(), dtype=np.float64)
        elapsed = time.perf_counter() - t1
        times.append(
            {
                "start": int(start),
                "end": int(end),
                "sec": round(elapsed, 3),
                "shape": list(dense.shape),
            }
        )
        del dense
        _trim()
    n_blocks = int(np.ceil(n_items / block))
    mean_sec = float(np.mean([row["sec"] for row in times]))
    _stage(
        doc,
        "rp3beta_blocks",
        {
            "matrix": "fit_train",
            "build_binary_sec": round(build_sec, 3),
            "normalize_sec": round(prep_sec, 3),
            "n_items": int(n_items),
            "block_rows": int(block),
            "n_blocks_full": n_blocks,
            "sampled_blocks": times,
            "mean_block_sec": round(mean_sec, 3),
            "extrapolated_full_fit_sec": round(mean_sec * n_blocks + prep_sec, 1),
            "note": (
                "Same blocked P_iu @ P_ui as RP3betaRecommender._fit_blocked. "
                "No top-k prune loop, no recommendations, no metrics."
            ),
            "rss": _rss(),
        },
    )
    del p_ui, p_iu
    _trim()


def stage_ease_fit(doc: dict, train: pd.DataFrame, n: int) -> None:
    from movielens_recommender.baselines.ease import EASERecommender

    t0 = time.perf_counter()
    model = EASERecommender(l2=5000.0, max_items=n, cache_user_scores=False)
    model.fit(train)
    elapsed = time.perf_counter() - t0
    _stage(
        doc,
        "ease_fit",
        {
            "l2": 5000.0,
            "max_items": n,
            "n_items_before_restriction": int(model.n_items_before_restriction),
            "n_items_fit": int(model.n_items_fit),
            "fit_sec": round(elapsed, 3),
            "cache_user_scores": False,
            "note": "One fit on fit-train. No recommendations or metrics.",
            "rss": _rss(),
        },
    )
    del model
    _trim()


def stage_als(doc: dict, train: pd.DataFrame) -> None:
    from movielens_recommender.baselines.als import ALSRecommender

    t0 = time.perf_counter()
    model = ALSRecommender(
        factors=64,
        regularization=0.01,
        iterations=2,
        alpha=40.0,
        random_state=42,
    )
    model.fit(train)
    elapsed = time.perf_counter() - t0
    _stage(
        doc,
        "als",
        {
            "iterations_run": 2,
            "factors": 64,
            "fit_sec": round(elapsed, 3),
            "extrapolated_15_iter_sec": round(elapsed * (15 / 2), 1),
            "note": (
                "Real ALSRecommender.fit on fit-train for 2 iterations. "
                "The 15-iteration figure multiplies wall time by 15/2, so it "
                "includes setup and overestimates the iteration-only cost. "
                "No recommendations or metrics."
            ),
            "rss": _rss(),
        },
    )
    del model
    _trim()


def stage_two_tower(doc: dict, split, movies: pd.DataFrame) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from movielens_recommender.two_tower.features import build_features
    from movielens_recommender.two_tower.model import TwoTowerModel
    from movielens_recommender.two_tower.recommender import TwoTowerRecommender
    from movielens_recommender.two_tower.train import _batch_loss, set_torch_seed

    t0 = time.perf_counter()
    features = build_features(
        split.train,
        dataset="ml-32m",
        data_dir=str(ROOT / "data"),
        movies=movies,
        relevance_threshold=4.0,
        max_history=50,
    )
    build_sec = time.perf_counter() - t0
    set_torch_seed(42)
    device = torch.device("cpu")
    model = TwoTowerModel(
        n_users=features.n_users,
        n_items=features.n_items,
        n_genres=features.n_genres,
        embedding_dim=64,
        temperature=0.1,
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(features.pos_user_idx),
        torch.from_numpy(features.pos_item_idx),
    )
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    n_batches = 20
    t1 = time.perf_counter()
    done = 0
    model.train()
    for user_idx, item_idx in loader:
        user_idx = user_idx.to(device)
        item_idx = item_idx.to(device)
        opt.zero_grad(set_to_none=True)
        loss = _batch_loss(model, features, user_idx, item_idx, device, 50)
        loss.backward()
        opt.step()
        done += 1
        if done >= n_batches:
            break
    batch_sec = time.perf_counter() - t1
    batches_per_epoch = int(np.ceil(len(dataset) / 1024))
    rec = TwoTowerRecommender.from_trained(model, features, max_history=50, device="cpu")
    sample_users = [int(u) for u in list(features.user_ids[:100])]
    t2 = time.perf_counter()
    for uid in sample_users:
        rec.recommend(uid, 10)
    rec_sec = time.perf_counter() - t2
    _stage(
        doc,
        "two_tower",
        {
            "build_features_sec": round(build_sec, 3),
            "n_users": int(features.n_users),
            "n_items": int(features.n_items),
            "n_positives": int(len(features.pos_user_idx)),
            "batches_timed": done,
            "batch_wall_sec": round(batch_sec, 3),
            "sec_per_batch": round(batch_sec / max(done, 1), 4),
            "batches_per_epoch": batches_per_epoch,
            "extrapolated_epoch_train_sec": round(
                (batch_sec / max(done, 1)) * batches_per_epoch, 1
            ),
            "recommend_users_timed": len(sample_users),
            "recommend_wall_sec": round(rec_sec, 3),
            "extrapolated_recommend_8000_sec": round(rec_sec / max(len(sample_users), 1) * 8000, 1),
            "note": (
                "20 AdamW batches on fit-train positives, ml-1m architecture "
                "(emb 64, batch 1024, temperature 0.1). recommend() timed on "
                "the first 100 fit users at k=10. Scores are discarded. "
                "No validation or test metric."
            ),
            "rss": _rss(),
        },
    )
    del rec, model, features, dataset, loader
    _trim()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from-stage",
        default="host",
        choices=["host", "load", "alloc", "item_item", "rp3", "ease", "als", "two_tower"],
    )
    parser.add_argument("--ease-n", type=int, default=0)
    args = parser.parse_args()
    doc = _load(OUT)
    doc["stages"].pop("probe_error", None)
    order = ["host", "load", "alloc", "item_item", "rp3", "ease", "als", "two_tower"]
    start = order.index(args.from_stage)
    ratings = split = movies = None
    if start <= order.index("load"):
        stage_host(doc)
        ratings, split = stage_load_split(doc)
    else:
        from movielens_recommender.data import load_ratings
        from movielens_recommender.movies import load_movies
        from movielens_recommender.split import SplitConfig, time_based_split

        ratings, _stats = load_ratings("ml-32m", ROOT / "data", clean=True)
        movies = load_movies("ml-32m", ROOT / "data")
        split = time_based_split(
            ratings,
            SplitConfig(
                min_ratings=5,
                test_fraction=0.2,
                val_fraction=0.1,
                relevance_threshold=4.0,
            ),
        )
    if movies is None:
        from movielens_recommender.movies import load_movies

        movies = load_movies("ml-32m", ROOT / "data")
    try:
        if start <= order.index("alloc"):
            stage_allocations(doc)
        if start <= order.index("item_item"):
            stage_item_item(doc, split.train)
        if start <= order.index("rp3"):
            stage_rp3(doc, split.train)
        if start <= order.index("ease"):
            n = args.ease_n
            if n <= 0:
                trials = doc["stages"].get("ease_allocations", {}).get("trials", [])
                ok = [t["n"] for t in trials if t.get("status") == "allocated_and_inverted"]
                n = int(max(ok)) if ok else 8000
            stage_ease_fit(doc, split.train, n)
        if start <= order.index("als"):
            stage_als(doc, split.train)
        if start <= order.index("two_tower"):
            stage_two_tower(doc, split, movies)
    except Exception as exc:
        _stage(
            doc,
            "probe_error",
            {"error": repr(exc), "traceback": traceback.format_exc(), "rss": _rss()},
        )
        raise
    doc["completed"] = "probe_error" not in doc["stages"]
    _save(doc)


if __name__ == "__main__":
    main()
