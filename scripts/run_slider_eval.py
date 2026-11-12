#!/usr/bin/env python3
"""Offline evaluation of the Familiar <-> Adventurous MMR slider (ADR-0032).

Evaluates all five slider positions (0.0, 0.25, 0.5, 0.75, 1.0) on the 604
held-out cold-start users (N in {3, 5, 10}) against the deployed cold-start
serving bundle, measuring NDCG@10, Recall@10, Coverage@10, Novelty@10,
Intra-list diversity@10, Mean popularity@10, and Long-tail share@10.
Does not modify results/cold-start/ml-1m.json or space-weights.tar.gz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
if str(SPACE) not in sys.path:
    sys.path.insert(0, str(SPACE))

from movielens_recommender.beyond_accuracy import (  # noqa: E402
    genre_item_vectors,
    long_tail_items,
    score_recommendation_lists,
)
from movielens_recommender.cli import host_info, library_versions  # noqa: E402
from movielens_recommender.cold_start import (  # noqa: E402
    HOLDOUT_FRACTION,
    _held_out_targets,
    eligible_user_ids,
    exclude_held_out_users,
    select_held_out_user_ids,
)
from movielens_recommender.config import load_config  # noqa: E402
from movielens_recommender.data import DATASET_SHA256, download_dataset, load_ratings  # noqa: E402
from movielens_recommender.metrics import (  # noqa: E402
    ndcg_at_k,
    paired_bootstrap_delta_ci,
    recall_at_k,
)
from movielens_recommender.movies import load_movies, movies_path  # noqa: E402
from movielens_recommender.serving.cold import (  # noqa: E402
    load_cold_start_bundle,
    recommend_new_user,
)
from movielens_recommender.split import SplitConfig, time_based_split  # noqa: E402

COLD_RESULTS_PATH = ROOT / "results" / "cold-start" / "ml-1m.json"
RESULTS_PATH = ROOT / "results" / "slider-eval" / "ml-1m.json"

POSITIONS: tuple[tuple[str, float], ...] = (
    ("Familiar", 0.0),
    ("Closer", 0.25),
    ("In between", 0.5),
    ("Further", 0.75),
    ("Adventurous", 1.0),
)
PROFILE_SIZES: tuple[int, ...] = (3, 5, 10)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _ensure_snapshot_and_data() -> tuple[Path, Path]:
    from runtime import install_space_weights, materialize_space, use_sparse_item_item_loader

    use_sparse_item_item_loader()
    runtime_dir = SPACE / ".runtime"
    cold_dir = runtime_dir / "cold_start"
    data_dir = runtime_dir / "data"
    if not (cold_dir / "manifest.json").is_file() or not movies_path("ml-1m", data_dir).is_file():
        weights = install_space_weights(SPACE)
        if weights is None:
            raise RuntimeError("Could not install space weights for slider evaluation")
        _prod, cold_dir = materialize_space(weights, runtime_dir)
    if not movies_path("ml-1m", data_dir).is_file():
        data_dir = ROOT / "data"
        if not movies_path("ml-1m", data_dir).is_file():
            download_dataset("ml-1m", data_dir)
    return cold_dir, data_dir


def _round_pair(raw: dict[str, Any]) -> dict[str, Any]:
    low = round(float(raw["low"]), 6)
    high = round(float(raw["high"]), 6)
    return {
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "high": high,
        "low": low,
        "mean": round(float(raw["mean"]), 6),
        "n_users": int(raw["n_users"]),
    }


def _evaluate_target_set(
    *,
    traces: dict[int, tuple[list[int], list[float], object, list[float]]],
    relevant: dict[int, set[int]],
    catalog: list[int],
    popularity: dict[int, float],
    genre_vecs: Any,
    tail: set[int],
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> list[dict[str, Any]]:
    from value import rerank_candidates

    eval_uids = sorted(relevant)
    base_ndcgs: list[float] | None = None
    rows: list[dict[str, Any]] = []
    for label, weight in POSITIONS:
        recs: dict[int, list[int]] = {}
        ndcgs: list[float] = []
        recalls: list[float] = []
        for uid in eval_uids:
            cands, scores, emb, pop_pct = traces[uid]
            picked = rerank_candidates(
                cands,
                scores,
                adventurous=weight,
                embeddings=emb,
                popularity=pop_pct,
                n=10,
            )
            recs[uid] = picked
            ndcgs.append(float(ndcg_at_k(picked, relevant[uid], 10)))
            recalls.append(float(recall_at_k(picked, relevant[uid], 10)))
        if base_ndcgs is None:
            base_ndcgs = ndcgs
            paired = None
        else:
            paired = _round_pair(
                paired_bootstrap_delta_ci(
                    base_ndcgs,
                    ndcgs,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                    seed=seed,
                )
            )
        scored = score_recommendation_lists(
            recs,
            relevant=relevant,
            catalog=catalog,
            item_popularity=popularity,
            item_vectors=genre_vecs,
            tail_items=tail,
            k=10,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        rows.append(
            {
                "adventurous": float(weight),
                "confidence_intervals": {
                    k: {b: round(float(v), 6) for b, v in bounds.items()}
                    for k, bounds in scored["confidence_intervals"].items()
                },
                "coverage@10": round(float(scored["coverage@10"]), 6),
                "intra_list_diversity@10": round(float(scored["intra_list_diversity@10"]), 6),
                "label": label,
                "long_tail_share@10": round(float(scored["long_tail_share@10"]), 6),
                "mean_popularity@10": round(float(scored["mean_popularity@10"]), 6),
                "n_eval_users": len(eval_uids),
                "ndcg@10": round(float(scored["ndcg@10"]), 6),
                "novelty@10": round(float(scored["novelty@10"]), 6),
                "paired_ndcg_minus_familiar": paired,
                "recall@10": round(float(sum(recalls) / len(recalls)), 6),
            }
        )
    return rows


def run_slider_eval() -> dict[str, Any]:
    t0 = time.perf_counter()
    cold_sha = _sha256(COLD_RESULTS_PATH)
    cold_committed = json.loads(COLD_RESULTS_PATH.read_text(encoding="utf-8"))

    cold_dir, data_dir = _ensure_snapshot_and_data()
    bundle = load_cold_start_bundle(cold_dir)
    config = load_config(ROOT / "configs" / "ml-1m.yaml")
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)

    ratings, _clean = load_ratings("ml-1m", data_dir, clean=True)
    movies = load_movies("ml-1m", data_dir)

    eligible = eligible_user_ids(ratings, config.split.min_ratings)
    held_ids = select_held_out_user_ids(eligible, fraction=HOLDOUT_FRACTION, seed=seed)
    recorded_held = [int(u) for u in cold_committed["holdout"]["held_out_user_ids"]]
    if [int(u) for u in held_ids.tolist()] != recorded_held:
        raise RuntimeError("held-out user ids do not match results/cold-start/ml-1m.json")

    held_ratings = ratings.loc[ratings["user_id"].isin(set(recorded_held))].copy()
    remain = exclude_held_out_users(ratings, held_ids)
    split = time_based_split(
        remain,
        SplitConfig(
            min_ratings=config.split.min_ratings,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            relevance_threshold=threshold,
        ),
    )
    full_train = split.full_train
    catalog_list = sorted(int(i) for i in full_train["item_id"].unique())
    catalog_set = set(catalog_list)
    popularity = {
        int(item): float(count)
        for item, count in full_train.groupby("item_id").size().items()
    }
    tail = long_tail_items(full_train)
    genre_vecs = genre_item_vectors(movies, catalog_list)

    from value import embedding_rows, popularity_percentiles

    pop_pct_map = popularity_percentiles(bundle.context.item_ids, bundle.context.popularity)
    tower = bundle.two_tower

    committed_by_n = {
        int(block["n"]): float(block["models"]["served"]["ndcg@10"])
        for block in cold_committed["by_n"]
    }

    by_n: list[dict[str, Any]] = []
    sensitivity_by_n: list[dict[str, Any]] = []
    for n in PROFILE_SIZES:
        print(f"Scoring 604 held-out cold-start users at N={n}...", flush=True)
        profiles, _hist, _frames, relevant_all, counts_all = _held_out_targets(
            held_ratings, n, catalog_set, threshold, target="all_later"
        )
        _p_s, _h_s, _f_s, relevant_tail, counts_tail = _held_out_targets(
            held_ratings,
            n,
            catalog_set,
            threshold,
            target="harness_tail",
            test_fraction=float(config.split.test_fraction),
        )
        needed_uids = sorted(set(relevant_all) | set(relevant_tail))
        traces: dict[int, tuple[list[int], list[float], object, list[float]]] = {}
        for uid in needed_uids:
            res = recommend_new_user(bundle, profiles[uid], n=10)
            pipe = res["pipeline"]
            cands = [int(i) for i in pipe["ranked_ids"]]
            scores = [float(v) for v in pipe["ranked_scores"]]
            emb = embedding_rows(tower._item_vectors, tower._features.item_index, cands)
            pop_pct = [float(pop_pct_map[int(i)]) for i in cands]
            traces[uid] = (cands, scores, emb, pop_pct)

        rows_all = _evaluate_target_set(
            traces=traces,
            relevant=relevant_all,
            catalog=catalog_list,
            popularity=popularity,
            genre_vecs=genre_vecs,
            tail=tail,
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=seed,
        )
        rows_tail = _evaluate_target_set(
            traces=traces,
            relevant=relevant_tail,
            catalog=catalog_list,
            popularity=popularity,
            genre_vecs=genre_vecs,
            tail=tail,
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=seed,
        )
        fam_ndcg = rows_all[0]["ndcg@10"]
        expected = committed_by_n[n]
        print(
            f"  N={n} Familiar NDCG@10={fam_ndcg} (committed={expected}) "
            f"Adventurous NDCG@10={rows_all[-1]['ndcg@10']} "
            f"cov {rows_all[0]['coverage@10']} -> {rows_all[-1]['coverage@10']} "
            f"tail {rows_all[0]['long_tail_share@10']} -> {rows_all[-1]['long_tail_share@10']}",
            flush=True,
        )
        by_n.append(
            {
                "committed_familiar_ndcg@10": expected,
                "counts": counts_all,
                "matches_committed_at_4_decimals": f"{fam_ndcg:.4f}" == f"{expected:.4f}",
                "n": int(n),
                "n_eval_users": int(counts_all["n_eval_users"]),
                "positions": rows_all,
                "target": "all_later_ratings",
            }
        )
        sensitivity_by_n.append(
            {
                "counts": counts_tail,
                "n": int(n),
                "n_eval_users": int(counts_tail["n_eval_users"]),
                "positions": rows_tail,
                "target": "harness_tail",
            }
        )

    payload = {
        "adr": "0032",
        "by_n": by_n,
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "host": host_info(),
        "library_versions": library_versions(),
        "positions": [{"adventurous": w, "label": lbl} for lbl, w in POSITIONS],
        "profile_sizes": list(PROFILE_SIZES),
        "protocol": {
            "bootstrap_alpha": config.eval.bootstrap_alpha,
            "held_out_users": len(recorded_held),
            "n_bootstrap": config.eval.n_bootstrap,
            "relevance_threshold": threshold,
            "seed": seed,
            "source_holdout": "results/cold-start/ml-1m.json",
            "weights_sha256": "f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd",
        },
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "sensitivity_last_20_percent": sensitivity_by_n,
    }
    _write(RESULTS_PATH, payload)
    if _sha256(COLD_RESULTS_PATH) != cold_sha:
        raise RuntimeError("results/cold-start/ml-1m.json changed during slider evaluation")
    print(f"Wrote {RESULTS_PATH}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0032 discovery slider offline evaluation.")
    parser.parse_args()
    run_slider_eval()


if __name__ == "__main__":
    main()
