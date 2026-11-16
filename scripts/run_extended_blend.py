#!/usr/bin/env python3
"""Similarity blended with popularity for the larger catalogue (ADR-0035).

Reuses ADR-0034's catalogue, neighbourhood configuration, and validation
users. Draws 5,000 new test users, fits with all 15,000 held-out users left
out, chooses the blend weight on validation, and scores the new test users
once. When the ship rule passes it writes the parameter files.
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

from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.cli import host_info, library_versions
from movielens_recommender.cold_start import _held_out_targets
from movielens_recommender.data import DATASET_SHA256, load_ratings
from movielens_recommender.metrics import paired_bootstrap_delta_ci
from movielens_recommender.movies import load_movies
from movielens_recommender.serving.cold import BLEND_METHOD, blend_rankings
from run_extended_catalog import (
    ALPHA,
    HEAD_FRACTION,
    MIN_FILM_RATINGS,
    MIN_USER_RATINGS,
    N_BOOTSTRAP,
    N_HELD_OUT,
    PROFILE_NS,
    SEED,
    SELECTION_N,
    SHIP_NS,
    THRESHOLD,
    TOP_K,
    WEIGHTS_DIR,
    WEIGHTS_VERSION,
    _fit,
    _popular_lists,
    _round,
    _summary,
    catalog_item_ids,
    export_weights,
    held_out_users,
)

ROOT = Path(__file__).resolve().parents[1]
PRIOR_PATH = ROOT / "results" / "extended-catalog" / "ml-32m.json"
RESULTS_PATH = ROOT / "results" / "extended-catalog" / "ml-32m-blend.json"

NEW_TEST_SEED = 43
N_NEW_TEST = 5_000
LAMBDAS = (0.25, 0.5, 1.0, 2.0, 4.0)
NEIGHBOURHOOD = {"name": "ml1m_choice", "k_neighbors": 200, "shrinkage": 100.0, "min_common": 1}


def new_test_users(
    ratings: pd.DataFrame, used: set[int], *, min_ratings: int, n: int, seed: int
) -> np.ndarray:
    """A seeded sample of eligible users that are not in ``used``."""
    per_user = ratings.groupby("user_id").size()
    eligible = np.sort(per_user.index[per_user.to_numpy() >= min_ratings].to_numpy(dtype=np.int64))
    pool = np.asarray([uid for uid in eligible.tolist() if uid not in used], dtype=np.int64)
    if n > len(pool):
        raise ValueError("not enough unused eligible users")
    return np.sort(np.random.default_rng(seed).choice(pool, size=n, replace=False))


def choose_lambda(scores: dict[float, float]) -> float:
    """Highest validation NDCG@10. A tie keeps the smaller weight."""
    best = min(scores)
    for lam in sorted(scores):
        if round(float(scores[lam]), 6) > round(float(scores[best]), 6):
            best = lam
    return float(best)


def choose_method(blend: float, popular: float) -> str:
    """Per-N serving method. A tie keeps most_popular."""
    return BLEND_METHOD if round(blend, 6) > round(popular, 6) else "most_popular"


def ship_decision(paired_by_n: dict[int, dict[str, Any]]) -> bool:
    """True when the blend beats most-popular (paired low end above 0) at every ship N."""
    return all(float(paired_by_n[n]["low"]) > 0.0 for n in SHIP_NS)


def _blend_lists(
    model: ItemItemCosineRecommender,
    counts: np.ndarray,
    profiles: dict[int, list[tuple[int, float]]],
    lam: float,
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    uids = list(profiles)
    for start in range(0, len(uids), 500):
        batch = {uid: profiles[uid] for uid in uids[start : start + 500]}
        for uid, rows in blend_rankings(model, counts, batch, lam, TOP_K).items():
            out[int(uid)] = [int(item) for item, _score in rows]
    return out


def evaluate(
    held: pd.DataFrame,
    model: ItemItemCosineRecommender,
    counts: np.ndarray,
    lambdas: tuple[float, ...],
    popular_ranked: list[int],
    catalog: list[int],
    tail: set[int],
) -> dict[int, dict[str, Any]]:
    by_n: dict[int, dict[str, Any]] = {}
    catalog_set = set(catalog)
    for n in PROFILE_NS:
        profiles, _hist, _frames, relevant, n_counts = _held_out_targets(
            held, n, catalog_set, THRESHOLD, target="all_later"
        )
        profiles = {uid: profiles[uid] for uid in relevant}
        popular, popular_values = _summary(
            _popular_lists(popular_ranked, profiles), relevant, catalog, tail
        )
        block: dict[str, Any] = {"counts": n_counts, "most_popular": popular, "blend": {}}
        for lam in lambdas:
            summary, values = _summary(
                _blend_lists(model, counts, profiles, lam), relevant, catalog, tail
            )
            paired = paired_bootstrap_delta_ci(
                popular_values, values, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=SEED
            )
            summary["paired_minus_most_popular"] = {
                "mean": _round(paired["mean"]),
                "low": _round(paired["low"]),
                "high": _round(paired["high"]),
                "n_users": int(paired["n_users"]),
                "excludes_zero": bool(paired["low"] > 0.0 or paired["high"] < 0.0),
            }
            block["blend"][str(lam)] = summary
        by_n[n] = block
        shown = {lam: row["ndcg@10"] for lam, row in block["blend"].items()}
        print(f"  N={n}: most_popular={popular['ndcg@10']} blend={shown}", flush=True)
    return by_n


def run(data_dir: Path, *, export: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    prior_bytes = PRIOR_PATH.read_bytes()
    prior = json.loads(prior_bytes.decode("utf-8"))
    if prior["decision"]["kept_config"] != NEIGHBOURHOOD["name"]:
        raise RuntimeError("ADR-0034 kept a different neighbourhood than ADR-0035 registered")
    listed = load_movies("ml-latest-small", data_dir)
    ratings, _stats = load_ratings("ml-32m", data_dir, clean=True)
    catalog_ids = catalog_item_ids(ratings, listed, floor=MIN_FILM_RATINGS)
    ratings = ratings[ratings["item_id"].isin(set(catalog_ids.tolist()))]
    val_ids, old_test_ids, _eligible = held_out_users(
        ratings, min_ratings=MIN_USER_RATINGS, n_held_out=N_HELD_OUT, seed=SEED
    )
    if hashlib.sha256(val_ids.tobytes()).hexdigest() != (
        prior["protocol"]["validation_user_ids_sha256"]
    ):
        raise RuntimeError("validation users differ from the ADR-0034 record")
    used = set(val_ids.tolist()) | set(old_test_ids.tolist())
    test_ids = new_test_users(
        ratings, used, min_ratings=MIN_USER_RATINGS, n=N_NEW_TEST, seed=NEW_TEST_SEED
    )
    left_out = used | set(test_ids.tolist())
    train = ratings[~ratings["user_id"].isin(left_out)]
    val = ratings[ratings["user_id"].isin(set(val_ids.tolist()))]
    test = ratings[ratings["user_id"].isin(set(test_ids.tolist()))]
    if set(train["user_id"].unique()) & left_out:
        raise AssertionError("a held-out user is in the fit rows")
    print(
        f"catalogue {len(catalog_ids)} films, fit rows {len(train)}, "
        f"validation users {len(val_ids)}, new test users {len(test_ids)}",
        flush=True,
    )

    popular = MostPopularRecommender().fit(train)
    popular_ranked = [int(item) for item in popular._ranked_items]
    counts = (
        train.groupby("item_id").size().reindex(catalog_ids).fillna(0).to_numpy(dtype=np.int64)
    )
    catalog = [int(item) for item in catalog_ids.tolist()]
    n_head = max(1, int(round(HEAD_FRACTION * len(catalog))))
    tail = set(popular_ranked[n_head:]) | (set(catalog) - set(popular_ranked))

    started = time.perf_counter()
    print(f"Fitting item-item {NEIGHBOURHOOD}...", flush=True)
    model = _fit(train, NEIGHBOURHOOD)
    fit_sec = round(time.perf_counter() - started, 3)
    n_fit_rows = int(len(train))
    del train
    if model._item_ids.tolist() != catalog:
        raise AssertionError("the fit catalogue is not the registered catalogue")

    print("Validation users:", flush=True)
    val_by_n = evaluate(val, model, counts, LAMBDAS, popular_ranked, catalog, tail)
    selection = {lam: val_by_n[SELECTION_N]["blend"][str(lam)]["ndcg@10"] for lam in LAMBDAS}
    kept = choose_lambda(selection)
    serving_by_n = {
        str(n): choose_method(
            val_by_n[n]["blend"][str(kept)]["ndcg@10"], val_by_n[n]["most_popular"]["ndcg@10"]
        )
        for n in PROFILE_NS
    }
    print(f"Kept lambda: {kept}. Serving by N: {serving_by_n}", flush=True)

    print("New test users (scored once):", flush=True)
    test_by_n = evaluate(test, model, counts, (kept,), popular_ranked, catalog, tail)
    paired = {n: test_by_n[n]["blend"][str(kept)]["paired_minus_most_popular"] for n in PROFILE_NS}
    ships = ship_decision(paired)
    print(f"Ship rule passes: {ships}", flush=True)

    payload: dict[str, Any] = {
        "adr": "0035",
        "catalog": {
            "listed_in": "ml-latest-small/movies.csv",
            "min_film_ratings": MIN_FILM_RATINGS,
            "n_films": len(catalog),
        },
        "dataset": "ml-32m",
        "dataset_sha256": DATASET_SHA256["ml-32m"],
        "titles_dataset_sha256": DATASET_SHA256["ml-latest-small"],
        "decision": {
            "kept_lambda": kept,
            "rule": (
                "Ships when paired test NDCG@10, blend minus most_popular, has an "
                "interval low end above 0 at N=3, N=5, and N=10."
            ),
            "selection_n": SELECTION_N,
            "selection_scores": {str(lam): score for lam, score in selection.items()},
            "serving_by_n": serving_by_n,
            "ships": bool(ships),
        },
        "fit_sec": fit_sec,
        "host": host_info(),
        "lambdas": list(LAMBDAS),
        "library_versions": library_versions(),
        "neighbourhood": dict(NEIGHBOURHOOD),
        "protocol": {
            "alpha": ALPHA,
            "min_user_ratings": MIN_USER_RATINGS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_fit_rows": n_fit_rows,
            "n_left_out_users": len(left_out),
            "new_test_seed": NEW_TEST_SEED,
            "profile_ns": list(PROFILE_NS),
            "relevance_threshold": THRESHOLD,
            "seed": SEED,
            "target": "all_later",
            "test_user_ids_sha256": hashlib.sha256(test_ids.tobytes()).hexdigest(),
            "validation_user_ids_sha256": hashlib.sha256(val_ids.tobytes()).hexdigest(),
            "prior_record_sha256": hashlib.sha256(prior_bytes).hexdigest(),
        },
        "test": {str(n): test_by_n[n] for n in PROFILE_NS},
        "validation": {str(n): val_by_n[n] for n in PROFILE_NS},
    }
    if ships and export:
        manifest = {
            "adr": "0035",
            "blend_lambda": kept,
            "candidate_k": int(NEIGHBOURHOOD["k_neighbors"]),
            "candidate_source": BLEND_METHOD,
            "catalog_weights_version": WEIGHTS_VERSION,
            "dataset": "ml-32m",
            "dataset_sha256": DATASET_SHA256["ml-32m"],
            "kind": "catalog",
            "min_film_ratings": MIN_FILM_RATINGS,
            "relevance_threshold": THRESHOLD,
            "results": "results/extended-catalog/ml-32m-blend.json",
            "serving_by_n": serving_by_n,
            "titles_dataset": "ml-latest-small",
            "titles_dataset_sha256": DATASET_SHA256["ml-latest-small"],
            "weights_only": True,
        }
        sizes = export_weights(model, counts, manifest, WEIGHTS_DIR)
        payload["weights"] = {"files": sizes, "total_bytes": int(sum(sizes.values()))}
        print(f"Wrote {WEIGHTS_DIR}: {payload['weights']['total_bytes']} bytes", flush=True)
    payload["runtime_sec"] = round(time.perf_counter() - t0, 3)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if PRIOR_PATH.read_bytes() != prior_bytes:
        raise RuntimeError("the ADR-0034 record changed during this run")
    print(f"Wrote {RESULTS_PATH}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0035 popularity blend, larger catalogue.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--no-export", action="store_true", help="Score only. Do not write space/weights/extended."
    )
    args = parser.parse_args()
    run(Path(args.data_dir), export=not args.no_export)


if __name__ == "__main__":
    main()
