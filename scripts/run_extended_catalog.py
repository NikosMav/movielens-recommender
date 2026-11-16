#!/usr/bin/env python3
"""A larger Discover catalogue from MovieLens 32M (ADR-0034).

Fits item-item cosine on ml-32m ratings restricted to the films listed in
ml-latest-small, with 10,000 users held out. Validation users choose the
configuration and the per-N serving method. Test users are scored once. When
the ship rule passes, the neighbourhood matrix and the popularity counts are
written as parameter files with no rating rows, user ids, or titles.
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
from scipy import sparse

from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.cli import host_info, library_versions
from movielens_recommender.cold_start import _held_out_targets
from movielens_recommender.data import DATASET_SHA256, load_ratings
from movielens_recommender.metrics import (
    bootstrap_mean_ci,
    catalog_coverage,
    long_tail_share,
    ndcg_at_k,
    paired_bootstrap_delta_ci,
)
from movielens_recommender.movies import load_movies

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "extended-catalog" / "ml-32m.json"
WEIGHTS_DIR = ROOT / "space" / "weights" / "extended"

SEED = 42
MIN_FILM_RATINGS = 50
MIN_USER_RATINGS = 20
N_HELD_OUT = 10_000
PROFILE_NS = (1, 3, 5, 10)
SELECTION_N = 5
SHIP_NS = (3, 5, 10)
THRESHOLD = 4.0
TOP_K = 10
N_BOOTSTRAP = 1000
ALPHA = 0.05
HEAD_FRACTION = 0.2
CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "ml32m_choice", "k_neighbors": 50, "shrinkage": 0.0, "min_common": 1},
    {"name": "ml1m_choice", "k_neighbors": 200, "shrinkage": 100.0, "min_common": 1},
)
METHODS = ("item_item_fold_in", "most_popular")
WEIGHTS_VERSION = 1


def catalog_item_ids(ratings: pd.DataFrame, listed: pd.DataFrame, *, floor: int) -> np.ndarray:
    """Listed films with at least ``floor`` ratings, sorted by id."""
    counts = ratings.groupby("item_id").size()
    kept = counts.reindex(listed["item_id"].astype(int)).fillna(0).astype(int)
    return np.sort(kept.index[kept.to_numpy() >= floor].to_numpy(dtype=np.int64))


def held_out_users(
    ratings: pd.DataFrame, *, min_ratings: int, n_held_out: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    """Seeded validation and test user ids, and the eligible-user count."""
    per_user = ratings.groupby("user_id").size()
    eligible = np.sort(per_user.index[per_user.to_numpy() >= min_ratings].to_numpy(dtype=np.int64))
    if n_held_out % 2 or n_held_out > len(eligible):
        raise ValueError("n_held_out must be even and no larger than the eligible users")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n_held_out, replace=False)
    half = n_held_out // 2
    return np.sort(chosen[:half]), np.sort(chosen[half:]), int(len(eligible))


def choose_config(scores: dict[str, float]) -> str:
    """Highest validation NDCG@10 at the selection N. A tie keeps the first config."""
    best = CONFIGS[0]["name"]
    for config in CONFIGS[1:]:
        if round(float(scores[config["name"]]), 6) > round(float(scores[best]), 6):
            best = config["name"]
    return str(best)


def choose_method(item_item: float, popular: float) -> str:
    """Per-N serving method. A tie keeps most_popular."""
    return "item_item_fold_in" if round(item_item, 6) > round(popular, 6) else "most_popular"


def ship_decision(paired_by_n: dict[int, dict[str, Any]]) -> bool:
    """True when fold-in beats most-popular (paired low end above 0) at every ship N."""
    return all(float(paired_by_n[n]["low"]) > 0.0 for n in SHIP_NS)


def _fit(train: pd.DataFrame, config: dict[str, Any]) -> ItemItemCosineRecommender:
    model = ItemItemCosineRecommender(
        min_common=int(config["min_common"]),
        k_neighbors=int(config["k_neighbors"]),
        shrinkage=float(config["shrinkage"]),
        sparse_topk=True,
    ).fit(train)
    # Scoring a profile reads the similarity only. Drop the fit users' rows.
    model._user_item = sparse.csr_matrix((0, len(model._item_ids)), dtype=np.float64)
    model._user_index = {}
    return model


def _fold_in_lists(
    model: ItemItemCosineRecommender, profiles: dict[int, list[tuple[int, float]]]
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    uids = list(profiles)
    for start in range(0, len(uids), 500):
        batch = {uid: profiles[uid] for uid in uids[start : start + 500]}
        for uid, rows in model.topk_for_profiles(batch, TOP_K).items():
            out[int(uid)] = [int(item) for item, _score in rows]
    return out


def _popular_lists(
    ranked: list[int], profiles: dict[int, list[tuple[int, float]]]
) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for uid, profile in profiles.items():
        banned = {int(item) for item, _rating in profile}
        picked: list[int] = []
        for item in ranked:
            if item not in banned:
                picked.append(item)
                if len(picked) >= TOP_K:
                    break
        out[int(uid)] = picked
    return out


def _per_user(lists: dict[int, list[int]], relevant: dict[int, set[int]]) -> list[float]:
    return [float(ndcg_at_k(lists.get(uid, []), relevant[uid], TOP_K)) for uid in relevant]


def _round(value: float) -> float:
    return round(float(value), 6)


def _summary(
    lists: dict[int, list[int]],
    relevant: dict[int, set[int]],
    catalog: list[int],
    tail: set[int],
) -> tuple[dict[str, Any], list[float]]:
    values = _per_user(lists, relevant)
    mean, low, high = bootstrap_mean_ci(values, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=SEED)
    scored = {uid: lists.get(uid, []) for uid in relevant}
    return (
        {
            "ndcg@10": _round(mean),
            "ndcg@10_ci": {"low": _round(low), "high": _round(high), "mean": _round(mean)},
            "coverage@10": _round(catalog_coverage(scored, catalog, TOP_K)),
            "long_tail_share@10": _round(long_tail_share(scored, tail, TOP_K)),
        },
        values,
    )


def evaluate(
    held: pd.DataFrame,
    models: dict[str, ItemItemCosineRecommender],
    popular_ranked: list[int],
    catalog: list[int],
    tail: set[int],
) -> dict[int, dict[str, Any]]:
    """NDCG@10 per profile size for each model and for most-popular, on one user group."""
    by_n: dict[int, dict[str, Any]] = {}
    catalog_set = set(catalog)
    for n in PROFILE_NS:
        profiles, _hist, _frames, relevant, counts = _held_out_targets(
            held, n, catalog_set, THRESHOLD, target="all_later"
        )
        profiles = {uid: profiles[uid] for uid in relevant}
        popular, popular_values = _summary(
            _popular_lists(popular_ranked, profiles), relevant, catalog, tail
        )
        block: dict[str, Any] = {"counts": counts, "most_popular": popular, "item_item": {}}
        for name, model in models.items():
            summary, values = _summary(_fold_in_lists(model, profiles), relevant, catalog, tail)
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
            block["item_item"][name] = summary
        by_n[n] = block
        shown = {name: row["ndcg@10"] for name, row in block["item_item"].items()}
        print(f"  N={n}: most_popular={popular['ndcg@10']} item_item={shown}", flush=True)
    return by_n


def export_weights(
    model: ItemItemCosineRecommender,
    counts: np.ndarray,
    manifest: dict[str, Any],
    dest: Path,
) -> dict[str, int]:
    """Write the parameter files. No rating rows, user ids, or titles."""
    dest.mkdir(parents=True, exist_ok=True)
    similarity = sparse.csr_matrix(model._similarity).astype(np.float32)
    similarity.eliminate_zeros()
    sparse.save_npz(dest / "item_item_similarity.npz", similarity)
    np.save(dest / "item_ids.npy", np.asarray(model._item_ids, dtype=np.int64))
    np.save(dest / "item_counts.npy", np.asarray(counts, dtype=np.int64))
    (dest / "item_item_meta.json").write_text(
        json.dumps(model.hyperparams(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (dest / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {path.name: path.stat().st_size for path in sorted(dest.iterdir())}


def run(data_dir: Path, *, export: bool) -> dict[str, Any]:
    t0 = time.perf_counter()
    listed = load_movies("ml-latest-small", data_dir)
    ratings, _stats = load_ratings("ml-32m", data_dir, clean=True)
    n_all = int(len(ratings))
    catalog_ids = catalog_item_ids(ratings, listed, floor=MIN_FILM_RATINGS)
    ratings = ratings[ratings["item_id"].isin(set(catalog_ids.tolist()))]
    val_ids, test_ids, n_eligible = held_out_users(
        ratings, min_ratings=MIN_USER_RATINGS, n_held_out=N_HELD_OUT, seed=SEED
    )
    held_mask = ratings["user_id"].isin(set(val_ids.tolist()) | set(test_ids.tolist()))
    train = ratings[~held_mask]
    val = ratings[ratings["user_id"].isin(set(val_ids.tolist()))]
    test = ratings[ratings["user_id"].isin(set(test_ids.tolist()))]
    if set(train["user_id"].unique()) & (set(val_ids.tolist()) | set(test_ids.tolist())):
        raise AssertionError("a held-out user is in the fit rows")
    print(
        f"catalogue {len(catalog_ids)} films, fit rows {len(train)}, "
        f"validation users {len(val_ids)}, test users {len(test_ids)}",
        flush=True,
    )

    popular = MostPopularRecommender().fit(train)
    popular_ranked = [int(item) for item in popular._ranked_items]
    train_counts = train.groupby("item_id").size()
    counts = train_counts.reindex(catalog_ids).fillna(0).to_numpy(dtype=np.int64)
    catalog = [int(item) for item in catalog_ids.tolist()]
    n_head = max(1, int(round(HEAD_FRACTION * len(catalog))))
    tail = set(popular_ranked[n_head:]) | (set(catalog) - set(popular_ranked))

    models: dict[str, ItemItemCosineRecommender] = {}
    fit_sec: dict[str, float] = {}
    for config in CONFIGS:
        started = time.perf_counter()
        print(f"Fitting item-item {config}...", flush=True)
        models[config["name"]] = _fit(train, config)
        fit_sec[config["name"]] = round(time.perf_counter() - started, 3)
        print(f"  fit in {fit_sec[config['name']]}s", flush=True)
    del train

    print("Validation users:", flush=True)
    val_by_n = evaluate(val, models, popular_ranked, catalog, tail)
    selection = {
        name: val_by_n[SELECTION_N]["item_item"][name]["ndcg@10"] for name in models
    }
    kept = choose_config(selection)
    serving_by_n = {
        str(n): choose_method(
            val_by_n[n]["item_item"][kept]["ndcg@10"], val_by_n[n]["most_popular"]["ndcg@10"]
        )
        for n in PROFILE_NS
    }
    print(f"Kept configuration: {kept}. Serving by N: {serving_by_n}", flush=True)

    print("Test users (scored once):", flush=True)
    test_by_n = evaluate(test, {kept: models[kept]}, popular_ranked, catalog, tail)
    paired = {n: test_by_n[n]["item_item"][kept]["paired_minus_most_popular"] for n in PROFILE_NS}
    ships = ship_decision(paired)
    print(f"Ship rule passes: {ships}", flush=True)

    kept_config = next(config for config in CONFIGS if config["name"] == kept)
    payload: dict[str, Any] = {
        "adr": "0034",
        "catalog": {
            "listed_in": "ml-latest-small/movies.csv",
            "min_film_ratings": MIN_FILM_RATINGS,
            "n_films": len(catalog),
            "n_listed": int(len(listed)),
            "n_ratings_all": n_all,
            "n_ratings_on_catalog": int(len(ratings)),
        },
        "configs": [dict(config) for config in CONFIGS],
        "dataset": "ml-32m",
        "dataset_sha256": DATASET_SHA256["ml-32m"],
        "titles_dataset_sha256": DATASET_SHA256["ml-latest-small"],
        "decision": {
            "kept_config": kept,
            "rule": (
                "Ships when paired test NDCG@10, item_item_fold_in minus most_popular, "
                "has an interval low end above 0 at N=3, N=5, and N=10."
            ),
            "selection_n": SELECTION_N,
            "selection_scores": selection,
            "serving_by_n": serving_by_n,
            "ships": bool(ships),
        },
        "fit_sec": fit_sec,
        "host": host_info(),
        "library_versions": library_versions(),
        "protocol": {
            "alpha": ALPHA,
            "head_fraction": HEAD_FRACTION,
            "min_user_ratings": MIN_USER_RATINGS,
            "n_bootstrap": N_BOOTSTRAP,
            "n_eligible_users": n_eligible,
            "n_held_out": N_HELD_OUT,
            "profile_ns": list(PROFILE_NS),
            "relevance_threshold": THRESHOLD,
            "seed": SEED,
            "target": "all_later",
            "test_user_ids_sha256": hashlib.sha256(test_ids.tobytes()).hexdigest(),
            "validation_user_ids_sha256": hashlib.sha256(val_ids.tobytes()).hexdigest(),
        },
        "test": {str(n): test_by_n[n] for n in PROFILE_NS},
        "validation": {str(n): val_by_n[n] for n in PROFILE_NS},
    }
    if ships and export:
        manifest = {
            "adr": "0034",
            "candidate_k": int(kept_config["k_neighbors"]),
            "candidate_source": "item_item_fold_in",
            "dataset": "ml-32m",
            "dataset_sha256": DATASET_SHA256["ml-32m"],
            "kind": "catalog",
            "min_film_ratings": MIN_FILM_RATINGS,
            "relevance_threshold": THRESHOLD,
            "results": "results/extended-catalog/ml-32m.json",
            "serving_by_n": serving_by_n,
            "titles_dataset": "ml-latest-small",
            "titles_dataset_sha256": DATASET_SHA256["ml-latest-small"],
            "catalog_weights_version": WEIGHTS_VERSION,
            "weights_only": True,
        }
        sizes = export_weights(models[kept], counts, manifest, WEIGHTS_DIR)
        payload["weights"] = {"files": sizes, "total_bytes": int(sum(sizes.values()))}
        print(f"Wrote {WEIGHTS_DIR}: {payload['weights']['total_bytes']} bytes", flush=True)
    payload["runtime_sec"] = round(time.perf_counter() - t0, 3)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {RESULTS_PATH}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0034 extended Discover catalogue.")
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument(
        "--no-export", action="store_true", help="Score only. Do not write space/weights/extended."
    )
    args = parser.parse_args()
    run(Path(args.data_dir), export=not args.no_export)


if __name__ == "__main__":
    main()
