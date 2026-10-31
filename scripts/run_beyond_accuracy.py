#!/usr/bin/env python3
"""Score the committed ml-1m models on the ADR-0017 metrics.

Rebuilds each model from ``results/ml-1m.json``. Does not tune, does not
change a hyperparameter, and does not write any other results file. ml-32M
is out of scope.

    python scripts/run_beyond_accuracy.py
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
COMMITTED_PATH = ROOT / "results" / "ml-1m.json"
DEFAULT_OUTPUT = ROOT / "results" / "beyond-accuracy" / "ml-1m.json"

MODEL_ORDER = [
    "most_popular",
    "item_item_cosine",
    "als_tuned",
    "ease",
    "rp3beta",
    "two_tower",
    "lambdarank",
]

FUTURE_WORK = "ml-32M is future work. This panel is ml-1m only."
REPRODUCTION = "python scripts/run_beyond_accuracy.py"


def _round(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def _matches_4(left: float, right: float) -> bool:
    return f"{left:.4f}" == f"{right:.4f}"


def _load_committed() -> dict[str, Any]:
    if not COMMITTED_PATH.is_file():
        raise SystemExit(f"Missing {COMMITTED_PATH}")
    return json.loads(COMMITTED_PATH.read_text(encoding="utf-8"))


def _two_tower_hp(block: dict[str, Any]) -> tuple[dict[str, Any], int]:
    hp = dict(block)
    epochs = int(hp.pop("refit_epochs"))
    best = int(hp.pop("best_epoch"))
    if epochs != best:
        raise SystemExit(f"two-tower refit_epochs {epochs} != best_epoch {best}")
    hp.pop("seeds", None)
    return hp, epochs


def _committed_ndcg(name: str, committed: dict[str, Any], seed: int) -> tuple[float, str, float]:
    """Return the list's committed NDCG, its source, and the headline number."""
    headline = float(committed["metrics"][name]["ndcg@10"])
    if name == "two_tower":
        rows = committed["two_tower"]["per_seed"]
        row = next(item for item in rows if int(item["seed"]) == seed)
        source = f"results/ml-1m.json two_tower.per_seed seed {seed}"
        return float(row["ndcg@10"]), source, headline
    if name == "lambdarank":
        rows = committed["ranker"]["per_seed"]
        row = next(item for item in rows if int(item["seed"]) == seed)
        source = f"results/ml-1m.json ranker.per_seed seed {seed}"
        return float(row["ndcg@10"]), source, headline
    return headline, f"results/ml-1m.json metrics.{name}.ndcg@10", headline


def _model_entry(
    *,
    scored: dict[str, Any],
    hyperparameters: dict[str, Any],
    committed_ndcg: float,
    source: str,
    headline: float,
    list_seed: int | None,
    fit_sec: float,
) -> dict[str, Any]:
    point_keys = [
        "coverage@10",
        "novelty@10",
        "intra_list_diversity@10",
        "mean_popularity@10",
        "long_tail_share@10",
        "ndcg@10",
    ]
    block: dict[str, Any] = {key: _round(scored[key]) for key in point_keys}
    block["n_eval_users"] = int(scored["n_eval_users"])
    block["hyperparameters"] = hyperparameters
    block["list_seed"] = list_seed
    block["fit_sec"] = round(float(fit_sec), 3)
    block["confidence_intervals"] = {
        key: {bound: _round(value) for bound, value in bounds.items()}
        for key, bounds in scored["confidence_intervals"].items()
    }
    committed_rounded = _round(committed_ndcg)
    headline_rounded = _round(headline)
    block["ndcg@10_committed"] = committed_rounded
    block["ndcg@10_committed_source"] = source
    block["ndcg@10_headline"] = headline_rounded
    block["ndcg@10_matches_committed_at_4_decimals"] = _matches_4(
        block["ndcg@10"], committed_rounded
    )
    block["ndcg@10_matches_headline_at_4_decimals"] = _matches_4(
        block["ndcg@10"], headline_rounded
    )
    return block


def _ensure_data(data_dir: Path, download: bool) -> None:
    from movielens_recommender.data import download_dataset
    from movielens_recommender.movies import movies_path

    if movies_path("ml-1m", data_dir).is_file():
        return
    if not download:
        raise SystemExit(
            f"ml-1m is not in {data_dir}. Re-run without --no-download."
        )
    print(f"Downloading ml-1m into {data_dir}", flush=True)
    download_dataset("ml-1m", data_dir)


def _prepare(committed: dict[str, Any], data_dir: Path):
    from movielens_recommender.data import load_ratings
    from movielens_recommender.movies import load_movies
    from movielens_recommender.split import (
        SplitConfig,
        SplitResult,
        apply_cold_start_policy,
        time_based_split,
    )

    ratings, _clean = load_ratings("ml-1m", data_dir, clean=True)
    movies = load_movies("ml-1m", data_dir)
    split_block = committed["split"]["config"]
    threshold = float(committed["relevance_threshold"])
    split = time_based_split(
        ratings,
        SplitConfig(
            min_ratings=int(split_block["min_ratings"]),
            test_fraction=float(split_block["test_fraction"]),
            val_fraction=float(split_block["val_fraction"]),
            relevance_threshold=threshold,
        ),
    )
    summary = split.summary()
    for key in (
        "n_train",
        "n_test",
        "n_val",
        "n_full_train",
        "n_full_train_items",
        "n_full_train_users",
        "n_users_kept",
    ):
        got = int(summary[key])
        expected = int(committed["split"][key])
        if got != expected:
            raise SystemExit(f"split {key} is {got}, committed file has {expected}")
    if split.val is None or split.val.empty:
        raise SystemExit("the committed split has a validation window")
    full_train = split.full_train
    seen, relevant, cold = apply_cold_start_policy(
        SplitResult(
            train=full_train,
            test=split.test,
            config=split.config,
            n_users_kept=split.n_users_kept,
            n_users_dropped=split.n_users_dropped,
        ),
        relevance_threshold=threshold,
    )
    expected_users = int(committed["metrics"]["most_popular"]["n_eval_users"])
    if len(relevant) != expected_users:
        raise SystemExit(
            f"eval users {len(relevant)} != committed {expected_users}"
        )
    return movies, split, full_train, seen, relevant, cold, summary


def _lists(model: Any, seen: dict, relevant: dict) -> dict[int, list[int]]:
    from movielens_recommender.evaluate import collect_recommendations

    return collect_recommendations(model.recommend, seen, relevant, max_k=10)


def _fit_baselines(committed, full_train, seen, relevant, seed: int, threshold: float):
    from movielens_recommender.scale import release_memory
    from movielens_recommender.tune import build_model

    specs = [
        "most_popular",
        "item_item_cosine",
        "als_tuned",
        "ease",
        "rp3beta",
    ]
    lists: dict[str, dict[int, list[int]]] = {}
    fit_sec: dict[str, float] = {}
    recorded: dict[str, dict[str, Any]] = {}
    for name in specs:
        hp = dict(committed["hyperparameters"][name])
        print(f"Fitting {name} on full train", flush=True)
        t0 = time.perf_counter()
        model, recorded_hp = build_model(
            name,
            full_train,
            hyperparams=hp,
            relevance_threshold=threshold,
            seed=seed,
        )
        lists[name] = _lists(model, seen, relevant)
        fit_sec[name] = time.perf_counter() - t0
        recorded[name] = recorded_hp
        del model
        release_memory()
        print(f"  {name} lists in {fit_sec[name]:.1f}s", flush=True)
    return lists, fit_sec, recorded


def _fit_two_tower(committed, movies, frame, data_dir: Path, seed: int, threshold: float):
    from movielens_recommender.cli import set_seeds
    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    hp, epochs = _two_tower_hp(committed["hyperparameters"]["two_tower"])
    set_seeds(seed)
    print(
        f"Fitting two-tower for {epochs} epochs on {len(frame)} rows (seed {seed})",
        flush=True,
    )
    t0 = time.perf_counter()
    model, _features, train_result = fit_two_tower_recommender(
        frame,
        dataset="ml-1m",
        data_dir=str(data_dir),
        movies=movies,
        hyperparams=hp,
        seed=seed,
        relevance_threshold=threshold,
        n_epochs=epochs,
        val_split=None,
        show_progress=True,
    )
    return model, hp, epochs, time.perf_counter() - t0, train_result


def _fit_item_item(frame, hp: dict[str, Any]):
    from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender

    return ItemItemCosineRecommender(
        min_common=int(hp["min_common"]),
        k_neighbors=int(hp["k_neighbors"]),
        shrinkage=float(hp["shrinkage"]),
    ).fit(frame)


def _ranker_lists(
    committed: dict[str, Any],
    *,
    movies,
    split,
    full_train,
    data_dir: Path,
    seen,
    relevant,
    seed: int,
    threshold: float,
    full_two_tower_topk,
):
    """Train the S4 LambdaRank on two-tower candidates and return test top-10 lists."""
    from movielens_recommender.ranker.candidates import (
        materialize_candidate_set,
        score_rank_maps,
    )
    from movielens_recommender.ranker.features import (
        build_feature_context,
        categorical_feature_names,
        relevance_pairs,
    )
    from movielens_recommender.ranker.pipeline import _blocks_for_users, _rerank
    from movielens_recommender.ranker.train import (
        assemble_matrices,
        train_with_early_stop_then_refit,
    )
    from movielens_recommender.scale import release_memory

    rank_hp = dict(committed["hyperparameters"]["lambdarank"])
    if rank_hp.get("candidate_set") != "two_tower":
        raise SystemExit(
            f"committed ranker candidate set is {rank_hp.get('candidate_set')!r}, "
            "not two_tower"
        )
    k = int(rank_hp["candidate_k"])
    ii_hp = dict(rank_hp["retriever_item_item"])
    committed_features = {
        row["feature"] for row in committed["ranker"]["feature_importance_gain"]
    }

    print("Fitting tuned item-item on full train for ranker features", flush=True)
    ii_full_model = _fit_item_item(full_train, ii_hp)
    ii_full = ii_full_model.topk_with_scores(k)
    del ii_full_model
    release_memory()

    print("Fitting ranker retrievers on fit-train", flush=True)
    ii_fit_model = _fit_item_item(split.train, ii_hp)
    ii_val = ii_fit_model.topk_with_scores(k)
    del ii_fit_model
    release_memory()
    tt_fit, _hp, _epochs, _sec, _train = _fit_two_tower(
        committed, movies, split.train, data_dir, seed, threshold
    )
    tt_val = tt_fit.topk_with_scores(k)
    del tt_fit
    release_memory()

    ctx_fit = build_feature_context(
        split.train,
        movies,
        users=None,
        mode="off",
        group_frame=split.train,
    )
    if set(ctx_fit.names) != committed_features:
        missing = sorted(committed_features - set(ctx_fit.names))
        extra = sorted(set(ctx_fit.names) - committed_features)
        raise SystemExit(
            "ranker features do not match the committed S4 set. "
            f"missing={missing} extra={extra}"
        )
    lists_val = materialize_candidate_set("two_tower", ii_val, tt_val, k)
    blocks = _blocks_for_users(
        ctx_fit,
        lists_val,
        score_rank_maps(ii_val),
        score_rank_maps(tt_val),
        relevance_pairs(split.val, relevance_threshold=threshold),
        users={int(uid) for uid in split.val["user_id"].unique()},
    )
    matrices = assemble_matrices(blocks, ctx_fit.names)
    matrices.categorical_features = categorical_feature_names("off")
    del ii_val, tt_val, lists_val
    release_memory()

    print(f"Training LambdaRank on two-tower candidates (seed {seed})", flush=True)
    t0 = time.perf_counter()
    trained = train_with_early_stop_then_refit(
        matrices,
        split_seed=int(rank_hp["early_stop_user_split_seed"]),
        early_stop_fraction=float(rank_hp["early_stop_fraction"]),
        ranker_seed=seed,
        num_boost_round=int(rank_hp["num_boost_round_cap"]),
        early_stopping_rounds=int(rank_hp["early_stopping_rounds"]),
        learning_rate=float(rank_hp["learning_rate"]),
        num_leaves=int(rank_hp["num_leaves"]),
        min_data_in_leaf=int(rank_hp["min_data_in_leaf"]),
        feature_fraction=float(rank_hp["feature_fraction"]),
        bagging_fraction=float(rank_hp["bagging_fraction"]),
    )
    train_sec = time.perf_counter() - t0
    print(
        f"  best_iteration {trained.best_iteration} "
        f"(committed primary seed {rank_hp['best_iteration_primary_seed']})",
        flush=True,
    )

    ctx_full = build_feature_context(
        full_train,
        movies,
        users=None,
        mode="off",
        group_frame=split.train,
    )
    lists_full = materialize_candidate_set("two_tower", ii_full, full_two_tower_topk, k)
    ranked = _rerank(
        trained.booster,
        ctx_full,
        lists_full,
        score_rank_maps(ii_full),
        score_rank_maps(full_two_tower_topk),
        trained.refit_info["feature_names"],
    )
    from movielens_recommender.ranker.candidates import PrecomputedRecommender

    lists = _lists(PrecomputedRecommender(ranked), seen, relevant)
    info = {
        "best_iteration": int(trained.best_iteration),
        "committed_best_iteration": int(rank_hp["best_iteration_primary_seed"]),
        "candidate_set": "two_tower",
        "candidate_k": k,
        "demographics": "off",
        "n_features": len(ctx_fit.names),
    }
    return lists, train_sec, info, rank_hp


def _score_all(lists, movies, full_train, relevant, committed):
    from movielens_recommender.beyond_accuracy import (
        ITEM_REPRESENTATION,
        METRIC_DEFINITIONS,
        genre_item_vectors,
        long_tail_items,
        score_recommendation_lists,
    )

    catalog = [int(i) for i in full_train["item_id"].unique()]
    popularity = {
        int(item): float(count)
        for item, count in full_train.groupby("item_id").size().items()
    }
    tail = long_tail_items(full_train)
    needed = set(catalog)
    for recs in lists.values():
        for items in recs.values():
            needed.update(int(i) for i in items)
    vectors = genre_item_vectors(movies, sorted(needed))
    boot = committed["bootstrap"]
    seed = int(committed["seed"])
    scored = {}
    for name, recs in lists.items():
        print(f"Scoring {name}", flush=True)
        scored[name] = score_recommendation_lists(
            recs,
            relevant=relevant,
            catalog=catalog,
            item_popularity=popularity,
            item_vectors=vectors,
            tail_items=tail,
            k=10,
            n_bootstrap=int(boot["n_bootstrap"]),
            alpha=float(boot["alpha"]),
            seed=seed,
        )
    return scored, {
        "definitions": METRIC_DEFINITIONS,
        "item_representation": ITEM_REPRESENTATION,
        "n_train_items": len(catalog),
        "n_tail_items": len(tail),
        "n_head_items": len(catalog) - len(tail),
        "tail_fraction": len(tail) / len(catalog),
    }


def _user_hash(relevant: dict) -> str:
    payload = ",".join(str(uid) for uid in sorted(int(uid) for uid in relevant))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_payload(committed: dict[str, Any], data_dir: Path) -> dict[str, Any]:
    from movielens_recommender.cli import host_info, library_versions
    from movielens_recommender.data import DATASET_SHA256, DATASET_VERSION_LABELS
    from movielens_recommender.scale import release_memory

    seed = int(committed["seed"])
    threshold = float(committed["relevance_threshold"])
    movies, split, full_train, seen, relevant, cold, summary = _prepare(
        committed, data_dir
    )
    started = time.perf_counter()
    lists, fit_sec, recorded = _fit_baselines(
        committed, full_train, seen, relevant, seed, threshold
    )

    tt_model, tt_hp, epochs, tt_sec, tt_train = _fit_two_tower(
        committed, movies, full_train, data_dir, seed, threshold
    )
    lists["two_tower"] = _lists(tt_model, seen, relevant)
    full_tt_topk = tt_model.topk_with_scores(
        int(committed["hyperparameters"]["lambdarank"]["candidate_k"])
    )
    fit_sec["two_tower"] = tt_sec
    del tt_model
    release_memory()
    print(f"  two_tower lists in {tt_sec:.1f}s", flush=True)

    rank_lists, rank_sec, rank_info, rank_hp = _ranker_lists(
        committed,
        movies=movies,
        split=split,
        full_train=full_train,
        data_dir=data_dir,
        seen=seen,
        relevant=relevant,
        seed=seed,
        threshold=threshold,
        full_two_tower_topk=full_tt_topk,
    )
    lists["lambdarank"] = rank_lists
    fit_sec["lambdarank"] = rank_sec
    del full_tt_topk
    release_memory()

    scored, definitions = _score_all(lists, movies, full_train, relevant, committed)
    hyperparameters = dict(recorded)
    hyperparameters["two_tower"] = {**tt_hp, "refit_epochs": epochs, "list_seed": seed}
    hyperparameters["lambdarank"] = {
        "candidate_k": rank_info["candidate_k"],
        "candidate_set": "two_tower",
        "demographics": "off",
        "list_seed": seed,
        "learning_rate": rank_hp["learning_rate"],
        "num_leaves": rank_hp["num_leaves"],
        "min_data_in_leaf": rank_hp["min_data_in_leaf"],
        "feature_fraction": rank_hp["feature_fraction"],
        "bagging_fraction": rank_hp["bagging_fraction"],
        "num_boost_round_cap": rank_hp["num_boost_round_cap"],
        "early_stopping_rounds": rank_hp["early_stopping_rounds"],
        "early_stop_fraction": rank_hp["early_stop_fraction"],
        "early_stop_user_split_seed": rank_hp["early_stop_user_split_seed"],
        "best_iteration": rank_info["best_iteration"],
        "committed_best_iteration": rank_info["committed_best_iteration"],
        "retriever_item_item": rank_hp["retriever_item_item"],
        "retriever_two_tower": rank_hp["retriever_two_tower"],
        "retriever_two_tower_epochs": rank_hp["retriever_two_tower_epochs"],
        "n_features": rank_info["n_features"],
    }

    models = {}
    for name in MODEL_ORDER:
        committed_ndcg, source, headline = _committed_ndcg(name, committed, seed)
        list_seed = seed if name in {"two_tower", "lambdarank"} else None
        models[name] = _model_entry(
            scored=scored[name],
            hyperparameters=hyperparameters[name],
            committed_ndcg=committed_ndcg,
            source=source,
            headline=headline,
            list_seed=list_seed,
            fit_sec=fit_sec[name],
        )

    note = (
        f"Two-tower and LambdaRank lists are seed {seed} only. "
        "Their committed NDCG@10 is that seed's test number in results/ml-1m.json. "
        "The headline NDCG@10 for those two is the mean over three seeds "
        "and is stored beside it."
    )
    return {
        "adr": "docs/adr/0017-beyond-accuracy.md",
        "bootstrap": {
            "alpha": float(committed["bootstrap"]["alpha"]),
            "n_bootstrap": int(committed["bootstrap"]["n_bootstrap"]),
            "seed": seed,
        },
        "cold_start": cold.to_dict(),
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "dataset_version": DATASET_VERSION_LABELS["ml-1m"],
        "definitions": definitions["definitions"],
        "eval_users": {
            "matches_committed_count": True,
            "n": len(relevant),
            "user_ids_sha256": _user_hash(relevant),
        },
        "future_work": FUTURE_WORK,
        "host": host_info(),
        "item_representation": definitions["item_representation"],
        "k": 10,
        "library_versions": library_versions(),
        "long_tail": {
            "n_head_items": definitions["n_head_items"],
            "n_tail_items": definitions["n_tail_items"],
            "n_train_items": definitions["n_train_items"],
            "tail_fraction": _round(definitions["tail_fraction"]),
        },
        "model_order": MODEL_ORDER,
        "models": models,
        "n_eval_users": len(relevant),
        "ndcg_comparison_note": note,
        "protocol": {
            "demographics": "off",
            "fit_matrix": "full_train",
            "no_tuning": True,
            "ranker_candidate_set": "two_tower",
            "relevance_threshold": threshold,
            "split": "per-user time split from results/ml-1m.json",
            "tie_break": "top_n_indices (ADR-0015)",
            "two_tower_epochs_trained": int(tt_train.epochs_trained),
        },
        "relevance_threshold": threshold,
        "reproduction_command": REPRODUCTION,
        "runtime_sec": round(time.perf_counter() - started, 3),
        "seed": seed,
        "split": summary,
        "uncertainty_policy": scored["most_popular"]["uncertainty_policy"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail if ml-1m is not already under --data-dir",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.resolve() == COMMITTED_PATH.resolve():
        raise SystemExit("refusing to overwrite results/ml-1m.json")
    data_dir = Path(args.data_dir)
    _ensure_data(data_dir, download=not args.no_download)
    payload = build_payload(_load_committed(), data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}", flush=True)
    for name in payload["model_order"]:
        block = payload["models"][name]
        flag = "yes" if block["ndcg@10_matches_committed_at_4_decimals"] else "NO"
        print(
            f"  {name}: ndcg@10 {block['ndcg@10']:.6f} "
            f"committed {block['ndcg@10_committed']:.6f} match={flag}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
