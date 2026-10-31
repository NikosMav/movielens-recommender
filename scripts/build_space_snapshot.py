"""Fit ml-1m serving models and write weight files under ``space/weights``.

The Space does not run this script. It downloads ml-1m and loads the files
this command writes. Results JSON is hashed before and after; any byte change
aborts. ``finish_round2`` is not called, because that function rewrites
``results/cold-start/ml-1m.json``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.cold_start import (
    HOLDOUT_FRACTION,
    _fit_ease,
    _fit_item_item,
    _load_tuned_hyperparams,
    assert_held_out_absent,
    eligible_user_ids,
    exclude_held_out_users,
    select_held_out_user_ids,
)
from movielens_recommender.cold_start_rank import (
    COLD_START_NS,
    PROFILE_BUCKETS,
    RANK_CANDIDATE_KS,
    _fit_tower,
    _matrices_for_k,
    _retrieve,
    _tower_hyperparams,
    simulated_queries,
)
from movielens_recommender.config import load_config
from movielens_recommender.data import DATASET_SHA256, load_movies, load_ratings
from movielens_recommender.ranker.features import build_feature_context
from movielens_recommender.ranker.train import fit_lambdarank, lgb_params
from movielens_recommender.serving.build import build_artifacts, git_sha
from movielens_recommender.serving.cold import save_cold_start_bundle
from movielens_recommender.split import SplitConfig, time_based_split

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
WEIGHTS = REPO_ROOT / "space" / "weights"


def _load_runtime():
    path = REPO_ROOT / "space" / "runtime.py"
    spec = importlib.util.spec_from_file_location("space_runtime_snapshot", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*.json"))
    }


def _cold_snapshot(config, choices: dict) -> Path:
    """Refit the new-user models the way ``finish_round2`` does, without writing JSON."""
    ratings, _stats = load_ratings(config.dataset, config.data_dir, clean=True)
    movies = load_movies(config.dataset, config.data_dir)
    threshold = float(config.eval.relevance_threshold)
    seed = int(config.seed)
    eligible = eligible_user_ids(ratings, config.split.min_ratings)
    held_ids = select_held_out_user_ids(eligible, fraction=HOLDOUT_FRACTION, seed=seed)
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
    assert_held_out_absent(held_ids, full_train, split.train)

    _tuned_tower, item_hp, ease_hp = _load_tuned_hyperparams(Path(config.results_dir))
    del _tuned_tower
    winner_trial = {
        "history_only": choices["history_only"],
        "user_id_dropout": choices["user_id_dropout"],
        "best_epoch": choices["representation_best_epoch"],
    }
    hyperparams = _tower_hyperparams(choices["fixed_hyperparams"], winner_trial)
    print("Refitting cold-start retrievers on the 90% complement...", flush=True)
    tower = _fit_tower(
        full_train,
        config=config,
        movies=movies,
        hyperparams=hyperparams,
        epochs=int(winner_trial["best_epoch"]),
        threshold=threshold,
    )
    item_item = _fit_item_item(full_train, dict(item_hp))
    ease = _fit_ease(full_train, dict(ease_hp))
    popular = MostPopularRecommender().fit(full_train)
    catalog = {int(item) for item in full_train["item_id"].unique()}
    names = list(choices["feature_names"])
    context = build_feature_context(
        full_train,
        movies,
        users=None,
        mode="off",
        group_frame=None,
        relevance_threshold=threshold,
    )
    context.names = names
    pool = np.unique(full_train["user_id"].to_numpy(dtype=np.int64))
    queries = simulated_queries(remain, pool, catalog, threshold)
    max_k = max(RANK_CANDIDATE_KS)
    retrieved = {
        n: _retrieve(
            queries[n],
            item_item=item_item,
            ease=ease,
            tower=tower,
            popular=popular,
            k=max_k,
        )
        for n in COLD_START_NS
    }
    print(
        f"Refitting LambdaRank for {choices['best_iteration']} trees, no second early stop...",
        flush=True,
    )
    matrices = _matrices_for_k(queries, retrieved, int(choices["candidate_k"]), context, names)
    rank_cfg = config.models.ranker
    params = lgb_params(
        seed=seed,
        learning_rate=float(rank_cfg.learning_rate),
        num_leaves=int(rank_cfg.num_leaves),
        min_data_in_leaf=int(rank_cfg.min_data_in_leaf),
        feature_fraction=float(rank_cfg.feature_fraction),
        bagging_fraction=float(rank_cfg.bagging_fraction),
    )
    booster, info = fit_lambdarank(
        matrices,
        params=params,
        num_boost_round=int(choices["best_iteration"]),
        valid=None,
        early_stopping_rounds=None,
    )
    if int(info["num_trees"]) != int(choices["best_iteration"]):
        raise AssertionError("cold-start refit did not keep the recorded iteration count")
    destination = Path("artifacts") / config.dataset / "cold_start"
    return save_cold_start_bundle(
        destination,
        movies=movies,
        item_item=item_item,
        ease=ease,
        two_tower=tower,
        booster=booster,
        context=context,
        popular_item_ids=np.asarray(popular._ranked_items, dtype=np.int64),
        manifest={
            "dataset": config.dataset,
            "dataset_sha256": DATASET_SHA256[config.dataset],
            "git_sha": git_sha(REPO_ROOT),
            "candidate_source": "cold_start_union",
            "candidate_k": int(choices["candidate_k"]),
            "ranker_mode": "cold_start_ranker",
            "demographics_missing": False,
            "history_only": bool(choices["history_only"]),
            "user_id_dropout": float(choices["user_id_dropout"]),
            "best_iteration": int(choices["best_iteration"]),
            "feature_names": list(names),
            "relevance_threshold": threshold,
            "serving_by_n": {str(n): choices["serving_by_n"][str(n)] for n in COLD_START_NS},
            "profile_buckets": list(PROFILE_BUCKETS),
        },
    )


def main() -> None:
    before = _result_bytes(RESULTS)
    config = load_config(REPO_ROOT / "configs" / "ml-1m.yaml")
    runtime = _load_runtime()
    choices_path = Path(config.results_dir) / "cold-start" / f"{config.dataset}.json"
    choices = runtime.load_round2_choices(REPO_ROOT / choices_path)
    production = build_artifacts(config, show_progress=True)
    cold = _cold_snapshot(config, choices)
    runtime.export_space_weights(
        production,
        cold,
        WEIGHTS,
        cold_train_protocol="ml1m_complement_full_train",
    )
    after = _result_bytes(RESULTS)
    if before != after:
        changed = sorted(set(before) | set(after))
        drifted = [name for name in changed if before.get(name) != after.get(name)]
        raise SystemExit(f"results JSON bytes changed: {drifted}")
    sizes = runtime.weight_sizes(WEIGHTS)
    total = 0
    for name, size in sizes:
        total += size
        print(f"{size} {name}")
    print(f"{total} space/weights total bytes")


if __name__ == "__main__":
    main()
