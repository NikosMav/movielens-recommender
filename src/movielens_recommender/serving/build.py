"""Fit the production ranker on the training matrix and write a serving snapshot.

This is the ADR-0007 refit, not the ablation suite and not a metric run.
Hyperparameters are read from the committed tuning files. The candidate set
and ``best_iteration`` are computed again on the validation window.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.cli import set_seeds
from movielens_recommender.config import RunConfig
from movielens_recommender.data import DATASET_SHA256, download_dataset, load_ratings, load_users
from movielens_recommender.movies import load_movies
from movielens_recommender.ranker.candidates import choose_candidate_set, score_rank_maps
from movielens_recommender.ranker.features import (
    assert_context_uses_only,
    assert_group_affinity_uses_only,
    build_feature_context,
    categorical_feature_names,
    relevance_pairs,
)
from movielens_recommender.ranker.pipeline import (
    _blocks_for_users,
    _candidate_reports,
    _ranker_kwargs,
)
from movielens_recommender.ranker.train import assemble_matrices, train_with_early_stop_then_refit
from movielens_recommender.serving.bundle import ServingBundle, manifest_template, save_bundle
from movielens_recommender.split import SplitConfig, SplitResult, time_based_split
from movielens_recommender.two_tower.train import fit_two_tower_recommender


def git_sha(cwd: Path | str | None = None) -> str:
    """HEAD sha, with a ``-dirty`` suffix when the worktree is not clean."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    try:
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return sha
    if dirty != 0:
        return f"{sha}-dirty"
    return sha


def load_production_hyperparams(config: RunConfig) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Validation-chosen item–item and two-tower hyperparameters.

    ``build-artifacts`` does not search again. The files are the ones written
    by ``movielens-recommender run``.
    """
    results = Path(config.results_dir)
    baseline_path = results / "tuning" / f"{config.dataset}.json"
    tower_path = results / "tuning" / f"two_tower_{config.dataset}.json"
    if not baseline_path.is_file() or not tower_path.is_file():
        raise FileNotFoundError(
            "Tuned hyperparameters were not found at "
            f"{baseline_path} and {tower_path}. "
            "Run movielens-recommender run for this dataset first. "
            "build-artifacts does not retune."
        )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    tower = json.loads(tower_path.read_text(encoding="utf-8"))
    if "item_item_cosine" not in baseline:
        raise ValueError(f"{baseline_path} has no item_item_cosine block")
    item_hp = dict(baseline["item_item_cosine"]["best_hyperparams"])
    block = tower["two_tower"]
    return item_hp, dict(block["best_hyperparams"]), int(block["best_epoch"])


def fit_serving_bundle(
    *,
    config: RunConfig,
    split: SplitResult,
    movies: pd.DataFrame,
    users: pd.DataFrame | None,
    item_item_hp: Mapping[str, Any],
    two_tower_hp: Mapping[str, Any],
    two_tower_epochs: int,
    dataset_sha256: str,
    git_sha_value: str,
    show_progress: bool = False,
) -> ServingBundle:
    """Fit retrievers and the primary-seed refit ranker. No test evaluation."""
    if split.val is None or split.val.empty or split.test is None:
        raise ValueError("serving fit requires a non-empty validation window and a test frame")
    if not config.models.two_tower.enabled or not config.models.ranker.enabled:
        raise ValueError("build-artifacts requires two_tower and ranker to be enabled")
    rank_cfg = config.models.ranker
    demo_mode = str(rank_cfg.demographics)
    if demo_mode != "off" and (users is None or users.empty):
        raise ValueError(f"demographics={demo_mode!r} requires a users frame")
    k = int(rank_cfg.candidate_k)
    threshold = float(config.eval.relevance_threshold)
    primary_seed = int(config.seed)
    set_seeds(primary_seed)

    print("Fitting item-item and two-tower on fit-train...", flush=True)
    item_fit = _fit_item_item(split.train, item_item_hp)
    tower_fit = _fit_tower(
        split.train,
        config=config,
        movies=movies,
        hyperparams=two_tower_hp,
        n_epochs=two_tower_epochs,
        seed=primary_seed,
        show_progress=show_progress,
    )
    print(f"Choosing the candidate set on validation recall at K={k}...", flush=True)
    ii_val = item_fit.topk_with_scores(k)
    tt_val = tower_fit.topk_with_scores(k)
    reports, lists_val = _candidate_reports(
        ii_val,
        tt_val,
        k=k,
        train=split.train,
        labels=split.val,
        relevance_threshold=threshold,
    )
    winner = choose_candidate_set(reports)
    print(f"Validation candidate-set winner: {winner}", flush=True)

    ctx_fit = build_feature_context(
        split.train,
        movies,
        users=users,
        mode=demo_mode,
        relevance_threshold=threshold,
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

    ii_maps = score_rank_maps(ii_val)
    tt_maps = score_rank_maps(tt_val)
    positives = relevance_pairs(split.val, relevance_threshold=threshold)
    label_users = {int(uid) for uid in split.val["user_id"].unique()}
    blocks = _blocks_for_users(
        ctx_fit,
        lists_val[winner],
        ii_maps,
        tt_maps,
        positives,
        users=label_users,
    )
    matrices = assemble_matrices(blocks, ctx_fit.names)
    matrices.categorical_features = categorical_feature_names(demo_mode)
    print(
        f"Training LambdaRank on {winner} (seed={primary_seed}, groups={len(matrices.group)})...",
        flush=True,
    )
    trained = train_with_early_stop_then_refit(
        matrices,
        split_seed=primary_seed,
        early_stop_fraction=float(rank_cfg.early_stop_fraction),
        ranker_seed=primary_seed,
        num_boost_round=int(rank_cfg.num_boost_round),
        early_stopping_rounds=int(rank_cfg.early_stopping_rounds),
        **_ranker_kwargs(config),
    )
    # Fit-train retrievers are only for ranker labels. Scoring uses full train.
    del item_fit, tower_fit, ii_maps, tt_maps

    print("Refitting item-item and two-tower on full train...", flush=True)
    item_full = _fit_item_item(split.full_train, item_item_hp)
    tower_full = _fit_tower(
        split.full_train,
        config=config,
        movies=movies,
        hyperparams=two_tower_hp,
        n_epochs=two_tower_epochs,
        seed=primary_seed,
        show_progress=show_progress,
    )
    ctx_full = build_feature_context(
        split.full_train,
        movies,
        users=users,
        mode=demo_mode,
        group_frame=split.train,
        relevance_threshold=threshold,
    )
    assert_context_uses_only(ctx_full, split.full_train, split.test)
    if demo_mode in {"affinity", "both"}:
        assert users is not None
        assert_group_affinity_uses_only(ctx_full, split.train, users, split.test)

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest = manifest_template(
        dataset=config.dataset,
        dataset_sha256=dataset_sha256,
        config=config.to_dict(),
        git_sha=git_sha_value,
        created_at=created_at,
        candidate_set=winner,
        candidate_k=k,
        demographics=demo_mode,
        ranker_seed=primary_seed,
        best_iteration=int(trained.best_iteration),
        two_tower_epochs=int(two_tower_epochs),
        item_item_hyperparams=dict(item_item_hp),
        two_tower_hyperparams=dict(two_tower_hp),
    )
    histories = split.full_train.loc[:, ["user_id", "item_id", "rating", "timestamp"]].copy()
    return ServingBundle(
        manifest=manifest,
        movies=movies.loc[:, ["item_id", "title", "genres", "year"]].copy(),
        histories=histories,
        item_item=item_full,
        two_tower=tower_full,
        booster=trained.booster,
        context=ctx_full,
    )


def build_artifacts(
    config: RunConfig,
    *,
    out_dir: Path | str | None = None,
    download: bool = True,
    show_progress: bool = True,
) -> Path:
    """Download if needed, fit the production pipeline, write ``artifacts/<dataset>``."""
    if config.dataset not in DATASET_SHA256:
        raise ValueError(
            f"build-artifacts supports {sorted(DATASET_SHA256)}; got {config.dataset!r}"
        )
    t0 = time.perf_counter()
    if download:
        download_dataset(config.dataset, config.data_dir)
    ratings, _stats = load_ratings(config.dataset, config.data_dir, clean=True)
    movies = load_movies(config.dataset, config.data_dir)
    users = None
    if config.models.ranker.demographics != "off":
        users = load_users(config.dataset, config.data_dir)
    item_hp, tower_hp, epochs = load_production_hyperparams(config)
    split = time_based_split(
        ratings,
        SplitConfig(
            min_ratings=config.split.min_ratings,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            relevance_threshold=config.eval.relevance_threshold,
        ),
    )
    bundle = fit_serving_bundle(
        config=config,
        split=split,
        movies=movies,
        users=users,
        item_item_hp=item_hp,
        two_tower_hp=tower_hp,
        two_tower_epochs=epochs,
        dataset_sha256=DATASET_SHA256[config.dataset],
        git_sha_value=git_sha(),
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - t0
    bundle.manifest["build_runtime_sec"] = round(float(elapsed), 3)
    destination = Path(out_dir) if out_dir is not None else Path("artifacts") / config.dataset
    save_bundle(bundle, destination)
    print(f"Wrote serving snapshot to {destination} in {elapsed:.1f}s", flush=True)
    return destination


def _fit_item_item(
    train: pd.DataFrame, hyperparams: Mapping[str, Any]
) -> ItemItemCosineRecommender:
    return ItemItemCosineRecommender(
        min_common=int(hyperparams.get("min_common", 1)),
        k_neighbors=int(hyperparams.get("k_neighbors", 0)),
        shrinkage=float(hyperparams.get("shrinkage", 0.0)),
    ).fit(train)


def _fit_tower(
    train: pd.DataFrame,
    *,
    config: RunConfig,
    movies: pd.DataFrame,
    hyperparams: Mapping[str, Any],
    n_epochs: int,
    seed: int,
    show_progress: bool,
):
    recommender, _features, _result = fit_two_tower_recommender(
        train,
        dataset=config.dataset,
        data_dir=config.data_dir,
        movies=movies,
        hyperparams=dict(hyperparams),
        seed=seed,
        relevance_threshold=config.eval.relevance_threshold,
        n_epochs=int(n_epochs),
        val_split=None,
        show_progress=show_progress,
    )
    return recommender
