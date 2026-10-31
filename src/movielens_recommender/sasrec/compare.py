"""Refit the committed two-tower and LambdaRank headlines for paired NDCG (ADR-0019).

Candidate-set selection is not redone. Demographics stay off. The ranker is
trained by the existing early-stop-then-refit function.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.movies import load_movies
from movielens_recommender.ranker.candidates import (
    PrecomputedRecommender,
    materialize_candidate_set,
    score_rank_maps,
)
from movielens_recommender.ranker.features import (
    assert_context_uses_only,
    build_feature_context,
    relevance_pairs,
)
from movielens_recommender.ranker.pipeline import _blocks_for_users, _rerank
from movielens_recommender.ranker.train import assemble_matrices, train_with_early_stop_then_refit
from movielens_recommender.sasrec.train import evaluate_split, per_user_ndcg
from movielens_recommender.scale import release_memory
from movielens_recommender.split import SplitResult
from movielens_recommender.two_tower.train import fit_two_tower_recommender


def _test_view(split: SplitResult) -> SplitResult:
    return SplitResult(
        train=split.full_train,
        test=split.test,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )


def _fit_item_item(frame: pd.DataFrame, hyperparams: dict[str, Any]) -> ItemItemCosineRecommender:
    return ItemItemCosineRecommender(
        min_common=int(hyperparams["min_common"]),
        k_neighbors=int(hyperparams["k_neighbors"]),
        shrinkage=float(hyperparams["shrinkage"]),
    ).fit(frame)


def score_headline_references(
    split: SplitResult,
    *,
    dataset: str,
    data_dir: str,
    relevance_threshold: float,
    two_tower_hp: dict[str, Any],
    two_tower_epochs: int,
    item_item_hp: dict[str, Any],
    ranker_hp: dict[str, Any],
    n_bootstrap: int,
    bootstrap_alpha: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Full-train two-tower and demographics-off LambdaRank, primary seed only."""
    movies = load_movies(dataset, data_dir)
    view = _test_view(split)
    k = int(ranker_hp["candidate_k"])

    print("Refitting headline two-tower on full train...", flush=True)
    tower, _features, tower_train = fit_two_tower_recommender(
        split.full_train,
        dataset=dataset,
        data_dir=data_dir,
        movies=movies,
        hyperparams=dict(two_tower_hp),
        seed=seed,
        relevance_threshold=relevance_threshold,
        n_epochs=int(two_tower_epochs),
        val_split=None,
        show_progress=True,
    )
    tt_metrics = evaluate_split(
        tower.recommend,
        split.full_train,
        split.test,
        relevance_threshold=relevance_threshold,
        split=view,
        n_bootstrap=n_bootstrap,
        bootstrap_alpha=bootstrap_alpha,
        seed=seed,
    )
    tt_users, tt_values = per_user_ndcg(
        tower.recommend,
        split.full_train,
        split.test,
        relevance_threshold=relevance_threshold,
        split=view,
    )
    print("Scoring two-tower candidates for the ranker...", flush=True)
    tt_full = tower.topk_with_scores(k)
    del tower, _features
    release_memory()

    print("Fitting fit-train two-tower for ranker labels...", flush=True)
    tower_fit, feat_fit, _train_fit = fit_two_tower_recommender(
        split.train,
        dataset=dataset,
        data_dir=data_dir,
        movies=movies,
        hyperparams=dict(two_tower_hp),
        seed=seed,
        relevance_threshold=relevance_threshold,
        n_epochs=int(two_tower_epochs),
        val_split=None,
        show_progress=True,
    )
    tt_val = tower_fit.topk_with_scores(k)
    del tower_fit, feat_fit, _train_fit
    release_memory()

    print("Fitting item-item retrievers for ranker features...", flush=True)
    item_full = _fit_item_item(split.full_train, item_item_hp)
    ii_full = item_full.topk_with_scores(k)
    del item_full
    item_fit = _fit_item_item(split.train, item_item_hp)
    ii_val = item_fit.topk_with_scores(k)
    del item_fit
    release_memory()

    ctx_fit = build_feature_context(
        split.train,
        movies,
        users=None,
        mode="off",
        group_frame=split.train,
        relevance_threshold=relevance_threshold,
    )
    assert_context_uses_only(ctx_fit, split.train, split.val)
    assert_context_uses_only(ctx_fit, split.train, split.test)
    lists_val = materialize_candidate_set("two_tower", ii_val, tt_val, k)
    positives = relevance_pairs(split.val, relevance_threshold=relevance_threshold)
    label_users = {int(uid) for uid in split.val["user_id"].unique()}
    blocks = _blocks_for_users(
        ctx_fit,
        lists_val,
        score_rank_maps(ii_val),
        score_rank_maps(tt_val),
        positives,
        users=label_users,
    )
    matrices = assemble_matrices(blocks, ctx_fit.names)
    print(
        f"Training LambdaRank (demographics off, candidates=two_tower, K={k})...",
        flush=True,
    )
    trained = train_with_early_stop_then_refit(
        matrices,
        split_seed=int(ranker_hp["early_stop_user_split_seed"]),
        early_stop_fraction=float(ranker_hp["early_stop_fraction"]),
        ranker_seed=seed,
        num_boost_round=int(ranker_hp["num_boost_round_cap"]),
        early_stopping_rounds=int(ranker_hp["early_stopping_rounds"]),
        learning_rate=float(ranker_hp["learning_rate"]),
        num_leaves=int(ranker_hp["num_leaves"]),
        min_data_in_leaf=int(ranker_hp["min_data_in_leaf"]),
        feature_fraction=float(ranker_hp["feature_fraction"]),
        bagging_fraction=float(ranker_hp["bagging_fraction"]),
    )
    ctx_full = build_feature_context(
        split.full_train,
        movies,
        users=None,
        mode="off",
        group_frame=split.train,
        relevance_threshold=relevance_threshold,
    )
    assert_context_uses_only(ctx_full, split.full_train, split.test)
    lists_full = materialize_candidate_set("two_tower", ii_full, tt_full, k)
    ranked = _rerank(
        trained.booster,
        ctx_full,
        lists_full,
        score_rank_maps(ii_full),
        score_rank_maps(tt_full),
        trained.refit_info["feature_names"],
    )

    ranker_model = PrecomputedRecommender(ranked)
    lr_metrics = evaluate_split(
        ranker_model.recommend,
        split.full_train,
        split.test,
        relevance_threshold=relevance_threshold,
        split=view,
        n_bootstrap=n_bootstrap,
        bootstrap_alpha=bootstrap_alpha,
        seed=seed,
    )
    lr_users, lr_values = per_user_ndcg(
        ranker_model.recommend,
        split.full_train,
        split.test,
        relevance_threshold=relevance_threshold,
        split=view,
    )
    return {
        "lambdarank": {
            "best_iteration": int(trained.best_iteration),
            "metrics": lr_metrics,
            "ndcg": lr_values,
            "user_ids": lr_users,
        },
        "two_tower": {
            "epochs": int(two_tower_epochs),
            "metrics": tt_metrics,
            "ndcg": tt_values,
            "train_wall_sec": float(tower_train.wall_time_sec),
            "user_ids": tt_users,
        },
    }
