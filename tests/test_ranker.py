"""Stage-4 ranker tests on synthetic data (no MovieLens download)."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.movies import GENRES
from movielens_recommender.ranker.candidates import (
    balanced_union,
    choose_candidate_set,
    dedupe_ids,
    unbalanced_union,
)
from movielens_recommender.ranker.explain import (
    explain_candidates,
    load_ranker,
    save_ranker,
)
from movielens_recommender.ranker.features import (
    RETRIEVER_SCORE_RANK_FEATURES,
    assert_context_uses_only,
    build_feature_context,
    build_feature_matrix,
    drop_retriever_score_rank,
    feature_names,
)
from movielens_recommender.ranker.gate import gate_against_s3
from movielens_recommender.ranker.train import (
    RankerMatrices,
    assemble_matrices,
    fit_lambdarank,
    lgb_params,
    split_users,
    train_fixed_rounds,
    train_with_early_stop_then_refit,
    validate_groups,
)
from movielens_recommender.split import chronological_tail_holdout


def _movies() -> pd.DataFrame:
    rows = []
    for i, genre in enumerate(GENRES[:8]):
        rows.append(
            {
                "item_id": 100 + i,
                "title": f"Movie {i} (199{i})",
                "genres": genre,
                "year": 1990.0 + i,
            }
        )
    return pd.DataFrame(rows)


def _frame(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": u, "item_id": i, "rating": r, "timestamp": t}
            for u, i, r, t in rows
        ]
    )


def test_balanced_union_dedupes_then_backfills_to_k():
    left = [1, 2, 3, 4, 5, 6]
    right = [2, 4, 6, 7, 8, 9]
    # Top 3 from each alternate to [1, 2, 4, 3, 6] after dedupe, then backfill 5.
    assert balanced_union(left, right, 6) == [1, 2, 4, 3, 6, 5]


def test_balanced_union_no_overlap_is_exactly_k():
    assert balanced_union([1, 2, 3, 4], [5, 6, 7, 8], 4) == [1, 5, 2, 6]


def test_balanced_union_stops_when_sources_run_out():
    assert balanced_union([1, 1, 1], [1], 4) == [1]
    assert dedupe_ids([1, 1, 2, 2, 3]) == [1, 2, 3]


def test_unbalanced_union_dedupes_and_reports_longer_list():
    merged = unbalanced_union([1, 2, 3, 4, 5], [2, 4, 6, 7, 8], 4)
    assert merged == [1, 2, 4, 3, 6, 7]
    assert len(merged) == 6
    assert len(set(merged)) == len(merged)


def test_choose_candidate_set_by_validation_recall_at_200():
    reports = {
        "item_item": {"recall@100": 0.4, "recall@200": 0.5},
        "two_tower": {"recall@100": 0.3, "recall@200": 0.7},
        "union_balanced": {"recall@100": 0.9, "recall@200": 0.6},
        "union_unbalanced": {"recall@100": 0.2, "recall@200": 0.7},
    }
    # two_tower and unbalanced tie on recall@200; recall@100 breaks the tie.
    assert choose_candidate_set(reports) == "two_tower"
    reports["union_balanced"]["recall@200"] = 0.7
    reports["union_balanced"]["recall@100"] = 0.3
    # three-way tie on both recalls: preference order picks union_balanced.
    assert choose_candidate_set(reports) == "union_balanced"


def test_feature_context_excludes_validation_and_test_timestamps():
    movies = _movies()
    # Distinct timestamps: fit-train <= 10, validation == 100, test == 1000.
    fit = _frame(
        [
            (1, 100, 5.0, 1),
            (1, 101, 4.0, 2),
            (1, 102, 3.0, 3),
            (2, 100, 5.0, 4),
            (2, 103, 4.0, 5),
            (2, 104, 2.0, 6),
        ]
    )
    val = _frame(
        [
            (1, 103, 5.0, 100),
            (2, 101, 4.0, 100),
        ]
    )
    test = _frame(
        [
            (1, 104, 5.0, 1000),
            (2, 102, 5.0, 1000),
        ]
    )
    ctx = build_feature_context(fit, movies)
    assert_context_uses_only(ctx, fit, val)
    assert_context_uses_only(ctx, fit, test)
    assert max(float(v) for v in ctx.recency) <= 10
    assert 100.0 not in set(ctx.recency)
    assert 1000.0 not in set(ctx.recency)

    full = pd.concat([fit, val], ignore_index=True)
    ctx_refit = build_feature_context(full, movies)
    assert_context_uses_only(ctx_refit, full, test)
    assert max(float(v) for v in ctx_refit.recency) <= 100
    assert 1000.0 not in set(ctx_refit.recency)
    # Item 103's only fit-train absence means its refit recency is the val timestamp.
    assert float(ctx_refit.recency[ctx_refit.item_index[103]]) == 100.0
    # Item 100 was rated at timestamp 1 and 4 in fit-train, never later in test.
    assert float(ctx_refit.recency[ctx_refit.item_index[100]]) == 4.0


def test_tail_holdout_labels_are_not_in_head_features():
    movies = _movies()
    ratings = _frame(
        [(1, 100 + i, 4.0, 10 + i) for i in range(8)]
        + [(2, 100 + i, 5.0, 30 + i) for i in range(8)]
    )
    head, tail = chronological_tail_holdout(ratings, 0.25)
    assert not tail.empty
    for uid in (1, 2):
        head_ts = head.loc[head["user_id"] == uid, "timestamp"]
        tail_ts = tail.loc[tail["user_id"] == uid, "timestamp"]
        assert head_ts.max() < tail_ts.min()
    ctx = build_feature_context(head, movies)
    assert_context_uses_only(ctx, head, tail)


def test_missing_retriever_score_and_rank_are_nan():
    movies = _movies()
    fit = _frame([(1, 100, 5.0, 1), (1, 101, 4.0, 2), (1, 102, 3.0, 3)])
    ctx = build_feature_context(fit, movies)
    matrix = build_feature_matrix(
        ctx,
        1,
        [100, 101],
        item_item={100: (0.8, 1)},
        two_tower={101: (0.3, 2)},
    )
    names = feature_names()
    col = {name: i for i, name in enumerate(names)}
    assert matrix[0, col["item_item_score"]] == pytest.approx(0.8)
    assert matrix[0, col["item_item_rank"]] == pytest.approx(1.0)
    assert np.isnan(matrix[0, col["two_tower_score"]])
    assert np.isnan(matrix[0, col["two_tower_rank"]])
    assert matrix[0, col["in_both"]] == pytest.approx(0.0)
    assert matrix[1, col["in_both"]] == pytest.approx(0.0)
    assert np.isnan(matrix[1, col["item_item_score"]])
    assert "item_popularity" in names
    assert "item_recency" in names
    assert "item_year" in names
    assert "user_n_ratings" in names
    assert any(name.startswith("genre_") for name in names)
    assert any(name.startswith("affinity_") for name in names)


def test_drop_retriever_score_rank_keeps_other_columns():
    names = feature_names()
    matrix = np.ones((2, len(names)))
    dropped, kept = drop_retriever_score_rank(matrix, names)
    assert kept == [name for name in names if name not in RETRIEVER_SCORE_RANK_FEATURES]
    assert "in_both" in kept
    assert dropped.shape[1] == len(kept)
    for name in RETRIEVER_SCORE_RANK_FEATURES:
        assert name not in kept


def _toy_blocks(n_users: int = 12, n_items: int = 6, seed: int = 0) -> RankerMatrices:
    rng = np.random.default_rng(seed)
    names = feature_names()
    blocks = []
    for uid in range(n_users):
        x = rng.normal(size=(n_items, len(names)))
        # First feature sorts relevance so the ranker has a signal.
        y = (x[:, 0] > 0).astype(np.float64)
        if y.min() == y.max():
            y[0] = 1.0
            y[1] = 0.0
        blocks.append((uid, x, y))
    return assemble_matrices(blocks, names)


def test_lambdarank_group_shapes_sum_to_rows():
    pytest.importorskip("lightgbm")
    matrices = _toy_blocks()
    validate_groups(matrices.y, matrices.group)
    assert int(matrices.group.sum()) == len(matrices.y)
    assert list(matrices.group) == [6] * 12
    params = lgb_params(
        seed=1,
        learning_rate=0.1,
        num_leaves=7,
        min_data_in_leaf=1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
    )
    _booster, info = fit_lambdarank(
        matrices,
        params=params,
        num_boost_round=4,
    )
    assert info["group_sizes"] == [6] * 12
    assert info["n_rows"] == 12 * 6
    assert info["n_groups"] == 12
    assert info["num_trees"] == 4
    assert info["early_stopping"] is False


def test_fixed_round_refit_uses_best_iteration():
    pytest.importorskip("lightgbm")
    matrices = _toy_blocks(n_users=16, n_items=8, seed=1)
    result = train_with_early_stop_then_refit(
        matrices,
        split_seed=7,
        early_stop_fraction=0.25,
        ranker_seed=3,
        num_boost_round=12,
        early_stopping_rounds=2,
        learning_rate=0.1,
        num_leaves=7,
        min_data_in_leaf=1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
    )
    best = result.best_iteration
    assert best >= 1
    assert result.refit_info["early_stopping"] is False
    assert result.refit_info["num_boost_round"] == best
    assert result.booster.num_trees() == best
    assert result.early_stop_info["best_iteration"] == best
    longer, longer_info = train_fixed_rounds(
        matrices,
        ranker_seed=3,
        num_boost_round=best + 2,
        learning_rate=0.1,
        num_leaves=7,
        min_data_in_leaf=1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
    )
    assert longer.num_trees() == best + 2
    assert longer_info["num_boost_round"] == best + 2
    assert longer_info["early_stopping"] is False


def test_early_stop_user_split_does_not_depend_on_ranker_seed():
    users = np.arange(40)
    train_a, early_a = split_users(users, 0.2, seed=42)
    train_b, early_b = split_users(users, 0.2, seed=42)
    assert list(train_a) == list(train_b)
    assert list(early_a) == list(early_b)
    assert set(train_a).isdisjoint(set(early_a))
    assert len(early_a) >= 1
    assert len(train_a) >= 1


def test_contributions_sum_to_raw_score(tmp_path):
    pytest.importorskip("lightgbm")
    matrices = _toy_blocks(n_users=10, n_items=5, seed=2)
    params = lgb_params(
        seed=0,
        learning_rate=0.2,
        num_leaves=7,
        min_data_in_leaf=1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
    )
    booster, info = fit_lambdarank(matrices, params=params, num_boost_round=5)
    rows = matrices.x[:4]
    item_ids = [100, 101, 102, 103]
    retrievers = [
        {"item_item": {"score": 0.5, "rank": 1}, "two_tower": None},
        {"item_item": None, "two_tower": {"score": 0.2, "rank": 4}},
        {
            "item_item": {"score": 0.1, "rank": 3},
            "two_tower": {"score": 0.4, "rank": 2},
        },
        {"item_item": None, "two_tower": None},
    ]
    explained = explain_candidates(
        booster,
        rows,
        info["feature_names"],
        item_ids=item_ids,
        retrievers=retrievers,
    )
    raw = np.asarray(booster.predict(rows, raw_score=True), dtype=np.float64)
    assert len(explained) == 4
    for i, row in enumerate(explained):
        total = row["bias"] + sum(row["contributions"].values())
        assert total == pytest.approx(raw[i], abs=1e-5)
        assert total == pytest.approx(row["raw_score"], abs=1e-5)
        assert row["item_id"] == item_ids[i]
        assert row["retrievers"] == retrievers[i]
        assert set(row["contributions"]) == set(info["feature_names"])

    save_ranker(
        booster,
        tmp_path,
        {
            "feature_names": info["feature_names"],
            "best_iteration": 5,
            "recreate_command": "movielens-recommender run --config configs/default.yaml",
        },
    )
    loaded, meta = load_ranker(tmp_path)
    assert meta["schema_version"] == 1
    assert meta["feature_names"] == info["feature_names"]
    reloaded = explain_candidates(
        loaded,
        rows,
        meta["feature_names"],
        item_ids=item_ids,
        retrievers=retrievers,
    )
    for i, row in enumerate(reloaded):
        total = row["bias"] + sum(row["contributions"].values())
        assert total == pytest.approx(explained[i]["raw_score"], abs=1e-5)


def test_gate_reads_item_item_metric_from_results():
    source = inspect.getsource(gate_against_s3)
    assert "0.1201" not in source
    metrics = {
        "item_item_cosine": {
            "ndcg@10": 0.2,
            "confidence_intervals": {
                "ndcg@10": {"mean": 0.2, "low": 0.1, "high": 0.3}
            },
        }
    }
    win = gate_against_s3(
        ranker_mean=0.25,
        per_seed_cis=[{"low": 0.22, "high": 0.28}],
        metrics=metrics,
    )
    assert win["bar_ndcg@10"] == 0.2
    assert win["bar_model"] == "item_item_cosine"
    assert win["negative_result"] is False
    lose = gate_against_s3(
        ranker_mean=0.2,
        per_seed_cis=[{"low": 0.15, "high": 0.25}],
        metrics=metrics,
    )
    assert lose["negative_result"] is True
    assert lose["mean_exceeds_bar_point"] is False
