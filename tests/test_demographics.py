"""Demographic feature tests on synthetic frames (no MovieLens download)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from movielens_recommender.data import (
    AGE_UNKNOWN,
    GENDER_UNKNOWN,
    OCCUPATION_UNKNOWN,
    REGION_UNKNOWN,
    parse_users,
    read_users_dat,
    region_from_zip,
    users_path,
)
from movielens_recommender.movies import GENRES
from movielens_recommender.ranker.features import (
    GROUP_AFFINITY_FEATURES,
    RAW_DEMOGRAPHIC_FEATURES,
    assert_group_affinity_uses_only,
    build_feature_context,
    build_feature_matrix,
    categorical_feature_names,
    feature_names,
    history_user_stats,
)
from movielens_recommender.ranker.train import (
    assemble_matrices,
    train_with_early_stop_then_refit,
)


def _movies() -> pd.DataFrame:
    rows = []
    for i, genre in enumerate(GENRES[:6]):
        rows.append(
            {
                "item_id": 100 + i,
                "title": f"Movie {i} (199{i})",
                "genres": genre,
                "year": 1990.0 + i,
            }
        )
    return pd.DataFrame(rows)


def _ratings(rows: list[tuple[int, int, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": u, "item_id": i, "rating": r, "timestamp": t}
            for u, i, r, t in rows
        ]
    )


def _users() -> pd.DataFrame:
    return parse_users(
        pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "gender": ["F", "F", "M"],
                "age": [25, 25, 18],
                "occupation": [4, 4, 7],
                "zip_code": ["12345", "12001", "67890"],
            }
        )
    )


def test_users_dat_parsing_and_region(tmp_path):
    path = tmp_path / "users.dat"
    path.write_text(
        "\n".join(
            [
                "1::F::25::12::12345",
                "2::M::1::0::T5J2L1",
                "3::M::56::20::90210",
                "4::X::99::40::",
                "1::M::18::1::99999",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    users = read_users_dat(path)
    assert list(users["user_id"]) == [1, 2, 3, 4]
    row1 = users.loc[users["user_id"] == 1].iloc[0]
    assert row1["gender"] == "F"
    assert int(row1["gender_code"]) == 1
    assert int(row1["age_code"]) == 25
    assert int(row1["occupation_code"]) == 12
    assert int(row1["region_code"]) == 1
    row2 = users.loc[users["user_id"] == 2].iloc[0]
    assert int(row2["region_code"]) == REGION_UNKNOWN
    assert int(row2["age_code"]) == 1
    row3 = users.loc[users["user_id"] == 3].iloc[0]
    assert int(row3["region_code"]) == 9
    assert int(row3["occupation_code"]) == 20
    row4 = users.loc[users["user_id"] == 4].iloc[0]
    assert int(row4["gender_code"]) == GENDER_UNKNOWN
    assert int(row4["age_code"]) == AGE_UNKNOWN
    assert int(row4["occupation_code"]) == OCCUPATION_UNKNOWN
    assert int(row4["region_code"]) == REGION_UNKNOWN
    assert region_from_zip(" 48103") == 4
    with pytest.raises(ValueError, match="ml-1m only"):
        users_path("ml-latest-small")


def test_feature_shapes_and_codes():
    movies = _movies()
    fit = _ratings(
        [
            (1, 100, 5.0, 1),
            (1, 101, 4.0, 2),
            (2, 100, 3.0, 3),
            (2, 102, 5.0, 4),
            (3, 103, 4.0, 5),
        ]
    )
    users = _users()
    off = build_feature_context(fit, movies)
    both = build_feature_context(fit, movies, users=users, mode="both")
    assert off.names == feature_names()
    assert off.names == feature_names("off")
    assert both.n_features == len(feature_names("both"))
    assert len(feature_names("both")) == (
        len(feature_names("off"))
        + len(RAW_DEMOGRAPHIC_FEATURES)
        + len(GROUP_AFFINITY_FEATURES)
    )
    assert feature_names("raw") == feature_names("off") + list(RAW_DEMOGRAPHIC_FEATURES)
    matrix = build_feature_matrix(both, 1, [100, 101, 104], {}, {})
    assert matrix.shape == (3, both.n_features)
    col = {name: i for i, name in enumerate(both.names)}
    assert matrix[0, col["demo_gender"]] == pytest.approx(1.0)
    assert matrix[0, col["demo_age"]] == pytest.approx(25.0)
    assert matrix[0, col["demo_occupation"]] == pytest.approx(4.0)
    assert matrix[0, col["demo_region"]] == pytest.approx(1.0)
    assert matrix[0, col["demo_gender"]] == matrix[1, col["demo_gender"]]
    assert np.isfinite(matrix[0, col["group_gender_pos_rate"]])
    assert np.isfinite(matrix[0, col["group_age_pop_share"]])
    assert np.isfinite(matrix[0, col["group_occupation_pos_rate"]])
    codes = matrix[0, [col[name] for name in RAW_DEMOGRAPHIC_FEATURES]]
    assert np.all(codes >= 0)
    off_matrix = build_feature_matrix(off, 1, [100], {100: (0.4, 1)}, {})
    both_matrix = build_feature_matrix(both, 1, [100], {100: (0.4, 1)}, {})
    for name in feature_names("off"):
        left = off_matrix[0, feature_names("off").index(name)]
        right = both_matrix[0, both.names.index(name)]
        assert np.isnan(left) == np.isnan(right)
        if np.isfinite(left):
            assert left == pytest.approx(right)


def test_group_affinity_uses_only_fit_train():
    movies = _movies()
    users = _users()
    fit = _ratings(
        [
            (1, 100, 5.0, 1),
            (2, 101, 2.0, 2),
            (3, 102, 4.0, 3),
        ]
    )
    forbidden = _ratings([(2, 100, 5.0, 50), (1, 100, 5.0, 60)])
    ctx = build_feature_context(fit, movies, users=users, mode="both", group_frame=fit)
    assert_group_affinity_uses_only(ctx, fit, users, forbidden)

    leaked = build_feature_context(
        pd.concat([fit, forbidden], ignore_index=True),
        movies,
        users=users,
        mode="both",
    )
    safe = build_feature_context(
        pd.concat([fit, forbidden], ignore_index=True),
        movies,
        users=users,
        mode="both",
        group_frame=fit,
    )
    col_name = "group_gender_pos_rate"
    fit_x = build_feature_matrix(ctx, 1, [100], {}, {})
    leaked_x = build_feature_matrix(leaked, 1, [100], {}, {})
    safe_x = build_feature_matrix(safe, 1, [100], {}, {})
    col = feature_names("both").index(col_name)
    assert fit_x[0, col] == pytest.approx(safe_x[0, col])
    assert fit_x[0, col] != pytest.approx(leaked_x[0, col])
    # Full-train item recency may include the forbidden timestamp; the group
    # rate must not.
    assert float(safe.recency[safe.item_index[100]]) == 60.0
    assert float(ctx.recency[ctx.item_index[100]]) == 1.0


def test_history_override_does_not_use_the_full_user_count():
    movies = _movies()
    users = _users()
    fit = _ratings(
        [
            (1, 100, 5.0, 1),
            (1, 101, 4.0, 2),
            (1, 102, 3.0, 3),
            (1, 103, 2.0, 4),
        ]
    )
    ctx = build_feature_context(fit, movies, users=users, mode="both")
    assert ctx.user_n_ratings[1] == pytest.approx(4.0)
    stats = history_user_stats(ctx, [100, 101], [5.0, 4.0])
    matrix = build_feature_matrix(
        ctx, 1, [103], {}, {}, user_stats=stats
    )
    col = {name: i for i, name in enumerate(ctx.names)}
    assert matrix[0, col["user_n_ratings"]] == pytest.approx(2.0)
    assert matrix[0, col["demo_age"]] == pytest.approx(25.0)
    full = build_feature_matrix(ctx, 1, [103], {}, {})
    assert full[0, col["user_n_ratings"]] == pytest.approx(4.0)
    assert matrix[0, col["group_age_pos_rate"]] == pytest.approx(
        full[0, col["group_age_pos_rate"]]
    )


def test_lambdarank_accepts_demographic_categoricals():
    pytest.importorskip("lightgbm")
    movies = _movies()
    rows = []
    for uid in range(1, 9):
        for item, rating, ts in (
            (100, 5.0, 1),
            (101, 3.0, 2),
            (102, 4.0, 3),
            (103, 1.0, 4),
        ):
            rows.append((uid, item, rating, ts))
    fit = _ratings(rows)
    users = parse_users(
        pd.DataFrame(
            {
                "user_id": list(range(1, 9)),
                "gender": ["F", "M"] * 4,
                "age": [25, 18, 35, 45, 25, 18, 35, 56],
                "occupation": [1, 2, 3, 4, 5, 6, 7, 8],
                "zip_code": ["10000", "20000", "30000", "40000"] * 2,
            }
        )
    )
    ctx = build_feature_context(fit, movies, users=users, mode="both")
    blocks = []
    for uid in range(1, 9):
        items = [100, 101, 102, 103]
        matrix = build_feature_matrix(ctx, uid, items, {100: (0.2, 1)}, {101: (0.3, 1)})
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        blocks.append((uid, matrix, labels))
    matrices = assemble_matrices(blocks, ctx.names)
    matrices.categorical_features = categorical_feature_names("both")
    result = train_with_early_stop_then_refit(
        matrices,
        split_seed=0,
        early_stop_fraction=0.25,
        ranker_seed=1,
        num_boost_round=8,
        early_stopping_rounds=2,
        learning_rate=0.1,
        num_leaves=4,
        min_data_in_leaf=1,
        feature_fraction=1.0,
        bagging_fraction=1.0,
    )
    pred = np.asarray(result.booster.predict(matrices.x), dtype=np.float64)
    assert pred.shape == (matrices.x.shape[0],)
    assert np.isfinite(pred).all()
