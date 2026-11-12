"""Unit tests for 3-way candidate unions and EASE-DAN ranker features (ADR-0031)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from movielens_recommender.baselines.ease_dan import EASEDANRecommender
from movielens_recommender.ranker.candidates import (
    balanced_three_way_union,
    unbalanced_three_way_union,
)
from movielens_recommender.ranker.features import (
    build_feature_context,
    build_feature_matrix,
    tri_ranker_feature_names,
)


def test_three_way_unions_respect_budget_and_dedupe():
    a = [1, 2, 3, 4]
    b = [2, 5, 6, 7]
    c = [1, 8, 9, 10]
    bal = balanced_three_way_union(a, b, c, 6)
    assert len(bal) == 6
    assert len(set(bal)) == 6
    # Round-robin picks 1 (from a), 2 (from b), 8 (from c), 3 (from a), 5 (from b), 9 (from c)
    assert bal == [1, 2, 8, 3, 5, 9]

    unbal = unbalanced_three_way_union(a, b, c, 4)
    assert len(unbal) == len({*a, *b, *c})
    assert len(set(unbal)) == len(unbal)


def test_ease_dan_topk_with_scores_matches_recommend_and_fills_tri_features():
    rows = []
    for uid in range(1, 10):
        for step, iid in enumerate([1, 2, 3, (uid % 4) + 4], start=1):
            rows.append(
                {"user_id": uid, "item_id": iid, "rating": 4.0 + (step % 2), "timestamp": step}
            )
    train = pd.DataFrame(rows)
    movies = pd.DataFrame(
        [
            {"item_id": i, "title": f"Film {i}", "genres": "Action|Drama", "year": 1999}
            for i in range(1, 9)
        ]
    )
    dan = EASEDANRecommender(l2=20.0, dropout_p=0.3, alpha=0.1, beta=0.2).fit(train)
    topk = dan.topk_with_scores(3, user_ids=[1, 2])
    assert [i for i, _s in topk[1]] == dan.recommend(1, 3)

    dan_uncached = EASEDANRecommender(
        l2=20.0, dropout_p=0.3, alpha=0.1, beta=0.2, cache_user_scores=False
    ).fit(train)
    topk_uncached = dan_uncached.topk_with_scores(3, user_ids=[1, 2])
    assert [i for i, _s in topk_uncached[1]] == dan.recommend(1, 3)

    ctx = build_feature_context(train, movies, mode="off")
    ctx.names = tri_ranker_feature_names("off")
    mat = build_feature_matrix(
        ctx,
        1,
        [5, 6],
        {5: (0.8, 1)},
        {5: (0.9, 1), 6: (0.4, 2)},
        ease_dan={5: (0.7, 1)},
    )
    col = {name: idx for idx, name in enumerate(ctx.names)}
    assert mat[0, col["ease_dan_score"]] == 0.7
    assert mat[0, col["ease_dan_rank"]] == 1.0
    assert mat[0, col["in_all_three"]] == 1.0
    assert np.isnan(mat[1, col["ease_dan_score"]])
    assert mat[1, col["in_all_three"]] == 0.0
