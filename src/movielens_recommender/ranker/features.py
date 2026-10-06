"""Ranker features built from one allowed interaction matrix.

Training features come from fit-train only (retrievers trained on fit-train).
Before test scoring, the same builder is called on full train (fit-train ∪
validation). Test rows are never passed in. Movie genres and release year are
static metadata from the movies file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from movielens_recommender.movies import GENRE_TO_IDX, GENRES, _genre_multi_hot

# Dropped together by the retriever-score ablation. ``in_both`` stays.
RETRIEVER_SCORE_RANK_FEATURES: tuple[str, ...] = (
    "item_item_score",
    "item_item_rank",
    "two_tower_score",
    "two_tower_rank",
)


def slug_genre(name: str) -> str:
    """Stable feature-name suffix for a MovieLens genre label."""
    text = str(name).lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def feature_names() -> list[str]:
    """Column order for the ranker matrix. Stable across runs."""
    slugs = [slug_genre(g) for g in GENRES]
    if len(slugs) != len(set(slugs)):
        raise RuntimeError(f"genre slugs are not unique: {slugs}")
    names = [
        *RETRIEVER_SCORE_RANK_FEATURES,
        "in_both",
        "item_popularity",
        "item_recency",
        "item_year",
        "user_n_ratings",
        "user_mean_rating",
        "user_std_rating",
    ]
    names.extend(f"genre_{slug}" for slug in slugs)
    names.extend(f"affinity_{slug}" for slug in slugs)
    return names


def drop_retriever_score_rank(
    matrix: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Drop retriever score and rank columns; keep every other feature."""
    keep = [i for i, name in enumerate(names) if name not in RETRIEVER_SCORE_RANK_FEATURES]
    kept_names = [names[i] for i in keep]
    return matrix[:, keep], kept_names


@dataclass
class FeatureContext:
    """Item and user statistics derived from a single interaction matrix."""

    item_ids: np.ndarray
    item_index: dict[int, int]
    popularity: np.ndarray
    recency: np.ndarray
    year: np.ndarray
    genres: np.ndarray
    user_n_ratings: dict[int, float]
    user_mean_rating: dict[int, float]
    user_std_rating: dict[int, float]
    user_affinity: dict[int, np.ndarray]
    names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.names)


def _item_side(movies: pd.DataFrame, item_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    by_id = movies.set_index("item_id")
    n = len(item_ids)
    genres = np.zeros((n, len(GENRES)), dtype=np.float64)
    years = np.full(n, np.nan, dtype=np.float64)
    for i, item_id in enumerate(item_ids):
        iid = int(item_id)
        if iid not in by_id.index:
            genres[i, GENRE_TO_IDX["(no genres listed)"]] = 1.0
            continue
        row = by_id.loc[iid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        genres[i] = _genre_multi_hot(str(row["genres"])).astype(np.float64)
        year = row["year"]
        if pd.notna(year):
            years[i] = float(year)
    return genres, years


def build_feature_context(train: pd.DataFrame, movies: pd.DataFrame) -> FeatureContext:
    """Statistics from ``train`` only. ``movies`` supplies genres and year."""
    if train.empty:
        raise ValueError("feature context requires a non-empty train matrix")
    item_ids = np.sort(train["item_id"].unique().astype(np.int64))
    item_index = {int(i): n for n, i in enumerate(item_ids)}
    genres, years = _item_side(movies, item_ids)

    popularity = np.zeros(len(item_ids), dtype=np.float64)
    recency = np.full(len(item_ids), np.nan, dtype=np.float64)
    pop = train.groupby("item_id").size()
    rec = train.groupby("item_id")["timestamp"].max()
    for item_id, count in pop.items():
        idx = item_index[int(item_id)]
        popularity[idx] = float(count)
        recency[idx] = float(rec.loc[item_id])

    user_n: dict[int, float] = {}
    user_mean: dict[int, float] = {}
    user_std: dict[int, float] = {}
    user_aff: dict[int, np.ndarray] = {}
    for uid, group in train.groupby("user_id", sort=True):
        uid_i = int(uid)
        ratings = group["rating"].to_numpy(dtype=np.float64)
        user_n[uid_i] = float(len(ratings))
        user_mean[uid_i] = float(ratings.mean())
        user_std[uid_i] = float(ratings.std(ddof=0))
        idxs = [item_index[int(i)] for i in group["item_id"].tolist()]
        user_aff[uid_i] = genres[idxs].mean(axis=0)

    return FeatureContext(
        item_ids=item_ids,
        item_index=item_index,
        popularity=popularity,
        recency=recency,
        year=years,
        genres=genres,
        user_n_ratings=user_n,
        user_mean_rating=user_mean,
        user_std_rating=user_std,
        user_affinity=user_aff,
        names=feature_names(),
    )


def assert_context_uses_only(
    ctx: FeatureContext,
    allowed: pd.DataFrame,
    forbidden: pd.DataFrame | None = None,
) -> None:
    """Fail if item recency or user counts disagree with ``allowed``.

    When ``forbidden`` is set, a later per-item timestamp that exists only in
    that frame must not be the recorded recency. Equal timestamps across
    frames are not treated as leakage; the check is per item against the
    allowed maximum.
    """
    allowed_rec = allowed.groupby("item_id")["timestamp"].max()
    allowed_n = allowed.groupby("user_id").size()
    for item_id, ts in allowed_rec.items():
        idx = ctx.item_index.get(int(item_id))
        if idx is None:
            raise AssertionError(f"item {item_id} missing from feature context")
        got = float(ctx.recency[idx])
        if got != float(ts):
            raise AssertionError(
                f"item {item_id} recency {got} != allowed max timestamp {float(ts)}"
            )
    for uid, count in allowed_n.items():
        got_n = ctx.user_n_ratings.get(int(uid))
        if got_n != float(count):
            raise AssertionError(
                f"user {uid} activity {got_n} != allowed count {float(count)}"
            )
    if forbidden is None or forbidden.empty:
        return
    forbidden_rec = forbidden.groupby("item_id")["timestamp"].max()
    for item_id, fts in forbidden_rec.items():
        idx = ctx.item_index.get(int(item_id))
        if idx is None:
            continue
        got = float(ctx.recency[idx])
        allowed_ts = (
            float(allowed_rec.loc[item_id]) if item_id in allowed_rec.index else None
        )
        if allowed_ts is None:
            raise AssertionError(
                f"item {item_id} has a feature recency but is absent from the allowed frame"
            )
        if float(fts) > allowed_ts and got >= float(fts):
            raise AssertionError(
                f"item {item_id} recency {got} includes forbidden timestamp {float(fts)}"
            )


def relevance_pairs(
    label_frame: pd.DataFrame,
    *,
    relevance_threshold: float,
) -> set[tuple[int, int]]:
    """``(user_id, item_id)`` pairs in the label window at or above the threshold."""
    if label_frame.empty:
        return set()
    positive = label_frame.loc[label_frame["rating"] >= relevance_threshold, ["user_id", "item_id"]]
    return {
        (int(uid), int(item))
        for uid, item in zip(
            positive["user_id"].tolist(),
            positive["item_id"].tolist(),
            strict=True,
        )
    }


def binary_relevance_labels(
    label_frame: pd.DataFrame,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    *,
    relevance_threshold: float,
) -> np.ndarray:
    """1 when ``label_frame`` rates the pair at or above the threshold, else 0.

    Pairs absent from the label window are 0. The label frame is the only
    source of labels; it must not be the test window.
    """
    if label_frame.empty:
        return np.zeros(len(user_ids), dtype=np.float64)
    rated = label_frame.groupby(["user_id", "item_id"], sort=False)["rating"].max()
    labels = np.zeros(len(user_ids), dtype=np.float64)
    for i, (uid, item) in enumerate(zip(user_ids, item_ids, strict=True)):
        key = (int(uid), int(item))
        if key not in rated.index:
            continue
        if float(rated.loc[key]) >= relevance_threshold:
            labels[i] = 1.0
    return labels


def build_feature_matrix(
    ctx: FeatureContext,
    user_id: int,
    item_ids: Sequence[int],
    item_item: Mapping[int, tuple[float, int]],
    two_tower: Mapping[int, tuple[float, int]],
) -> np.ndarray:
    """One row per candidate. Missing retriever score/rank is NaN.

    An item that appears in only one retriever's top-K gets NaN for the other
    retriever's score and rank. ``in_both`` is 1 only when both maps contain it.
    """
    items = [int(i) for i in item_ids]
    n = len(items)
    names = ctx.names
    col = {name: i for i, name in enumerate(names)}
    x = np.full((n, len(names)), np.nan, dtype=np.float64)
    if n == 0:
        return x

    uid = int(user_id)
    n_ratings = ctx.user_n_ratings.get(uid, np.nan)
    mean_rating = ctx.user_mean_rating.get(uid, np.nan)
    std_rating = ctx.user_std_rating.get(uid, np.nan)
    affinity = ctx.user_affinity.get(uid)
    n_genres = len(GENRES)
    genre0 = col[f"genre_{slug_genre(GENRES[0])}"]
    aff0 = col[f"affinity_{slug_genre(GENRES[0])}"]

    for row, item in enumerate(items):
        ii = item_item.get(item)
        tt = two_tower.get(item)
        if ii is not None:
            x[row, col["item_item_score"]] = float(ii[0])
            x[row, col["item_item_rank"]] = float(ii[1])
        if tt is not None:
            x[row, col["two_tower_score"]] = float(tt[0])
            x[row, col["two_tower_rank"]] = float(tt[1])
        x[row, col["in_both"]] = 1.0 if ii is not None and tt is not None else 0.0

        idx = ctx.item_index.get(item)
        if idx is not None:
            x[row, col["item_popularity"]] = float(ctx.popularity[idx])
            x[row, col["item_recency"]] = float(ctx.recency[idx])
            year = ctx.year[idx]
            if np.isfinite(year):
                x[row, col["item_year"]] = float(year)
            x[row, genre0 : genre0 + n_genres] = ctx.genres[idx]
        x[row, col["user_n_ratings"]] = float(n_ratings)
        x[row, col["user_mean_rating"]] = float(mean_rating)
        x[row, col["user_std_rating"]] = float(std_rating)
        if affinity is not None:
            x[row, aff0 : aff0 + n_genres] = affinity
    return x


# The slice above assumes genre columns are contiguous in GENRES order, which
# feature_names() guarantees. Keep the helper explicit for the affinity block.


def _genre_slice_ok() -> None:
    names = feature_names()
    genre_cols = [n for n in names if n.startswith("genre_")]
    aff_cols = [n for n in names if n.startswith("affinity_")]
    expected = [f"genre_{slug_genre(g)}" for g in GENRES]
    expected_aff = [f"affinity_{slug_genre(g)}" for g in GENRES]
    if genre_cols != expected or aff_cols != expected_aff:
        raise RuntimeError("genre feature order drifted from GENRES")


_genre_slice_ok()
