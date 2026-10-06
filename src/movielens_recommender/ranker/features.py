"""Ranker features built from one allowed interaction matrix.

Training features come from fit-train only (retrievers trained on fit-train).
Before test scoring, the same builder is called on full train (fit-train ∪
validation). Test rows are never passed in. Movie genres and release year are
static metadata from the movies file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from movielens_recommender.data import (
    AGE_UNKNOWN,
    GENDER_UNKNOWN,
    OCCUPATION_UNKNOWN,
    REGION_UNKNOWN,
)
from movielens_recommender.movies import GENRE_TO_IDX, GENRES, _genre_multi_hot

# Dropped together by the retriever-score ablation. ``in_both`` stays.
RETRIEVER_SCORE_RANK_FEATURES: tuple[str, ...] = (
    "item_item_score",
    "item_item_rank",
    "two_tower_score",
    "two_tower_rank",
)

# Experiment modes. ``off`` is the S4 feature set and the production default.
DEMO_MODES: tuple[str, ...] = ("off", "raw", "affinity", "both")

# User-level codes. Non-negative integers so LightGBM can treat them as
# categoricals. Unknown sentinels live in ``data.py``.
RAW_DEMOGRAPHIC_FEATURES: tuple[str, ...] = (
    "demo_gender",
    "demo_age",
    "demo_occupation",
    "demo_region",
)

# Item × the user's demographic group, smoothed toward the global rate.
# Computed from the allowed group frame only (fit-train in this experiment).
GROUP_AFFINITY_FEATURES: tuple[str, ...] = (
    "group_age_pos_rate",
    "group_age_pop_share",
    "group_gender_pos_rate",
    "group_gender_pop_share",
    "group_occupation_pos_rate",
    "group_occupation_pop_share",
)

CATEGORICAL_DEMOGRAPHIC_FEATURES: tuple[str, ...] = RAW_DEMOGRAPHIC_FEATURES

# Additive prior strength for group rates. A group count of 0 returns the
# global positive rate; popularity share shrinks toward the item's global share.
GROUP_PRIOR_STRENGTH = 20.0

_GROUP_COLUMNS: tuple[tuple[str, str], ...] = (
    ("age", "age_code"),
    ("gender", "gender_code"),
    ("occupation", "occupation_code"),
)

_UNKNOWN_DEMO_CODES = np.array(
    [GENDER_UNKNOWN, AGE_UNKNOWN, OCCUPATION_UNKNOWN, REGION_UNKNOWN],
    dtype=np.float64,
)


def slug_genre(name: str) -> str:
    """Stable feature-name suffix for a MovieLens genre label."""
    text = str(name).lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _check_mode(mode: str) -> str:
    text = str(mode)
    if text not in DEMO_MODES:
        raise ValueError(f"unknown demographic feature mode {mode!r}; choose from {DEMO_MODES}")
    return text


def feature_names(mode: str = "off") -> list[str]:
    """Column order for the ranker matrix. Stable across runs.

    ``mode='off'`` is the S4 set. ``raw`` adds user demographic codes,
    ``affinity`` adds group-affinity rates, ``both`` adds both. Columns of
    the smaller modes are a prefix-stable subset of ``both`` except that
    ``affinity`` omits the raw block (it is baseline columns, then affinity).
    """
    mode = _check_mode(mode)
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
    if mode in {"raw", "both"}:
        names.extend(RAW_DEMOGRAPHIC_FEATURES)
    if mode in {"affinity", "both"}:
        names.extend(GROUP_AFFINITY_FEATURES)
    return names


def categorical_feature_names(mode: str = "off") -> list[str]:
    """Demographic columns LightGBM should split as categoricals."""
    mode = _check_mode(mode)
    if mode in {"raw", "both"}:
        return list(CATEGORICAL_DEMOGRAPHIC_FEATURES)
    return []


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
    demo_mode: str = "off"
    # user_id -> four non-negative codes (gender, age, occupation, region).
    demo_codes: dict[int, np.ndarray] = field(default_factory=dict)
    # user_id -> {age, gender, occupation} -> code used to index group stats.
    user_group_codes: dict[int, dict[str, int]] = field(default_factory=dict)
    # group name -> code -> (pos_rate, pop_share), each aligned to ``item_ids``.
    group_item_stats: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    prior_strength: float = GROUP_PRIOR_STRENGTH
    relevance_threshold: float = 4.0

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


def compute_group_item_stats(
    frame: pd.DataFrame,
    users: pd.DataFrame,
    item_ids: np.ndarray,
    *,
    relevance_threshold: float,
    prior_strength: float,
) -> dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Smoothed item positive rate and popularity share inside demographic groups.

    Global rates use every row of ``frame``. Group counts use only rows whose
    user is in ``users``. An item with no interactions in a group shrinks to
    the global positive rate, and its popularity share shrinks toward that
    item's global share with strength ``prior_strength``.

    Returned arrays are aligned to ``item_ids``. Items outside that catalog
    are ignored.
    """
    if frame.empty:
        raise ValueError("group-affinity stats require a non-empty interaction frame")
    strength = float(prior_strength)
    if strength < 0:
        raise ValueError("prior_strength must be >= 0")
    item_index = {int(i): n for n, i in enumerate(item_ids)}
    n_items = len(item_ids)
    n_total = float(len(frame))
    n_pos = float((frame["rating"] >= relevance_threshold).sum())
    global_pos = n_pos / n_total
    global_share = np.zeros(n_items, dtype=np.float64)
    for item_id, count in frame.groupby("item_id").size().items():
        idx = item_index.get(int(item_id))
        if idx is not None:
            global_share[idx] = float(count) / n_total

    cols = ["user_id", "age_code", "gender_code", "occupation_code"]
    missing = [c for c in cols if c not in users.columns]
    if missing:
        raise ValueError(f"users frame is missing demographic columns: {missing}")
    user_side = users[cols].drop_duplicates("user_id")
    merged = frame.merge(user_side, on="user_id", how="inner")

    out: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for key, col in _GROUP_COLUMNS:
        tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if merged.empty:
            out[key] = tables
            continue
        for code, sub in merged.groupby(col, sort=True):
            code_i = int(code)
            n_g = float(len(sub))
            pos_rate = np.full(n_items, global_pos, dtype=np.float64)
            pop_share = (strength * global_share) / (n_g + strength)
            item_n = sub.groupby("item_id").size()
            positive = sub.loc[sub["rating"] >= relevance_threshold]
            if positive.empty:
                item_pos = pd.Series(dtype=np.float64)
            else:
                item_pos = positive.groupby("item_id").size()
            for item_id, n_ig in item_n.items():
                idx = item_index.get(int(item_id))
                if idx is None:
                    continue
                n_ig_f = float(n_ig)
                p_ig = float(item_pos.loc[item_id]) if item_id in item_pos.index else 0.0
                pos_rate[idx] = (p_ig + strength * global_pos) / (n_ig_f + strength)
                pop_share[idx] = (n_ig_f + strength * global_share[idx]) / (n_g + strength)
            tables[code_i] = (pos_rate, pop_share)
        out[key] = tables
    return out


def _demo_lookup(
    users: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, int]]]:
    codes: dict[int, np.ndarray] = {}
    groups: dict[int, dict[str, int]] = {}
    for row in users.itertuples(index=False):
        uid = int(row.user_id)
        codes[uid] = np.array(
            [
                float(row.gender_code),
                float(row.age_code),
                float(row.occupation_code),
                float(row.region_code),
            ],
            dtype=np.float64,
        )
        groups[uid] = {
            "age": int(row.age_code),
            "gender": int(row.gender_code),
            "occupation": int(row.occupation_code),
        }
    return codes, groups


def build_feature_context(
    train: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    users: pd.DataFrame | None = None,
    mode: str = "off",
    group_frame: pd.DataFrame | None = None,
    relevance_threshold: float = 4.0,
    prior_strength: float = GROUP_PRIOR_STRENGTH,
) -> FeatureContext:
    """Statistics from ``train`` only. ``movies`` supplies genres and year.

    Demographic group-affinity tables are computed from ``group_frame`` when
    that argument is set, and from ``train`` otherwise. Pass fit-train as
    ``group_frame`` even when ``train`` is full train so validation rows do
    not enter the group rates. ``mode='off'`` ignores ``users``.
    """
    mode = _check_mode(mode)
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

    demo_codes: dict[int, np.ndarray] = {}
    user_groups: dict[int, dict[str, int]] = {}
    group_stats: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    if mode != "off":
        if users is None or users.empty:
            raise ValueError(f"demographic mode {mode!r} requires a users frame")
        demo_codes, user_groups = _demo_lookup(users)
        if mode in {"affinity", "both"}:
            source = train if group_frame is None else group_frame
            group_stats = compute_group_item_stats(
                source,
                users,
                item_ids,
                relevance_threshold=float(relevance_threshold),
                prior_strength=float(prior_strength),
            )

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
        names=feature_names(mode),
        demo_mode=mode,
        demo_codes=demo_codes,
        user_group_codes=user_groups,
        group_item_stats=group_stats,
        prior_strength=float(prior_strength),
        relevance_threshold=float(relevance_threshold),
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


def _stats_allclose(
    left: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
    right: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> bool:
    if set(left) != set(right):
        return False
    for key, tables in left.items():
        other = right[key]
        if set(tables) != set(other):
            return False
        for code, (pos, share) in tables.items():
            o_pos, o_share = other[code]
            if not np.allclose(pos, o_pos, rtol=0.0, atol=1e-12):
                return False
            if not np.allclose(share, o_share, rtol=0.0, atol=1e-12):
                return False
    return True


def assert_group_affinity_uses_only(
    ctx: FeatureContext,
    allowed: pd.DataFrame,
    users: pd.DataFrame,
    forbidden: pd.DataFrame | None = None,
) -> None:
    """Fail if group-affinity tables disagree with ``allowed``.

    When ``forbidden`` is set, folding it into the group frame must change at
    least one stored rate (otherwise the check cannot see leakage) and the
    context must still match ``allowed`` alone.
    """
    if ctx.demo_mode not in {"affinity", "both"}:
        raise AssertionError(
            f"group-affinity check requires mode affinity or both, got {ctx.demo_mode!r}"
        )
    expected = compute_group_item_stats(
        allowed,
        users,
        ctx.item_ids,
        relevance_threshold=ctx.relevance_threshold,
        prior_strength=ctx.prior_strength,
    )
    if not _stats_allclose(ctx.group_item_stats, expected):
        raise AssertionError("group-affinity stats do not match the allowed frame")
    if forbidden is None or forbidden.empty:
        return
    leaked = compute_group_item_stats(
        pd.concat([allowed, forbidden], ignore_index=True),
        users,
        ctx.item_ids,
        relevance_threshold=ctx.relevance_threshold,
        prior_strength=ctx.prior_strength,
    )
    if _stats_allclose(expected, leaked):
        raise AssertionError(
            "forbidden interactions did not change group-affinity stats; "
            "the leakage check is vacuous"
        )
    if _stats_allclose(ctx.group_item_stats, leaked):
        raise AssertionError("group-affinity stats include forbidden interactions")


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


def history_user_stats(
    ctx: FeatureContext,
    item_ids: Sequence[int],
    ratings: Sequence[float],
) -> tuple[float, float, float, np.ndarray | None]:
    """User activity and genre affinity from an explicit history, not ``ctx``.

    Used by the truncated-history cold-start check. Item popularity and
    demographic group tables on ``ctx`` are left unchanged.
    """
    ratings_arr = np.asarray(list(ratings), dtype=np.float64)
    n = float(len(ratings_arr))
    if n == 0:
        return n, float("nan"), float("nan"), None
    idxs = [ctx.item_index[int(i)] for i in item_ids if int(i) in ctx.item_index]
    affinity = ctx.genres[idxs].mean(axis=0) if idxs else None
    return n, float(ratings_arr.mean()), float(ratings_arr.std(ddof=0)), affinity


def build_feature_matrix(
    ctx: FeatureContext,
    user_id: int,
    item_ids: Sequence[int],
    item_item: Mapping[int, tuple[float, int]],
    two_tower: Mapping[int, tuple[float, int]],
    *,
    user_stats: tuple[float, float, float, np.ndarray | None] | None = None,
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
    if user_stats is None:
        n_ratings = ctx.user_n_ratings.get(uid, np.nan)
        mean_rating = ctx.user_mean_rating.get(uid, np.nan)
        std_rating = ctx.user_std_rating.get(uid, np.nan)
        affinity = ctx.user_affinity.get(uid)
    else:
        n_ratings, mean_rating, std_rating, affinity = user_stats
    n_genres = len(GENRES)
    genre0 = col[f"genre_{slug_genre(GENRES[0])}"]
    aff0 = col[f"affinity_{slug_genre(GENRES[0])}"]
    demo = None
    if ctx.demo_mode in {"raw", "both"}:
        demo = ctx.demo_codes.get(uid, _UNKNOWN_DEMO_CODES)
    if demo is not None:
        for offset, name in enumerate(RAW_DEMOGRAPHIC_FEATURES):
            x[:, col[name]] = float(demo[offset])
    group_codes = ctx.user_group_codes.get(uid) if ctx.demo_mode in {"affinity", "both"} else None

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
        if group_codes is not None and idx is not None:
            _fill_group_affinity(x, row, col, ctx, group_codes, idx)
    return x


def _fill_group_affinity(
    x: np.ndarray,
    row: int,
    col: dict[str, int],
    ctx: FeatureContext,
    group_codes: Mapping[str, int],
    item_idx: int,
) -> None:
    for key, prefix in (
        ("age", "group_age"),
        ("gender", "group_gender"),
        ("occupation", "group_occupation"),
    ):
        code = group_codes.get(key)
        tables = ctx.group_item_stats.get(key, {})
        stats = tables.get(int(code)) if code is not None else None
        if stats is None:
            continue
        pos_rate, pop_share = stats
        x[row, col[f"{prefix}_pos_rate"]] = float(pos_rate[item_idx])
        x[row, col[f"{prefix}_pop_share"]] = float(pop_share[item_idx])


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
