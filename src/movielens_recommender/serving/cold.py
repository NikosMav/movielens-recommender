"""Snapshot and scoring for a person who has no MovieLens user id.

The files live under ``artifacts/<dataset>/cold_start/`` and are gitignored.
Training stays in :mod:`movielens_recommender.cold_start`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.cold_start_rank import profile_bucket, union_candidate_ids
from movielens_recommender.ranker.candidates import score_rank_maps
from movielens_recommender.ranker.explain import explain_candidates
from movielens_recommender.ranker.features import (
    FeatureContext,
    annotate_cold_start_ranker_rows,
    build_feature_matrix,
    history_user_stats,
)
from movielens_recommender.serving.reasons import (
    POPULAR_WITH_MANY_VIEWERS,
    HistoryItem,
    RecommendationExplanation,
    display_title,
    explain_recommendation,
)
from movielens_recommender.two_tower.recommender import TwoTowerRecommender

SCHEMA_VERSION = 1
KIND = "cold_start"
_NEW_USER_ID = 0
_CANDIDATE_SOURCES = ("item_item_fold_in", "ease_fold_in", "history_two_tower")


@dataclass
class ColdStartBundle:
    """Retrievers and ranker for a profile that is only a list of ratings."""

    manifest: dict[str, Any]
    movies: pd.DataFrame
    item_item: ItemItemCosineRecommender
    ease: EASERecommender
    two_tower: TwoTowerRecommender
    booster: Any
    context: FeatureContext
    popular_item_ids: np.ndarray | None = None

    @property
    def candidate_source(self) -> str:
        return str(self.manifest["candidate_source"])

    @property
    def candidate_k(self) -> int:
        return int(self.manifest["candidate_k"])


def save_cold_start_bundle(
    directory: Path | str,
    *,
    movies: pd.DataFrame,
    item_item: ItemItemCosineRecommender,
    ease: EASERecommender,
    two_tower: TwoTowerRecommender,
    booster: Any,
    context: FeatureContext,
    manifest: Mapping[str, Any],
    popular_item_ids: np.ndarray | None = None,
) -> Path:
    """Write the new-user snapshot. ``directory`` is created and gitignored."""
    from movielens_recommender.ranker.explain import save_ranker
    from movielens_recommender.serving.bundle import (
        _save_context,
        _save_item_item,
        _save_movies,
        _save_two_tower,
    )

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _save_movies(movies, directory / "movies.csv")
    _save_item_item(item_item, directory)
    _save_ease(ease, directory)
    _save_two_tower(two_tower, directory)
    _save_context(context, directory)
    payload = dict(manifest)
    payload["schema_version"] = SCHEMA_VERSION
    payload["kind"] = KIND
    save_ranker(
        booster,
        directory,
        {
            "feature_names": list(context.names),
            "best_iteration": int(payload.get("best_iteration", 0)),
            "num_boost_round": int(payload.get("best_iteration", 0)),
            "candidate_set": str(payload.get("candidate_source", "")),
            "objective": "lambdarank",
            "dataset": payload.get("dataset"),
            "demographics": payload.get("ranker_mode"),
        },
    )
    if popular_item_ids is not None:
        np.save(
            directory / "popular_items.npy",
            np.asarray(popular_item_ids, dtype=np.int64),
        )
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def load_cold_start_bundle(directory: Path | str) -> ColdStartBundle:
    """Load a snapshot written by :func:`save_cold_start_bundle`."""
    from movielens_recommender.ranker.explain import load_ranker
    from movielens_recommender.serving.bundle import (
        _load_context,
        _load_item_item,
        _load_two_tower,
    )

    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No cold-start snapshot at {directory}. Build one with "
            "movielens-recommender cold-start --config configs/ml-1m.yaml"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("kind")) != KIND:
        raise ValueError(f"{directory} is not a cold-start snapshot")
    version = int(manifest.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"cold-start schema_version {version} does not match supported {SCHEMA_VERSION}"
        )
    movies = pd.read_csv(directory / "movies.csv")
    booster, _meta = load_ranker(directory)
    context = _load_context(directory)
    model_names = list(booster.feature_name())
    if model_names != list(context.names):
        raise ValueError("ranker feature names do not match the feature context")
    popular_path = directory / "popular_items.npy"
    if manifest.get("serving_by_n") is not None and not popular_path.is_file():
        raise FileNotFoundError(f"cold-start snapshot at {directory} is missing popular_items.npy")
    popular = np.load(popular_path).astype(np.int64) if popular_path.is_file() else None
    return ColdStartBundle(
        manifest=manifest,
        movies=movies,
        item_item=_load_item_item(directory),
        ease=_load_ease(directory),
        two_tower=_load_two_tower(directory),
        booster=booster,
        context=context,
        popular_item_ids=popular,
    )


def method_sentence(method: str, n_ratings: int, bucket: int) -> str:
    """Plain description of the method chosen for this profile length."""
    sentences = {
        "most_popular": "The list is the titles most people have rated.",
        "item_item_fold_in": "The list is neighbours of the films you rated.",
        "history_two_tower": (
            "The list comes from the neural retriever, using only the films you rated."
        ),
        "cold_start_ranker": (
            "The list is a ranker trained on short profiles. "
            "It re-ranks popular titles and neighbours of the films you rated."
        ),
    }
    if method not in sentences:
        raise ValueError(f"unknown serving method {method!r}")
    return (
        f"{sentences[method]} For a profile of {int(n_ratings)} ratings, "
        f"using the rule chosen for N={int(bucket)}."
    )


def recommend_new_user(
    bundle: ColdStartBundle,
    ratings: Sequence[tuple[int, float]],
    *,
    n: int = 10,
) -> dict[str, Any]:
    """Top-n for a profile of ``(item_id, rating)`` pairs, oldest first.

    Rated items are excluded. Reasons never include demographic or group
    feature names. ``latency_sec`` is the wall time of this call.

    Snapshots without ``serving_by_n`` keep the single-source ranker path.
    """
    started = time.perf_counter()
    profile = _collapse_profile(ratings)
    if not profile:
        raise ValueError("a new-user profile needs at least one rating")
    serving = bundle.manifest.get("serving_by_n")
    if serving:
        result = _recommend_served(bundle, profile, serving, n=n)
    else:
        result = _recommend_v1(bundle, profile, n=n)
    result["latency_sec"] = time.perf_counter() - started
    return result


def _recommend_v1(
    bundle: ColdStartBundle,
    profile: list[tuple[int, float]],
    *,
    n: int,
) -> dict[str, Any]:
    rated = [item for item, _rating in profile]
    rated_set = set(rated)
    k = bundle.candidate_k
    uid = _NEW_USER_ID
    ii_rows = bundle.item_item.topk_for_profiles({uid: profile}, k).get(uid, [])
    ease_rows = bundle.ease.topk_for_profiles({uid: profile}, k).get(uid, [])
    tt_rows = bundle.two_tower.topk_for_new_histories({uid: rated}, k).get(uid, [])
    ii_map = score_rank_maps({uid: ii_rows}).get(uid, {})
    ease_map = score_rank_maps({uid: ease_rows}).get(uid, {})
    tt_map = score_rank_maps({uid: tt_rows}).get(uid, {})
    source = bundle.candidate_source
    if source not in _CANDIDATE_SOURCES:
        raise ValueError(f"unknown candidate source {source!r}")
    by_source = {
        "item_item_fold_in": ii_rows,
        "ease_fold_in": ease_rows,
        "history_two_tower": tt_rows,
    }
    candidates = [int(item) for item, _score in by_source[source] if int(item) not in rated_set]
    stats = history_user_stats(
        bundle.context,
        rated,
        [rating for _item, rating in profile],
    )
    matrix = build_feature_matrix(
        bundle.context,
        uid,
        candidates,
        ii_map,
        tt_map,
        user_stats=stats,
        ease=ease_map,
        demographics_missing=bool(bundle.manifest.get("demographics_missing", False)),
    )
    history = _history_items(bundle, profile)
    cards = _rank_and_explain(
        bundle,
        candidates,
        matrix,
        history,
        ii_map,
        tt_map,
        ease_map,
        n=n,
    )
    for card in cards:
        if int(card["item_id"]) in rated_set:
            raise AssertionError("a rated movie was recommended")
        _assert_no_demographic_reason(card)
    return {
        "n_ratings": len(profile),
        "candidate_source": source,
        "recommendations": cards,
    }


def _recommend_served(
    bundle: ColdStartBundle,
    profile: list[tuple[int, float]],
    serving: Mapping[str, str],
    *,
    n: int,
) -> dict[str, Any]:
    """Per-N rule. The ranker, when selected, sees the union candidate pool."""
    rated = [item for item, _rating in profile]
    rated_set = set(rated)
    bucket = profile_bucket(len(profile))
    method = str(serving[str(bucket)])
    history = _history_items(bundle, profile)
    if method == "cold_start_ranker":
        cards = _ranker_union_cards(bundle, profile, history, n=n)
    elif method == "most_popular":
        cards = [
            _plain_card(bundle, item, history, method)
            for item in _popular_top(bundle, rated_set, n)
        ]
    elif method == "item_item_fold_in":
        rows = bundle.item_item.topk_for_profiles({_NEW_USER_ID: profile}, n).get(_NEW_USER_ID, [])
        cards = [
            _plain_card(bundle, int(item), history, method)
            for item, _score in rows
            if int(item) not in rated_set
        ]
    elif method == "history_two_tower":
        rows = bundle.two_tower.topk_for_new_histories({_NEW_USER_ID: rated}, n).get(
            _NEW_USER_ID, []
        )
        cards = [
            _plain_card(bundle, int(item), history, method)
            for item, _score in rows
            if int(item) not in rated_set
        ]
    else:
        raise ValueError(f"unknown serving method {method!r}")
    cards = cards[:n]
    for card in cards:
        if int(card["item_id"]) in rated_set:
            raise AssertionError("a rated movie was recommended")
        _assert_no_demographic_reason(card)
    return {
        "n_ratings": len(profile),
        "profile_bucket": int(bucket),
        "method": method,
        "method_sentence": method_sentence(method, len(profile), bucket),
        "candidate_source": method,
        "recommendations": cards,
    }


def _popular_top(bundle: ColdStartBundle, banned: set[int], n: int) -> list[int]:
    ranked = bundle.popular_item_ids
    if ranked is None:
        raise ValueError("this snapshot has no popularity ranking")
    picked: list[int] = []
    for item in ranked.tolist():
        iid = int(item)
        if iid in banned:
            continue
        picked.append(iid)
        if len(picked) >= n:
            break
    return picked


def _plain_card(
    bundle: ColdStartBundle,
    item_id: int,
    history: list[HistoryItem],
    method: str,
) -> dict[str, Any]:
    """A reason sentence without presenting a fake ranker contribution."""
    contributions = {
        "most_popular": {"item_popularity": 1.0},
        "item_item_fold_in": {"item_item_score": 1.0},
        "history_two_tower": {"two_tower_score": 1.0},
    }
    sims = bundle.item_item.neighbor_similarities(
        int(item_id), [item.item_id for item in history]
    )
    explanation = explain_recommendation(
        contributions[method],
        history=history,
        neighbor_similarity=sims,
        include_demographics=False,
        because_rated=True,
        popularity_text=POPULAR_WITH_MANY_VIEWERS,
    )
    card = _card(bundle, int(item_id), explanation)
    card["details"] = []
    card["bias"] = None
    card["raw_score"] = None
    return card


def _ranker_union_cards(
    bundle: ColdStartBundle,
    profile: list[tuple[int, float]],
    history: list[HistoryItem],
    *,
    n: int,
) -> list[dict[str, Any]]:
    rated = [item for item, _rating in profile]
    rated_set = set(rated)
    k = bundle.candidate_k
    uid = _NEW_USER_ID
    ii_rows = bundle.item_item.topk_for_profiles({uid: profile}, k).get(uid, [])
    ease_rows = bundle.ease.topk_for_profiles({uid: profile}, k).get(uid, [])
    tt_rows = bundle.two_tower.topk_for_new_histories({uid: rated}, k).get(uid, [])
    popular_ids = _popular_top(bundle, rated_set, k)
    items, ease_set, pop_set, tower_set = union_candidate_ids(
        [int(item) for item, _score in ease_rows],
        popular_ids,
        [int(item) for item, _score in tt_rows],
        banned=rated_set,
    )
    ii_map = score_rank_maps({uid: ii_rows}).get(uid, {})
    ease_map = score_rank_maps({uid: ease_rows}).get(uid, {})
    tt_map = score_rank_maps({uid: tt_rows}).get(uid, {})
    stats = history_user_stats(
        bundle.context,
        rated,
        [rating for _item, rating in profile],
    )
    names = list(bundle.context.names)
    matrix = build_feature_matrix(
        bundle.context,
        uid,
        items,
        ii_map,
        tt_map,
        user_stats=stats,
        ease=ease_map,
        demographics_missing=False,
    )
    matrix = annotate_cold_start_ranker_rows(
        matrix,
        names,
        items,
        profile_n=len(profile),
        ease_ids=ease_set,
        popular_ids=pop_set,
        tower_ids=tower_set,
    )
    return _rank_and_explain(
        bundle,
        items,
        matrix,
        history,
        ii_map,
        tt_map,
        ease_map,
        n=n,
    )


def measure_new_user_latency(directory: Path | str, train: pd.DataFrame) -> dict[str, Any]:
    """Warmed latency of one top-10 call on five popular training titles.

    The profile is the five items with the most training ratings, ties broken
    by item id, each rated 5. The first call warms caches. The stored time is
    the second call.
    """
    bundle = load_cold_start_bundle(directory)
    counts = train.groupby("item_id", sort=False).size().rename("n").reset_index()
    ordered = counts.sort_values(["n", "item_id"], ascending=[False, True], kind="mergesort")
    chosen = [int(item) for item in ordered["item_id"].head(5).tolist()]
    profile = [(item, 5.0) for item in chosen]
    recommend_new_user(bundle, profile, n=10)
    result = recommend_new_user(bundle, profile, n=10)
    payload = {
        "latency_sec": round(float(result["latency_sec"]), 6),
        "n_ratings": 5,
        "rating": 5.0,
        "profile_item_ids": chosen,
        "n": 10,
        "warmed": True,
        "note": (
            "Second call to recommend_new_user in one process, after one "
            "warmup call on the same profile. The profile is the five most "
            "common training items, ties broken by item id, each rated 5."
        ),
    }
    if result.get("method"):
        payload["method"] = result["method"]
        payload["profile_bucket"] = int(result["profile_bucket"])
        payload["method_sentence"] = result["method_sentence"]
    return payload


def _collapse_profile(ratings: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    latest: dict[int, float] = {}
    order: list[int] = []
    for item_id, rating in ratings:
        item = int(item_id)
        if item not in latest:
            order.append(item)
        latest[item] = float(rating)
    return [(item, latest[item]) for item in order]


def _history_items(
    bundle: ColdStartBundle,
    profile: Sequence[tuple[int, float]],
) -> list[HistoryItem]:
    items: list[HistoryItem] = []
    for step, (item_id, rating) in enumerate(profile, start=1):
        meta = _movie(bundle, item_id)
        items.append(
            HistoryItem(
                item_id=int(item_id),
                title=str(meta["title"]),
                genres=str(meta["genres"]),
                rating=float(rating),
                timestamp=step,
            )
        )
    return items


def _movie(bundle: ColdStartBundle, item_id: int) -> dict[str, Any]:
    frame = bundle.movies
    hit = frame.loc[frame["item_id"] == int(item_id)]
    if hit.empty:
        return {"title": f"Item {int(item_id)}", "year": None, "genres": ""}
    row = hit.iloc[0]
    year = row["year"]
    year_out: int | None
    if pd.isna(year):
        year_out = None
    else:
        year_out = int(year)
    return {
        "title": str(row["title"]),
        "year": year_out,
        "genres": str(row["genres"]),
    }


def _rank_and_explain(
    bundle: ColdStartBundle,
    candidates: list[int],
    matrix: np.ndarray,
    history: list[HistoryItem],
    ii_map: Mapping[int, tuple[float, int]],
    tt_map: Mapping[int, tuple[float, int]],
    ease_map: Mapping[int, tuple[float, int]],
    *,
    n: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    scores = np.asarray(bundle.booster.predict(matrix), dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    top_idx = [int(i) for i in order[:n]]
    top_items = [candidates[i] for i in top_idx]
    history_ids = [item.item_id for item in history]
    provenance = []
    for item in top_items:
        provenance.append(
            {
                "item_item": _hit(ii_map.get(int(item))),
                "two_tower": _hit(tt_map.get(int(item))),
                "ease": _hit(ease_map.get(int(item))),
            }
        )
    explained = explain_candidates(
        bundle.booster,
        matrix[top_idx],
        list(bundle.context.names),
        item_ids=top_items,
        retrievers=provenance,
    )
    cards: list[dict[str, Any]] = []
    for row in explained:
        item_id = int(row["item_id"])
        sims = bundle.item_item.neighbor_similarities(item_id, history_ids)
        explanation = explain_recommendation(
            row["contributions"],
            history=history,
            neighbor_similarity=sims,
            bias=float(row["bias"]),
            raw_score=float(row["raw_score"]),
            include_demographics=False,
            because_rated=True,
            popularity_text=POPULAR_WITH_MANY_VIEWERS,
        )
        cards.append(_card(bundle, item_id, explanation))
    return cards


def _card(
    bundle: ColdStartBundle,
    item_id: int,
    explanation: RecommendationExplanation,
) -> dict[str, Any]:
    meta = _movie(bundle, item_id)
    return {
        "item_id": int(item_id),
        "title": display_title(str(meta["title"])),
        "year": meta["year"],
        "genres": meta["genres"],
        "reasons": [reason.text for reason in explanation.reasons],
        "details": [
            {"feature": name, "contribution": value} for name, value in explanation.details
        ],
        "bias": explanation.bias,
        "raw_score": explanation.raw_score,
    }


def _hit(pair: tuple[float, int] | None) -> dict[str, float | int] | None:
    if pair is None:
        return None
    return {"score": float(pair[0]), "rank": int(pair[1])}


def _assert_no_demographic_reason(card: Mapping[str, Any]) -> None:
    blob = " ".join(str(text) for text in card.get("reasons") or [])
    if "demo_" in blob or "group_" in blob:
        raise AssertionError("a new-user reason named a demographic feature")
    if "Popular with viewers similar to you" in blob:
        raise AssertionError("a new-user reason used the demographic sentence")
    for row in card.get("details") or []:
        name = str(row["feature"])
        if name.startswith("demo_") or name.startswith("group_"):
            raise AssertionError(f"new-user details include {name}")


def _save_ease(model: EASERecommender, directory: Path) -> None:
    if model._similarity is None:
        raise ValueError("EASE model is not fit")
    np.savez_compressed(
        directory / "ease.npz",
        similarity=np.asarray(model._similarity, dtype=np.float64),
        item_ids=np.asarray(model._item_ids, dtype=np.int64),
    )
    (directory / "ease_meta.json").write_text(
        json.dumps({"l2": float(model.l2)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_ease(directory: Path) -> EASERecommender:
    meta = json.loads((directory / "ease_meta.json").read_text(encoding="utf-8"))
    model = EASERecommender(l2=float(meta["l2"]))
    with np.load(directory / "ease.npz") as packed:
        item_ids = packed["item_ids"].astype(np.int64)
        similarity = np.asarray(packed["similarity"], dtype=np.float64)
    model._item_ids = item_ids
    model._item_index = {int(item): i for i, item in enumerate(item_ids.tolist())}
    model._similarity = similarity
    return model
