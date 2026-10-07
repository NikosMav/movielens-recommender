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
from movielens_recommender.ranker.candidates import score_rank_maps
from movielens_recommender.ranker.explain import explain_candidates
from movielens_recommender.ranker.features import (
    FeatureContext,
    build_feature_matrix,
    history_user_stats,
)
from movielens_recommender.serving.reasons import (
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
    return ColdStartBundle(
        manifest=manifest,
        movies=movies,
        item_item=_load_item_item(directory),
        ease=_load_ease(directory),
        two_tower=_load_two_tower(directory),
        booster=booster,
        context=context,
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
    """
    started = time.perf_counter()
    profile = _collapse_profile(ratings)
    if not profile:
        raise ValueError("a new-user profile needs at least one rating")
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
        "latency_sec": time.perf_counter() - started,
    }


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
    return {
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
