"""Load and query a serving snapshot. Training lives in :mod:`.build`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.movies import GENRES
from movielens_recommender.ranker.candidates import candidate_ids_for_user, score_rank_maps
from movielens_recommender.ranker.explain import explain_candidates
from movielens_recommender.ranker.features import (
    FeatureContext,
    build_feature_matrix,
    cold_start_feature_names,
    feature_names,
)
from movielens_recommender.serving.reasons import (
    HistoryItem,
    RecommendationExplanation,
    display_title,
    explain_recommendation,
    format_stars,
)
from movielens_recommender.two_tower.features import TwoTowerFeatures
from movielens_recommender.two_tower.model import TwoTowerModel
from movielens_recommender.two_tower.recommender import TwoTowerRecommender

SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"

_GROUP_KEYS = ("age", "gender", "occupation")


@dataclass
class ServingBundle:
    """Retrievers, refit ranker, feature context, and the histories they used."""

    manifest: dict[str, Any]
    movies: pd.DataFrame
    histories: pd.DataFrame
    item_item: ItemItemCosineRecommender
    two_tower: TwoTowerRecommender
    booster: Any
    context: FeatureContext

    @property
    def candidate_set(self) -> str:
        return str(self.manifest["candidate_set"])

    @property
    def candidate_k(self) -> int:
        return int(self.manifest["candidate_k"])


def save_bundle(bundle: ServingBundle, directory: Path | str) -> Path:
    """Write the snapshot. ``directory`` is created and is meant to be gitignored."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _save_movies(bundle.movies, directory / "movies.csv")
    _save_histories(bundle.histories, directory / "histories.npz")
    _save_item_item(bundle.item_item, directory)
    _save_two_tower(bundle.two_tower, directory)
    _save_context(bundle.context, directory)
    from movielens_recommender.ranker.explain import save_ranker

    save_ranker(
        bundle.booster,
        directory,
        {
            "feature_names": list(bundle.context.names),
            "best_iteration": int(bundle.manifest.get("best_iteration", 0)),
            "num_boost_round": int(bundle.manifest.get("best_iteration", 0)),
            "candidate_set": bundle.candidate_set,
            "objective": "lambdarank",
            "dataset": bundle.manifest.get("dataset"),
            "seed": bundle.manifest.get("ranker_seed"),
            "demographics": bundle.manifest.get("demographics"),
        },
    )
    manifest = dict(bundle.manifest)
    manifest["schema_version"] = SCHEMA_VERSION
    (directory / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return directory


def load_bundle(directory: Path | str) -> ServingBundle:
    """Load a snapshot written by :func:`save_bundle`.

    Callers that serve many requests should call this once and reuse the bundle.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No serving snapshot at {directory}. Build one with "
            "movielens-recommender build-artifacts --config configs/ml-1m.yaml"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = int(manifest.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"serving schema_version {version} does not match supported {SCHEMA_VERSION}"
        )
    from movielens_recommender.ranker.explain import load_ranker

    booster, _meta = load_ranker(directory)
    context = _load_context(directory)
    model_names = list(booster.feature_name())
    if model_names != list(context.names):
        raise ValueError(
            "Ranker feature names do not match the saved feature context. "
            f"ranker={model_names[:4]}... context={list(context.names)[:4]}..."
        )
    movies = pd.read_csv(directory / "movies.csv")
    if "year" in movies.columns:
        movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    histories = _load_histories(directory / "histories.npz")
    return ServingBundle(
        manifest=manifest,
        movies=movies,
        histories=histories,
        item_item=_load_item_item(directory),
        two_tower=_load_two_tower(directory),
        booster=booster,
        context=context,
    )


def movie_row(bundle: ServingBundle, item_id: int) -> dict[str, Any]:
    """Title, year, and genres for one item. Missing metadata gets a plain fallback."""
    frame = bundle.movies
    hit = frame.loc[frame["item_id"] == int(item_id)]
    if hit.empty:
        return {
            "item_id": int(item_id),
            "title": f"Item {int(item_id)}",
            "year": None,
            "genres": "",
        }
    row = hit.iloc[0]
    year = row["year"]
    year_out: int | None
    if pd.isna(year):
        year_out = None
    else:
        year_out = int(year)
    return {
        "item_id": int(item_id),
        "title": str(row["title"]),
        "year": year_out,
        "genres": str(row["genres"]),
    }


def user_history_items(bundle: ServingBundle, user_id: int) -> list[HistoryItem]:
    """Training-profile rows for ``user_id``, oldest first."""
    frame = bundle.histories
    rows = frame.loc[frame["user_id"] == int(user_id)]
    if rows.empty:
        return []
    ordered = rows.sort_values(["timestamp", "item_id"], kind="mergesort")
    items: list[HistoryItem] = []
    for record in ordered.itertuples(index=False):
        meta = movie_row(bundle, int(record.item_id))
        items.append(
            HistoryItem(
                item_id=int(record.item_id),
                title=str(meta["title"]),
                genres=str(meta["genres"]),
                rating=float(record.rating),
                timestamp=int(record.timestamp),
            )
        )
    return items


def recent_history(bundle: ServingBundle, user_id: int, n: int = 10) -> list[dict[str, Any]]:
    """Most recent training ratings, newest first, ready for a table."""
    items = user_history_items(bundle, user_id)
    chosen = list(reversed(items[-n:])) if n > 0 else []
    table = []
    for item in chosen:
        meta = movie_row(bundle, item.item_id)
        table.append(
            {
                "title": display_title(item.title),
                "year": meta["year"] if meta["year"] is not None else "",
                "genres": item.genres,
                "stars": format_stars(item.rating),
            }
        )
    return table


def user_summary(bundle: ServingBundle, user_id: int, n_genres: int = 3) -> dict[str, Any]:
    """Rating count and the most common genres in one user's training profile."""
    items = user_history_items(bundle, user_id)
    counts: dict[str, int] = {}
    for item in items:
        for genre in str(item.genres).split("|"):
            name = genre.strip()
            if not name or name == "(no genres listed)":
                continue
            counts[name] = counts.get(name, 0) + 1
    top = sorted(counts, key=lambda name: (-counts[name], name))[:n_genres]
    return {"user_id": int(user_id), "n_ratings": len(items), "top_genres": top}


def user_summaries(bundle: ServingBundle, n_genres: int = 3) -> dict[int, dict[str, Any]]:
    """One pass over the training profile. Used when the page lists every user."""
    genre_of = {
        int(item): str(genres)
        for item, genres in zip(
            bundle.movies["item_id"].tolist(),
            bundle.movies["genres"].tolist(),
            strict=True,
        )
    }
    n_ratings: dict[int, int] = {}
    genre_counts: dict[int, dict[str, int]] = {}
    users = bundle.histories["user_id"].to_numpy()
    item_ids = bundle.histories["item_id"].to_numpy()
    for uid, item_id in zip(users.tolist(), item_ids.tolist(), strict=True):
        uid_i = int(uid)
        n_ratings[uid_i] = n_ratings.get(uid_i, 0) + 1
        counts = genre_counts.setdefault(uid_i, {})
        for genre in genre_of.get(int(item_id), "").split("|"):
            name = genre.strip()
            if not name or name == "(no genres listed)":
                continue
            counts[name] = counts.get(name, 0) + 1
    summaries: dict[int, dict[str, Any]] = {}
    for uid, n in n_ratings.items():
        counts = genre_counts.get(uid, {})
        top = sorted(counts, key=lambda name: (-counts[name], name))[:n_genres]
        summaries[uid] = {"user_id": uid, "n_ratings": n, "top_genres": top}
    return summaries


def known_user_ids(bundle: ServingBundle) -> list[int]:
    return sorted(int(uid) for uid in bundle.histories["user_id"].unique())


def recommend_for_user(
    bundle: ServingBundle,
    user_id: int,
    *,
    n: int = 10,
) -> dict[str, Any]:
    """Top-n production list, plus item–item and two-tower lists for comparison.

    Explanations are attached only to the production list. An unknown user id
    raises ``KeyError``; the two-tower has no embedding for them.
    """
    uid = int(user_id)
    if uid not in set(known_user_ids(bundle)):
        raise KeyError(f"user {uid} is not in the serving snapshot")
    k = bundle.candidate_k
    ii_rows = bundle.item_item.topk_for_user(uid, k)
    tt_rows = bundle.two_tower.topk_for_user(uid, k)
    left_ids = [item for item, _score in ii_rows]
    right_ids = [item for item, _score in tt_rows]
    candidates = candidate_ids_for_user(bundle.candidate_set, left_ids, right_ids, k)
    ii_map = score_rank_maps({uid: ii_rows}).get(uid, {})
    tt_map = score_rank_maps({uid: tt_rows}).get(uid, {})
    matrix = build_feature_matrix(bundle.context, uid, candidates, ii_map, tt_map)
    if len(candidates) == 0:
        production: list[dict[str, Any]] = []
    else:
        scores = np.asarray(bundle.booster.predict(matrix), dtype=np.float64)
        order = np.argsort(-scores, kind="mergesort")
        top_idx = [int(i) for i in order[:n]]
        top_items = [candidates[i] for i in top_idx]
        top_matrix = matrix[top_idx]
        provenance = []
        for item in top_items:
            provenance.append(
                {
                    "item_item": _hit(ii_map.get(int(item))),
                    "two_tower": _hit(tt_map.get(int(item))),
                }
            )
        explained = explain_candidates(
            bundle.booster,
            top_matrix,
            list(bundle.context.names),
            item_ids=top_items,
            retrievers=provenance,
        )
        history = user_history_items(bundle, uid)
        history_ids = [item.item_id for item in history]
        production = []
        for row in explained:
            item_id = int(row["item_id"])
            sims = bundle.item_item.neighbor_similarities(item_id, history_ids)
            explanation = explain_recommendation(
                row["contributions"],
                history=history,
                neighbor_similarity=sims,
                bias=float(row["bias"]),
                raw_score=float(row["raw_score"]),
            )
            production.append(_card(bundle, item_id, explanation))
    return {
        "user_id": uid,
        "candidate_set": bundle.candidate_set,
        "cold_start": str(bundle.manifest.get("cold_start", "existing_users_only")),
        "production": production,
        "item_item": [_card(bundle, item) for item, _score in ii_rows[:n]],
        "two_tower": [_card(bundle, item) for item, _score in tt_rows[:n]],
    }


def _card(
    bundle: ServingBundle,
    item_id: int,
    explanation: RecommendationExplanation | None = None,
) -> dict[str, Any]:
    meta = movie_row(bundle, item_id)
    card: dict[str, Any] = {
        "item_id": int(item_id),
        "title": display_title(str(meta["title"])),
        "year": meta["year"],
        "genres": meta["genres"],
    }
    if explanation is not None:
        card["reasons"] = [reason.text for reason in explanation.reasons]
        card["details"] = [
            {"feature": name, "contribution": value} for name, value in explanation.details
        ]
        card["bias"] = explanation.bias
        card["raw_score"] = explanation.raw_score
    return card


def _hit(pair: tuple[float, int] | None) -> dict[str, float | int] | None:
    if pair is None:
        return None
    return {"score": float(pair[0]), "rank": int(pair[1])}


def _save_movies(movies: pd.DataFrame, path: Path) -> None:
    frame = movies.loc[:, ["item_id", "title", "genres", "year"]].copy()
    frame.to_csv(path, index=False)


def _save_histories(histories: pd.DataFrame, path: Path) -> None:
    frame = histories.loc[:, ["user_id", "item_id", "rating", "timestamp"]]
    np.savez_compressed(
        path,
        user_id=frame["user_id"].to_numpy(dtype=np.int64),
        item_id=frame["item_id"].to_numpy(dtype=np.int64),
        rating=frame["rating"].to_numpy(dtype=np.float64),
        timestamp=frame["timestamp"].to_numpy(dtype=np.int64),
    )


def _load_histories(path: Path) -> pd.DataFrame:
    with np.load(path) as packed:
        return pd.DataFrame(
            {
                "user_id": packed["user_id"].astype(np.int64),
                "item_id": packed["item_id"].astype(np.int64),
                "rating": packed["rating"].astype(np.float64),
                "timestamp": packed["timestamp"].astype(np.int64),
            }
        )


def _save_item_item(model: ItemItemCosineRecommender, directory: Path) -> None:
    if model._similarity is None or model._user_item is None:
        raise ValueError("item-item model is not fit")
    user_ids = np.empty(len(model._user_index), dtype=np.int64)
    for uid, uidx in model._user_index.items():
        user_ids[uidx] = int(uid)
    sparse.save_npz(directory / "item_item_similarity.npz", sparse.csr_matrix(model._similarity))
    sparse.save_npz(directory / "item_item_user_item.npz", model._user_item.tocsr())
    np.savez(
        directory / "item_item_index.npz",
        item_ids=np.asarray(model._item_ids, dtype=np.int64),
        user_ids=user_ids,
    )
    (directory / "item_item_meta.json").write_text(
        json.dumps(model.hyperparams(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_item_item(directory: Path) -> ItemItemCosineRecommender:
    meta = json.loads((directory / "item_item_meta.json").read_text(encoding="utf-8"))
    model = ItemItemCosineRecommender(
        min_common=int(meta["min_common"]),
        k_neighbors=int(meta["k_neighbors"]),
        shrinkage=float(meta["shrinkage"]),
    )
    with np.load(directory / "item_item_index.npz") as packed:
        item_ids = packed["item_ids"].astype(np.int64)
        user_ids = packed["user_ids"].astype(np.int64)
    similarity = sparse.load_npz(directory / "item_item_similarity.npz")
    model._similarity = similarity.toarray().astype(np.float64)
    model._user_item = sparse.load_npz(directory / "item_item_user_item.npz").tocsr()
    model._item_ids = item_ids
    model._item_index = {int(item): i for i, item in enumerate(item_ids)}
    model._user_index = {int(user): i for i, user in enumerate(user_ids)}
    return model


def _pack_ragged(
    mapping: Mapping[int, np.ndarray | set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.asarray(sorted(int(k) for k in mapping), dtype=np.int64)
    if len(keys) == 0:
        empty = np.asarray([], dtype=np.int64)
        return empty, np.asarray([0], dtype=np.int64), empty
    chunks = [np.asarray(list(mapping[int(key)]), dtype=np.int64) for key in keys]
    lengths = np.asarray([len(chunk) for chunk in chunks], dtype=np.int64)
    offsets = np.zeros(len(keys) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    values = np.concatenate(chunks) if chunks else np.asarray([], dtype=np.int64)
    return keys, offsets, values


def _unpack_ragged(
    keys: np.ndarray, offsets: np.ndarray, values: np.ndarray
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for i, key in enumerate(keys.tolist()):
        out[int(key)] = np.asarray(values[int(offsets[i]) : int(offsets[i + 1])], dtype=np.int64)
    return out


def _save_two_tower(model: TwoTowerRecommender, directory: Path) -> None:
    if model._model is None or model._features is None:
        raise ValueError("two-tower model is not fit")
    torch_model = model._model
    features = model._features
    torch.save(torch_model.state_dict(), directory / "two_tower_model.pt")
    hist_keys, hist_offsets, hist_values = _pack_ragged(features.user_history)
    seen_keys, seen_offsets, seen_values = _pack_ragged(features.seen_item_ids)
    np.savez_compressed(
        directory / "two_tower_features.npz",
        user_ids=np.asarray(features.user_ids, dtype=np.int64),
        item_ids=np.asarray(features.item_ids, dtype=np.int64),
        genres=np.asarray(features.genres, dtype=np.float32),
        years=np.asarray(features.years, dtype=np.float32),
        item_q=np.asarray(features.item_q, dtype=np.float64),
        hist_keys=hist_keys,
        hist_offsets=hist_offsets,
        hist_values=hist_values,
        seen_keys=seen_keys,
        seen_offsets=seen_offsets,
        seen_values=seen_values,
    )
    meta = {
        "embedding_dim": int(torch_model.embedding_dim),
        "temperature": float(torch_model.temperature),
        "n_users": int(torch_model.n_users),
        "n_items": int(torch_model.n_items),
        "n_genres": int(features.n_genres),
        "max_history": int(model._max_history),
        "year_mean": float(features.year_mean),
        "year_std": float(features.year_std),
        "relevance_threshold": float(features.relevance_threshold),
        "hyperparams": model._hyperparams,
        "history_only": bool(torch_model.history_only),
        "score_without_user_id": bool(model.score_without_user_id),
    }
    (directory / "two_tower_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_two_tower(directory: Path) -> TwoTowerRecommender:
    meta = json.loads((directory / "two_tower_meta.json").read_text(encoding="utf-8"))
    with np.load(directory / "two_tower_features.npz") as packed:
        user_ids = packed["user_ids"].astype(np.int64)
        item_ids = packed["item_ids"].astype(np.int64)
        genres = packed["genres"].astype(np.float32)
        years = packed["years"].astype(np.float32)
        item_q = packed["item_q"].astype(np.float64)
        user_history = _unpack_ragged(
            packed["hist_keys"], packed["hist_offsets"], packed["hist_values"]
        )
        seen_raw = _unpack_ragged(
            packed["seen_keys"], packed["seen_offsets"], packed["seen_values"]
        )
    seen_item_ids = {uid: set(int(i) for i in items.tolist()) for uid, items in seen_raw.items()}
    features = TwoTowerFeatures(
        user_ids=user_ids,
        item_ids=item_ids,
        user_index={int(uid): i for i, uid in enumerate(user_ids)},
        item_index={int(iid): i for i, iid in enumerate(item_ids)},
        user_history=user_history,
        seen_item_ids=seen_item_ids,
        pos_user_idx=np.asarray([], dtype=np.int64),
        pos_item_idx=np.asarray([], dtype=np.int64),
        genres=genres,
        years=years,
        year_mean=float(meta["year_mean"]),
        year_std=float(meta["year_std"]),
        item_q=item_q,
        relevance_threshold=float(meta["relevance_threshold"]),
    )
    network = TwoTowerModel(
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
        n_genres=int(meta["n_genres"]),
        embedding_dim=int(meta["embedding_dim"]),
        temperature=float(meta["temperature"]),
        history_only=bool(meta.get("history_only", False)),
    )
    state = torch.load(directory / "two_tower_model.pt", map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    recommender = TwoTowerRecommender.from_trained(
        network,
        features,
        max_history=int(meta["max_history"]),
        device="cpu",
        hyperparams=dict(meta.get("hyperparams") or {}),
    )
    recommender.score_without_user_id = bool(meta.get("score_without_user_id", False)) or bool(
        network.history_only
    )
    return recommender


def _save_context(ctx: FeatureContext, directory: Path) -> None:
    user_ids = np.asarray(sorted(ctx.user_n_ratings), dtype=np.int64)
    if len(user_ids):
        affinity = np.vstack([ctx.user_affinity[int(uid)] for uid in user_ids])
    else:
        affinity = np.zeros((0, len(GENRES)))
    demo_ids = np.asarray(sorted(ctx.demo_codes), dtype=np.int64)
    demo_codes = (
        np.vstack([ctx.demo_codes[int(uid)] for uid in demo_ids])
        if len(demo_ids)
        else np.zeros((0, 4), dtype=np.float64)
    )
    group_ids = np.asarray(sorted(ctx.user_group_codes), dtype=np.int64)
    group_codes = {key: [] for key in _GROUP_KEYS}
    for uid in group_ids:
        codes = ctx.user_group_codes[int(uid)]
        for key in _GROUP_KEYS:
            group_codes[key].append(int(codes[key]))
    arrays: dict[str, np.ndarray] = {
        "item_ids": np.asarray(ctx.item_ids, dtype=np.int64),
        "popularity": np.asarray(ctx.popularity, dtype=np.float64),
        "recency": np.asarray(ctx.recency, dtype=np.float64),
        "year": np.asarray(ctx.year, dtype=np.float64),
        "genres": np.asarray(ctx.genres, dtype=np.float64),
        "user_ids": user_ids,
        "user_n": np.asarray([ctx.user_n_ratings[int(uid)] for uid in user_ids], dtype=np.float64),
        "user_mean": np.asarray(
            [ctx.user_mean_rating[int(uid)] for uid in user_ids], dtype=np.float64
        ),
        "user_std": np.asarray(
            [ctx.user_std_rating[int(uid)] for uid in user_ids], dtype=np.float64
        ),
        "user_affinity": np.asarray(affinity, dtype=np.float64),
        "demo_user_ids": demo_ids,
        "demo_codes": np.asarray(demo_codes, dtype=np.float64),
        "group_user_ids": group_ids,
        "group_age_user": np.asarray(group_codes["age"], dtype=np.int64),
        "group_gender_user": np.asarray(group_codes["gender"], dtype=np.int64),
        "group_occupation_user": np.asarray(group_codes["occupation"], dtype=np.int64),
    }
    for key in _GROUP_KEYS:
        tables = ctx.group_item_stats.get(key, {})
        codes = np.asarray(sorted(tables), dtype=np.int64)
        if len(codes) == 0:
            pos = np.zeros((0, len(ctx.item_ids)), dtype=np.float64)
            share = np.zeros((0, len(ctx.item_ids)), dtype=np.float64)
        else:
            pos = np.vstack([tables[int(code)][0] for code in codes])
            share = np.vstack([tables[int(code)][1] for code in codes])
        arrays[f"group_{key}_codes"] = codes
        arrays[f"group_{key}_pos"] = np.asarray(pos, dtype=np.float64)
        arrays[f"group_{key}_share"] = np.asarray(share, dtype=np.float64)
    np.savez_compressed(directory / "feature_context.npz", **arrays)
    meta = {
        "names": list(ctx.names),
        "demo_mode": ctx.demo_mode,
        "prior_strength": float(ctx.prior_strength),
        "relevance_threshold": float(ctx.relevance_threshold),
    }
    expected = feature_names(ctx.demo_mode)
    allowed = [expected]
    if ctx.demo_mode in {"off", "both"}:
        allowed.append(cold_start_feature_names(ctx.demo_mode))
    if list(ctx.names) not in allowed:
        raise ValueError("feature context names drifted from feature_names(mode)")
    (directory / "feature_context_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_context(directory: Path) -> FeatureContext:
    meta = json.loads((directory / "feature_context_meta.json").read_text(encoding="utf-8"))
    with np.load(directory / "feature_context.npz") as packed:
        item_ids = packed["item_ids"].astype(np.int64)
        user_ids = packed["user_ids"].astype(np.int64)
        user_n = {
            int(uid): float(value)
            for uid, value in zip(user_ids.tolist(), packed["user_n"].tolist(), strict=True)
        }
        user_mean = {
            int(uid): float(value)
            for uid, value in zip(user_ids.tolist(), packed["user_mean"].tolist(), strict=True)
        }
        user_std = {
            int(uid): float(value)
            for uid, value in zip(user_ids.tolist(), packed["user_std"].tolist(), strict=True)
        }
        affinity_matrix = packed["user_affinity"]
        user_aff = {
            int(uid): np.asarray(affinity_matrix[i], dtype=np.float64)
            for i, uid in enumerate(user_ids.tolist())
        }
        demo_ids = packed["demo_user_ids"].astype(np.int64)
        demo_matrix = packed["demo_codes"]
        demo_codes = {
            int(uid): np.asarray(demo_matrix[i], dtype=np.float64)
            for i, uid in enumerate(demo_ids.tolist())
        }
        group_ids = packed["group_user_ids"].astype(np.int64)
        user_groups: dict[int, dict[str, int]] = {}
        age_u = packed["group_age_user"]
        gender_u = packed["group_gender_user"]
        occ_u = packed["group_occupation_user"]
        for i, uid in enumerate(group_ids.tolist()):
            user_groups[int(uid)] = {
                "age": int(age_u[i]),
                "gender": int(gender_u[i]),
                "occupation": int(occ_u[i]),
            }
        group_stats: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
        for key in _GROUP_KEYS:
            codes = packed[f"group_{key}_codes"].astype(np.int64)
            pos = packed[f"group_{key}_pos"]
            share = packed[f"group_{key}_share"]
            group_stats[key] = {
                int(code): (
                    np.asarray(pos[i], dtype=np.float64),
                    np.asarray(share[i], dtype=np.float64),
                )
                for i, code in enumerate(codes.tolist())
            }
        genres = np.asarray(packed["genres"], dtype=np.float64)
        popularity = np.asarray(packed["popularity"], dtype=np.float64)
        recency = np.asarray(packed["recency"], dtype=np.float64)
        year = np.asarray(packed["year"], dtype=np.float64)
    return FeatureContext(
        item_ids=item_ids,
        item_index={int(item): i for i, item in enumerate(item_ids.tolist())},
        popularity=popularity,
        recency=recency,
        year=year,
        genres=genres,
        user_n_ratings=user_n,
        user_mean_rating=user_mean,
        user_std_rating=user_std,
        user_affinity=user_aff,
        names=list(meta["names"]),
        demo_mode=str(meta["demo_mode"]),
        demo_codes=demo_codes,
        user_group_codes=user_groups,
        group_item_stats=group_stats,
        prior_strength=float(meta["prior_strength"]),
        relevance_threshold=float(meta["relevance_threshold"]),
    )


def manifest_template(
    *,
    dataset: str,
    dataset_sha256: str,
    config: Mapping[str, Any],
    git_sha: str,
    created_at: str,
    candidate_set: str,
    candidate_k: int,
    demographics: str,
    ranker_seed: int,
    best_iteration: int,
    two_tower_epochs: int,
    item_item_hyperparams: Mapping[str, Any],
    two_tower_hyperparams: Mapping[str, Any],
    build_runtime_sec: float | None = None,
) -> dict[str, Any]:
    """Small manifest. Large arrays stay in the sibling files."""
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_sha256": dataset_sha256,
        "config": dict(config),
        "git_sha": git_sha,
        "created_at": created_at,
        "candidate_set": candidate_set,
        "candidate_k": int(candidate_k),
        "demographics": demographics,
        "ranker_seed": int(ranker_seed),
        "best_iteration": int(best_iteration),
        "two_tower_epochs": int(two_tower_epochs),
        "item_item_hyperparams": dict(item_item_hyperparams),
        "two_tower_hyperparams": dict(two_tower_hyperparams),
        "cold_start": "existing_users_only",
        "history": "full_train (fit-train union validation); per-user test holdout excluded",
        "protocol": (
            "ADR-0007 refit. Retrievers used at scoring are fit on full train. "
            "The ranker is trained on validation-window labels with fit-train "
            "features, early-stopped on a held-out slice of validation users, "
            "then refit on all validation users for best_iteration rounds. "
            "Group-affinity features, when enabled, are computed from fit-train "
            "only. Test labels are not used. Hyperparameters are the committed "
            "validation-chosen values, not a new search."
        ),
    }
    if build_runtime_sec is not None:
        payload["build_runtime_sec"] = round(float(build_runtime_sec), 3)
    return payload
