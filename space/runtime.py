"""Rebuild serving snapshots from parameter files that contain no rating rows.

Streamlit Community Cloud runs ``app.py`` from this directory and installs
``movielens-recommender`` from the GitHub commit pinned in ``requirements.txt``.
Loading parameters is not training: item–item and EASE similarity matrices are
copied, and the user–item matrix, histories, and feature tables are rebuilt
from a GroupLens download (or from a caller-supplied frame in tests).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy import sparse

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.baselines.popular import MostPopularRecommender
from movielens_recommender.cold_start import (
    HOLDOUT_FRACTION,
    eligible_user_ids,
    exclude_held_out_users,
    select_held_out_user_ids,
)
from movielens_recommender.ranker.explain import load_ranker
from movielens_recommender.ranker.features import (
    build_feature_context,
    cold_start_ranker_feature_names,
)
from movielens_recommender.serving.bundle import ServingBundle, save_bundle
from movielens_recommender.serving.cold import save_cold_start_bundle
from movielens_recommender.split import SplitConfig, time_based_split
from movielens_recommender.two_tower.features import build_features
from movielens_recommender.two_tower.model import TwoTowerModel
from movielens_recommender.two_tower.recommender import TwoTowerRecommender

SPACE_WEIGHTS_VERSION = 1

# Gzip of production/ and cold_start/. This GitHub repo gitignores the file.
WEIGHTS_ARCHIVE_NAME = "space-weights.tar.gz"
WEIGHTS_URL = (
    "https://github.com/NikosMav/movielens-recommender/releases/download/"
    "space-weights-v1/space-weights.tar.gz"
)
WEIGHTS_SHA256 = "f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd"

# Files a full serving snapshot writes that must not be uploaded with the Space.
OMITTED_FILES: tuple[str, ...] = (
    "movies.csv",
    "histories.npz",
    "item_item_user_item.npz",
    "item_item_index.npz",
    "two_tower_features.npz",
    "feature_context.npz",
    "feature_context_meta.json",
    "popular_items.npy",
)

FORBIDDEN_FILENAMES = frozenset(
    OMITTED_FILES
    + (
        "ratings.dat",
        "ratings.csv",
        "movies.dat",
        "users.dat",
    )
)

# Do not scan JSON text for the word "rating": feature names contain it.
FORBIDDEN_NPZ_KEYS = frozenset(
    {
        "rating",
        "ratings",
        "user_id",
        "user_ids",
        "title",
        "titles",
    }
)
FORBIDDEN_NPZ_PREFIXES = ("hist_", "seen_")

_COPIED_FILES = (
    "item_item_similarity.npz",
    "item_item_meta.json",
    "two_tower_model.pt",
    "two_tower_meta.json",
    "ranker.txt",
    "ranker_meta.json",
)
_COLD_FILES = ("ease.npz", "ease_meta.json")
_STRIP_ON_REBUILD = (
    "weights_only",
    "space_weights_version",
    "omitted",
    "cold_train_protocol",
    "train_split",
)

_DEMO_TEXT = re.compile(
    r"demo_|group_|\b(?:gender|occupation|zip|region|male|female)\b|\bage\b",
    re.IGNORECASE,
)


def visible_reasons(reasons: list[str]) -> list[str]:
    """Drop reason sentences that name a demographic attribute."""
    return [text for text in reasons if not _DEMO_TEXT.search(text)]


def visible_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ranker-detail rows whose feature is demographic."""
    kept = []
    for row in details:
        feature = str(row.get("feature", ""))
        if feature.startswith(("demo_", "group_")) or _DEMO_TEXT.search(feature):
            continue
        kept.append(row)
    return kept


def assert_weights_only(root: Path | str) -> None:
    """Raise if ``root`` contains a forbidden file or a rating array."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_FILENAMES:
            raise ValueError(f"weight tree contains forbidden file {path.name}")
        if path.suffix == ".npz":
            _assert_npz_keys(path)


def weight_sizes(root: Path | str) -> list[tuple[str, int]]:
    """``(relative posix path, size in bytes)`` for every file under ``root``."""
    root = Path(root)
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rows.append((path.relative_to(root).as_posix(), path.stat().st_size))
    return rows


def export_space_weights(
    production_dir: Path | str,
    cold_dir: Path | str,
    dest_root: Path | str,
    *,
    cold_train_protocol: str,
) -> Path:
    """Copy parameter files out of two full snapshots.

    ``cold_train_protocol`` is ``ml1m_complement_full_train`` for the real
    ml-1m snapshot (the complement split is copied from the production
    manifest) or ``caller`` when the caller will pass the train frame itself.
    """
    if cold_train_protocol not in {"ml1m_complement_full_train", "caller"}:
        raise ValueError(f"unknown cold_train_protocol {cold_train_protocol!r}")
    production_dir = Path(production_dir)
    cold_dir = Path(cold_dir)
    dest_root = Path(dest_root)
    prod_manifest = _read_json(production_dir / "manifest.json")
    _export_one(production_dir, dest_root / "production", extra={})
    extra: dict[str, Any] = {"cold_train_protocol": cold_train_protocol}
    if cold_train_protocol == "ml1m_complement_full_train":
        extra["train_split"] = _train_split_record(prod_manifest)
    _export_one(cold_dir, dest_root / "cold_start", extra=extra)
    assert_weights_only(dest_root)
    return dest_root


def load_round2_choices(path: Path | str) -> dict[str, Any]:
    """Read the committed cold-start JSON. This function does not write it."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    round2 = payload["round2"]
    ranker = payload["ranker"]
    representation = payload["representation"]
    if int(round2["best_iteration"]) != int(ranker["best_iteration"]):
        raise ValueError("round2 best_iteration does not match ranker.best_iteration")
    if list(round2["feature_names"]) != list(ranker["feature_names"]):
        raise ValueError("round2 feature names do not match ranker.feature_names")
    expected = cold_start_ranker_feature_names()
    if list(round2["feature_names"]) != expected:
        raise ValueError("round2 feature names do not match cold_start_ranker_feature_names()")
    if str(representation["winner"]) != str(round2["representation_winner"]):
        raise ValueError("representation.winner does not match round2.representation_winner")
    if bool(representation["history_only"]) != bool(round2["history_only"]):
        raise ValueError("representation.history_only does not match round2.history_only")
    serving = {str(key): str(value) for key, value in round2["serving_by_n"].items()}
    if any(method != "cold_start_ranker" for method in serving.values()):
        raise ValueError(f"round2 serving_by_n is not all cold_start_ranker: {serving}")
    return {
        "best_iteration": int(round2["best_iteration"]),
        "candidate_k": int(round2["candidate_k"]),
        "feature_names": list(round2["feature_names"]),
        "serving_by_n": serving,
        "history_only": bool(round2["history_only"]),
        "representation_best_epoch": int(round2["representation_best_epoch"]),
        "representation_winner": str(round2["representation_winner"]),
        "user_id_dropout": float(representation["user_id_dropout"]),
        "fixed_hyperparams": dict(representation["fixed_hyperparams"]),
    }


def materialize_space(
    weights_root: Path | str,
    runtime_dir: Path | str,
    *,
    ratings: pd.DataFrame | None = None,
    movies: pd.DataFrame | None = None,
    users: pd.DataFrame | None = None,
    cold_ratings: pd.DataFrame | None = None,
    cold_movies: pd.DataFrame | None = None,
    data_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Write full snapshots a serving loader can open.

    With no frames, download ml-1m into ``data_dir`` (default
    ``<runtime_dir>/data``) and rebuild both train matrices from it. Tests pass
    synthetic frames and never touch the network.
    """
    weights_root = Path(weights_root)
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    if ratings is None:
        ratings, movies, users, data_dir = _download_ml1m(
            Path(data_dir) if data_dir is not None else runtime_dir / "data"
        )
        cold_ratings = ratings
        cold_movies = movies
    if movies is None:
        raise ValueError("movies are required when ratings are passed in")
    if cold_ratings is None:
        cold_ratings = ratings
    if cold_movies is None:
        cold_movies = movies
    production = _materialize_production(
        weights_root / "production",
        runtime_dir / "production",
        ratings=ratings,
        movies=movies,
        users=users,
    )
    cold_weights = weights_root / "cold_start"
    cold_runtime = runtime_dir / "cold_start"
    if (cold_weights / "manifest.json").is_file():
        cold_runtime = _materialize_cold(
            cold_weights,
            cold_runtime,
            ratings=cold_ratings,
            movies=cold_movies,
        )
    return production, cold_runtime


def _download_ml1m(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    from movielens_recommender.data import download_dataset, load_ratings, load_users
    from movielens_recommender.movies import load_movies

    download_dataset("ml-1m", data_dir)
    ratings, _stats = load_ratings("ml-1m", data_dir, clean=True)
    movies = load_movies("ml-1m", data_dir)
    users = load_users("ml-1m", data_dir)
    return ratings, movies, users, data_dir


def _materialize_production(
    weights_dir: Path,
    runtime_dir: Path,
    *,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame | None,
) -> Path:
    manifest = _read_json(weights_dir / "manifest.json")
    _require_weights_manifest(manifest)
    mode = str(manifest.get("demographics", "off"))
    if mode != "off" and (users is None or users.empty):
        raise ValueError(f"demographics={mode!r} requires a users frame")
    split = _split_ratings(ratings, manifest["config"])
    train = split.full_train
    threshold = float(manifest["config"]["eval"]["relevance_threshold"])
    item_item = _attach_item_item(weights_dir, train)
    tower = _attach_tower(weights_dir, train, movies, dataset=str(manifest.get("dataset", "ml-1m")))
    context = build_feature_context(
        train,
        movies,
        users=users,
        mode=mode,
        group_frame=split.train,
        relevance_threshold=threshold,
    )
    booster, _meta = load_ranker(weights_dir)
    model_names = list(booster.feature_name())
    if model_names != list(context.names):
        raise ValueError(
            "rebuilt production feature names do not match the ranker "
            f"({model_names[:4]}... vs {list(context.names)[:4]}...)"
        )
    bundle = ServingBundle(
        manifest=_serving_manifest(manifest),
        movies=movies.loc[:, ["item_id", "title", "genres", "year"]].copy(),
        histories=train.loc[:, ["user_id", "item_id", "rating", "timestamp"]].copy(),
        item_item=item_item,
        two_tower=tower,
        booster=booster,
        context=context,
    )
    return save_bundle(bundle, runtime_dir)


def _materialize_cold(
    weights_dir: Path,
    runtime_dir: Path,
    *,
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
) -> Path:
    manifest = _read_json(weights_dir / "manifest.json")
    _require_weights_manifest(manifest)
    train = _cold_train_frame(ratings, manifest)
    threshold = float(manifest.get("relevance_threshold", 4.0))
    item_item = _attach_item_item(weights_dir, train)
    tower = _attach_tower(weights_dir, train, movies, dataset=str(manifest.get("dataset", "ml-1m")))
    ease = _attach_ease(weights_dir)
    context = build_feature_context(
        train,
        movies,
        users=None,
        mode="off",
        group_frame=None,
        relevance_threshold=threshold,
    )
    booster, _meta = load_ranker(weights_dir)
    model_names = list(booster.feature_name())
    recorded = manifest.get("feature_names")
    if recorded is not None and list(recorded) != model_names:
        raise ValueError("cold ranker feature names do not match the weight manifest")
    context.names = model_names
    popular_ids = None
    if manifest.get("serving_by_n") is not None:
        popular = MostPopularRecommender().fit(train)
        popular_ids = np.asarray(popular._ranked_items, dtype=np.int64)
    return save_cold_start_bundle(
        runtime_dir,
        movies=movies.loc[:, ["item_id", "title", "genres", "year"]].copy(),
        item_item=item_item,
        ease=ease,
        two_tower=tower,
        booster=booster,
        context=context,
        popular_item_ids=popular_ids,
        manifest=_serving_manifest(manifest),
    )


def _cold_train_frame(ratings: pd.DataFrame, manifest: dict[str, Any]) -> pd.DataFrame:
    protocol = str(manifest.get("cold_train_protocol", ""))
    if protocol == "caller":
        return ratings
    if protocol != "ml1m_complement_full_train":
        raise ValueError(f"unknown cold_train_protocol {protocol!r}")
    spec = manifest.get("train_split")
    if not isinstance(spec, dict):
        raise ValueError("ml1m cold weights are missing train_split")
    eligible = eligible_user_ids(ratings, int(spec["min_ratings"]))
    held = select_held_out_user_ids(
        eligible,
        fraction=HOLDOUT_FRACTION,
        seed=int(spec["seed"]),
    )
    remain = exclude_held_out_users(ratings, held)
    split = time_based_split(
        remain,
        SplitConfig(
            min_ratings=int(spec["min_ratings"]),
            test_fraction=float(spec["test_fraction"]),
            val_fraction=float(spec["val_fraction"]),
            relevance_threshold=float(spec["relevance_threshold"]),
        ),
    )
    return split.full_train


def _split_ratings(ratings: pd.DataFrame, config: dict[str, Any]):
    return time_based_split(
        ratings,
        SplitConfig(
            min_ratings=int(config["split"]["min_ratings"]),
            test_fraction=float(config["split"]["test_fraction"]),
            val_fraction=float(config["split"]["val_fraction"]),
            relevance_threshold=float(config["eval"]["relevance_threshold"]),
        ),
    )


def _attach_item_item(directory: Path, train: pd.DataFrame) -> ItemItemCosineRecommender:
    meta = _read_json(directory / "item_item_meta.json")
    model = ItemItemCosineRecommender(
        min_common=int(meta["min_common"]),
        k_neighbors=int(meta["k_neighbors"]),
        shrinkage=float(meta["shrinkage"]),
    )
    item_ids = np.load(directory / "item_ids.npy").astype(np.int64)
    expected = np.sort(train["item_id"].unique()).astype(np.int64)
    if not np.array_equal(item_ids, expected):
        raise ValueError(
            "saved item ids do not match the rebuilt train catalog "
            f"({len(item_ids)} saved, {len(expected)} rebuilt)"
        )
    similarity = sparse.load_npz(directory / "item_item_similarity.npz").tocsr()
    if similarity.dtype != np.float64:
        similarity = similarity.astype(np.float64)
    users = np.sort(train["user_id"].unique()).astype(np.int64)
    user_index = {int(user): index for index, user in enumerate(users.tolist())}
    item_index = {int(item): index for index, item in enumerate(item_ids.tolist())}
    rows = train["user_id"].map(user_index).to_numpy()
    cols = train["item_id"].map(item_index).to_numpy()
    data = train["rating"].to_numpy(dtype=np.float64)
    model._similarity = similarity
    model._user_item = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(users), len(item_ids)),
        dtype=np.float64,
    )
    model._item_ids = item_ids
    model._item_index = item_index
    model._user_index = user_index
    return model


def _attach_ease(directory: Path) -> EASERecommender:
    meta = _read_json(directory / "ease_meta.json")
    model = EASERecommender(l2=float(meta["l2"]))
    with np.load(directory / "ease.npz") as packed:
        item_ids = packed["item_ids"].astype(np.int64)
        similarity = np.asarray(packed["similarity"], dtype=np.float64)
    model._item_ids = item_ids
    model._item_index = {int(item): index for index, item in enumerate(item_ids.tolist())}
    model._similarity = similarity
    return model


def _attach_tower(
    directory: Path,
    train: pd.DataFrame,
    movies: pd.DataFrame,
    *,
    dataset: str,
) -> TwoTowerRecommender:
    meta = _read_json(directory / "two_tower_meta.json")
    features = build_features(
        train,
        dataset=dataset,
        movies=movies,
        relevance_threshold=float(meta["relevance_threshold"]),
        max_history=int(meta["max_history"]),
    )
    if features.n_users != int(meta["n_users"]) or features.n_items != int(meta["n_items"]):
        raise ValueError(
            "rebuilt two-tower catalog does not match the checkpoint "
            f"(users {features.n_users} vs {meta['n_users']}, "
            f"items {features.n_items} vs {meta['n_items']})"
        )
    if features.n_genres != int(meta["n_genres"]):
        raise ValueError(
            f"rebuilt genre count {features.n_genres} does not match {meta['n_genres']}"
        )
    if not np.allclose(features.year_mean, float(meta["year_mean"])):
        raise ValueError(
            f"year_mean {features.year_mean} does not match checkpoint {meta['year_mean']}"
        )
    if not np.allclose(features.year_std, float(meta["year_std"])):
        raise ValueError(
            f"year_std {features.year_std} does not match checkpoint {meta['year_std']}"
        )
    network = TwoTowerModel(
        n_users=int(meta["n_users"]),
        n_items=int(meta["n_items"]),
        n_genres=int(meta["n_genres"]),
        embedding_dim=int(meta["embedding_dim"]),
        temperature=float(meta["temperature"]),
        history_only=bool(meta.get("history_only", False)),
    )
    state = _torch().load(directory / "two_tower_model.pt", map_location="cpu", weights_only=True)
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


def _torch():
    import torch

    return torch


def _export_one(source: Path, dest: Path, *, extra: dict[str, Any]) -> None:
    manifest = _read_json(source / "manifest.json")
    dest.mkdir(parents=True, exist_ok=True)
    for name in _COPIED_FILES:
        _copy(source / name, dest / name)
    if str(manifest.get("kind")) == "cold_start":
        for name in _COLD_FILES:
            _copy(source / name, dest / name)
    with np.load(source / "item_item_index.npz") as packed:
        item_ids = np.asarray(packed["item_ids"], dtype=np.int64)
    np.save(dest / "item_ids.npy", item_ids)
    payload = dict(manifest)
    payload["weights_only"] = True
    payload["space_weights_version"] = SPACE_WEIGHTS_VERSION
    payload["omitted"] = list(OMITTED_FILES)
    payload.update(extra)
    (dest / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.write_bytes(source.read_bytes())


def _train_split_record(production_manifest: dict[str, Any]) -> dict[str, Any]:
    config = production_manifest["config"]
    return {
        "min_ratings": int(config["split"]["min_ratings"]),
        "test_fraction": float(config["split"]["test_fraction"]),
        "val_fraction": float(config["split"]["val_fraction"]),
        "relevance_threshold": float(config["eval"]["relevance_threshold"]),
        "seed": int(config["seed"]),
    }


def _serving_manifest(weights_manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in weights_manifest.items() if key not in _STRIP_ON_REBUILD}


def _require_weights_manifest(manifest: dict[str, Any]) -> None:
    if not manifest.get("weights_only"):
        raise ValueError("manifest is not a weights-only Space snapshot")
    version = int(manifest.get("space_weights_version", -1))
    if version != SPACE_WEIGHTS_VERSION:
        raise ValueError(
            f"space_weights_version {version} does not match {SPACE_WEIGHTS_VERSION}"
        )


def _assert_npz_keys(path: Path) -> None:
    with np.load(path) as packed:
        for key in packed.files:
            if key in FORBIDDEN_NPZ_KEYS or key.startswith(FORBIDDEN_NPZ_PREFIXES):
                raise ValueError(f"{path.name} contains forbidden array {key!r}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def install_space_weights(root: Path | str, *, sha256: str | None = None) -> Path | None:
    """Read ``space-weights.tar.gz`` beside the app, or download it, then extract.

    The local file wins. Otherwise the bytes come from ``WEIGHTS_URL`` and are
    stored under ``.runtime/cache/``. Returns ``weights/``, or ``None`` when
    the archive is missing or the checksum fails.
    """
    root = Path(root)
    dest = root / "weights"
    expected = WEIGHTS_SHA256 if sha256 is None else sha256
    cache = root / ".runtime" / "cache" / WEIGHTS_ARCHIVE_NAME
    try:
        payload = _weight_archive_bytes(root / WEIGHTS_ARCHIVE_NAME, cache, expected)
        if payload is None or hashlib.sha256(payload).hexdigest() != expected:
            return None
        if dest.exists() and any(dest.iterdir()):
            if (dest / "production" / "manifest.json").is_file():
                assert_weights_only(dest)
                return dest
            raise ValueError(f"weight destination is not empty: {dest}")
        dest.mkdir(parents=True, exist_ok=True)
        _extract_weight_archive(payload, dest)
        assert_weights_only(dest)
    except Exception:
        return None
    if not (dest / "production" / "manifest.json").is_file():
        return None
    return dest


def _weight_archive_bytes(archive: Path, cache: Path, expected: str) -> bytes | None:
    if archive.is_file():
        return archive.read_bytes()
    return _download_weights(cache, expected)


def _download_weights(cache: Path, expected: str) -> bytes | None:
    """Download ``WEIGHTS_URL`` into ``cache`` after the SHA-256 matches.

    Tests set ``MOVIELENS_ALLOW_DOWNLOAD=0`` so this returns before any network
    call. A checksum mismatch is not written to the cache.
    """
    if os.environ.get("MOVIELENS_ALLOW_DOWNLOAD") == "0":
        return None
    if cache.is_file():
        cached = cache.read_bytes()
        if hashlib.sha256(cached).hexdigest() == expected:
            return cached
    try:
        with urlopen(WEIGHTS_URL, timeout=120) as response:  # noqa: S310 — pinned URL
            payload = response.read()
    except Exception:
        return None
    if hashlib.sha256(payload).hexdigest() != expected:
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(cache.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(cache)
    return payload


def use_sparse_item_item_loader() -> None:
    """Keep item–item similarity sparse when a snapshot is loaded.

    The library loader densifies a full item-by-item matrix to float64. The
    stored matrix is already sparse, and the same matmul accepts that form.
    Community Cloud's RAM budget is 1 GB.
    """
    import movielens_recommender.serving.bundle as bundle_mod

    if getattr(bundle_mod, "_space_sparse_item_item", False):
        return

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
        similarity = sparse.load_npz(directory / "item_item_similarity.npz").tocsr()
        if similarity.dtype != np.float64:
            similarity = similarity.astype(np.float64)
        model._similarity = similarity
        model._user_item = sparse.load_npz(directory / "item_item_user_item.npz").tocsr()
        model._item_ids = item_ids
        model._item_index = {int(item): index for index, item in enumerate(item_ids)}
        model._user_index = {int(user): index for index, user in enumerate(user_ids)}
        return model

    bundle_mod._load_item_item = _load_item_item
    bundle_mod._space_sparse_item_item = True


def _extract_weight_archive(payload: bytes, dest: Path) -> None:
    dest_root = dest.resolve()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError("weight archive contains a link")
            name = member.name
            if name.startswith("/") or ".." in Path(name).parts:
                raise ValueError(f"weight archive path is not safe: {name}")
            target = (dest_root / name).resolve()
            if target != dest_root and dest_root not in target.parents:
                raise ValueError(f"weight archive path escapes the destination: {name}")
        try:
            archive.extractall(dest_root, filter="data")
        except TypeError:
            archive.extractall(dest_root)
