"""Snapshot loading for the public demo. The page does not train.

``MOVIELENS_ARTIFACTS`` and ``MOVIELENS_COLD_ARTIFACTS`` point at full snapshots
(the test path). Otherwise the page reads ``space-weights.tar.gz`` beside
``app.py``, or downloads the pinned release archive, checks its SHA-256,
extracts it into ``weights/``, and rebuilds the snapshots after ml-1m.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

from catalog import clean_year, movie_records, popularity_order
from posters import (
    POSTER_CACHE_TTL_SECONDS,
    lookup_poster,
    mark_transient,
    transient_blocked,
)

_SPACE_DIR = Path(__file__).resolve().parent
N_RECS = 10


class PosterUnavailable(Exception):
    """A TMDB search failed after retries. Not a definitive miss."""


def weights_root() -> Path:
    return _SPACE_DIR / "weights"


def runtime_root() -> Path:
    override = os.environ.get("MOVIELENS_SPACE_RUNTIME")
    if override:
        return Path(override)
    return _SPACE_DIR / ".runtime"


def results_dir() -> Path:
    override = os.environ.get("MOVIELENS_RESULTS")
    if override:
        return Path(override)
    return _SPACE_DIR.parent / "results"


def _read_manifest(path: Path) -> dict | None:
    file = path / "manifest.json"
    if not file.is_file():
        return None
    import json

    payload = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def usable_snapshot(path: Path) -> bool:
    manifest = _read_manifest(path)
    return manifest is not None and not manifest.get("weights_only")


def tmdb_api_key() -> str:
    """Environment first, then a Streamlit secret. Never a committed value."""
    env = os.environ.get("TMDB_API_KEY", "").strip()
    if env:
        return env
    secret_paths = (
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if not any(path.is_file() for path in secret_paths):
        return ""
    try:
        value = st.secrets["TMDB_API_KEY"]
    except Exception:
        return ""
    return str(value).strip()


@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str):
    from movielens_recommender.serving.bundle import load_bundle as _load
    from movielens_recommender.serving.bundle import user_summaries
    from runtime import use_sparse_item_item_loader

    use_sparse_item_item_loader()
    bundle = _load(path_str)
    return bundle, user_summaries(bundle)


@st.cache_resource(show_spinner=False)
def load_cold(path_str: str):
    from movielens_recommender.serving.cold import load_cold_start_bundle
    from runtime import use_sparse_item_item_loader

    use_sparse_item_item_loader()
    return load_cold_start_bundle(path_str)


@st.cache_data(show_spinner=False)
def recommendations(path_str: str, user_id: int, n: int) -> dict[str, Any]:
    from movielens_recommender.serving.bundle import recommend_for_user

    bundle, _summaries = load_bundle(path_str)
    return recommend_for_user(bundle, int(user_id), n=n)


@st.cache_data(show_spinner=False)
def catalog(path_str: str) -> tuple[list[dict[str, Any]], dict[int, int]]:
    bundle = load_cold(path_str)
    records = movie_records(bundle.movies)
    popular = (
        None
        if bundle.popular_item_ids is None
        else [int(item) for item in bundle.popular_item_ids]
    )
    item_ids = [int(item) for item in bundle.context.item_ids]
    popularity = [float(value) for value in bundle.context.popularity]
    return records, popularity_order(popular, item_ids, popularity)


@st.cache_data(show_spinner=False, ttl=POSTER_CACHE_TTL_SECONDS)
def _poster_hit(title: str, year: int, api_key: str) -> str:
    """Cache a hit or a definitive miss. A transient failure is raised and not stored."""
    url, definitive = lookup_poster(title, None if year < 0 else year, api_key)
    if url:
        return url
    if definitive:
        return ""
    raise PosterUnavailable(title)


def poster_for(title: str, year: object, api_key: str) -> str | None:
    """Poster URL from the one-day cache, or None for the placeholder.

    Confirmed misses are cached as an empty string. Rate limits and network
    errors are remembered for a short window and then tried again.
    """
    if not api_key:
        return None
    cleaned = clean_year(year)
    year_key = -1 if cleaned is None else int(cleaned)
    if transient_blocked(title, year_key):
        return None
    try:
        value = _poster_hit(title, year_key, api_key)
    except PosterUnavailable:
        mark_transient(title, year_key)
        return None
    return value or None


@st.cache_data(show_spinner=False)
def quality_view(path_str: str) -> dict[str, Any]:
    from quality import load_quality

    return load_quality(path_str)


@st.cache_resource(show_spinner="Downloading MovieLens ml-1m and loading the model…")
def _materialize(weights_str: str, runtime_str: str) -> tuple[str, str]:
    from runtime import materialize_space

    production, cold = materialize_space(weights_str, runtime_str)
    return str(production), str(cold)


def warm(prod: Path, cold: Path, *, has_prod: bool, has_cold: bool) -> None:
    """Load both modes once so a later section reads the cache."""
    if has_prod:
        _bundle, summaries = load_bundle(str(prod.resolve()))
        if summaries:
            user_id = sorted(summaries)[0]
            recommendations(str(prod.resolve()), int(user_id), N_RECS)
    if has_cold:
        catalog(str(cold.resolve()))


def _downloaded_weights() -> Path | None:
    from runtime import install_space_weights

    return install_space_weights(_SPACE_DIR)


def resolve_snapshots() -> tuple[Path, Path] | None:
    prod_env = os.environ.get("MOVIELENS_ARTIFACTS")
    cold_env = os.environ.get("MOVIELENS_COLD_ARTIFACTS")
    if prod_env and usable_snapshot(Path(prod_env)):
        prod = Path(prod_env)
        cold = Path(cold_env) if cold_env else prod / "cold_start"
        return prod, cold
    try:
        weights = _downloaded_weights()
    except Exception:
        return None
    if weights is None:
        return None
    try:
        prod_str, cold_str = _materialize(str(weights), str(runtime_root()))
    except Exception:
        return None
    return Path(prod_str), Path(cold_str)
