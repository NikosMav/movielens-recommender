"""Load MovieLens movie metadata (genres, release year)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from movielens_recommender.data import DEFAULT_DATA_DIR, dataset_dir

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")

# Canonical MovieLens genre vocabulary (ml-1m / ml-latest-small).
GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)",
)

GENRE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GENRES)}


def movies_path(name: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Return the expected path to the movies metadata file."""
    root = dataset_dir(name, data_dir)
    if name == "ml-latest-small":
        return root / "ml-latest-small" / "movies.csv"
    if name == "ml-1m":
        return root / "ml-1m" / "movies.dat"
    raise ValueError(f"Unknown dataset: {name!r}")


def parse_year_from_title(title: str) -> float | None:
    """Extract a trailing ``(YYYY)`` year from a MovieLens title, if present."""
    match = _YEAR_RE.search(str(title).strip())
    if not match:
        return None
    year = int(match.group(1))
    if 1870 <= year <= 2100:
        return float(year)
    return None


def _genre_multi_hot(genres_str: str) -> np.ndarray:
    vec = np.zeros(len(GENRES), dtype=np.float32)
    parts = [p.strip() for p in str(genres_str).split("|") if p.strip()]
    if not parts:
        vec[GENRE_TO_IDX["(no genres listed)"]] = 1.0
        return vec
    unknown = True
    for part in parts:
        # ml-1m uses "Children's"; map onto "Children".
        key = "Children" if part in {"Children's", "Children"} else part
        idx = GENRE_TO_IDX.get(key)
        if idx is not None:
            vec[idx] = 1.0
            unknown = False
    if unknown:
        vec[GENRE_TO_IDX["(no genres listed)"]] = 1.0
    return vec


def load_movies(
    name: str = "ml-latest-small",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load movies with ``item_id``, ``title``, ``genres``, ``year``.

    ``year`` may be NaN when the title has no parseable ``(YYYY)`` suffix.
    """
    path = movies_path(name, data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Movies metadata not found at {path}. Run download first."
        )

    if name == "ml-latest-small":
        df = pd.read_csv(path)
        df = df.rename(columns={"movieId": "item_id"})
    elif name == "ml-1m":
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            header=None,
            encoding="latin-1",
        )
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    df["item_id"] = df["item_id"].astype(int)
    df["title"] = df["title"].astype(str)
    df["genres"] = df["genres"].astype(str)
    df["year"] = df["title"].map(parse_year_from_title)
    return df[["item_id", "title", "genres", "year"]].copy()


def build_item_side_features(
    movies: pd.DataFrame,
    item_ids: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Build genre multi-hot and normalized year vectors aligned to ``item_ids``.

    Returns
    -------
    genres : (n_items, n_genres) float32
    years_norm : (n_items,) float32 — missing years filled with 0 after z-score
    year_mean, year_std : scalars used for the transform (std floored at 1.0)
    """
    by_id = movies.set_index("item_id")
    n = len(item_ids)
    genres = np.zeros((n, len(GENRES)), dtype=np.float32)
    years_raw = np.full(n, np.nan, dtype=np.float64)

    for i, item_id in enumerate(item_ids):
        iid = int(item_id)
        if iid not in by_id.index:
            genres[i, GENRE_TO_IDX["(no genres listed)"]] = 1.0
            continue
        row = by_id.loc[iid]
        # Duplicate item_ids are unexpected; take the first row if a frame.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        genres[i] = _genre_multi_hot(str(row["genres"]))
        year = row["year"]
        if pd.notna(year):
            years_raw[i] = float(year)

    observed = years_raw[np.isfinite(years_raw)]
    if len(observed) == 0:
        year_mean, year_std = 1995.0, 1.0
    else:
        year_mean = float(observed.mean())
        year_std = float(max(observed.std(ddof=0), 1.0))

    years_norm = np.zeros(n, dtype=np.float32)
    for i, y in enumerate(years_raw):
        if np.isfinite(y):
            years_norm[i] = np.float32((y - year_mean) / year_std)
    return genres, years_norm, year_mean, year_std
