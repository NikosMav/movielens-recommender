"""Download, verify, clean, and load MovieLens datasets from GroupLens."""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# Official GroupLens URLs — see https://grouplens.org/datasets/movielens/
DATASET_URLS: dict[str, str] = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}

# SHA-256 of the official zip bytes (computed from GroupLens archives at pin time).
DATASET_SHA256: dict[str, str] = {
    "ml-latest-small": "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436",
    "ml-1m": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
}

DATASET_VERSION_LABELS: dict[str, str] = {
    "ml-latest-small": (
        "ml-latest-small@sha256:"
        "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
    ),
    "ml-1m": (
        "ml-1m@sha256:a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
    ),
}

LICENSE_URL = "https://grouplens.org/datasets/movielens/"
DEFAULT_DATA_DIR = Path("data")


@dataclass(frozen=True)
class CleanStats:
    """Counts from :func:`clean_ratings`."""

    n_raw: int
    n_dropped_null: int
    n_dropped_invalid_rating: int
    n_dropped_invalid_ids: int
    n_dropped_duplicates: int
    n_clean: int

    def to_dict(self) -> dict:
        return asdict(self)


def dataset_dir(name: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Return the on-disk directory for a named dataset."""
    return Path(data_dir) / name


def ratings_path(name: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Return the expected path to the ratings file after download/extract."""
    root = dataset_dir(name, data_dir)
    if name == "ml-latest-small":
        return root / "ml-latest-small" / "ratings.csv"
    if name == "ml-1m":
        return root / "ml-1m" / "ratings.dat"
    raise ValueError(f"Unknown dataset: {name!r}. Choose from {sorted(DATASET_URLS)}")


def is_downloaded(name: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> bool:
    """True if ratings file is present locally."""
    return ratings_path(name, data_dir).is_file()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download_dataset(
    name: str = "ml-latest-small",
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    force: bool = False,
) -> Path:
    """Download and extract a MovieLens dataset into ``data_dir``.

    Verifies the zip against the pinned SHA-256 in :data:`DATASET_SHA256`.
    Data is never redistributed by this project — download only from GroupLens.
    See the license/terms at: https://grouplens.org/datasets/movielens/

    Returns
    -------
    Path
        Path to the ratings file.
    """
    if name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {name!r}. Choose from {sorted(DATASET_URLS)}")

    out = ratings_path(name, data_dir)
    if out.is_file() and not force:
        return out

    root = dataset_dir(name, data_dir)
    root.mkdir(parents=True, exist_ok=True)

    url = DATASET_URLS[name]
    with urlopen(url) as resp:  # noqa: S310 — fixed HTTPS GroupLens URLs only
        payload = resp.read()

    digest = _sha256_bytes(payload)
    expected = DATASET_SHA256[name]
    if digest != expected:
        raise ValueError(
            f"SHA-256 mismatch for {name}: got {digest}, expected {expected}. "
            "Refusing to extract. If GroupLens updated the archive, bump the pin "
            "in DATASET_SHA256 deliberately."
        )

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(root)

    if not out.is_file():
        raise FileNotFoundError(f"Expected ratings file missing after extract: {out}")
    return out


def load_raw_ratings(
    name: str = "ml-latest-small",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load ratings before cleaning (columns: user_id, item_id, rating, timestamp)."""
    path = ratings_path(name, data_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Ratings not found at {path}. Run download first "
            f"(e.g. `movielens-recommender download --dataset {name}`)."
        )

    if name == "ml-latest-small":
        df = pd.read_csv(path)
        df = df.rename(
            columns={
                "userId": "user_id",
                "movieId": "item_id",
                "rating": "rating",
                "timestamp": "timestamp",
            }
        )
    elif name == "ml-1m":
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            header=None,
        )
    else:
        raise ValueError(f"Unknown dataset: {name!r}")

    return df[["user_id", "item_id", "rating", "timestamp"]].copy()


def clean_ratings(ratings: pd.DataFrame) -> tuple[pd.DataFrame, CleanStats]:
    """Apply explicit cleaning rules; return cleaned frame + stats.

    Rules
    -----
    1. Drop rows with any null in user_id, item_id, rating, timestamp.
    2. Keep ratings in ``[0.5, 5.0]`` (MovieLens half-star / five-star scale).
    3. Require ``user_id > 0``, ``item_id > 0``, ``timestamp > 0``.
    4. For duplicate ``(user_id, item_id)``, keep the latest timestamp; if still
       tied, keep the higher rating (deterministic).
    """
    n_raw = len(ratings)
    df = ratings.copy()

    before = len(df)
    df = df.dropna(subset=["user_id", "item_id", "rating", "timestamp"])
    n_dropped_null = before - len(df)

    before = len(df)
    df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]
    n_dropped_invalid_rating = before - len(df)

    before = len(df)
    df = df[(df["user_id"] > 0) & (df["item_id"] > 0) & (df["timestamp"] > 0)]
    n_dropped_invalid_ids = before - len(df)

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)

    before = len(df)
    df = df.sort_values(
        ["user_id", "item_id", "timestamp", "rating"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
    n_dropped_duplicates = before - len(df)

    df = df.reset_index(drop=True)
    stats = CleanStats(
        n_raw=n_raw,
        n_dropped_null=int(n_dropped_null),
        n_dropped_invalid_rating=int(n_dropped_invalid_rating),
        n_dropped_invalid_ids=int(n_dropped_invalid_ids),
        n_dropped_duplicates=int(n_dropped_duplicates),
        n_clean=len(df),
    )
    return df, stats


def load_ratings(
    name: str = "ml-latest-small",
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    clean: bool = True,
) -> pd.DataFrame | tuple[pd.DataFrame, CleanStats]:
    """Load (and optionally clean) ratings.

    When ``clean=True`` (default), returns ``(dataframe, CleanStats)``.
    When ``clean=False``, returns the raw dataframe only.
    """
    raw = load_raw_ratings(name, data_dir)
    if not clean:
        raw["user_id"] = raw["user_id"].astype(int)
        raw["item_id"] = raw["item_id"].astype(int)
        raw["rating"] = raw["rating"].astype(float)
        raw["timestamp"] = raw["timestamp"].astype(int)
        return raw
    return clean_ratings(raw)
