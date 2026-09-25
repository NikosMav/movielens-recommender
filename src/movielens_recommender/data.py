"""Download and load MovieLens datasets from GroupLens."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# Official GroupLens URLs — see https://grouplens.org/datasets/movielens/
DATASET_URLS: dict[str, str] = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
}

LICENSE_URL = "https://grouplens.org/datasets/movielens/"

DEFAULT_DATA_DIR = Path("data")


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


def download_dataset(
    name: str = "ml-latest-small",
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    force: bool = False,
) -> Path:
    """Download and extract a MovieLens dataset into ``data_dir``.

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

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(root)

    if not out.is_file():
        raise FileNotFoundError(f"Expected ratings file missing after extract: {out}")
    return out


def load_ratings(
    name: str = "ml-latest-small",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load ratings as a DataFrame with columns user_id, item_id, rating, timestamp.

    Parameters
    ----------
    name
        ``ml-latest-small`` (CSV) or ``ml-1m`` (:: -separated DAT).
    data_dir
        Root data directory (gitignored).
    """
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

    df = df[["user_id", "item_id", "rating", "timestamp"]].copy()
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df
