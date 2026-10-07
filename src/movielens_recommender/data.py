"""Download, verify, clean, and load MovieLens datasets from GroupLens."""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

# Official GroupLens URLs — see https://grouplens.org/datasets/movielens/
DATASET_URLS: dict[str, str] = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}

# SHA-256 of the official zip bytes (computed from GroupLens archives at pin time).
DATASET_SHA256: dict[str, str] = {
    "ml-latest-small": "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436",
    "ml-1m": "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20",
    "ml-32m": "e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee",
}

DATASET_VERSION_LABELS: dict[str, str] = {
    "ml-latest-small": (
        "ml-latest-small@sha256:"
        "696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436"
    ),
    "ml-1m": (
        "ml-1m@sha256:a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
    ),
    "ml-32m": (
        "ml-32m@sha256:e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee"
    ),
}

# CSV layout shared by ml-latest-small and ml-32m (userId, movieId, rating, timestamp).
CSV_DATASETS = frozenset({"ml-latest-small", "ml-32m"})

LICENSE_URL = "https://grouplens.org/datasets/movielens/"
DEFAULT_DATA_DIR = Path("data")

# MovieLens 1M age codes are buckets, not raw ages. The labels are the
# GroupLens README legend. Codes used as LightGBM categoricals are
# non-negative; unknown sentinels are outside the observed sets.
AGE_BUCKETS: tuple[int, ...] = (1, 18, 25, 35, 45, 50, 56)
AGE_BUCKET_LABELS: dict[int, str] = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}
GENDER_TO_CODE: dict[str, int] = {"M": 0, "F": 1}
GENDER_UNKNOWN = 2
AGE_UNKNOWN = 0
OCCUPATION_MAX = 20
OCCUPATION_UNKNOWN = 21
REGION_UNKNOWN = 10


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


def users_path(name: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Return the expected path to ml-1m ``users.dat``.

    ml-latest-small does not ship demographics. Callers that need them must
    stay on ml-1m.
    """
    if name != "ml-1m":
        raise ValueError(
            f"{name} has no users.dat demographics. The demographic experiment is ml-1m only."
        )
    return dataset_dir(name, data_dir) / "ml-1m" / "users.dat"


def region_from_zip(zip_code: object) -> int:
    """Coarse US region: the first ZIP digit, or ``REGION_UNKNOWN``.

    The raw ZIP is not a model feature. A non-digit prefix (Canadian and
    other non-US codes in ml-1m) maps to the unknown sentinel.
    """
    text = str(zip_code).strip()
    if text and text[0].isdigit():
        return int(text[0])
    return REGION_UNKNOWN


def _code_or_unknown(value: object, allowed: set[int], unknown: int) -> int:
    try:
        code = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return unknown
    if code in allowed:
        return code
    return unknown


def parse_users(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a users table to the demographic schema.

    Expected columns are ``user_id``, ``gender``, ``age``, ``occupation``,
    ``zip_code`` (the ``users.dat`` fields). Duplicate user ids keep the
    first row. Gender, age bucket, occupation, and ZIP region are stored
    both as raw values and as non-negative categorical codes.
    """
    required = ["user_id", "gender", "age", "occupation", "zip_code"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"users frame is missing columns: {missing}")
    df = frame.loc[:, required].copy()
    before = len(df)
    df = df.dropna(subset=["user_id"])
    df = df[df["user_id"].astype(float) > 0]
    df["user_id"] = df["user_id"].astype(int)
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()
    df["zip_code"] = df["zip_code"].astype(str).str.strip()
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    df = df.reset_index(drop=True)

    age_ok = set(AGE_BUCKETS)
    occ_ok = set(range(OCCUPATION_MAX + 1))
    df["gender_code"] = df["gender"].map(
        lambda g: GENDER_TO_CODE.get(g, GENDER_UNKNOWN)
    ).astype(int)
    df["age_code"] = [
        _code_or_unknown(value, age_ok, AGE_UNKNOWN) for value in df["age"].tolist()
    ]
    df["occupation_code"] = [
        _code_or_unknown(value, occ_ok, OCCUPATION_UNKNOWN) for value in df["occupation"].tolist()
    ]
    df["region"] = [region_from_zip(value) for value in df["zip_code"].tolist()]
    df["region_code"] = df["region"].astype(int)
    df.attrs["n_dropped"] = before - len(df)
    return df[
        [
            "user_id",
            "gender",
            "age",
            "occupation",
            "zip_code",
            "region",
            "gender_code",
            "age_code",
            "occupation_code",
            "region_code",
        ]
    ].copy()


def read_users_dat(path: Path | str) -> pd.DataFrame:
    """Parse an ml-1m ``users.dat`` file (``UserID::Gender::Age::Occupation::Zip``)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"users.dat not found at {path}")
    raw = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        header=None,
        encoding="latin-1",
    )
    return parse_users(raw)


def load_users(
    name: str = "ml-1m",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Load ml-1m demographics. Other datasets raise; they have no users file."""
    return read_users_dat(users_path(name, data_dir))


def ratings_path(name: str, data_dir: Path | str = DEFAULT_DATA_DIR) -> Path:
    """Return the expected path to the ratings file after download/extract."""
    root = dataset_dir(name, data_dir)
    if name in CSV_DATASETS:
        return root / name / "ratings.csv"
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
    if os.environ.get("MOVIELENS_ALLOW_DOWNLOAD") == "0":
        raise RuntimeError(
            "MovieLens download is disabled (MOVIELENS_ALLOW_DOWNLOAD=0). "
            "CI must not fetch GroupLens archives."
        )
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

    if name in CSV_DATASETS:
        df = pd.read_csv(
            path,
            dtype={
                "userId": np.int32,
                "movieId": np.int32,
                "rating": np.float32,
                "timestamp": np.int64,
            },
        )
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
