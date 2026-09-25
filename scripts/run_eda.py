#!/usr/bin/env python3
"""Exploratory data analysis for MovieLens (figures + aggregate stats only).

Writes committed artifacts under ``docs/eda/`` — never raw ratings.
Requires the dataset to already be downloaded (or pass --download).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from movielens_recommender.data import download_dataset, load_ratings  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "eda"


def run_eda(dataset: str, data_dir: Path, out_dir: Path, *, download: bool) -> Path:
    if download:
        download_dataset(dataset, data_dir)
    ratings, clean_stats = load_ratings(dataset, data_dir, clean=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "dataset": dataset,
        "cleaning": clean_stats.to_dict(),
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["user_id"].nunique()),
        "n_items": int(ratings["item_id"].nunique()),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_std": float(ratings["rating"].std()),
        "rating_value_counts": {
            str(k): int(v) for k, v in sorted(ratings["rating"].value_counts().items())
        },
        "user_activity": {
            "min": int(ratings.groupby("user_id").size().min()),
            "median": float(ratings.groupby("user_id").size().median()),
            "mean": float(ratings.groupby("user_id").size().mean()),
            "max": int(ratings.groupby("user_id").size().max()),
        },
        "item_popularity": {
            "min": int(ratings.groupby("item_id").size().min()),
            "median": float(ratings.groupby("item_id").size().median()),
            "mean": float(ratings.groupby("item_id").size().mean()),
            "max": int(ratings.groupby("item_id").size().max()),
        },
        "timestamp_min": int(ratings["timestamp"].min()),
        "timestamp_max": int(ratings["timestamp"].max()),
    }
    stats_path = out_dir / f"{dataset}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # 1) Rating distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="#2c5f7c")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title(f"{dataset}: rating distribution")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_rating_distribution.png", dpi=120)
    plt.close(fig)

    # 2) User activity (log histogram)
    user_counts = ratings.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(user_counts, bins=40, color="#2c5f7c", edgecolor="white")
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Users")
    ax.set_title(f"{dataset}: user activity")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_user_activity.png", dpi=120)
    plt.close(fig)

    # 3) Item popularity (log-x histogram)
    item_counts = ratings.groupby("item_id").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(item_counts, bins=40, color="#2c5f7c", edgecolor="white")
    ax.set_xlabel("Ratings per item")
    ax.set_ylabel("Items")
    ax.set_yscale("log")
    ax.set_title(f"{dataset}: item popularity")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_item_popularity.png", dpi=120)
    plt.close(fig)

    # 4) Ratings over time (monthly)
    ts = pd.to_datetime(ratings["timestamp"], unit="s")
    monthly = ts.dt.to_period("M").value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, color="#2c5f7c", linewidth=1.2)
    ax.set_xlabel("Month")
    ax.set_ylabel("Ratings")
    ax.set_title(f"{dataset}: ratings over time")
    fig.tight_layout()
    fig.savefig(out_dir / f"{dataset}_ratings_over_time.png", dpi=120)
    plt.close(fig)

    return stats_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MovieLens EDA → docs/eda/")
    parser.add_argument("--dataset", default="ml-latest-small")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    path = run_eda(
        args.dataset,
        Path(args.data_dir),
        Path(args.out_dir),
        download=args.download,
    )
    print(f"Wrote EDA stats to {path} and figures under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
