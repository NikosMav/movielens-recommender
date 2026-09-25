"""CLI: download data, run split/train/eval pipeline, write results JSON."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from movielens_recommender import __version__
from movielens_recommender.baselines import (
    ALSRecommender,
    ItemItemCosineRecommender,
    MostPopularRecommender,
)
from movielens_recommender.data import DATASET_URLS, download_dataset, load_ratings
from movielens_recommender.evaluate import evaluate_recommender, format_metrics
from movielens_recommender.split import SplitConfig, time_based_split

DEFAULT_KS = (10, 20)
DEFAULT_SEED = 42
DEFAULT_RESULTS_DIR = Path("results")


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def library_versions() -> dict[str, str]:
    return {
        "movielens-recommender": __version__,
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "pandas": _pkg_version("pandas"),
        "implicit": _pkg_version("implicit"),
    }


def run_pipeline(
    *,
    dataset: str = "ml-latest-small",
    data_dir: Path = Path("data"),
    results_dir: Path = DEFAULT_RESULTS_DIR,
    min_ratings: int = 5,
    test_fraction: float = 0.2,
    relevance_threshold: float = 4.0,
    seed: int = DEFAULT_SEED,
    ks: tuple[int, ...] = DEFAULT_KS,
    download: bool = True,
) -> Path:
    """Download (optional), split, train baselines, evaluate, write JSON."""
    if download:
        download_dataset(dataset, data_dir)

    ratings = load_ratings(dataset, data_dir)
    config = SplitConfig(
        min_ratings=min_ratings,
        test_fraction=test_fraction,
        relevance_threshold=relevance_threshold,
    )
    split = time_based_split(ratings, config)

    models: dict[str, object] = {}
    hyperparams: dict[str, dict] = {}

    popular = MostPopularRecommender().fit(split.train)
    models["most_popular"] = popular
    hyperparams["most_popular"] = {}

    item_knn = ItemItemCosineRecommender(min_common=1).fit(split.train)
    models["item_item_cosine"] = item_knn
    hyperparams["item_item_cosine"] = {"min_common": 1}

    als = ALSRecommender(
        factors=64,
        regularization=0.01,
        iterations=15,
        alpha=40.0,
        confidence_threshold=relevance_threshold,
        random_state=seed,
    ).fit(split.train)
    models["als"] = als
    hyperparams["als"] = als.hyperparams()

    results: dict[str, dict] = {}
    for name, model in models.items():
        metrics = evaluate_recommender(
            model.recommend,
            split.train,
            split.test,
            relevance_threshold=relevance_threshold,
            ks=ks,
        )
        results[name] = format_metrics(metrics)

    payload = {
        "dataset": dataset,
        "split": split.summary(),
        "relevance_threshold": relevance_threshold,
        "ks": list(ks),
        "seed": seed,
        "library_versions": library_versions(),
        "hyperparameters": hyperparams,
        "metrics": results,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _cmd_download(args: argparse.Namespace) -> int:
    path = download_dataset(args.dataset, args.data_dir, force=args.force)
    print(f"Downloaded {args.dataset} -> {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    out = run_pipeline(
        dataset=args.dataset,
        data_dir=Path(args.data_dir),
        results_dir=Path(args.results_dir),
        min_ratings=args.min_ratings,
        test_fraction=args.test_fraction,
        relevance_threshold=args.relevance_threshold,
        seed=args.seed,
        ks=tuple(args.k),
        download=not args.no_download,
    )
    print(f"Wrote results to {out}")
    print(out.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movielens-recommender",
        description="MovieLens recommender: download, split, baselines, evaluate.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_dl = sub.add_parser("download", help="Download a MovieLens dataset from GroupLens")
    p_dl.add_argument(
        "--dataset",
        default="ml-latest-small",
        choices=sorted(DATASET_URLS),
        help="Dataset name (default: ml-latest-small)",
    )
    p_dl.add_argument("--data-dir", default="data", help="Root data directory (gitignored)")
    p_dl.add_argument("--force", action="store_true", help="Re-download even if present")
    p_dl.set_defaults(func=_cmd_download)

    p_run = sub.add_parser(
        "run",
        help="Download (if needed), split, train baselines, evaluate, write results/*.json",
    )
    p_run.add_argument(
        "--dataset",
        default="ml-latest-small",
        choices=sorted(DATASET_URLS),
        help="Dataset name (default: ml-latest-small)",
    )
    p_run.add_argument("--data-dir", default="data", help="Root data directory (gitignored)")
    p_run.add_argument("--results-dir", default="results", help="Where to write metrics JSON")
    p_run.add_argument("--min-ratings", type=int, default=5, help="Min ratings per user to keep")
    p_run.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of each user's latest interactions held out as test",
    )
    p_run.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Ratings >= this count as relevant for metrics",
    )
    p_run.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed (ALS)")
    p_run.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_KS),
        help="Cutoff values for precision/recall/NDCG (default: 10 20)",
    )
    p_run.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download; require data already on disk",
    )
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
