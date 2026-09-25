"""CLI: download data, run config-driven baseline pipeline, write results JSON."""

from __future__ import annotations

import argparse
import json
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from movielens_recommender import __version__
from movielens_recommender.baselines import (
    ALSRecommender,
    ItemItemCosineRecommender,
    MostPopularRecommender,
)
from movielens_recommender.config import RunConfig, load_config
from movielens_recommender.data import (
    DATASET_SHA256,
    DATASET_URLS,
    DATASET_VERSION_LABELS,
    download_dataset,
    load_ratings,
)
from movielens_recommender.evaluate import evaluate_recommender, format_metrics
from movielens_recommender.split import SplitConfig, time_based_split


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
        "pyyaml": _pkg_version("PyYAML"),
    }


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_pipeline(config: RunConfig, *, download: bool = True) -> Path:
    """Download (optional), clean, split, train baselines, evaluate, write JSON."""
    set_seeds(config.seed)
    data_dir = Path(config.data_dir)
    results_dir = Path(config.results_dir)
    dataset = config.dataset

    if download:
        download_dataset(dataset, data_dir)

    ratings, clean_stats = load_ratings(dataset, data_dir, clean=True)
    split_cfg = SplitConfig(
        min_ratings=config.split.min_ratings,
        test_fraction=config.split.test_fraction,
        relevance_threshold=config.eval.relevance_threshold,
    )
    split = time_based_split(ratings, split_cfg)

    models: dict[str, object] = {}
    hyperparams: dict[str, dict] = {}

    popular = MostPopularRecommender().fit(split.train)
    models["most_popular"] = popular
    hyperparams["most_popular"] = {}

    knn_cfg = config.models.item_item_cosine
    item_knn = ItemItemCosineRecommender(min_common=knn_cfg.min_common).fit(split.train)
    models["item_item_cosine"] = item_knn
    hyperparams["item_item_cosine"] = {"min_common": knn_cfg.min_common}

    als_cfg = config.models.als
    als = ALSRecommender(
        factors=als_cfg.factors,
        regularization=als_cfg.regularization,
        iterations=als_cfg.iterations,
        alpha=als_cfg.alpha,
        confidence_threshold=config.eval.relevance_threshold,
        random_state=config.seed,
    ).fit(split.train)
    models["als"] = als
    hyperparams["als"] = als.hyperparams()

    results: dict[str, dict] = {}
    for name, model in models.items():
        metrics = evaluate_recommender(
            model.recommend,
            split.train,
            split.test,
            relevance_threshold=config.eval.relevance_threshold,
            ks=tuple(config.eval.ks),
            n_bootstrap=config.eval.n_bootstrap,
            bootstrap_alpha=config.eval.bootstrap_alpha,
            seed=config.seed,
            split=split,
        )
        results[name] = format_metrics(metrics)

    payload = {
        "dataset": dataset,
        "dataset_version": DATASET_VERSION_LABELS[dataset],
        "dataset_sha256": DATASET_SHA256[dataset],
        "cleaning": clean_stats.to_dict(),
        "split": split.summary(),
        "relevance_threshold": config.eval.relevance_threshold,
        "ks": list(config.eval.ks),
        "seed": config.seed,
        "bootstrap": {
            "n_bootstrap": config.eval.n_bootstrap,
            "alpha": config.eval.bootstrap_alpha,
        },
        "config": config.to_dict(),
        "library_versions": library_versions(),
        "hyperparameters": hyperparams,
        "metrics": results,
        "primary_metric": "ndcg@10",
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _cmd_download(args: argparse.Namespace) -> int:
    path = download_dataset(args.dataset, args.data_dir, force=args.force)
    print(f"Downloaded {args.dataset} (sha256={DATASET_SHA256[args.dataset]}) -> {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    # CLI overrides for convenience / backwards compatibility.
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if args.seed is not None:
        config.seed = args.seed

    out = run_pipeline(config, download=not args.no_download)
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
        help="Config-driven pipeline: download, clean, split, train, evaluate → results/*.json",
    )
    p_run.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML run config (default: configs/default.yaml)",
    )
    p_run.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASET_URLS),
        help="Override dataset from config",
    )
    p_run.add_argument("--data-dir", default=None, help="Override data_dir from config")
    p_run.add_argument("--results-dir", default=None, help="Override results_dir from config")
    p_run.add_argument("--seed", type=int, default=None, help="Override seed from config")
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
