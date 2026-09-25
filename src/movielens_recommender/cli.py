"""CLI: download data, run config-driven baseline pipeline, write results JSON."""

from __future__ import annotations

import argparse
import json
import random
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from movielens_recommender import __version__
from movielens_recommender.config import RunConfig, load_config
from movielens_recommender.data import (
    DATASET_SHA256,
    DATASET_URLS,
    DATASET_VERSION_LABELS,
    download_dataset,
    load_ratings,
)
from movielens_recommender.evaluate import evaluate_recommender, format_metrics
from movielens_recommender.split import (
    GlobalCutoffConfig,
    SplitConfig,
    SplitResult,
    global_time_cutoff_split,
    time_based_split,
)
from movielens_recommender.tune import build_model, tune_als, tune_item_knn


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


def _eval_model(
    model: Any,
    train,
    test,
    *,
    config: RunConfig,
    split,
    include_segments: bool,
) -> dict[str, Any]:
    metrics = evaluate_recommender(
        model.recommend,
        train,
        test,
        relevance_threshold=config.eval.relevance_threshold,
        ks=tuple(config.eval.ks),
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
        split=split,
        include_segments=include_segments,
    )
    return format_metrics(metrics)


def run_pipeline(config: RunConfig, *, download: bool = True) -> Path:
    """Download (optional), clean, split, tune, train baselines, evaluate, write JSON."""
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
        val_fraction=config.split.val_fraction,
        relevance_threshold=config.eval.relevance_threshold,
    )
    split = time_based_split(ratings, split_cfg)
    full_train = split.full_train

    # Eval SplitResult uses full_train as the fit matrix (seen / catalog / cold-start).
    eval_split = SplitResult(
        train=full_train,
        test=split.test,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )

    tuning_payload: dict[str, Any] | None = None
    tuned_hps: dict[str, dict[str, Any]] = {}

    if config.tune:
        if split.val is None or split.val.empty:
            raise ValueError("tune=true requires split.val_fraction > 0")
        print("Tuning ALS on validation (NDCG@10)...", flush=True)
        als_tune = tune_als(
            split,
            relevance_threshold=config.eval.relevance_threshold,
            seed=config.seed,
        )
        print("Tuning item-item on validation (NDCG@10)...", flush=True)
        knn_tune = tune_item_knn(
            split,
            relevance_threshold=config.eval.relevance_threshold,
        )
        tuning_payload = {
            "protocol": (
                "Fit each trial on fit-train only; select by validation NDCG@10 "
                "(point estimate, no bootstrap). Never uses test. Chosen configs "
                "are refit on full_train (= fit-train ∪ val) before test eval."
            ),
            "primary_metric": "ndcg@10",
            "val_fraction": config.split.val_fraction,
            "als": als_tune.to_dict(),
            "item_item_cosine": knn_tune.to_dict(),
        }
        tuned_hps["als_tuned"] = als_tune.best_hyperparams
        tuned_hps["item_item_cosine_tuned"] = knn_tune.best_hyperparams

        tuning_dir = results_dir / "tuning"
        tuning_dir.mkdir(parents=True, exist_ok=True)
        tuning_path = tuning_dir / f"{dataset}.json"
        tuning_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "dataset_sha256": DATASET_SHA256[dataset],
                    "seed": config.seed,
                    **tuning_payload,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote tuning results to {tuning_path}", flush=True)

    # Default (S2) hyperparams from config.
    default_als = {
        "factors": config.models.als.factors,
        "regularization": config.models.als.regularization,
        "iterations": config.models.als.iterations,
        "alpha": config.models.als.alpha,
    }
    default_knn = {
        "min_common": config.models.item_item_cosine.min_common,
        "k_neighbors": config.models.item_item_cosine.k_neighbors,
        "shrinkage": config.models.item_item_cosine.shrinkage,
    }

    model_specs: list[tuple[str, dict[str, Any], bool]] = [
        ("most_popular", {}, False),
        ("item_item_cosine", default_knn, False),
        ("als", default_als, False),
    ]
    if config.tune:
        model_specs.extend(
            [
                ("item_item_cosine_tuned", tuned_hps["item_item_cosine_tuned"], True),
                ("als_tuned", tuned_hps["als_tuned"], True),
            ]
        )

    results: dict[str, dict] = {}
    hyperparams: dict[str, dict] = {}
    tuned_flags: dict[str, bool] = {}

    for name, hp, is_tuned in model_specs:
        print(f"Fitting {name} on full_train and evaluating on test...", flush=True)
        model, recorded = build_model(
            name,
            full_train,
            hyperparams=hp,
            relevance_threshold=config.eval.relevance_threshold,
            seed=config.seed,
        )
        hyperparams[name] = recorded
        tuned_flags[name] = is_tuned
        results[name] = _eval_model(
            model,
            full_train,
            split.test,
            config=config,
            split=eval_split,
            include_segments=True,
        )

    payload: dict[str, Any] = {
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
        "tuned": tuned_flags,
        "metrics": results,
        "primary_metric": "ndcg@10",
        "protocol_notes": {
            "validation": (
                "Per-user val holdout from train pool "
                f"(val_fraction={config.split.val_fraction}); "
                "tune on val only; refit on full_train; evaluate once on test "
                "(ADR-0005)."
            ),
            "tuned_models": (
                "Models whose name ends with _tuned (or tuned[name]=true) used "
                "hyperparameters selected on validation NDCG@10. Untuned names "
                "keep the S2 YAML defaults."
            ),
        },
    }
    if tuning_payload is not None:
        payload["tuning_summary"] = {
            "als_best": tuning_payload["als"]["best_hyperparams"],
            "als_best_val_ndcg@10": tuning_payload["als"]["best_val_score"],
            "item_item_cosine_best": tuning_payload["item_item_cosine"]["best_hyperparams"],
            "item_item_cosine_best_val_ndcg@10": tuning_payload["item_item_cosine"][
                "best_val_score"
            ],
            "tuning_json": f"results/tuning/{dataset}.json",
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if config.global_cutoff.enabled:
        gc_path = run_global_cutoff(
            ratings,
            clean_stats=clean_stats.to_dict(),
            config=config,
            tuned_hps=tuned_hps if config.tune else {},
            default_als=default_als,
            default_knn=default_knn,
        )
        print(f"Wrote global-cutoff results to {gc_path}", flush=True)

    return out_path


def run_global_cutoff(
    ratings,
    *,
    clean_stats: dict[str, Any],
    config: RunConfig,
    tuned_hps: dict[str, dict[str, Any]],
    default_als: dict[str, Any],
    default_knn: dict[str, Any],
) -> Path:
    """Secondary global-time-cutoff evaluation (does not re-tune)."""
    gc_cfg = GlobalCutoffConfig(
        timestamp_quantile=config.global_cutoff.timestamp_quantile,
        min_train_ratings=config.global_cutoff.min_train_ratings,
        relevance_threshold=config.eval.relevance_threshold,
    )
    gc = global_time_cutoff_split(ratings, gc_cfg)
    eval_split = gc.as_split_result()

    # Prefer tuned configs when available; otherwise S2 defaults. Do not re-tune.
    als_hp = tuned_hps.get("als_tuned", default_als)
    knn_hp = tuned_hps.get("item_item_cosine_tuned", default_knn)
    used_tuned = bool(tuned_hps)

    model_specs = [
        ("most_popular", {}, False),
        ("item_item_cosine", knn_hp, used_tuned),
        ("als", als_hp, used_tuned),
    ]

    results: dict[str, dict] = {}
    hyperparams: dict[str, dict] = {}
    for name, hp, _is_tuned in model_specs:
        print(f"[global_cutoff] Fitting {name}...", flush=True)
        model, recorded = build_model(
            name,
            gc.train,
            hyperparams=hp,
            relevance_threshold=config.eval.relevance_threshold,
            seed=config.seed,
        )
        hyperparams[name] = recorded
        results[name] = _eval_model(
            model,
            gc.train,
            gc.test,
            config=config,
            split=eval_split,
            include_segments=False,
        )
        if eval_split.cold_start is not None:
            gc.cold_start = eval_split.cold_start

    # Surviving counts after cold-start relevance filtering.
    cold = results[next(iter(results))].get("cold_start", {})
    surviving = {
        "n_train_users": int(gc.train["user_id"].nunique()),
        "n_train_items": int(gc.train["item_id"].nunique()),
        "n_test_interactions": int(len(gc.test)),
        "n_eval_users": int(cold.get("n_eval_users", 0)),
        "n_users_kept_after_min_train": gc.n_users_kept,
        "n_test_interactions_after_user_filter": gc.n_test_interactions_kept,
    }

    payload = {
        "dataset": config.dataset,
        "dataset_version": DATASET_VERSION_LABELS[config.dataset],
        "dataset_sha256": DATASET_SHA256[config.dataset],
        "protocol": "global_time_cutoff",
        "secondary": True,
        "retuned": False,
        "hyperparams_source": (
            "Per-user-protocol tuned configs (validation NDCG@10); not re-tuned "
            "for the global cutoff. most_popular has no hyperparameters."
            if used_tuned
            else "S2 YAML defaults (no tuning run)."
        ),
        "cleaning": clean_stats,
        "split": gc.summary(),
        "surviving": surviving,
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
        "notes": (
            "Secondary sanity check: whether model ranking holds when cross-user "
            "future signal is removed (ADR-0002). Not the headline table."
        ),
    }

    out_dir = Path(config.results_dir) / "global_cutoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{config.dataset}.json"
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
    if args.no_tune:
        config.tune = False

    out = run_pipeline(config, download=not args.no_download)
    print(f"Wrote results to {out}")
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
        help="Config-driven pipeline: download, clean, split, tune, evaluate → results/*.json",
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
    p_run.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip validation-grid tuning; evaluate YAML defaults only",
    )
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
