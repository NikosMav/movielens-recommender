"""CLI: download data, run config-driven baseline + two-tower pipeline, write results JSON."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from movielens_recommender.movies import load_movies
from movielens_recommender.split import (
    GlobalCutoffConfig,
    SplitConfig,
    SplitResult,
    global_time_cutoff_split,
    time_based_split,
)
from movielens_recommender.tune import (
    build_model,
    tune_als,
    tune_ease,
    tune_item_knn,
    tune_rp3beta,
)


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def library_versions() -> dict[str, str]:
    versions = {
        "movielens-recommender": __version__,
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "pandas": _pkg_version("pandas"),
        "implicit": _pkg_version("implicit"),
        "pyyaml": _pkg_version("PyYAML"),
    }
    torch_v = _pkg_version("torch")
    if torch_v != "unknown":
        versions["torch"] = torch_v
    lgb_v = _pkg_version("lightgbm")
    if lgb_v != "unknown":
        versions["lightgbm"] = lgb_v
    return versions


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


def _eval_ks(config: RunConfig) -> list[int]:
    return sorted(set(int(k) for k in list(config.eval.ks) + list(config.eval.retrieval_ks)))


def _eval_model(
    model: Any,
    train,
    test,
    *,
    config: RunConfig,
    split,
    include_segments: bool,
    ks: list[int] | None = None,
) -> dict[str, Any]:
    metrics = evaluate_recommender(
        model.recommend,
        train,
        test,
        relevance_threshold=config.eval.relevance_threshold,
        ks=tuple(ks if ks is not None else _eval_ks(config)),
        n_bootstrap=config.eval.n_bootstrap,
        bootstrap_alpha=config.eval.bootstrap_alpha,
        seed=config.seed,
        split=split,
        include_segments=include_segments,
    )
    return format_metrics(metrics)


def _seed_summary(per_seed: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(s["metrics"][key]) for s in per_seed if key in s["metrics"]]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        out[key] = {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std(ddof=0)), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "values": [round(float(v), 6) for v in vals],
        }
    return out


def _load_baseline_tuning(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_baseline_tuning(
    path: Path, dataset: str, seed: int, payload: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "dataset_sha256": DATASET_SHA256[dataset],
                "seed": seed,
                **payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_pipeline(
    config: RunConfig,
    *,
    download: bool = True,
    reuse_baseline_tuning: bool = False,
    reuse_two_tower_tuning: bool = False,
) -> Path:
    """Download (optional), clean, split, tune, train, evaluate, write JSON."""
    set_seeds(config.seed)
    data_dir = Path(config.data_dir)
    results_dir = Path(config.results_dir)
    dataset = config.dataset
    pipeline_t0 = time.perf_counter()

    if download:
        download_dataset(dataset, data_dir)

    ratings, clean_stats = load_ratings(dataset, data_dir, clean=True)
    movies = load_movies(dataset, data_dir)
    split_cfg = SplitConfig(
        min_ratings=config.split.min_ratings,
        test_fraction=config.split.test_fraction,
        val_fraction=config.split.val_fraction,
        relevance_threshold=config.eval.relevance_threshold,
    )
    split = time_based_split(ratings, split_cfg)
    full_train = split.full_train

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
    tuning_dir = results_dir / "tuning"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    baseline_tuning_path = tuning_dir / f"{dataset}.json"

    if config.tune:
        if split.val is None or split.val.empty:
            raise ValueError("tune=true requires split.val_fraction > 0")

        need_classic = True
        need_linear = True
        if reuse_baseline_tuning:
            existing = _load_baseline_tuning(baseline_tuning_path)
            if existing is not None and "als" in existing and "item_item_cosine" in existing:
                print(f"Reusing baseline tuning from {baseline_tuning_path}", flush=True)
                als_block = existing["als"]
                knn_block = existing["item_item_cosine"]
                tuning_payload = {
                    "protocol": existing.get(
                        "protocol",
                        "Fit each trial on fit-train only; select by validation NDCG@10.",
                    ),
                    "primary_metric": "ndcg@10",
                    "val_fraction": config.split.val_fraction,
                    "als": als_block,
                    "item_item_cosine": knn_block,
                    "reused_from": str(baseline_tuning_path),
                }
                tuned_hps["als_tuned"] = dict(als_block["best_hyperparams"])
                tuned_hps["item_item_cosine_tuned"] = dict(knn_block["best_hyperparams"])
                need_classic = False
                if "ease" in existing and "rp3beta" in existing:
                    tuning_payload["ease"] = existing["ease"]
                    tuning_payload["rp3beta"] = existing["rp3beta"]
                    tuned_hps["ease"] = dict(existing["ease"]["best_hyperparams"])
                    tuned_hps["rp3beta"] = dict(existing["rp3beta"]["best_hyperparams"])
                    need_linear = False

        if need_classic or need_linear:
            if tuning_payload is None:
                tuning_payload = {
                    "protocol": (
                        "Fit each trial on fit-train only; select by validation NDCG@10 "
                        "(point estimate, no bootstrap). Never uses test. Chosen configs "
                        "are refit on full_train (= fit-train ∪ val) before test eval."
                    ),
                    "primary_metric": "ndcg@10",
                    "val_fraction": config.split.val_fraction,
                }
            if need_classic:
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
                tuning_payload["als"] = als_tune.to_dict()
                tuning_payload["item_item_cosine"] = knn_tune.to_dict()
                tuned_hps["als_tuned"] = als_tune.best_hyperparams
                tuned_hps["item_item_cosine_tuned"] = knn_tune.best_hyperparams
            if need_linear:
                print("Tuning EASE on validation (NDCG@10)...", flush=True)
                ease_tune = tune_ease(
                    split,
                    relevance_threshold=config.eval.relevance_threshold,
                )
                print("Tuning RP3beta on validation (NDCG@10)...", flush=True)
                rp3_tune = tune_rp3beta(
                    split,
                    relevance_threshold=config.eval.relevance_threshold,
                )
                tuning_payload["ease"] = ease_tune.to_dict()
                tuning_payload["rp3beta"] = rp3_tune.to_dict()
                tuned_hps["ease"] = ease_tune.best_hyperparams
                tuned_hps["rp3beta"] = rp3_tune.best_hyperparams
            _write_baseline_tuning(
                baseline_tuning_path, dataset, config.seed, tuning_payload
            )
            print(f"Wrote tuning results to {baseline_tuning_path}", flush=True)

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
                ("ease", tuned_hps["ease"], True),
                ("rp3beta", tuned_hps["rp3beta"], True),
            ]
        )

    results: dict[str, dict] = {}
    hyperparams: dict[str, dict] = {}
    tuned_flags: dict[str, bool] = {}
    fitted_models: dict[str, Any] = {}
    eval_ks = _eval_ks(config)

    for name, hp, is_tuned in model_specs:
        print(f"Fitting {name} on full_train and evaluating on test...", flush=True)
        model, recorded = build_model(
            name,
            full_train,
            hyperparams=hp,
            relevance_threshold=config.eval.relevance_threshold,
            seed=config.seed,
        )
        fitted_models[name] = model
        hyperparams[name] = recorded
        tuned_flags[name] = is_tuned
        results[name] = _eval_model(
            model,
            full_train,
            split.test,
            config=config,
            split=eval_split,
            include_segments=True,
            ks=eval_ks,
        )

    two_tower_meta: dict[str, Any] | None = None
    two_tower_primary: Any | None = None
    if config.models.two_tower.enabled:
        if not _torch_available():
            raise ImportError(
                "models.two_tower.enabled=true but PyTorch is not installed. "
                "Install with: pip install torch==2.6.0 "
                "--index-url https://download.pytorch.org/whl/cpu "
                "&& pip install -e '.[deep]'"
            )
        reused_tt = None
        tt_tuning_path = tuning_dir / f"two_tower_{dataset}.json"
        if reuse_two_tower_tuning and tt_tuning_path.is_file():
            print(f"Reusing two-tower tuning from {tt_tuning_path}", flush=True)
            reused_tt = json.loads(tt_tuning_path.read_text(encoding="utf-8"))
        two_tower_meta, two_tower_primary = _run_two_tower(
            split=split,
            eval_split=eval_split,
            full_train=full_train,
            movies=movies,
            config=config,
            results=results,
            hyperparams=hyperparams,
            tuned_flags=tuned_flags,
            tuning_dir=tuning_dir,
            reused_tuning=reused_tt,
        )

    ranker_block: dict[str, Any] | None = None
    ranker_later: dict[str, Any] | None = None
    if config.models.ranker.enabled:
        if not config.tune:
            raise ValueError(
                "models.ranker.enabled=true requires tune=true so item-item "
                "hyperparameters are the validation-chosen ones."
            )
        if two_tower_meta is None or two_tower_primary is None:
            raise ValueError(
                "models.ranker.enabled=true requires models.two_tower.enabled=true."
            )
        if not _lightgbm_available():
            raise ImportError(
                "models.ranker.enabled=true but LightGBM is not installed. "
                "Install with: pip install -e '.[rank]' (lightgbm==4.6.0)."
            )
        from movielens_recommender.ranker.pipeline import run_ranker_stage

        ranker_out = run_ranker_stage(
            split=split,
            movies=movies,
            config=config,
            tuned_item_hp=tuned_hps["item_item_cosine_tuned"],
            two_tower_hp=two_tower_meta["best_hyperparams"],
            two_tower_epochs=int(two_tower_meta["best_epoch"]),
            full_train_item_model=fitted_models["item_item_cosine_tuned"],
            full_train_two_tower=two_tower_primary,
            bar_metrics=results["item_item_cosine"],
        )
        results.update(ranker_out["metrics"])
        hyperparams.update(ranker_out["hyperparameters"])
        tuned_flags.update(ranker_out["tuned"])
        ranker_block = ranker_out["ranker"]
        ranker_later = ranker_out["later_window"]

    payload: dict[str, Any] = {
        "dataset": dataset,
        "dataset_version": DATASET_VERSION_LABELS[dataset],
        "dataset_sha256": DATASET_SHA256[dataset],
        "cleaning": clean_stats.to_dict(),
        "split": split.summary(),
        "relevance_threshold": config.eval.relevance_threshold,
        "ks": list(config.eval.ks),
        "retrieval_ks": list(config.eval.retrieval_ks),
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
            "two_tower": (
                "Optional PyTorch two-tower (ADR-0006): in-batch sampled softmax "
                "with log-q correction; exact top-k; history from fit-train (val) "
                "or full-train (test) only. Test metrics reported over multiple "
                "seeds (mean/spread) plus per-seed user-bootstrap CIs."
            ),
            "retrieval_ks": (
                "Recall@100 / Recall@200 are reported for retrieval comparison "
                "(candidate generation for S4)."
            ),
            "ease_rp3beta": (
                "EASE^R and RP3beta (ADR-0008): binary interaction matrix "
                "(any train rating is an edge). λ, and (alpha, beta, top_k), "
                "are chosen on validation NDCG@10, refit on full_train, and "
                "evaluated once on test. Seen train items are excluded. "
                "The S4 ranker is unchanged."
            ),
            "ranker": (
                "Optional LightGBM LambdaRank (ADR-0007). Candidate set chosen "
                "by validation Recall@200. Ranker labels are the validation "
                "window; training features come from fit-train only. Before "
                "test scoring, retrievers and features are rebuilt on full "
                "train. Test labels never enter ranker training. Ranker seeds "
                "share retrievers and candidate sets."
            ),
        },
        "runtime_sec": round(time.perf_counter() - pipeline_t0, 3),
    }
    if tuning_payload is not None:
        payload["tuning_summary"] = {
            "als_best": tuning_payload["als"]["best_hyperparams"],
            "als_best_val_ndcg@10": tuning_payload["als"]["best_val_score"],
            "item_item_cosine_best": tuning_payload["item_item_cosine"]["best_hyperparams"],
            "item_item_cosine_best_val_ndcg@10": tuning_payload["item_item_cosine"][
                "best_val_score"
            ],
            "ease_best": tuning_payload["ease"]["best_hyperparams"],
            "ease_best_val_ndcg@10": tuning_payload["ease"]["best_val_score"],
            "rp3beta_best": tuning_payload["rp3beta"]["best_hyperparams"],
            "rp3beta_best_val_ndcg@10": tuning_payload["rp3beta"]["best_val_score"],
            "tuning_json": f"results/tuning/{dataset}.json",
        }
    if two_tower_meta is not None:
        payload["two_tower"] = two_tower_meta
        payload["tuning_summary"] = payload.get("tuning_summary") or {}
        payload["tuning_summary"]["two_tower_best"] = two_tower_meta["best_hyperparams"]
        payload["tuning_summary"]["two_tower_best_val_ndcg@10"] = two_tower_meta[
            "best_val_ndcg@10"
        ]
        payload["tuning_summary"]["two_tower_best_epoch"] = two_tower_meta["best_epoch"]
        payload["tuning_summary"]["two_tower_tuning_json"] = two_tower_meta["tuning_json"]
    if ranker_block is not None:
        payload["ranker"] = ranker_block

    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{dataset}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if config.global_cutoff.enabled:
        gc_path = run_global_cutoff(
            ratings,
            movies=movies,
            clean_stats=clean_stats.to_dict(),
            config=config,
            tuned_hps=tuned_hps if config.tune else {},
            default_als=default_als,
            default_knn=default_knn,
            two_tower_hp=two_tower_meta["best_hyperparams"] if two_tower_meta else None,
            two_tower_epochs=two_tower_meta["best_epoch"] if two_tower_meta else None,
            ranker_later=ranker_later,
        )
        print(f"Wrote global-cutoff results to {gc_path}", flush=True)

    return out_path


def _run_two_tower(
    *,
    split: SplitResult,
    eval_split: SplitResult,
    full_train: pd.DataFrame,
    movies: pd.DataFrame,
    config: RunConfig,
    results: dict[str, dict],
    hyperparams: dict[str, dict],
    tuned_flags: dict[str, bool],
    tuning_dir: Path,
    reused_tuning: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Any]:
    from movielens_recommender.two_tower.train import (
        fit_two_tower_recommender,
        tune_two_tower,
    )

    dataset = config.dataset
    tt_tuning_path = tuning_dir / f"two_tower_{dataset}.json"
    if reused_tuning is not None:
        tt_block = reused_tuning["two_tower"]
        best_hp = dict(tt_block["best_hyperparams"])
        best_epoch = int(tt_block["best_epoch"])
        best_val_score = float(tt_block["best_val_score"])
    else:
        print("Tuning two-tower on validation (NDCG@10)...", flush=True)
        tt_tune = tune_two_tower(
            split,
            dataset=dataset,
            data_dir=config.data_dir,
            movies=movies,
            relevance_threshold=config.eval.relevance_threshold,
            seed=config.seed,
            show_progress=True,
        )
        tt_doc = {
            "dataset": dataset,
            "dataset_sha256": DATASET_SHA256[dataset],
            "seed": config.seed,
            "protocol": (
                "Fit each trial on fit-train only; select by validation NDCG@10 "
                "with early stopping. Never uses test. Refit on full_train for "
                "best_epoch epochs before test eval (ADR-0005 / ADR-0006)."
            ),
            "primary_metric": "ndcg@10",
            "val_fraction": config.split.val_fraction,
            "two_tower": tt_tune.to_dict(),
        }
        tt_tuning_path.write_text(
            json.dumps(tt_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote two-tower tuning results to {tt_tuning_path}", flush=True)
        best_hp = dict(tt_tune.best_hyperparams)
        best_epoch = int(tt_tune.best_epoch)
        best_val_score = float(tt_tune.best_val_score)
    seeds = list(config.models.two_tower.seeds)
    per_seed: list[dict[str, Any]] = []
    eval_ks = _eval_ks(config)
    primary_seed = config.seed if config.seed in seeds else seeds[0]
    primary_rec: Any | None = None

    for seed in seeds:
        print(
            f"Refitting two_tower on full_train "
            f"(epochs={best_epoch}, seed={seed}) and evaluating...",
            flush=True,
        )
        set_seeds(seed)
        rec, _feat, train_result = fit_two_tower_recommender(
            full_train,
            dataset=dataset,
            data_dir=config.data_dir,
            movies=movies,
            hyperparams=best_hp,
            seed=seed,
            relevance_threshold=config.eval.relevance_threshold,
            n_epochs=best_epoch,
            val_split=None,
            show_progress=True,
        )
        if seed == primary_seed:
            primary_rec = rec
        metrics = _eval_model(
            rec,
            full_train,
            split.test,
            config=config,
            split=eval_split,
            include_segments=True,
            ks=eval_ks,
        )
        per_seed.append(
            {
                "seed": seed,
                "metrics": metrics,
                "train": {
                    "epochs": train_result.epochs_trained,
                    "wall_time_sec": train_result.wall_time_sec,
                },
            }
        )

    summary_keys = [
        "ndcg@10",
        "ndcg@20",
        "precision@10",
        "recall@10",
        "precision@20",
        "recall@20",
        "recall@100",
        "recall@200",
        "coverage@10",
        "mean_popularity@10",
    ]
    across = _seed_summary(per_seed, summary_keys)

    # Headline metrics entry: mean across seeds for point estimates; attach
    # CI from the config seed when present, else the first seed.
    primary = next(s for s in per_seed if s["seed"] == primary_seed)
    headline = dict(primary["metrics"])
    for key, block in across.items():
        headline[key] = block["mean"]
    headline["seed_summary"] = across
    headline["per_seed_confidence_intervals"] = {
        str(s["seed"]): s["metrics"].get("confidence_intervals", {}) for s in per_seed
    }
    headline["n_seeds"] = float(len(seeds))

    results["two_tower"] = headline
    recorded_hp = {
        **best_hp,
        "best_epoch": best_epoch,
        "refit_epochs": best_epoch,
        "seeds": seeds,
    }
    hyperparams["two_tower"] = recorded_hp
    tuned_flags["two_tower"] = True

    # Gate vs item-item bar (documented in ADR-0006 / README).
    bar_names = ["item_item_cosine", "item_item_cosine_tuned"]
    bars = {}
    for name in bar_names:
        if name in results:
            ci = results[name]["confidence_intervals"]["ndcg@10"]
            bars[name] = {
                "ndcg@10": results[name]["ndcg@10"],
                "ci": ci,
            }
    tt_mean = across["ndcg@10"]["mean"]
    tt_cis = [s["metrics"]["confidence_intervals"]["ndcg@10"] for s in per_seed]
    beats = {}
    for name, bar in bars.items():
        # Win only if mean exceeds bar point estimate AND every seed CI low
        # is above the bar's CI high (conservative) — also record soft comparison.
        bar_point = float(bar["ndcg@10"])
        bar_high = float(bar["ci"]["high"])
        beats[name] = {
            "two_tower_mean_ndcg@10": tt_mean,
            "bar_ndcg@10": bar_point,
            "bar_ci_high": bar_high,
            "mean_exceeds_bar_point": bool(tt_mean > bar_point),
            "all_seed_ci_low_above_bar_ci_high": all(
                float(c["low"]) > bar_high for c in tt_cis
            ),
            "any_seed_mean_exceeds_bar_point": any(
                float(s["metrics"]["ndcg@10"]) > bar_point for s in per_seed
            ),
        }
    beats_both = all(
        beats[n]["mean_exceeds_bar_point"] and beats[n]["all_seed_ci_low_above_bar_ci_high"]
        for n in beats
    ) if beats else False

    meta = {
        "best_hyperparams": best_hp,
        "best_val_ndcg@10": best_val_score,
        "best_epoch": best_epoch,
        "early_stopping": (
            f"Refit used fixed epoch count = best validation epoch ({best_epoch})."
        ),
        "seeds": seeds,
        "across_seeds": across,
        "per_seed": [
            {
                "seed": s["seed"],
                "ndcg@10": s["metrics"]["ndcg@10"],
                "ndcg@10_ci": s["metrics"]["confidence_intervals"]["ndcg@10"],
                "recall@100": s["metrics"].get("recall@100"),
                "recall@200": s["metrics"].get("recall@200"),
                "train_wall_time_sec": s["train"]["wall_time_sec"],
            }
            for s in per_seed
        ],
        "gate": {
            "primary_metric": "ndcg@10",
            "bars": bars,
            "beats": beats,
            "beats_both_item_item_bars": beats_both,
            "negative_result": not beats_both,
        },
        "tuning_json": f"results/tuning/two_tower_{dataset}.json",
    }
    if primary_rec is None:
        raise RuntimeError("two-tower primary-seed model was not retained")
    return meta, primary_rec


def run_global_cutoff(
    ratings,
    *,
    movies: pd.DataFrame,
    clean_stats: dict[str, Any],
    config: RunConfig,
    tuned_hps: dict[str, dict[str, Any]],
    default_als: dict[str, Any],
    default_knn: dict[str, Any],
    two_tower_hp: dict[str, Any] | None = None,
    two_tower_epochs: int | None = None,
    ranker_later: dict[str, Any] | None = None,
) -> Path:
    """Secondary global-time-cutoff evaluation (does not re-tune)."""
    gc_cfg = GlobalCutoffConfig(
        timestamp_quantile=config.global_cutoff.timestamp_quantile,
        min_train_ratings=config.global_cutoff.min_train_ratings,
        relevance_threshold=config.eval.relevance_threshold,
    )
    gc = global_time_cutoff_split(ratings, gc_cfg)
    eval_split = gc.as_split_result()

    als_hp = tuned_hps.get("als_tuned", default_als)
    knn_hp = tuned_hps.get("item_item_cosine_tuned", default_knn)
    used_tuned = bool(tuned_hps)
    eval_ks = _eval_ks(config)

    model_specs: list[tuple[str, dict[str, Any], bool]] = [
        ("most_popular", {}, False),
        ("item_item_cosine", knn_hp, used_tuned),
        ("als", als_hp, used_tuned),
    ]
    if "ease" in tuned_hps:
        model_specs.append(("ease", tuned_hps["ease"], True))
    if "rp3beta" in tuned_hps:
        model_specs.append(("rp3beta", tuned_hps["rp3beta"], True))

    results: dict[str, dict] = {}
    hyperparams: dict[str, dict] = {}
    fitted_gc: dict[str, Any] = {}
    for name, hp, _is_tuned in model_specs:
        print(f"[global_cutoff] Fitting {name}...", flush=True)
        model, recorded = build_model(
            name,
            gc.train,
            hyperparams=hp,
            relevance_threshold=config.eval.relevance_threshold,
            seed=config.seed,
        )
        fitted_gc[name] = model
        hyperparams[name] = recorded
        results[name] = _eval_model(
            model,
            gc.train,
            gc.test,
            config=config,
            split=eval_split,
            include_segments=False,
            ks=eval_ks,
        )
        if eval_split.cold_start is not None:
            gc.cold_start = eval_split.cold_start

    if two_tower_hp is not None and two_tower_epochs is not None and _torch_available():
        from movielens_recommender.two_tower.train import fit_two_tower_recommender

        print(
            f"[global_cutoff] Fitting two_tower "
            f"(not re-tuned; epochs={two_tower_epochs})...",
            flush=True,
        )
        set_seeds(config.seed)
        rec, _feat, _tr = fit_two_tower_recommender(
            gc.train,
            dataset=config.dataset,
            data_dir=config.data_dir,
            movies=movies,
            hyperparams=two_tower_hp,
            seed=config.seed,
            relevance_threshold=config.eval.relevance_threshold,
            n_epochs=two_tower_epochs,
            val_split=None,
            show_progress=True,
        )
        fitted_gc["two_tower"] = rec
        hyperparams["two_tower"] = {
            **dict(two_tower_hp),
            "refit_epochs": two_tower_epochs,
            "retuned": False,
        }
        results["two_tower"] = _eval_model(
            rec,
            gc.train,
            gc.test,
            config=config,
            split=eval_split,
            include_segments=False,
            ks=eval_ks,
        )

    if (
        ranker_later is not None
        and "item_item_cosine" in fitted_gc
        and "two_tower" in fitted_gc
        and _lightgbm_available()
    ):
        from movielens_recommender.ranker.pipeline import evaluate_ranker_later_window

        later_metrics = evaluate_ranker_later_window(
            train=gc.train,
            test=gc.test,
            movies=movies,
            config=config,
            tuned_item_hp=ranker_later["tuned_item_hp"],
            two_tower_hp=ranker_later["two_tower_hp"],
            two_tower_epochs=int(ranker_later["two_tower_epochs"]),
            full_train_item_model=fitted_gc["item_item_cosine"],
            full_train_two_tower=fitted_gc["two_tower"],
            winner=str(ranker_later["winner"]),
            best_iteration=int(ranker_later["best_iteration"]),
            split=eval_split,
        )
        results.update(later_metrics)
        hyperparams["lambdarank"] = {
            "candidate_set": ranker_later["winner"],
            "num_boost_round": int(ranker_later["best_iteration"]),
            "early_stopping": False,
            "retuned": False,
        }
        hyperparams["no_ranker"] = {
            "candidate_set": ranker_later["winner"],
            "retuned": False,
        }

    cold = results[next(iter(results))].get("cold_start", {})
    surviving = {
        "n_train_users": int(gc.train["user_id"].nunique()),
        "n_train_items": int(gc.train["item_id"].nunique()),
        "n_test_interactions": int(len(gc.test)),
        "n_eval_users": int(cold.get("n_eval_users", 0)),
        "n_users_kept_after_min_train": gc.n_users_kept,
        "n_test_interactions_after_user_filter": gc.n_test_interactions_kept,
    }

    hp_source = (
        "Per-user-protocol tuned configs (validation NDCG@10); not re-tuned "
        "for the global cutoff. most_popular has no hyperparameters."
        if used_tuned
        else "S2 YAML defaults (no tuning run)."
    )
    if two_tower_hp is not None:
        hp_source += (
            " two_tower uses the per-user-protocol chosen hyperparameters and "
            "early-stopping epoch; not re-tuned on the global cutoff."
        )
    if "ease" in tuned_hps or "rp3beta" in tuned_hps:
        hp_source += (
            " ease and rp3beta reuse the per-user-protocol validation-chosen "
            "hyperparameters; not re-tuned on the global cutoff."
        )
    if ranker_later is not None:
        hp_source += (
            " lambdarank reuses the per-user candidate set and primary-seed "
            "best_iteration as a fixed num_boost_round. Labels are the "
            "chronological tail of the pre-cutoff train; features for that "
            "fit come from the head only. Retrievers and features are rebuilt "
            "on the full pre-cutoff train before scoring. Post-cutoff labels "
            "are not used. Not re-tuned."
        )

    payload = {
        "dataset": config.dataset,
        "dataset_version": DATASET_VERSION_LABELS[config.dataset],
        "dataset_sha256": DATASET_SHA256[config.dataset],
        "protocol": "global_time_cutoff",
        "secondary": True,
        "retuned": False,
        "hyperparams_source": hp_source,
        "cleaning": clean_stats,
        "split": gc.summary(),
        "surviving": surviving,
        "relevance_threshold": config.eval.relevance_threshold,
        "ks": list(config.eval.ks),
        "retrieval_ks": list(config.eval.retrieval_ks),
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
    if args.no_two_tower:
        config.models.two_tower.enabled = False
    if args.no_ranker:
        config.models.ranker.enabled = False

    out = run_pipeline(
        config,
        download=not args.no_download,
        reuse_baseline_tuning=args.reuse_baseline_tuning,
        reuse_two_tower_tuning=args.reuse_two_tower_tuning,
    )
    print(f"Wrote results to {out}")
    return 0


def _cmd_demographics(args: argparse.Namespace) -> int:
    """S4b ranker experiment. Refuses anything other than ml-1m."""
    config = load_config(args.config)
    if args.data_dir is not None:
        config.data_dir = args.data_dir
    if args.results_dir is not None:
        config.results_dir = args.results_dir
    if config.dataset != "ml-1m":
        raise ValueError(
            "demographics is ml-1m only. ml-latest-small has no users.dat "
            "and its results must not change."
        )
    from movielens_recommender.ranker.demographics import run_demographic_experiment

    out = run_demographic_experiment(config, download=not args.no_download)
    print(f"Wrote demographic experiment to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="movielens-recommender",
        description="MovieLens recommender: download, split, baselines, two-tower, evaluate.",
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
    p_run.add_argument(
        "--reuse-baseline-tuning",
        action="store_true",
        help=(
            "Reuse results/tuning/{dataset}.json for ALS/item-knn/EASE/RP3beta "
            "when those blocks are present; still tune two-tower"
        ),
    )
    p_run.add_argument(
        "--no-two-tower",
        action="store_true",
        help="Skip the optional two-tower stage even if enabled in YAML",
    )
    p_run.add_argument(
        "--reuse-two-tower-tuning",
        action="store_true",
        help="Reuse results/tuning/two_tower_{dataset}.json; still refit and evaluate",
    )
    p_run.add_argument(
        "--no-ranker",
        action="store_true",
        help="Skip the optional LambdaRank stage even if enabled in YAML",
    )
    p_run.set_defaults(func=_cmd_run)

    p_demo = sub.add_parser(
        "demographics",
        help=(
            "S4b user-demographic ranker experiment "
            "(ml-1m only; does not rewrite results/ml-1m.json)"
        ),
    )
    p_demo.add_argument(
        "--config",
        default="configs/ml-1m.yaml",
        help="Path to the ml-1m YAML config",
    )
    p_demo.add_argument("--data-dir", default=None, help="Override data_dir from config")
    p_demo.add_argument("--results-dir", default=None, help="Override results_dir from config")
    p_demo.add_argument(
        "--no-download",
        action="store_true",
        help="Do not download; require ml-1m already on disk",
    )
    p_demo.set_defaults(func=_cmd_demographics)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
