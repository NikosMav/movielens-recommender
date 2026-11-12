#!/usr/bin/env python3
"""Evaluate EASE-DAN as a 3rd retriever in LambdaRank on ml-1m (ADR-0031).

Compares 2-retriever LambdaRank (item-item + two-tower) against 3-retriever
LambdaRank (item-item + two-tower + EASE-DAN) under both demographics='off'
(S4) and demographics='both' (S4b) across 3 ranker seeds (42, 43, 44).
Does not modify results/ml-1m.json or results/demographics/ml-1m.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.baselines.ease_dan import EASEDANRecommender
from movielens_recommender.cli import host_info, library_versions
from movielens_recommender.config import load_config
from movielens_recommender.data import DATASET_SHA256, download_dataset, load_ratings, load_users
from movielens_recommender.metrics import ndcg_at_k, paired_bootstrap_delta_ci
from movielens_recommender.movies import load_movies, movies_path
from movielens_recommender.ranker.candidates import (
    balanced_three_way_union,
    balanced_union,
    dedupe_ids,
    retrieval_recall,
    score_rank_maps,
    unbalanced_three_way_union,
)
from movielens_recommender.ranker.features import (
    GROUP_PRIOR_STRENGTH,
    FeatureContext,
    assert_context_uses_only,
    assert_group_affinity_uses_only,
    build_feature_context,
    build_feature_matrix,
    categorical_feature_names,
    feature_names,
    relevance_pairs,
    tri_ranker_feature_names,
)
from movielens_recommender.ranker.pipeline import (
    _evaluate_lists,
    _fit_item_item,
    _fit_two_tower,
    _ranker_kwargs,
    _relevant_users,
    _round,
    _seed_summary,
    _with_columns,
)
from movielens_recommender.ranker.train import (
    RankerMatrices,
    assemble_matrices,
    feature_importance_gain,
    train_with_early_stop_then_refit,
)
from movielens_recommender.scale import release_memory
from movielens_recommender.split import (
    SplitConfig,
    SplitResult,
    apply_cold_start_policy,
    time_based_split,
)

ROOT = Path(__file__).resolve().parents[1]
HEADLINE_PATH = ROOT / "results" / "ml-1m.json"
EASE_DAN_PATH = ROOT / "results" / "ease-dan" / "ml-1m.json"
RESULTS_PATH = ROOT / "results" / "ranker-ease-dan" / "ml-1m.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_data_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    default_dir = ROOT / "data"
    if movies_path("ml-1m", default_dir).is_file():
        return default_dir
    space_dir = ROOT / "space" / ".runtime" / "data"
    if movies_path("ml-1m", space_dir).is_file():
        return space_dir
    download_dataset("ml-1m", default_dir)
    return default_dir


def _build_lists(
    name: str,
    ii: dict[int, list[tuple[int, float]]],
    tt: dict[int, list[tuple[int, float]]],
    ed: dict[int, list[tuple[int, float]]],
    k: int,
) -> dict[int, list[int]]:
    users = set(ii) | set(tt) | set(ed)
    out: dict[int, list[int]] = {}
    for uid in users:
        a = [int(i) for i, _s in ii.get(uid, ())]
        b = [int(i) for i, _s in tt.get(uid, ())]
        c = [int(i) for i, _s in ed.get(uid, ())]
        if name == "two_tower":
            out[int(uid)] = dedupe_ids(b)[:k]
        elif name == "union_balanced":
            out[int(uid)] = balanced_union(a, b, k)
        elif name == "ease_dan":
            out[int(uid)] = dedupe_ids(c)[:k]
        elif name == "tri_union_balanced":
            out[int(uid)] = balanced_three_way_union(a, b, c, k)
        elif name == "tri_union_unbalanced":
            out[int(uid)] = unbalanced_three_way_union(a, b, c, k)
        else:
            raise ValueError(f"unknown candidate set {name!r}")
    return out


def _blocks_tri(
    ctx: FeatureContext,
    lists: dict[int, list[int]],
    ii_maps: dict[int, dict[int, tuple[float, int]]],
    tt_maps: dict[int, dict[int, tuple[float, int]]],
    ed_maps: dict[int, dict[int, tuple[float, int]]] | None,
    positives: set[tuple[int, int]],
    users: set[int],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    blocks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for uid in sorted(lists):
        if int(uid) not in users:
            continue
        items = [int(i) for i in lists[uid]]
        if len(items) < 2:
            continue
        matrix = build_feature_matrix(
            ctx,
            int(uid),
            items,
            ii_maps.get(int(uid), {}),
            tt_maps.get(int(uid), {}),
            ease_dan=None if ed_maps is None else ed_maps.get(int(uid), {}),
        )
        labels = np.asarray(
            [1.0 if (int(uid), item) in positives else 0.0 for item in items],
            dtype=np.float64,
        )
        blocks.append((int(uid), matrix, labels))
    return blocks


def _rerank_tri(
    booster: Any,
    ctx: FeatureContext,
    lists: dict[int, list[int]],
    ii_maps: dict[int, dict[int, tuple[float, int]]],
    tt_maps: dict[int, dict[int, tuple[float, int]]],
    ed_maps: dict[int, dict[int, tuple[float, int]]] | None,
    wanted_names: list[str],
) -> dict[int, list[int]]:
    col_idx = None if wanted_names == ctx.names else [ctx.names.index(n) for n in wanted_names]
    blocks: list[np.ndarray] = []
    owners: list[tuple[int, list[int]]] = []
    ranked: dict[int, list[int]] = {}
    for uid in sorted(lists):
        items = [int(i) for i in lists[uid]]
        if not items:
            ranked[int(uid)] = []
            continue
        matrix = build_feature_matrix(
            ctx,
            int(uid),
            items,
            ii_maps.get(int(uid), {}),
            tt_maps.get(int(uid), {}),
            ease_dan=None if ed_maps is None else ed_maps.get(int(uid), {}),
        )
        if col_idx is not None:
            matrix = matrix[:, col_idx]
        blocks.append(matrix)
        owners.append((int(uid), items))
    if not blocks:
        return ranked
    scores = np.asarray(booster.predict(np.vstack(blocks)), dtype=np.float64)
    offset = 0
    for uid, items in owners:
        n = len(items)
        order = np.argsort(-scores[offset : offset + n], kind="mergesort")
        ranked[uid] = [items[int(i)] for i in order]
        offset += n
    return ranked


def _aligned_ndcg(
    ranked: dict[int, list[int]],
    seen: dict[int, set[int]],
    relevant: dict[int, set[int]],
) -> tuple[list[int], list[float]]:
    uids: list[int] = []
    vals: list[float] = []
    for uid, rel in relevant.items():
        hide = seen.get(int(uid), set())
        recs = [int(i) for i in ranked.get(int(uid), []) if int(i) not in hide]
        uids.append(int(uid))
        vals.append(float(ndcg_at_k(recs, rel, 10)))
    return uids, vals


def _round_pair(raw: dict[str, Any]) -> dict[str, Any]:
    low = round(float(raw["low"]), 6)
    high = round(float(raw["high"]), 6)
    return {
        "excludes_zero": bool(low > 0.0 or high < 0.0),
        "high": high,
        "low": low,
        "mean": round(float(raw["mean"]), 6),
        "n_users": int(raw["n_users"]),
    }


def run_experiment(data_dir: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    headline_sha = _sha256(HEADLINE_PATH)
    dan_sha = _sha256(EASE_DAN_PATH)
    headline = _read(HEADLINE_PATH)
    dan_doc = _read(EASE_DAN_PATH)

    config = load_config(ROOT / "configs" / "ml-1m.yaml")
    config.data_dir = str(data_dir)
    ratings, _clean = load_ratings("ml-1m", data_dir, clean=True)
    movies = load_movies("ml-1m", data_dir)
    users = load_users("ml-1m", data_dir)
    split = time_based_split(
        ratings,
        SplitConfig(
            min_ratings=config.split.min_ratings,
            test_fraction=config.split.test_fraction,
            val_fraction=config.split.val_fraction,
            relevance_threshold=config.eval.relevance_threshold,
        ),
    )
    assert split.val is not None and not split.val.empty

    k = int(config.models.ranker.candidate_k)
    threshold = float(config.eval.relevance_threshold)
    primary_seed = int(config.seed)
    seeds = [int(s) for s in config.models.ranker.seeds]

    ii_hp = dict(headline["hyperparameters"]["item_item_cosine_tuned"])
    tt_hp = dict(headline["two_tower"]["best_hyperparams"])
    tt_epochs = int(headline["two_tower"]["best_epoch"])
    ed_hp = dict(dan_doc["selected"]["hyperparams"])

    print("Fitting fit-train retrievers (item-item, two-tower, EASE-DAN)...", flush=True)
    ii_fit = _fit_item_item(split.train, ii_hp)
    tt_fit, _f1, _tr1 = _fit_two_tower(
        split.train,
        config=config,
        movies=movies,
        hyperparams=tt_hp,
        n_epochs=tt_epochs,
        seed=primary_seed,
    )
    ed_fit = EASEDANRecommender(
        l2=float(ed_hp["l2"]),
        dropout_p=float(ed_hp["dropout_p"]),
        alpha=float(ed_hp["alpha"]),
        beta=float(ed_hp["beta"]),
    ).fit(split.train)

    ii_val = ii_fit.topk_with_scores(k)
    tt_val = tt_fit.topk_with_scores(k)
    ed_val = ed_fit.topk_with_scores(k)
    del ii_fit, tt_fit, ed_fit, _f1, _tr1
    release_memory()

    val_rel = _relevant_users(split.train, split.val, relevance_threshold=threshold)
    cand_names = (
        "two_tower",
        "union_balanced",
        "ease_dan",
        "tri_union_balanced",
        "tri_union_unbalanced",
    )
    val_lists: dict[str, dict[int, list[int]]] = {}
    val_reports: dict[str, dict[str, float]] = {}
    for cname in cand_names:
        lists = _build_lists(cname, ii_val, tt_val, ed_val, k)
        val_lists[cname] = lists
        rep = retrieval_recall(lists, val_rel, ks=(100, 200))
        val_reports[cname] = {rk: _round(rv) for rk, rv in rep.items()}
        print(f"  val candidate {cname}: {val_reports[cname]}", flush=True)

    # Pre-registered K=200 budget winner among 3-retriever-aware sets vs two_tower
    budget_sets = ("tri_union_balanced", "two_tower", "union_balanced")
    tri_winner = min(
        budget_sets,
        key=lambda c: (
            -val_reports[c]["recall@200"],
            -val_reports[c]["recall@100"],
            budget_sets.index(c),
        ),
    )
    print(f"Selected K=200 candidate set for tri-ranker: {tri_winner}", flush=True)

    # Also evaluate tri_union_unbalanced so we see whether full 3-way union lifts further
    print("Fitting full-train retrievers for test scoring...", flush=True)
    ii_full_m = _fit_item_item(split.full_train, ii_hp)
    tt_full_m, _f2, _tr2 = _fit_two_tower(
        split.full_train,
        config=config,
        movies=movies,
        hyperparams=tt_hp,
        n_epochs=tt_epochs,
        seed=primary_seed,
    )
    ed_full_m = EASEDANRecommender(
        l2=float(ed_hp["l2"]),
        dropout_p=float(ed_hp["dropout_p"]),
        alpha=float(ed_hp["alpha"]),
        beta=float(ed_hp["beta"]),
    ).fit(split.full_train)

    ii_full = ii_full_m.topk_with_scores(k)
    tt_full = tt_full_m.topk_with_scores(k)
    ed_full = ed_full_m.topk_with_scores(k)
    del ii_full_m, tt_full_m, ed_full_m, _f2, _tr2
    release_memory()

    ctx_fit_bi = build_feature_context(
        split.train,
        movies,
        users=users,
        mode="both",
        group_frame=split.train,
        relevance_threshold=threshold,
        prior_strength=GROUP_PRIOR_STRENGTH,
    )
    assert_context_uses_only(ctx_fit_bi, split.train, split.val)
    assert_group_affinity_uses_only(
        ctx_fit_bi,
        split.train,
        users,
        pd.concat([split.val, split.test], ignore_index=True),
    )
    ctx_fit_tri = build_feature_context(
        split.train,
        movies,
        users=users,
        mode="both",
        group_frame=split.train,
        relevance_threshold=threshold,
        prior_strength=GROUP_PRIOR_STRENGTH,
    )
    ctx_fit_tri.names = tri_ranker_feature_names("both")

    ctx_full_bi = build_feature_context(
        split.full_train,
        movies,
        users=users,
        mode="both",
        group_frame=split.train,
        relevance_threshold=threshold,
        prior_strength=GROUP_PRIOR_STRENGTH,
    )
    ctx_full_tri = build_feature_context(
        split.full_train,
        movies,
        users=users,
        mode="both",
        group_frame=split.train,
        relevance_threshold=threshold,
        prior_strength=GROUP_PRIOR_STRENGTH,
    )
    ctx_full_tri.names = tri_ranker_feature_names("both")

    ii_maps_val = score_rank_maps(ii_val)
    tt_maps_val = score_rank_maps(tt_val)
    ed_maps_val = score_rank_maps(ed_val)
    ii_maps_full = score_rank_maps(ii_full)
    tt_maps_full = score_rank_maps(tt_full)
    ed_maps_full = score_rank_maps(ed_full)

    positives = relevance_pairs(split.val, relevance_threshold=threshold)
    label_users = {int(u) for u in split.val["user_id"].unique()}

    full_lists_by_set = {
        cname: _build_lists(cname, ii_full, tt_full, ed_full, k)
        for cname in ("two_tower", tri_winner, "tri_union_unbalanced")
    }
    eval_split = SplitResult(
        train=split.full_train,
        test=split.test,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    seen, test_rel, _stats = apply_cold_start_policy(
        eval_split, relevance_threshold=threshold
    )

    # Build base RankerMatrices for bi (two_tower) and tri (tri_winner & tri_union_unbalanced)
    bi_blocks = _blocks_tri(
        ctx_fit_bi,
        val_lists["two_tower"],
        ii_maps_val,
        tt_maps_val,
        None,
        positives,
        label_users,
    )
    bi_mat_both = assemble_matrices(bi_blocks, ctx_fit_bi.names)
    bi_mat_both.categorical_features = categorical_feature_names("both")

    tri_blocks = _blocks_tri(
        ctx_fit_tri,
        val_lists[tri_winner],
        ii_maps_val,
        tt_maps_val,
        ed_maps_val,
        positives,
        label_users,
    )
    tri_mat_both = assemble_matrices(tri_blocks, ctx_fit_tri.names)
    tri_mat_both.categorical_features = categorical_feature_names("both")

    tri_unb_blocks = _blocks_tri(
        ctx_fit_tri,
        val_lists["tri_union_unbalanced"],
        ii_maps_val,
        tt_maps_val,
        ed_maps_val,
        positives,
        label_users,
    )
    tri_unb_mat_both = assemble_matrices(tri_unb_blocks, ctx_fit_tri.names)
    tri_unb_mat_both.categorical_features = categorical_feature_names("both")

    specs = [
        ("bi_off", bi_mat_both, feature_names("off"), ctx_full_bi, "two_tower", False),
        ("tri_off", tri_mat_both, tri_ranker_feature_names("off"), ctx_full_tri, tri_winner, True),
        ("bi_both", bi_mat_both, feature_names("both"), ctx_full_bi, "two_tower", False),
        (
            "tri_both",
            tri_mat_both,
            tri_ranker_feature_names("both"),
            ctx_full_tri,
            tri_winner,
            True,
        ),
        (
            "tri_unbalanced_both",
            tri_unb_mat_both,
            tri_ranker_feature_names("both"),
            ctx_full_tri,
            "tri_union_unbalanced",
            True,
        ),
    ]

    variants_out: dict[str, Any] = {}
    per_user_ndcg: dict[str, list[float]] = {}
    ref_uids: list[int] | None = None

    for vname, base_mat, wanted_cols, ctx_full, cset_name, use_ed in specs:
        sub_mat: RankerMatrices = _with_columns(base_mat, wanted_cols)
        print(
            f"Training {vname} on {cset_name} ({len(wanted_cols)} features)...",
            flush=True,
        )
        per_seed: list[dict[str, Any]] = []
        primary_metrics: dict[str, Any] | None = None
        primary_gains: list[dict[str, Any]] = []
        for s in seeds:
            trained = train_with_early_stop_then_refit(
                sub_mat,
                split_seed=primary_seed,
                early_stop_fraction=float(config.models.ranker.early_stop_fraction),
                ranker_seed=s,
                num_boost_round=int(config.models.ranker.num_boost_round),
                early_stopping_rounds=int(config.models.ranker.early_stopping_rounds),
                **_ranker_kwargs(config),
            )
            ranked = _rerank_tri(
                trained.booster,
                ctx_full,
                full_lists_by_set[cset_name],
                ii_maps_full,
                tt_maps_full,
                ed_maps_full if use_ed else None,
                list(trained.refit_info["feature_names"]),
            )
            metrics = _evaluate_lists(
                ranked,
                split.full_train,
                split.test,
                config=config,
                split=eval_split,
                include_segments=s == primary_seed,
            )
            per_seed.append(
                {
                    "best_iteration": int(trained.best_iteration),
                    "coverage@10": metrics["coverage@10"],
                    "metrics": metrics,
                    "ndcg@10": metrics["ndcg@10"],
                    "ndcg@10_ci": metrics["confidence_intervals"]["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "seed": int(s),
                }
            )
            if s == primary_seed:
                primary_metrics = metrics
                primary_gains = feature_importance_gain(
                    trained.booster, list(trained.refit_info["feature_names"])
                )[:12]
                uids, vals = _aligned_ndcg(ranked, seen, test_rel)
                if ref_uids is None:
                    ref_uids = uids
                elif uids != ref_uids:
                    raise RuntimeError("user alignment mismatch")
                per_user_ndcg[vname] = vals
        assert primary_metrics is not None
        across = _seed_summary(
            per_seed,
            ["ndcg@10", "ndcg@20", "recall@10", "recall@100", "recall@200", "coverage@10"],
        )
        tail = (
            (primary_metrics.get("segments") or {})
            .get("item_head_tail", {})
            .get("tail", {})
        )
        variants_out[vname] = {
            "across_seeds": across,
            "candidate_set": cset_name,
            "n_features": len(wanted_cols),
            "ndcg@10_mean": across["ndcg@10"]["mean"],
            "ndcg@10_std": across["ndcg@10"]["std"],
            "per_seed": [
                {
                    "best_iteration": r["best_iteration"],
                    "coverage@10": r["coverage@10"],
                    "ndcg@10": r["ndcg@10"],
                    "ndcg@10_ci": r["ndcg@10_ci"],
                    "recall@10": r["recall@10"],
                    "seed": r["seed"],
                }
                for r in per_seed
            ],
            "primary_ndcg@10_ci": primary_metrics["confidence_intervals"]["ndcg@10"],
            "recall@10_mean": across["recall@10"]["mean"],
            "coverage@10_mean": across["coverage@10"]["mean"],
            "tail_ndcg@10": tail.get("ndcg@10"),
            "tail_ndcg@10_ci": (tail.get("confidence_intervals") or {}).get("ndcg@10"),
            "top_feature_gains": primary_gains,
        }
        print(
            f"  {vname}: NDCG@10 mean={across['ndcg@10']['mean']} "
            f"±{across['ndcg@10']['std']} cov={across['coverage@10']['mean']}",
            flush=True,
        )

    paired_off = _round_pair(
        paired_bootstrap_delta_ci(
            per_user_ndcg["bi_off"],
            per_user_ndcg["tri_off"],
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=primary_seed,
        )
    )
    paired_both = _round_pair(
        paired_bootstrap_delta_ci(
            per_user_ndcg["bi_both"],
            per_user_ndcg["tri_both"],
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=primary_seed,
        )
    )
    paired_unb_both = _round_pair(
        paired_bootstrap_delta_ci(
            per_user_ndcg["bi_both"],
            per_user_ndcg["tri_unbalanced_both"],
            n_bootstrap=config.eval.n_bootstrap,
            alpha=config.eval.bootstrap_alpha,
            seed=primary_seed,
        )
    )

    payload = {
        "adr": "0031",
        "candidate_k": k,
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "decision": {
            "rule": (
                "lifts_baseline when 3-seed mean test NDCG@10 exceeds the "
                "2-retriever baseline and primary-seed paired 95% CI low > 0"
            ),
            "tri_both_lifts_bi_both": bool(
                variants_out["tri_both"]["ndcg@10_mean"]
                > variants_out["bi_both"]["ndcg@10_mean"]
                and paired_both["low"] > 0.0
            ),
            "tri_off_lifts_bi_off": bool(
                variants_out["tri_off"]["ndcg@10_mean"]
                > variants_out["bi_off"]["ndcg@10_mean"]
                and paired_off["low"] > 0.0
            ),
            "tri_unbalanced_both_lifts_bi_both": bool(
                variants_out["tri_unbalanced_both"]["ndcg@10_mean"]
                > variants_out["bi_both"]["ndcg@10_mean"]
                and paired_unb_both["low"] > 0.0
            ),
        },
        "ease_dan_hyperparams": ed_hp,
        "host": host_info(),
        "library_versions": library_versions(),
        "paired": {
            "tri_both_minus_bi_both": paired_both,
            "tri_off_minus_bi_off": paired_off,
            "tri_unbalanced_both_minus_bi_both": paired_unb_both,
        },
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "seeds": seeds,
        "selected_tri_candidate_set": tri_winner,
        "split": split.summary(),
        "validation_candidates": val_reports,
        "variants": variants_out,
    }
    _write(RESULTS_PATH, payload)
    if _sha256(HEADLINE_PATH) != headline_sha or _sha256(EASE_DAN_PATH) != dan_sha:
        raise RuntimeError("committed results files changed during ranker-ease-dan run")
    print(f"Wrote {RESULTS_PATH}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0031 EASE-DAN 3rd retriever in LambdaRank.")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    run_experiment(_resolve_data_dir(args.data_dir))


if __name__ == "__main__":
    main()
