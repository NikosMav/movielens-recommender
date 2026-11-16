#!/usr/bin/env python3
"""Pairwise transitions as a retriever and ranker feature on ml-1m (ADR-0033).

Compares the served ranker (two-tower candidates, demographics on) with the
PCTM list on its own and with LambdaRank over a validation-chosen
sequence-aware candidate set, across ranker seeds 42, 43, and 44. Applies the
pre-registered serving rule. Does not modify any other results file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.cli import host_info, library_versions
from movielens_recommender.config import load_config
from movielens_recommender.data import DATASET_SHA256, load_ratings, load_users
from movielens_recommender.metrics import paired_bootstrap_delta_ci
from movielens_recommender.movies import load_movies
from movielens_recommender.ranker.candidates import (
    SEQUENCE_CANDIDATE_SETS,
    retrieval_recall,
    score_rank_maps,
    sequence_candidate_ids,
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
    pctm_ranker_feature_names,
    relevance_pairs,
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
from movielens_recommender.sasrec.pctm import PCTMRecommender
from movielens_recommender.scale import release_memory
from movielens_recommender.split import (
    SplitConfig,
    SplitResult,
    apply_cold_start_policy,
    time_based_split,
)
from run_ranker_ease_dan import (
    _aligned_ndcg,
    _read,
    _resolve_data_dir,
    _round_pair,
    _sha256,
    _write,
)

ROOT = Path(__file__).resolve().parents[1]
HEADLINE_PATH = ROOT / "results" / "ml-1m.json"
SEQUENCE_PATH = ROOT / "results" / "sequence-modern" / "ml-1m.json"
RESULTS_PATH = ROOT / "results" / "ranker-pctm" / "ml-1m.json"
MODEL_DIR = ROOT / "models" / "ml-1m" / "ranker-pctm"

SEQUENCE_SETS = SEQUENCE_CANDIDATE_SETS
PAIRS = (
    ("pctm_only", "bi_both"),
    ("seq_off", "pctm_only"),
    ("seq_both", "pctm_only"),
    ("seq_both", "seq_off"),
    ("seq_off", "bi_both"),
    ("seq_both", "bi_both"),
)

Scored = dict[int, list[tuple[int, float]]]
RankMaps = dict[int, dict[int, tuple[float, int]]]


def build_lists(name: str, ii: Scored, tt: Scored, pc: Scored, k: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for uid in set(ii) | set(tt) | set(pc):
        out[int(uid)] = sequence_candidate_ids(
            name,
            [int(i) for i, _s in ii.get(uid, ())],
            [int(i) for i, _s in tt.get(uid, ())],
            [int(i) for i, _s in pc.get(uid, ())],
            k,
        )
    return out


def choose_sequence_set(reports: dict[str, dict[str, float]]) -> str:
    """Validation Recall@200, then Recall@100, then the ADR-0033 order."""
    return min(
        SEQUENCE_SETS,
        key=lambda name: (
            -float(reports[name]["recall@200"]),
            -float(reports[name]["recall@100"]),
            SEQUENCE_SETS.index(name),
        ),
    )


def serving_choice(wins: dict[str, bool]) -> dict[str, str]:
    """The ADR-0033 serving rule, from the stored paired-win flags."""
    ranker = "seq_both" if wins["seq_both_minus_seq_off"] else "seq_off"
    if wins[f"{ranker}_minus_pctm_only"]:
        return {"ranker_candidate": ranker, "serve": ranker, "serving_rule": "2"}
    if wins["pctm_only_minus_bi_both"]:
        return {"ranker_candidate": ranker, "serve": "pctm_only", "serving_rule": "3"}
    return {"ranker_candidate": ranker, "serve": "bi_both", "serving_rule": "4"}


def _matrix(
    ctx: FeatureContext,
    uid: int,
    items: list[int],
    ii_maps: RankMaps,
    tt_maps: RankMaps,
    pc_maps: RankMaps | None,
) -> np.ndarray:
    return build_feature_matrix(
        ctx,
        uid,
        items,
        ii_maps.get(uid, {}),
        tt_maps.get(uid, {}),
        pctm=None if pc_maps is None else pc_maps.get(uid, {}),
    )


def _blocks(
    ctx: FeatureContext,
    lists: dict[int, list[int]],
    ii_maps: RankMaps,
    tt_maps: RankMaps,
    pc_maps: RankMaps | None,
    positives: set[tuple[int, int]],
    users: set[int],
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    blocks: list[tuple[int, np.ndarray, np.ndarray]] = []
    for uid in sorted(lists):
        items = [int(i) for i in lists[uid]]
        if int(uid) not in users or len(items) < 2:
            continue
        labels = np.asarray(
            [1.0 if (int(uid), item) in positives else 0.0 for item in items],
            dtype=np.float64,
        )
        blocks.append((int(uid), _matrix(ctx, int(uid), items, ii_maps, tt_maps, pc_maps), labels))
    return blocks


def _rerank(
    booster: Any,
    ctx: FeatureContext,
    lists: dict[int, list[int]],
    ii_maps: RankMaps,
    tt_maps: RankMaps,
    pc_maps: RankMaps | None,
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
        matrix = _matrix(ctx, int(uid), items, ii_maps, tt_maps, pc_maps)
        blocks.append(matrix if col_idx is None else matrix[:, col_idx])
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


def run_experiment(data_dir: Path) -> dict[str, Any]:
    t0 = time.perf_counter()
    guarded = {path: _sha256(path) for path in (HEADLINE_PATH, SEQUENCE_PATH)}
    headline = _read(HEADLINE_PATH)
    sequence = _read(SEQUENCE_PATH)

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
    pc_hp = dict(sequence["models"]["pctm"]["hyperparams"])

    def retrieve(frame: pd.DataFrame) -> tuple[Scored, Scored, Scored]:
        item_item = _fit_item_item(frame, ii_hp)
        tower, _features, _trace = _fit_two_tower(
            frame,
            config=config,
            movies=movies,
            hyperparams=tt_hp,
            n_epochs=tt_epochs,
            seed=primary_seed,
        )
        pctm = PCTMRecommender(relevance_threshold=threshold, **pc_hp).fit(frame)
        out = (
            item_item.topk_with_scores(k),
            tower.topk_with_scores(k),
            pctm.topk_with_scores(k),
        )
        del item_item, tower, pctm, _features, _trace
        release_memory()
        return out

    def context(frame: pd.DataFrame, names: list[str] | None) -> FeatureContext:
        ctx = build_feature_context(
            frame,
            movies,
            users=users,
            mode="both",
            group_frame=split.train,
            relevance_threshold=threshold,
            prior_strength=GROUP_PRIOR_STRENGTH,
        )
        if names is not None:
            ctx.names = names
        return ctx

    print("Fitting fit-train retrievers (item-item, two-tower, PCTM)...", flush=True)
    ii_val, tt_val, pc_val = retrieve(split.train)
    val_rel = _relevant_users(split.train, split.val, relevance_threshold=threshold)
    val_lists: dict[str, dict[int, list[int]]] = {}
    val_reports: dict[str, dict[str, float]] = {}
    for cname in ("two_tower", *SEQUENCE_SETS):
        val_lists[cname] = build_lists(cname, ii_val, tt_val, pc_val, k)
        report = retrieval_recall(val_lists[cname], val_rel, ks=(100, 200))
        val_reports[cname] = {key: _round(value) for key, value in report.items()}
        print(f"  val candidate {cname}: {val_reports[cname]}", flush=True)
    seq_set = choose_sequence_set(val_reports)
    print(f"Selected sequence-aware candidate set: {seq_set}", flush=True)

    print("Fitting full-train retrievers for test scoring...", flush=True)
    ii_full, tt_full, pc_full = retrieve(split.full_train)

    seq_names = pctm_ranker_feature_names("both")
    ctx_fit_bi = context(split.train, None)
    assert_context_uses_only(ctx_fit_bi, split.train, split.val)
    assert_group_affinity_uses_only(
        ctx_fit_bi,
        split.train,
        users,
        pd.concat([split.val, split.test], ignore_index=True),
    )
    ctx_fit_seq = context(split.train, seq_names)
    ctx_full_bi = context(split.full_train, None)
    ctx_full_seq = context(split.full_train, seq_names)

    val_maps = tuple(score_rank_maps(rows) for rows in (ii_val, tt_val, pc_val))
    full_maps = tuple(score_rank_maps(rows) for rows in (ii_full, tt_full, pc_full))
    positives = relevance_pairs(split.val, relevance_threshold=threshold)
    label_users = {int(u) for u in split.val["user_id"].unique()}
    full_lists = {
        cname: build_lists(cname, ii_full, tt_full, pc_full, k)
        for cname in ("two_tower", "pctm", seq_set)
    }
    eval_split = SplitResult(
        train=split.full_train,
        test=split.test,
        val=split.val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
    )
    seen, test_rel, _stats = apply_cold_start_policy(eval_split, relevance_threshold=threshold)

    bi_mat = assemble_matrices(
        _blocks(
            ctx_fit_bi,
            val_lists["two_tower"],
            val_maps[0],
            val_maps[1],
            None,
            positives,
            label_users,
        ),
        ctx_fit_bi.names,
    )
    bi_mat.categorical_features = categorical_feature_names("both")
    seq_mat = assemble_matrices(
        _blocks(
            ctx_fit_seq,
            val_lists[seq_set],
            val_maps[0],
            val_maps[1],
            val_maps[2],
            positives,
            label_users,
        ),
        ctx_fit_seq.names,
    )
    seq_mat.categorical_features = categorical_feature_names("both")

    variants: dict[str, Any] = {}
    per_user: dict[str, list[float]] = {}
    reference: list[int] = []

    def keep_aligned(name: str, ranked: dict[int, list[int]]) -> None:
        uids, values = _aligned_ndcg(ranked, seen, test_rel)
        if not reference:
            reference.extend(uids)
        elif uids != reference:
            raise RuntimeError("user alignment mismatch")
        per_user[name] = values

    pctm_metrics = _evaluate_lists(
        full_lists["pctm"],
        split.full_train,
        split.test,
        config=config,
        split=eval_split,
        include_segments=False,
    )
    keep_aligned("pctm_only", full_lists["pctm"])
    variants["pctm_only"] = {
        "candidate_set": "pctm",
        "coverage@10_mean": pctm_metrics["coverage@10"],
        "n_features": 0,
        "ndcg@10_mean": pctm_metrics["ndcg@10"],
        "ndcg@10_std": 0.0,
        "primary_ndcg@10_ci": pctm_metrics["confidence_intervals"]["ndcg@10"],
        "recall@10_mean": pctm_metrics["recall@10"],
    }
    print(f"  pctm_only: NDCG@10={pctm_metrics['ndcg@10']}", flush=True)

    specs = [
        ("bi_both", bi_mat, feature_names("both"), ctx_full_bi, "two_tower", False),
        ("seq_off", seq_mat, pctm_ranker_feature_names("off"), ctx_full_seq, seq_set, True),
        ("seq_both", seq_mat, seq_names, ctx_full_seq, seq_set, True),
    ]
    for vname, base_mat, wanted, ctx_full, cset, use_pctm in specs:
        sub_mat: RankerMatrices = _with_columns(base_mat, wanted)
        print(f"Training {vname} on {cset} ({len(wanted)} features)...", flush=True)
        per_seed: list[dict[str, Any]] = []
        primary_ci: dict[str, Any] = {}
        gains: list[dict[str, Any]] = []
        for seed in seeds:
            trained = train_with_early_stop_then_refit(
                sub_mat,
                split_seed=primary_seed,
                early_stop_fraction=float(config.models.ranker.early_stop_fraction),
                ranker_seed=seed,
                num_boost_round=int(config.models.ranker.num_boost_round),
                early_stopping_rounds=int(config.models.ranker.early_stopping_rounds),
                **_ranker_kwargs(config),
            )
            names = list(trained.refit_info["feature_names"])
            ranked = _rerank(
                trained.booster,
                ctx_full,
                full_lists[cset],
                full_maps[0],
                full_maps[1],
                full_maps[2] if use_pctm else None,
                names,
            )
            metrics = _evaluate_lists(
                ranked,
                split.full_train,
                split.test,
                config=config,
                split=eval_split,
                include_segments=False,
            )
            per_seed.append(
                {
                    "best_iteration": int(trained.best_iteration),
                    "coverage@10": metrics["coverage@10"],
                    "metrics": metrics,
                    "ndcg@10": metrics["ndcg@10"],
                    "ndcg@10_ci": metrics["confidence_intervals"]["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "seed": int(seed),
                }
            )
            if seed == primary_seed:
                primary_ci = metrics["confidence_intervals"]["ndcg@10"]
                gains = feature_importance_gain(trained.booster, names)[:12]
                keep_aligned(vname, ranked)
                # Gitignored. The serving build reads the refit round count from the JSON.
                MODEL_DIR.mkdir(parents=True, exist_ok=True)
                trained.booster.save_model(str(MODEL_DIR / f"{vname}.txt"))
        across = _seed_summary(
            per_seed,
            ["ndcg@10", "ndcg@20", "recall@10", "recall@100", "recall@200", "coverage@10"],
        )
        variants[vname] = {
            "across_seeds": across,
            "candidate_set": cset,
            "coverage@10_mean": across["coverage@10"]["mean"],
            "feature_names": wanted,
            "n_features": len(wanted),
            "ndcg@10_mean": across["ndcg@10"]["mean"],
            "ndcg@10_std": across["ndcg@10"]["std"],
            "per_seed": [
                {key: row[key] for key in row if key != "metrics"} for row in per_seed
            ],
            "primary_ndcg@10_ci": primary_ci,
            "recall@10_mean": across["recall@10"]["mean"],
            "top_feature_gains": gains,
        }
        print(
            f"  {vname}: NDCG@10 mean={across['ndcg@10']['mean']} "
            f"±{across['ndcg@10']['std']} cov={across['coverage@10']['mean']}",
            flush=True,
        )

    paired = {
        f"{right}_minus_{left}": _round_pair(
            paired_bootstrap_delta_ci(
                per_user[left],
                per_user[right],
                n_bootstrap=config.eval.n_bootstrap,
                alpha=config.eval.bootstrap_alpha,
                seed=primary_seed,
            )
        )
        for right, left in PAIRS
    }
    wins = {name: bool(block["low"] > 0.0) for name, block in paired.items()}
    choice = serving_choice(wins)

    payload = {
        "adr": "0033",
        "candidate_k": k,
        "dataset": "ml-1m",
        "dataset_sha256": DATASET_SHA256["ml-1m"],
        "decision": {
            "rule": (
                "A comparison is a win when the primary-seed paired 95% interval "
                "low end is above 0. Serving follows ADR-0033 rules 1 to 4."
            ),
            "wins": wins,
            **choice,
        },
        "host": host_info(),
        "library_versions": library_versions(),
        "paired": paired,
        "pctm_hyperparams": pc_hp,
        "runtime_sec": round(time.perf_counter() - t0, 3),
        "seeds": seeds,
        "selected_sequence_candidate_set": seq_set,
        "split": split.summary(),
        "validation_candidates": val_reports,
        "variants": variants,
    }
    _write(RESULTS_PATH, payload)
    for path, digest in guarded.items():
        if _sha256(path) != digest:
            raise RuntimeError(f"committed results file changed during the run: {path}")
    print(f"Wrote {RESULTS_PATH}", flush=True)
    print(f"Serving choice: {choice}", flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="ADR-0033 PCTM retriever and ranker feature.")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    run_experiment(_resolve_data_dir(args.data_dir))


if __name__ == "__main__":
    main()
