#!/usr/bin/env python3
"""Fairness round added after the first S3e test files existed.

ml-32m retrains the sampled-softmax winner at the published epoch budget.
ml-1m extends the three high edges on validation only. Test and the ranker
are re-scored only when that validation winner changes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from movielens_recommender.scale import release_memory
from movielens_recommender.two_tower.plan import (
    edge_extension_decision,
    grid_edges,
    matched_verdict,
)
from run_two_tower_v2 import (
    CHECKPOINT_DIR,
    ROOT,
    _compact_model,
    _fit_and_eval,
    _json_topk,
    _pair,
    _published_tuning,
    _read,
    _restore_topk,
    _run_rankers,
    _segment_value,
    _val_view,
    _write,
    load_split,
)

MATCHED_PLAN = ROOT / "results" / "budget" / "two-tower-v2-ml32m-matched.json"
MATCHED_TUNING = ROOT / "results" / "tuning" / "two_tower_v2_ml-32m-matched.json"
MATCHED_OUT = ROOT / "results" / "two-tower-v2" / "ml-32m-matched.json"
MATCHED_STATE = CHECKPOINT_DIR / "ml-32m-matched-state.json"
EDGE_PLAN = ROOT / "results" / "budget" / "two-tower-v2-ml1m-edges.json"
EDGE_TUNING = ROOT / "results" / "tuning" / "two_tower_v2_ml-1m-edges.json"
EDGE_OUT = ROOT / "results" / "two-tower-v2" / "ml-1m-edges.json"


def _plan(path: Path, status: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Missing fairness budget {path}")
    doc = _read(path)
    if doc.get("status") != status:
        raise SystemExit(f"{path} is not marked {status}")
    return doc


def _remaining(doc: dict[str, Any], cap_sec: float) -> float:
    return float(cap_sec) - (time.time() - float(doc["started_unix"]))


def _not_run(reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "not_run", "reason": reason}
    out.update(extra)
    return out


def _publish_matched(doc: dict[str, Any]) -> None:
    _write(MATCHED_TUNING, doc)
    public = {key: value for key, value in doc.items() if key != "started_unix"}
    _write(MATCHED_OUT, public)


def _load_state() -> dict[str, Any]:
    if not MATCHED_STATE.is_file():
        return {}
    return _read(MATCHED_STATE)


def _save_state(state: dict[str, Any]) -> None:
    _write(MATCHED_STATE, state)


def cmd_fairness_ml32m() -> None:
    import torch

    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    torch.set_num_threads(os.cpu_count() or 1)
    plan = _plan(MATCHED_PLAN, "matched_epoch_budget")
    hp = dict(plan["trial"])
    cap = float(plan["cap_sec"])
    doc = _read(MATCHED_TUNING) if MATCHED_TUNING.is_file() else {
        "experiment": "s3e_matched_epoch_budget",
        "dataset": "ml-32m",
        "added_after": plan["added_after"],
        "budget_source": "results/budget/two-tower-v2-ml32m-matched.json",
        "seed": 42,
        "hyperparams": hp,
        "reference_val_ndcg@10": plan["reference_val_ndcg@10"],
        "started_unix": time.time(),
        "stage": "running",
    }
    if doc.get("stage") == "complete":
        print("matched round already complete", flush=True)
        return
    if "started_unix" not in doc:
        doc["started_unix"] = time.time()
    _write(MATCHED_TUNING, doc)

    config, movies, split = load_split("ml-32m")
    published_sample = _read(ROOT / "results" / "ml-32m.json")["eval_user_sample"]
    got = (split.eval_user_sample or {}).get("user_ids_sha256")
    if got != published_sample["user_ids_sha256"]:
        raise SystemExit(
            f"eval sample hash {got} != published {published_sample['user_ids_sha256']}"
        )
    doc["eval_user_sample"] = split.eval_user_sample

    if "validation" not in doc:
        print(f"matched validation: {hp}", flush=True)
        _rec, _feat, result = fit_two_tower_recommender(
            split.train,
            dataset="ml-32m",
            data_dir=str(ROOT / "data"),
            movies=movies,
            hyperparams=hp,
            seed=42,
            relevance_threshold=config.eval.relevance_threshold,
            val_split=_val_view(split),
            show_progress=True,
            epoch_checkpoint=CHECKPOINT_DIR / "ml-32m-epochs" / "matched.pt",
        )
        score = float("-inf") if result.best_val_ndcg10 is None else float(result.best_val_ndcg10)
        reference_val = float(plan["reference_val_ndcg@10"])
        doc["validation"] = {
            "val_ndcg@10": round(score, 6),
            "best_epoch": int(result.best_epoch),
            "epochs_trained": int(result.epochs_trained),
            "wall_time_sec": result.wall_time_sec,
            "history": result.history,
            "beats_reference_validation": bool(score > reference_val),
            "truncated": int(result.best_epoch) >= int(hp["max_epochs"]),
        }
        doc["stage"] = "validation"
        _publish_matched(doc)
        del _rec, _feat, result
        release_memory()
        print(
            f"validation {doc['validation']['val_ndcg@10']} "
            f"beats={doc['validation']['beats_reference_validation']} "
            f"truncated={doc['validation']['truncated']}",
            flush=True,
        )

    validation = doc["validation"]
    beats = bool(validation["beats_reference_validation"])
    best_epoch = int(validation["best_epoch"])
    if not beats:
        doc["test"] = _not_run("Validation did not beat the published reference.")
        doc["ranker"] = _not_run("Validation did not beat the published reference.")
        doc["reference_refit"] = _not_run(
            "The matched run has no test score, so a paired interval was not computed."
        )
        doc["paired_vs_reference"] = None
        doc["verdict"] = matched_verdict(
            beats_validation=False,
            best_epoch=best_epoch,
            max_epochs=int(hp["max_epochs"]),
            test_status="not_run",
        )
        doc["stage"] = "complete"
        _finish_matched(doc, plan)
        return

    state = _load_state()
    if doc.get("test", {}).get("status") != "ran":
        projected = best_epoch * float(plan["sampled_test_epoch_sec"])
        remaining = _remaining(doc, cap)
        if remaining < projected:
            doc["test"] = _not_run(
                "Validation beat the reference, but the remaining cap does not "
                "cover the test refit.",
                remaining_sec=round(remaining, 1),
                projected_sec=round(projected, 1),
            )
            doc["ranker"] = _not_run("Test was not scored.")
            doc["reference_refit"] = _not_run("Test was not scored.")
            doc["paired_vs_reference"] = None
            doc["verdict"] = "inconclusive"
            doc["stage"] = "complete"
            _finish_matched(doc, plan)
            return
        print(f"matched test refit epochs={best_epoch}", flush=True)
        rec, metrics, per_user, wall = _fit_and_eval(
            full_train=split.full_train,
            movies=movies,
            split=split,
            config=config,
            hyperparams=hp,
            n_epochs=best_epoch,
            seed=42,
            epoch_checkpoint=CHECKPOINT_DIR / "ml-32m-epochs" / "matched-test-seed42.pt",
        )
        test_users = [int(uid) for uid in split.test["user_id"].unique()]
        state["topk"] = _json_topk(
            rec.topk_with_scores(int(config.models.ranker.candidate_k), user_ids=test_users)
        )
        state["per_user"] = per_user
        _save_state(state)
        spec = {
            "role": "candidate",
            "name": "sampled_softmax",
            "hyperparams": hp,
            "n_epochs": best_epoch,
        }
        doc["test"] = {
            "status": "ran",
            "model": _compact_model(
                [{"seed": 42, "metrics": metrics, "train_wall_sec": wall}],
                42,
                spec,
            ),
        }
        doc["stage"] = "test"
        _publish_matched(doc)
        del rec
        release_memory()

    if doc.get("ranker", {}).get("status") != "ran":
        projected = best_epoch * float(plan["sampled_val_epoch_sec"])
        projected += float(plan["item_item_extrapolation_sec"])
        remaining = _remaining(doc, cap)
        if remaining < projected:
            doc["ranker"] = _not_run(
                "The remaining cap does not cover LambdaRank.",
                remaining_sec=round(remaining, 1),
                projected_sec=round(projected, 1),
            )
        else:
            print("matched LambdaRank", flush=True)
            tuning = {
                "selected_loss": "sampled_softmax",
                "best_by_loss": {
                    "sampled_softmax": {"hyperparams": hp, "best_epoch": best_epoch}
                },
            }
            doc["ranker"] = _run_rankers(
                dataset="ml-32m",
                config=config,
                movies=movies,
                split=split,
                tuning=tuning,
                towers={"new": {"topk": _restore_topk(state["topk"])}},
                train_reference=False,
            )
        doc["stage"] = "ranker"
        _publish_matched(doc)
        release_memory()

    if doc.get("reference_refit", {}).get("status") != "ran":
        projected = int(plan["reference_epochs"]) * float(plan["in_batch_epoch_sec"])
        remaining = _remaining(doc, cap)
        if remaining < projected:
            doc["reference_refit"] = _not_run(
                "The remaining cap does not cover a reference refit, so the "
                "comparison stays unpaired.",
                remaining_sec=round(remaining, 1),
                projected_sec=round(projected, 1),
            )
            doc["paired_vs_reference"] = None
        else:
            print("matched reference refit", flush=True)
            published = _published_tuning("ml-32m")
            ref_hp = dict(published["best_hyperparams"])
            ref_hp["loss"] = "in_batch"
            _rec, ref_metrics, ref_user, ref_wall = _fit_and_eval(
                full_train=split.full_train,
                movies=movies,
                split=split,
                config=config,
                hyperparams=ref_hp,
                n_epochs=int(published["best_epoch"]),
                seed=42,
                epoch_checkpoint=CHECKPOINT_DIR / "ml-32m-epochs" / "matched-reference-seed42.pt",
            )
            del _rec
            release_memory()
            paired = _pair(
                ref_user,
                state["per_user"],
                seed=config.seed,
                n_bootstrap=config.eval.n_bootstrap,
                alpha=config.eval.bootstrap_alpha,
            )
            doc["reference_refit"] = {
                "status": "ran",
                "n_epochs": int(published["best_epoch"]),
                "wall_time_sec": ref_wall,
                "ndcg@10": ref_metrics["ndcg@10"],
                "ndcg@10_ci": ref_metrics["confidence_intervals"]["ndcg@10"],
                "head_ndcg@10": _segment_value(ref_metrics, "head"),
                "tail_ndcg@10": _segment_value(ref_metrics, "tail"),
            }
            doc["paired_vs_reference"] = paired
        doc["stage"] = "reference"

    paired = (doc.get("paired_vs_reference") or {}).get("ndcg@10") or {}
    doc["verdict"] = matched_verdict(
        beats_validation=True,
        best_epoch=best_epoch,
        max_epochs=int(hp["max_epochs"]),
        test_status=str(doc["test"]["status"]),
        paired_mean=None if not paired else float(paired["mean"]),
        paired_excludes_zero=None if not paired else bool(paired["excludes_zero"]),
    )
    doc["stage"] = "complete"
    _finish_matched(doc, plan)


def _finish_matched(doc: dict[str, Any], plan: dict[str, Any]) -> None:
    doc["compute"] = {
        "cap_sec": plan["cap_sec"],
        "elapsed_sec": round(time.time() - float(doc["started_unix"]), 3),
        "validation_wall_sec": doc["validation"]["wall_time_sec"],
        "test_train_sec": (doc.get("test") or {}).get("model", {}).get("per_seed", [{}])[0].get(
            "train_wall_time_sec"
        ),
        "ranker_wall_sec": (doc.get("ranker") or {}).get("runtime_sec"),
        "reference_refit_wall_sec": (doc.get("reference_refit") or {}).get("wall_time_sec"),
        "cpu_count": os.cpu_count(),
        "seeds": [42],
    }
    _publish_matched(doc)
    print(f"verdict {doc['verdict']}", flush=True)


def _trial_id(hp: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(hp["embedding_dim"]),
        round(float(hp["learning_rate"]), 6),
        round(float(hp["temperature"]), 6),
    )


def _fill_edge_tower_pair() -> None:
    """Paired CI of the re-scored tower against the saved reference refit.

    Training already finished. This only scores the reference checkpoint and
    pairs it with the primary-seed per-user scores already on disk.
    """
    if not EDGE_TUNING.is_file():
        return
    doc = _read(EDGE_TUNING)
    if doc.get("status") != "complete" or doc.get("winner_held"):
        return
    if doc.get("paired_vs_reference"):
        return
    state_path = CHECKPOINT_DIR / "ml-1m-edges-state.json"
    if not state_path.is_file():
        return
    state = _read(state_path)
    config, movies, split = load_split("ml-1m")
    primary = int(config.seed)
    new_row = next(row for row in state["per_seed"] if int(row["seed"]) == primary)
    published = _published_tuning("ml-1m")
    ref_hp = dict(published["best_hyperparams"])
    ref_hp["loss"] = "in_batch"
    print("edge paired interval, scoring the saved reference tower", flush=True)
    _rec, _metrics, ref_user, _wall = _fit_and_eval(
        full_train=split.full_train,
        movies=movies,
        split=split,
        config=config,
        hyperparams=ref_hp,
        n_epochs=int(published["best_epoch"]),
        seed=primary,
        epoch_checkpoint=CHECKPOINT_DIR / "ml-1m-edges" / "reference-seed42.pt",
    )
    del _rec
    release_memory()
    paired = _pair(
        ref_user,
        new_row["per_user"],
        seed=config.seed,
        n_bootstrap=config.eval.n_bootstrap,
        alpha=config.eval.bootstrap_alpha,
    )
    doc["paired_vs_reference"] = paired
    _write(EDGE_TUNING, doc)
    if EDGE_OUT.is_file():
        out = _read(EDGE_OUT)
        out["paired_vs_reference"] = paired
        _write(EDGE_OUT, out)
    print(f"paired ndcg@10 {paired['ndcg@10']}", flush=True)


def cmd_fairness_ml1m() -> None:
    import torch

    from movielens_recommender.two_tower.train import fit_two_tower_recommender

    torch.set_num_threads(os.cpu_count() or 1)
    plan = _plan(EDGE_PLAN, "edge_extension")
    grid = list(plan["grid"])
    if EDGE_TUNING.is_file() and _read(EDGE_TUNING).get("status") == "complete":
        _fill_edge_tower_pair()
        print("edge extension already complete", flush=True)
        return
    doc = _read(EDGE_TUNING) if EDGE_TUNING.is_file() else {
        "experiment": "s3e_ml1m_edge_extension",
        "dataset": "ml-1m",
        "added_after": plan["added_after"],
        "chosen_after_test_results": True,
        "budget_source": "results/budget/two-tower-v2-ml1m-edges.json",
        "prior_hyperparams": plan["prior_hyperparams"],
        "prior_val_ndcg@10": plan["prior_val_ndcg@10"],
        "grid": grid,
        "trials": [],
        "status": "running",
    }
    done = {_trial_id(row["hyperparams"]) for row in doc["trials"]}
    config, movies, split = load_split("ml-1m")
    val_split = _val_view(split)
    for i, hp in enumerate(grid, start=1):
        if _trial_id(hp) in done:
            print(f"  skip edge trial {i}/{len(grid)}", flush=True)
            continue
        print(f"  edge trial {i}/{len(grid)}: {hp}", flush=True)
        _rec, _feat, result = fit_two_tower_recommender(
            split.train,
            dataset="ml-1m",
            data_dir=str(ROOT / "data"),
            movies=movies,
            hyperparams=hp,
            seed=int(config.seed),
            relevance_threshold=config.eval.relevance_threshold,
            val_split=val_split,
            show_progress=True,
            epoch_checkpoint=CHECKPOINT_DIR / "ml-1m-edges" / f"trial-{i}.pt",
        )
        score = float("-inf") if result.best_val_ndcg10 is None else float(result.best_val_ndcg10)
        doc["trials"].append(
            {
                "hyperparams": hp,
                "val_ndcg@10": round(score, 6),
                "best_epoch": int(result.best_epoch),
                "epochs_trained": int(result.epochs_trained),
                "wall_time_sec": result.wall_time_sec,
                "grid_edges": grid_edges(hp, grid),
            }
        )
        doc["tune_wall_sec"] = round(
            sum(float(row["wall_time_sec"]) for row in doc["trials"]), 3
        )
        _write(EDGE_TUNING, doc)
        del _rec, _feat, result
        release_memory()
        done.add(_trial_id(hp))

    decision = edge_extension_decision(doc["trials"], plan["prior_hyperparams"])
    winner = decision["winner"]
    doc["winner_held"] = bool(decision["winner_held"])
    doc["selected"] = {
        "hyperparams": winner["hyperparams"],
        "val_ndcg@10": winner["val_ndcg@10"],
        "best_epoch": winner["best_epoch"],
        "grid_edges": winner["grid_edges"],
    }
    if doc["winner_held"]:
        doc["test"] = _not_run(
            "The edges were checked and the winner held, so test and the ranker were not re-scored."
        )
        doc["ranker"] = _not_run(
            "The edges were checked and the winner held, so test and the ranker were not re-scored."
        )
        doc["status"] = "complete"
        _write(EDGE_TUNING, doc)
        _write(EDGE_OUT, doc)
        print("winner held", flush=True)
        return

    if doc.get("test", {}).get("status") != "ran":
        hp = dict(winner["hyperparams"])
        epochs = int(winner["best_epoch"])
        seeds = [int(seed) for seed in plan["test_seeds_if_winner_changes"]]
        per_seed = []
        state_path = CHECKPOINT_DIR / "ml-1m-edges-state.json"
        state = _read(state_path) if state_path.is_file() else {"per_seed": []}
        have = {int(row["seed"]) for row in state["per_seed"]}
        kept_topk = None
        for seed in seeds:
            if seed in have:
                row = next(item for item in state["per_seed"] if int(item["seed"]) == seed)
                per_seed.append(row)
                if seed == int(config.seed):
                    kept_topk = row.get("topk")
                continue
            print(f"edge test seed {seed} epochs {epochs}", flush=True)
            rec, metrics, per_user, wall = _fit_and_eval(
                full_train=split.full_train,
                movies=movies,
                split=split,
                config=config,
                hyperparams=hp,
                n_epochs=epochs,
                seed=seed,
                epoch_checkpoint=CHECKPOINT_DIR / "ml-1m-edges" / f"test-seed{seed}.pt",
            )
            row = {
                "seed": seed,
                "metrics": metrics,
                "per_user": per_user,
                "train_wall_sec": wall,
            }
            if seed == int(config.seed):
                test_users = [int(uid) for uid in split.test["user_id"].unique()]
                row["topk"] = _json_topk(
                    rec.topk_with_scores(
                        int(config.models.ranker.candidate_k), user_ids=test_users
                    )
                )
                kept_topk = row["topk"]
            state["per_seed"].append(row)
            _write(state_path, state)
            per_seed.append(row)
            del rec
            release_memory()
        spec = {
            "role": "candidate",
            "name": "sampled_softmax",
            "hyperparams": hp,
            "n_epochs": epochs,
        }
        doc["test"] = {
            "status": "ran",
            "seeds": seeds,
            "model": _compact_model(per_seed, int(config.seed), spec),
        }
        doc["status"] = "test_scored"
        _write(EDGE_TUNING, doc)
        state["kept_topk"] = kept_topk
        _write(state_path, state)

    if doc.get("ranker", {}).get("status") != "ran":
        state = _read(CHECKPOINT_DIR / "ml-1m-edges-state.json")
        published = _published_tuning("ml-1m")
        ref_hp = dict(published["best_hyperparams"])
        ref_hp["loss"] = "in_batch"
        tuning = {
            "selected_loss": "sampled_softmax",
            "best_by_loss": {
                "sampled_softmax": {
                    "hyperparams": winner["hyperparams"],
                    "best_epoch": winner["best_epoch"],
                }
            },
        }
        print("edge ranker, including the reference tower", flush=True)
        # The reference tower is refit inside _run_rankers. Its test topk is not
        # required because train_reference builds validation candidates itself
        # and the new tower's test topk is the saved one.
        ref_rec, _metrics, _user, _wall = _fit_and_eval(
            full_train=split.full_train,
            movies=movies,
            split=split,
            config=config,
            hyperparams=ref_hp,
            n_epochs=int(published["best_epoch"]),
            seed=int(config.seed),
            epoch_checkpoint=CHECKPOINT_DIR / "ml-1m-edges" / "reference-seed42.pt",
        )
        test_users = [int(uid) for uid in split.test["user_id"].unique()]
        ref_topk = ref_rec.topk_with_scores(
            int(config.models.ranker.candidate_k), user_ids=test_users
        )
        del ref_rec
        release_memory()
        doc["ranker"] = _run_rankers(
            dataset="ml-1m",
            config=config,
            movies=movies,
            split=split,
            tuning=tuning,
            towers={
                "reference": {"topk": ref_topk},
                "new": {"topk": _restore_topk(state["kept_topk"])},
            },
            train_reference=True,
        )
    doc["status"] = "complete"
    _write(EDGE_TUNING, doc)
    _write(EDGE_OUT, {key: value for key, value in doc.items() if key != "trials"} | {
        "trials": [
            {
                "hyperparams": row["hyperparams"],
                "val_ndcg@10": row["val_ndcg@10"],
                "best_epoch": row["best_epoch"],
                "grid_edges": row["grid_edges"],
            }
            for row in doc["trials"]
        ]
    })
    _fill_edge_tower_pair()
    print(f"winner changed val {doc['selected']['val_ndcg@10']}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="S3e fairness round")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ml32m")
    sub.add_parser("ml1m")
    args = parser.parse_args()
    if args.cmd == "ml32m":
        cmd_fairness_ml32m()
    elif args.cmd == "ml1m":
        cmd_fairness_ml1m()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
