"""Replay ratings in time order to measure model decay and refresh policies (ADR-0016)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.baselines.ease import EASERecommender
from movielens_recommender.baselines.item_knn import ItemItemCosineRecommender
from movielens_recommender.metrics import bootstrap_mean_ci, ndcg_at_k, paired_bootstrap_delta_ci
from movielens_recommender.ops.drift import (
    Period,
    item_divergence,
    new_item_share,
    new_user_share,
    select_tau,
    window_before,
)

K = 10
CHUNK = 500


@dataclass
class _EvalSet:
    users: list[int]
    profiles: dict[int, list[tuple[int, float]]]
    targets: dict[int, set[int]]


def _eval_set(
    ratings: pd.DataFrame, period: Period, *, min_history: int, relevance: float
) -> _EvalSet:
    before = ratings.loc[ratings["timestamp"] < period.start]
    inside = ratings.loc[
        (ratings["timestamp"] >= period.start) & (ratings["timestamp"] < period.end)
    ]
    liked = inside.loc[inside["rating"] >= relevance]
    history_n = before.groupby("user_id").size()
    eligible = sorted(
        int(uid) for uid in liked["user_id"].unique() if history_n.get(uid, 0) >= min_history
    )
    chosen = set(eligible)
    profiles: dict[int, list[tuple[int, float]]] = {uid: [] for uid in eligible}
    ordered = before.loc[before["user_id"].isin(chosen)].sort_values(
        ["timestamp", "item_id"], kind="mergesort"
    )
    for uid, item, rating in zip(
        ordered["user_id"].tolist(), ordered["item_id"].tolist(), ordered["rating"].tolist(),
        strict=True,
    ):
        profiles[int(uid)].append((int(item), float(rating)))
    targets: dict[int, set[int]] = {uid: set() for uid in eligible}
    for uid, item in zip(liked["user_id"].tolist(), liked["item_id"].tolist(), strict=True):
        if int(uid) in chosen:
            targets[int(uid)].add(int(item))
    return _EvalSet(users=eligible, profiles=profiles, targets=targets)


def _fit(name: str, params: Mapping[str, Any], train: pd.DataFrame) -> Any:
    if name == "most_popular":
        counts = train.groupby("item_id").size().reset_index(name="count")
        ranked = counts.sort_values(["count", "item_id"], ascending=[False, True], kind="mergesort")
        return ranked["item_id"].to_numpy(dtype=np.int64)
    if name == "item_item":
        return ItemItemCosineRecommender(**params).fit(train)
    if name == "ease":
        return EASERecommender(**params).fit(train)
    raise ValueError(f"unknown replay model {name!r}")


def _score(model: Any, evals: _EvalSet) -> np.ndarray:
    """Per-user NDCG@10, in ``evals.users`` order."""
    out = np.zeros(len(evals.users), dtype=np.float64)
    if isinstance(model, np.ndarray):
        for i, uid in enumerate(evals.users):
            seen = {item for item, _r in evals.profiles[uid]}
            recs = [int(item) for item in model[: K + len(seen)] if int(item) not in seen][:K]
            out[i] = ndcg_at_k(recs, evals.targets[uid], K)
        return out
    for begin in range(0, len(evals.users), CHUNK):
        chunk = evals.users[begin : begin + CHUNK]
        tops = model.topk_for_profiles({uid: evals.profiles[uid] for uid in chunk}, K)
        for offset, uid in enumerate(chunk):
            recs = [item for item, _s in tops.get(uid, [])]
            out[begin + offset] = ndcg_at_k(recs, evals.targets[uid], K)
    return out


def _spearman(signal: Sequence[float], gap: Sequence[float]) -> float | None:
    from scipy.stats import spearmanr

    if len(signal) < 3 or len(set(signal)) < 2 or len(set(gap)) < 2:
        return None
    rho = spearmanr(signal, gap).statistic
    return None if rho != rho else round(float(rho), 6)


def _policy_block(
    periods: Sequence[Period],
    cutoffs: Sequence[int],
    scores: Mapping[tuple[int, int], np.ndarray],
    n_tuning: int,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for index, (period, cutoff) in enumerate(zip(periods, cutoffs, strict=True)):
        values = scores[(cutoff, index)]
        mean, low, high = bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                "period": period.name,
                "trained_before": pd.Timestamp(cutoff, unit="s", tz="UTC").strftime("%Y-%m-%d"),
                "ndcg@10": round(mean, 6),
                "low": round(low, 6),
                "high": round(high, 6),
                "n_users": int(values.size),
            }
        )
    retrain_at = [
        periods[i].name for i in range(1, len(periods)) if cutoffs[i] != cutoffs[i - 1]
    ]
    means = [row["ndcg@10"] for row in rows]
    return {
        "per_period": rows,
        "tuning_mean": round(float(np.mean(means[:n_tuning])), 6),
        "test_mean": round(float(np.mean(means[n_tuning:])), 6) if len(means) > n_tuning else None,
        "retrains": len(retrain_at),
        "retrain_periods": retrain_at,
    }


def run_replay(
    ratings: pd.DataFrame,
    periods: Sequence[Period],
    *,
    models: Mapping[str, Mapping[str, Any]],
    taus: Sequence[float],
    n_tuning: int,
    sample_size: int = 2000,
    min_history: int = 5,
    relevance: float = 4.0,
    min_item_count: int = 5,
    tolerance: float = 0.005,
    tau_model: str = "item_item",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Score frozen, periodic, and drift-triggered policies over ``periods``.

    Every model is fit once per period start. A policy is a choice of which
    fit serves each period, so the per-user scores are computed once per
    (fit, period) pair and shared by every policy.
    """
    ratings = ratings.sort_values(["timestamp", "user_id", "item_id"], kind="mergesort")
    starts = [p.start for p in periods]
    first_seen = ratings.groupby("user_id")["timestamp"].min().astype(int).to_dict()
    evals = [
        _eval_set(ratings, p, min_history=min_history, relevance=relevance) for p in periods
    ]
    last_windows = [window_before(periods, i) for i in range(len(periods))]
    last_frames = [
        ratings.loc[(ratings["timestamp"] >= w.start) & (ratings["timestamp"] < w.end)]
        for w in last_windows
    ]

    scores: dict[str, dict[tuple[int, int], np.ndarray]] = {name: {} for name in models}
    divergence: dict[tuple[int, int], float] = {}
    period_rows: list[dict[str, Any]] = []
    frozen_counts: Counter[int] = Counter()
    for c_index, cutoff in enumerate(starts):
        train = ratings.loc[ratings["timestamp"] < cutoff]
        train_items = train["item_id"].to_numpy()
        if c_index == 0:
            frozen_counts = Counter(train_items.tolist())
            frozen_items = train_items
        for p_index in range(c_index, len(periods)):
            divergence[(cutoff, p_index)] = item_divergence(
                last_frames[p_index]["item_id"].to_numpy(),
                train_items,
                sample_size=sample_size,
                seed=seed,
            )
        for name, params in models.items():
            model = _fit(name, params, train)
            for p_index in range(c_index, len(periods)):
                scores[name][(cutoff, p_index)] = _score(model, evals[p_index])
            del model

    for p_index, period in enumerate(periods):
        last = last_frames[p_index]
        last_items = last["item_id"].tolist()
        targets = [item for uid in evals[p_index].users for item in evals[p_index].targets[uid]]
        unseen = sum(1 for item in targets if frozen_counts.get(item, 0) == 0)
        period_rows.append(
            {
                "name": period.name,
                "start": pd.Timestamp(period.start, unit="s", tz="UTC").strftime("%Y-%m-%d"),
                "end": pd.Timestamp(period.end, unit="s", tz="UTC").strftime("%Y-%m-%d"),
                "n_ratings": int(
                    ((ratings["timestamp"] >= period.start) & (ratings["timestamp"] < period.end))
                    .sum()
                ),
                "n_eval_users": len(evals[p_index].users),
                "target_unseen_by_frozen": round(unseen / len(targets), 6) if targets else 0.0,
                "drift_vs_frozen": {
                    "item_divergence": round(divergence[(starts[0], p_index)], 6),
                    "new_item_share": round(
                        new_item_share(last_items, frozen_counts, min_count=min_item_count), 6
                    ),
                    "new_user_share": round(
                        new_user_share(last, first_seen, last_windows[p_index]), 6
                    ),
                },
            }
        )

    def drift_cutoffs(tau: float) -> list[int]:
        current = starts[0]
        chosen = []
        for p_index in range(len(periods)):
            if p_index > 0 and divergence[(current, p_index)] > tau:
                current = starts[p_index]
            chosen.append(current)
        return chosen

    frozen = [starts[0]] * len(periods)
    periodic = list(starts)
    common = {"n_bootstrap": n_bootstrap, "seed": seed}
    tuning_grid = {}
    for tau in taus:
        block = _policy_block(periods, drift_cutoffs(tau), scores[tau_model], n_tuning, **common)
        tuning_retrains = sum(
            1 for name in block["retrain_periods"]
            if [p.name for p in periods].index(name) < n_tuning
        )
        tuning_grid[tau] = {"mean_ndcg": block["tuning_mean"], "retrains": tuning_retrains}
    periodic_tuning = _policy_block(periods, periodic, scores[tau_model], n_tuning, **common)
    selection = select_tau(
        tuning_grid, periodic_mean=periodic_tuning["tuning_mean"], tolerance=tolerance
    )
    tau = selection["tau"]

    model_blocks: dict[str, Any] = {}
    for name in models:
        gaps = []
        for p_index in range(len(periods)):
            gaps.append(
                paired_bootstrap_delta_ci(
                    scores[name][(starts[0], p_index)],
                    scores[name][(starts[p_index], p_index)],
                    **common,
                )
            )
        gap_means = [g["mean"] for g in gaps]
        drift_block = _policy_block(periods, drift_cutoffs(tau), scores[name], n_tuning, **common)
        drift_block["tau"] = tau
        model_blocks[name] = {
            "params": dict(models[name]),
            "policies": {
                "frozen": _policy_block(periods, frozen, scores[name], n_tuning, **common),
                "periodic": _policy_block(periods, periodic, scores[name], n_tuning, **common),
                "drift": drift_block,
            },
            "paired_periodic_minus_frozen": [
                {"period": p.name, **{k: (round(v, 6) if isinstance(v, float) else v)
                                      for k, v in g.items()}}
                for p, g in zip(periods, gaps, strict=True)
            ],
            "spearman_vs_gap": {
                key: _spearman([row["drift_vs_frozen"][key] for row in period_rows], gap_means)
                for key in ("item_divergence", "new_item_share", "new_user_share")
            },
        }

    return {
        "periods": period_rows,
        "divergence_floor": round(
            item_divergence(frozen_items, frozen_items, sample_size=sample_size, seed=seed), 6
        ),
        "tau_selection": {
            "chosen_on": tau_model,
            "n_tuning_periods": n_tuning,
            "tuning_periods": [p.name for p in periods[:n_tuning]],
            "periodic_tuning_mean": periodic_tuning["tuning_mean"],
            "grid": {str(t): v for t, v in tuning_grid.items()},
            **{k: v for k, v in selection.items()},
        },
        "models": model_blocks,
    }
