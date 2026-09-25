"""Segment definitions for breakdown evaluation (activity terciles, head/tail)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from movielens_recommender.metrics import bootstrap_mean_ci, ndcg_at_k


@dataclass(frozen=True)
class SegmentDefinitions:
    """How user and item segments are formed from the training matrix."""

    user_activity: str = (
        "Eval users partitioned into low/mid/high terciles by train rating count "
        "(full_train interaction count). Ties broken by user_id ascending before "
        "assigning ranks so boundaries are deterministic."
    )
    item_head_tail: str = (
        "Head items = top 20% of train catalog by interaction count (ties broken "
        "by item_id ascending). Tail = remaining train items."
    )
    item_segment_metric: str = (
        "For an item segment S: relevant_S = relevant ∩ S; recommendations are "
        "filtered to items in S (order preserved) and scored at k against "
        "relevant_S. Users with empty relevant_S are excluded from that "
        "segment's average."
    )
    head_fraction: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEGMENT_DEFINITIONS = SegmentDefinitions()


def user_activity_counts(train: pd.DataFrame) -> dict[int, int]:
    """Map user_id → number of train interactions."""
    return {int(uid): int(c) for uid, c in train.groupby("user_id").size().items()}


def assign_activity_terciles(counts: Mapping[int, int]) -> dict[str, list[int]]:
    """Partition users into low/mid/high by interaction count.

    Users are sorted by (count ascending, user_id ascending), then split into
    three contiguous blocks as evenly as possible.
    """
    if not counts:
        return {"low": [], "mid": [], "high": []}
    ordered = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered)
    # Boundaries via linspace so leftover users go to later buckets deterministically.
    cuts = [int(round(x)) for x in np.linspace(0, n, 4)]
    labels = ("low", "mid", "high")
    out: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        out[label] = [uid for uid, _ in ordered[cuts[i] : cuts[i + 1]]]
    return out


def head_tail_items(train: pd.DataFrame, *, head_fraction: float = 0.2) -> dict[str, set[int]]:
    """Split train catalog into head (top fraction by popularity) and tail."""
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction must be in (0, 1)")
    counts = train.groupby("item_id").size().reset_index(name="count")
    ranked = counts.sort_values(
        ["count", "item_id"], ascending=[False, True], kind="mergesort"
    )
    n_items = len(ranked)
    n_head = max(1, int(n_items * head_fraction))
    if n_head >= n_items:
        n_head = n_items - 1 if n_items > 1 else 1
    head_ids = set(int(x) for x in ranked["item_id"].iloc[:n_head].tolist())
    all_ids = set(int(x) for x in ranked["item_id"].tolist())
    return {"head": head_ids, "tail": all_ids - head_ids}


def filter_recs_to_segment(recs: Sequence[int], segment: set[int], k: int) -> list[int]:
    """Keep recommendation order, restrict to segment, truncate to k."""
    out = [int(i) for i in recs if int(i) in segment]
    return out[:k]


def ndcg_for_user_subset(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, set[int]],
    user_ids: Sequence[int],
    *,
    k: int = 10,
    n_bootstrap: int = 1000,
    bootstrap_alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """NDCG@k mean + bootstrap CI over a subset of eval users."""
    values: list[float] = []
    used = 0
    for uid in user_ids:
        uid_i = int(uid)
        rel = relevant.get(uid_i)
        if not rel:
            continue
        recs = recommendations.get(uid_i, [])
        values.append(ndcg_at_k(recs, rel, k))
        used += 1
    if not values:
        return {
            f"ndcg@{k}": 0.0,
            "n_users": 0,
            "confidence_intervals": {
                f"ndcg@{k}": {"mean": 0.0, "low": 0.0, "high": 0.0},
            },
        }
    mean, low, high = bootstrap_mean_ci(
        values, n_bootstrap=n_bootstrap, alpha=bootstrap_alpha, seed=seed
    )
    return {
        f"ndcg@{k}": round(mean, 6),
        "n_users": used,
        "confidence_intervals": {
            f"ndcg@{k}": {
                "mean": round(mean, 6),
                "low": round(low, 6),
                "high": round(high, 6),
            },
        },
    }


def ndcg_for_item_segment(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, set[int]],
    segment_items: set[int],
    *,
    k: int = 10,
    n_bootstrap: int = 1000,
    bootstrap_alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """NDCG@k with relevant and recommended items restricted to a segment."""
    values: list[float] = []
    used = 0
    for uid, rel in relevant.items():
        rel_s = rel & segment_items
        if not rel_s:
            continue
        recs_s = filter_recs_to_segment(recommendations.get(uid, []), segment_items, k)
        values.append(ndcg_at_k(recs_s, rel_s, k))
        used += 1
    if not values:
        return {
            f"ndcg@{k}": 0.0,
            "n_users": 0,
            "n_segment_items": len(segment_items),
            "confidence_intervals": {
                f"ndcg@{k}": {"mean": 0.0, "low": 0.0, "high": 0.0},
            },
        }
    mean, low, high = bootstrap_mean_ci(
        values, n_bootstrap=n_bootstrap, alpha=bootstrap_alpha, seed=seed
    )
    return {
        f"ndcg@{k}": round(mean, 6),
        "n_users": used,
        "n_segment_items": len(segment_items),
        "confidence_intervals": {
            f"ndcg@{k}": {
                "mean": round(mean, 6),
                "low": round(low, 6),
                "high": round(high, 6),
            },
        },
    }


def evaluate_segments(
    recommendations: Mapping[int, Sequence[int]],
    relevant: Mapping[int, set[int]],
    train: pd.DataFrame,
    *,
    k: int = 10,
    n_bootstrap: int = 1000,
    bootstrap_alpha: float = 0.05,
    seed: int = 42,
    head_fraction: float = 0.2,
) -> dict[str, Any]:
    """Build user-activity and head/tail NDCG@k breakdowns."""
    counts = user_activity_counts(train)
    # Only segment users that are in the eval set.
    eval_counts = {uid: counts[uid] for uid in relevant if uid in counts}
    terciles = assign_activity_terciles(eval_counts)
    head_tail = head_tail_items(train, head_fraction=head_fraction)

    activity: dict[str, Any] = {}
    for label, users in terciles.items():
        # Restrict to eval users in this bucket (already true via eval_counts).
        bucket_users = [u for u in users if u in relevant]
        activity[label] = ndcg_for_user_subset(
            recommendations,
            relevant,
            bucket_users,
            k=k,
            n_bootstrap=n_bootstrap,
            bootstrap_alpha=bootstrap_alpha,
            seed=seed,
        )
        if bucket_users:
            bucket_counts = [eval_counts[u] for u in bucket_users]
            activity[label]["train_rating_count_min"] = int(min(bucket_counts))
            activity[label]["train_rating_count_max"] = int(max(bucket_counts))

    item_seg: dict[str, Any] = {}
    for label, items in head_tail.items():
        item_seg[label] = ndcg_for_item_segment(
            recommendations,
            relevant,
            items,
            k=k,
            n_bootstrap=n_bootstrap,
            bootstrap_alpha=bootstrap_alpha,
            seed=seed,
        )

    return {
        "definitions": SEGMENT_DEFINITIONS.to_dict(),
        "user_activity_terciles": activity,
        "item_head_tail": item_seg,
    }
