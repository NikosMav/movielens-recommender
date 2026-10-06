"""Candidate-set construction and validation recall.

Item–item is source A and two-tower is source B. Alternation always takes A
then B. Ranks elsewhere are 1-indexed positions in each source's top-K list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from movielens_recommender.metrics import recall_at_k

# Compared on validation Recall@200. ``union`` in ablations is the balanced set.
CANDIDATE_SET_NAMES: tuple[str, ...] = (
    "item_item",
    "two_tower",
    "union_balanced",
    "union_unbalanced",
)

# Exact metric ties: higher Recall@100 wins, then this order.
_TIE_PREFERENCE: tuple[str, ...] = (
    "union_balanced",
    "item_item",
    "two_tower",
    "union_unbalanced",
)

# Ranker ablations always cover these three. Unbalanced is added when it wins.
RANKER_CANDIDATE_SETS: tuple[str, ...] = (
    "item_item",
    "two_tower",
    "union_balanced",
)


def dedupe_ids(ids: Sequence[int]) -> list[int]:
    """Stable de-duplication, preserving first-seen order."""
    seen: set[int] = set()
    out: list[int] = []
    for raw in ids:
        item = int(raw)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def balanced_union(left: Sequence[int], right: Sequence[int], k: int) -> list[int]:
    """Top ``k/2`` from each source, dedupe, then backfill alternately to ``k``.

    ``k/2`` is integer division. The initial pass alternates through those
    prefixes (left, then right). Overlaps are kept at the earlier position, so
    the prefix can be shorter than ``k``. Backfill then alternates through the
    remainder of each source, starting with left, until the list has length
    ``k`` or both sources are exhausted.

    Each input is de-duplicated first. The result is shorter than ``k`` only
    when the two sources together have fewer than ``k`` unique ids.
    """
    if k <= 0:
        return []
    left_ids = dedupe_ids(left)
    right_ids = dedupe_ids(right)
    half = k // 2
    selected: list[int] = []
    seen: set[int] = set()

    def _take(item: int) -> None:
        if item not in seen and len(selected) < k:
            seen.add(item)
            selected.append(item)

    for i in range(half):
        if i < len(left_ids):
            _take(left_ids[i])
        if i < len(right_ids):
            _take(right_ids[i])

    li = half
    ri = half
    turn_left = True
    stalled = 0
    while len(selected) < k and stalled < 2:
        if turn_left:
            while li < len(left_ids) and left_ids[li] in seen:
                li += 1
            if li < len(left_ids):
                _take(left_ids[li])
                li += 1
                stalled = 0
            else:
                stalled += 1
        else:
            while ri < len(right_ids) and right_ids[ri] in seen:
                ri += 1
            if ri < len(right_ids):
                _take(right_ids[ri])
                ri += 1
                stalled = 0
            else:
                stalled += 1
        turn_left = not turn_left
    return selected


def unbalanced_union(left: Sequence[int], right: Sequence[int], k: int) -> list[int]:
    """Alternate through the top ``k`` of each source, dropping duplicates.

    The list is not truncated to ``k``. Its length is the number of unique ids
    in the two prefixes (between 0 and ``2k``). Order is the ranking used for
    Recall@100 / Recall@200.
    """
    if k <= 0:
        return []
    left_ids = dedupe_ids(left)[:k]
    right_ids = dedupe_ids(right)[:k]
    selected: list[int] = []
    seen: set[int] = set()
    n = max(len(left_ids), len(right_ids))
    for i in range(n):
        if i < len(left_ids) and left_ids[i] not in seen:
            seen.add(left_ids[i])
            selected.append(left_ids[i])
        if i < len(right_ids) and right_ids[i] not in seen:
            seen.add(right_ids[i])
            selected.append(right_ids[i])
    return selected


def candidate_ids_for_user(
    name: str,
    left: Sequence[int],
    right: Sequence[int],
    k: int,
) -> list[int]:
    """Build one user's candidate list for a named set."""
    if name == "item_item":
        return dedupe_ids(left)[:k]
    if name == "two_tower":
        return dedupe_ids(right)[:k]
    if name == "union_balanced":
        return balanced_union(left, right, k)
    if name == "union_unbalanced":
        return unbalanced_union(left, right, k)
    raise ValueError(f"Unknown candidate set: {name!r}")


def materialize_candidate_set(
    name: str,
    item_item: Mapping[int, Sequence[tuple[int, float]]],
    two_tower: Mapping[int, Sequence[tuple[int, float]]],
    k: int,
) -> dict[int, list[int]]:
    """Per-user ordered candidate ids for ``name``.

    Users are the union of the two retrieval maps. A missing source is an
    empty list for that user.
    """
    users = set(item_item) | set(two_tower)
    out: dict[int, list[int]] = {}
    for uid in users:
        left = [int(i) for i, _ in item_item.get(uid, ())]
        right = [int(i) for i, _ in two_tower.get(uid, ())]
        out[int(uid)] = candidate_ids_for_user(name, left, right, k)
    return out


def score_rank_maps(
    retrieved: Mapping[int, Sequence[tuple[int, float]]],
) -> dict[int, dict[int, tuple[float, int]]]:
    """``user -> item -> (score, 1-indexed rank)`` for one retriever's top-K."""
    out: dict[int, dict[int, tuple[float, int]]] = {}
    for uid, rows in retrieved.items():
        mapped: dict[int, tuple[float, int]] = {}
        for rank, (item, score) in enumerate(rows, start=1):
            mapped[int(item)] = (float(score), int(rank))
        out[int(uid)] = mapped
    return out


def retrieval_recall(
    lists: Mapping[int, Sequence[int]],
    relevant: Mapping[int, set[int]],
    *,
    ks: Sequence[int] = (100, 200),
) -> dict[str, float]:
    """Mean Recall@k and mean list length over users with relevant items.

    Point estimate only (candidate-set selection does not bootstrap).
    """
    if not relevant:
        raise ValueError("retrieval_recall requires at least one relevant user")
    user_ids = sorted(relevant)
    size_vals = [float(len(lists.get(uid, ()))) for uid in user_ids]
    out: dict[str, float] = {
        "mean_size": float(np.mean(size_vals)),
        "n_eval_users": float(len(user_ids)),
    }
    for k in ks:
        vals = [
            recall_at_k(lists.get(uid, ()), relevant[uid], int(k)) for uid in user_ids
        ]
        out[f"recall@{int(k)}"] = float(np.mean(vals))
    return out


def choose_candidate_set(reports: Mapping[str, Mapping[str, float]]) -> str:
    """Winner by validation Recall@200, then Recall@100, then a fixed order.

    ``reports`` must include every name in :data:`CANDIDATE_SET_NAMES`.
    """
    missing = [name for name in CANDIDATE_SET_NAMES if name not in reports]
    if missing:
        raise ValueError(f"candidate reports missing: {missing}")

    def _key(name: str) -> tuple[float, float, int]:
        block = reports[name]
        return (
            -float(block["recall@200"]),
            -float(block["recall@100"]),
            _TIE_PREFERENCE.index(name),
        )

    return min(CANDIDATE_SET_NAMES, key=_key)


class PrecomputedRecommender:
    """``recommend(user, n)`` over a stored ranking (already excludes seen)."""

    def __init__(self, ranked: Mapping[int, Sequence[int]]) -> None:
        self._ranked = {int(uid): [int(i) for i in items] for uid, items in ranked.items()}

    def recommend(self, user_id: int, n: int) -> list[int]:
        if n <= 0:
            return []
        return self._ranked.get(int(user_id), [])[:n]
