"""Scale helpers for catalogs that do not fit a dense item-item matrix.

Sampling and the sparse-catalog threshold are part of ADR-0011. Training
matrices stay complete; only metric, tuning, and ranker-label users are sampled.
"""

from __future__ import annotations

import gc
import hashlib

import numpy as np
import pandas as pd

from movielens_recommender.split import SplitResult

# Above this catalog size the dense Gram (and a second co-occurrence matrix)
# is not allocated. 20_000² × 8 bytes is 3.2 GiB for one matrix.
SPARSE_CATALOG_ITEMS = 20_000

# Precomputed score rows (users × items, float64) above this are skipped.
# ml-1m is about 180 MiB and still caches; ml-32M does not.
USER_SCORE_CACHE_BYTES = 256 * 1024 * 1024


def release_memory() -> None:
    """Collect cyclic garbage and return free arenas to the OS.

    Used between large fits so the next model does not sit on top of the
    previous Gram or feature pack (ADR-0011).
    """
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        return


def sample_user_ids(user_ids: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Deterministic sample of user ids.

    Ids are uniqued and sorted before sampling so the result depends only on
    the set of ids, ``n``, and ``seed``. The returned array is sorted.
    ``n <= 0`` or ``n`` at least the population returns every id.
    """
    ids = np.unique(np.asarray(user_ids, dtype=np.int64))
    if n <= 0 or n >= len(ids):
        return ids.copy()
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(ids, size=int(n), replace=False)
    chosen.sort()
    return chosen


def should_cache_user_scores(
    n_users: int,
    n_items: int,
    override: bool | None,
) -> bool:
    """Whether to materialize a dense user-item score matrix."""
    if override is not None:
        return bool(override)
    return int(n_users) * int(n_items) * 8 <= USER_SCORE_CACHE_BYTES


def warm_relevant_user_ids(
    train: pd.DataFrame,
    test: pd.DataFrame,
    relevance_threshold: float,
) -> np.ndarray:
    """Users with at least one test item that is relevant and in ``train``.

    Same population the harness averages over: rating ``>= threshold`` and
    the item appears in the train catalog. Does not build per-user Python sets.
    """
    rel = test.loc[
        test["rating"] >= relevance_threshold, ["user_id", "item_id"]
    ]
    if rel.empty or train.empty:
        return np.array([], dtype=np.int64)
    train_items = pd.Index(train["item_id"].unique())
    train_users = pd.Index(train["user_id"].unique())
    rel = rel[rel["item_id"].isin(train_items) & rel["user_id"].isin(train_users)]
    if rel.empty:
        return np.array([], dtype=np.int64)
    return np.unique(rel["user_id"].to_numpy(dtype=np.int64))


def apply_eval_user_sample(
    split: SplitResult,
    *,
    n_users: int,
    seed: int,
    relevance_threshold: float,
) -> SplitResult:
    """Restrict validation and test rows to one seeded user sample.

    ``train`` stays the full fit-train matrix. ``full_train`` is frozen to the
    pre-sample train ∪ validation matrix, so other users' training interactions
    remain available. Sampled users' histories are not truncated.
    """
    if n_users <= 0:
        return split
    full = split.full_train
    eligible = warm_relevant_user_ids(full, split.test, relevance_threshold)
    if len(eligible) == 0:
        raise ValueError("No warm-relevant users available to sample.")
    chosen = sample_user_ids(eligible, n_users, seed)
    chosen_set = set(int(u) for u in chosen.tolist())
    test = split.test.loc[split.test["user_id"].isin(chosen_set)].reset_index(drop=True)
    if split.val is None:
        val = None
        n_val_rows = 0
    else:
        val = split.val.loc[split.val["user_id"].isin(chosen_set)].reset_index(drop=True)
        n_val_rows = int(len(val))
    info = {
        "enabled": True,
        "seed": int(seed),
        "requested_users": int(n_users),
        "n_eligible_users": int(len(eligible)),
        "n_sampled_users": int(len(chosen)),
        "n_test_rows": int(len(test)),
        "n_val_rows": n_val_rows,
        "user_ids_sha256": hashlib.sha256(chosen.tobytes()).hexdigest(),
        "histories": "not_truncated",
        "train_matrix": "all_training_interactions",
        "selection": (
            "Users with at least one warm relevant test item "
            f"(rating >= {relevance_threshold}, item in full train). "
            "The same ids are used for validation metrics, ranker labels, "
            "and test metrics."
        ),
    }
    return SplitResult(
        train=split.train,
        test=test,
        val=val,
        config=split.config,
        n_users_kept=split.n_users_kept,
        n_users_dropped=split.n_users_dropped,
        frozen_full_train=full,
        eval_user_sample=info,
    )
