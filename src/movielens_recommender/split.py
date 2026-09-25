"""Time-based train/test split and cold-start filtering.

The default protocol is a **per-user chronological holdout**. It prevents
within-user leakage (a user's test rows never appear in that user's train) but
does **not** prevent cross-user / global temporal leakage: other users'
later interactions can still sit in the training matrix (see ADR-0002).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for the per-user time-based holdout.

    Rule
    ----
    For each user:

    1. Drop the user if they have fewer than ``min_ratings`` interactions.
    2. Sort their interactions by ``timestamp`` ascending (stable sort; ties
       broken by original row order via a stable sort).
    3. Hold out the last ``max(1, floor(n * test_fraction))`` interactions as
       test; the remainder are train. Require at least one train interaction
       (users who would otherwise have an empty train set are dropped).

    Prevents within-user leakage; does not prevent cross-user / global
    temporal leakage (ADR-0002). Deterministic given the same ratings and config.
    """

    min_ratings: int = 5
    test_fraction: float = 0.2
    relevance_threshold: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ColdStartStats:
    """Cold-start filtering summary applied after the split."""

    n_train_users: int
    n_train_items: int
    n_cold_items_in_test: int
    n_relevant_dropped_cold_item: int
    n_users_excluded_no_warm_relevant: int
    n_eval_users: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SplitResult:
    """Train/test frames plus the config used to produce them."""

    train: pd.DataFrame
    test: pd.DataFrame
    config: SplitConfig
    n_users_kept: int
    n_users_dropped: int
    cold_start: ColdStartStats | None = None

    def summary(self) -> dict:
        out = {
            "n_train": int(len(self.train)),
            "n_test": int(len(self.test)),
            "n_users_kept": self.n_users_kept,
            "n_users_dropped": self.n_users_dropped,
            "n_train_users": int(self.train["user_id"].nunique()),
            "n_test_users": int(self.test["user_id"].nunique()),
            "n_train_items": int(self.train["item_id"].nunique()),
            "config": self.config.to_dict(),
        }
        if self.cold_start is not None:
            out["cold_start"] = self.cold_start.to_dict()
        return out


def time_based_split(ratings: pd.DataFrame, config: SplitConfig | None = None) -> SplitResult:
    """Per-user chronological holdout of the latest interactions.

    Prevents within-user leakage; does not prevent cross-user / global
    temporal leakage (see ADR-0002).

    Parameters
    ----------
    ratings
        Must contain columns: user_id, item_id, rating, timestamp.
    config
        Split hyperparameters. Defaults to :class:`SplitConfig`.
    """
    if config is None:
        config = SplitConfig()

    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing columns: {sorted(missing)}")

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    kept = 0
    dropped = 0

    # Group in a deterministic user order.
    for _, group in ratings.groupby("user_id", sort=True):
        n = len(group)
        if n < config.min_ratings:
            dropped += 1
            continue

        # Stable sort by timestamp so ties keep relative order.
        ordered = group.sort_values("timestamp", kind="mergesort")
        n_test = max(1, int(n * config.test_fraction))
        # Ensure at least one training interaction.
        if n_test >= n:
            n_test = n - 1
        if n_test < 1:
            dropped += 1
            continue

        train_parts.append(ordered.iloc[:-n_test])
        test_parts.append(ordered.iloc[-n_test:])
        kept += 1

    if not train_parts:
        raise ValueError("No users remaining after split filters.")

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return SplitResult(
        train=train,
        test=test,
        config=config,
        n_users_kept=kept,
        n_users_dropped=dropped,
    )


def apply_cold_start_policy(
    split: SplitResult,
    *,
    relevance_threshold: float | None = None,
) -> tuple[dict[int, set[int]], dict[int, set[int]], ColdStartStats]:
    """Build seen/relevant maps with cold items removed from relevance.

    Cold-start policy
    -----------------
    - Cold users (absent from train) never appear after :func:`time_based_split`.
    - Cold items (absent from train) are dropped from each user's relevant test
      set. Users with no remaining warm relevant items are excluded from
      ranking-metric averages (counted in ``n_users_excluded_no_warm_relevant``).

    Returns
    -------
    seen, relevant, stats
        ``relevant`` only includes users with ≥1 warm relevant item.
    """
    threshold = (
        split.config.relevance_threshold if relevance_threshold is None else relevance_threshold
    )
    train_items = set(split.train["item_id"].astype(int))
    train_users = set(split.train["user_id"].astype(int))

    seen: dict[int, set[int]] = {
        int(uid): set(g["item_id"].astype(int)) for uid, g in split.train.groupby("user_id")
    }

    test_items = set(split.test["item_id"].astype(int))
    cold_items = test_items - train_items

    relevant: dict[int, set[int]] = {}
    n_relevant_dropped = 0
    n_excluded = 0

    for uid, g in split.test.groupby("user_id"):
        uid_i = int(uid)
        if uid_i not in train_users:
            n_excluded += 1
            continue
        rel_all = set(g.loc[g["rating"] >= threshold, "item_id"].astype(int))
        cold_rel = rel_all - train_items
        n_relevant_dropped += len(cold_rel)
        rel_warm = rel_all & train_items
        if not rel_warm:
            n_excluded += 1
            continue
        relevant[uid_i] = rel_warm

    stats = ColdStartStats(
        n_train_users=len(train_users),
        n_train_items=len(train_items),
        n_cold_items_in_test=len(cold_items),
        n_relevant_dropped_cold_item=int(n_relevant_dropped),
        n_users_excluded_no_warm_relevant=int(n_excluded),
        n_eval_users=len(relevant),
    )
    split.cold_start = stats
    return seen, relevant, stats
