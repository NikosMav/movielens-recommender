"""Time-based train/test split for recommendation evaluation."""

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

    The split is deterministic given the same ratings rows and config.
    """

    min_ratings: int = 5
    test_fraction: float = 0.2
    relevance_threshold: float = 4.0

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

    def summary(self) -> dict:
        return {
            "n_train": int(len(self.train)),
            "n_test": int(len(self.test)),
            "n_users_kept": self.n_users_kept,
            "n_users_dropped": self.n_users_dropped,
            "n_train_users": int(self.train["user_id"].nunique()),
            "n_test_users": int(self.test["user_id"].nunique()),
            "n_train_items": int(self.train["item_id"].nunique()),
            "config": self.config.to_dict(),
        }


def time_based_split(ratings: pd.DataFrame, config: SplitConfig | None = None) -> SplitResult:
    """Per-user time-based holdout of the latest interactions.

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
