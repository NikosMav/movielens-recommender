"""Time-based train/val/test splits and cold-start filtering.

The default protocol is a **per-user chronological holdout**. It prevents
within-user leakage (a user's test/val rows never appear in that user's earlier
slices) but does **not** prevent cross-user / global temporal leakage: other
users' later interactions can still sit in the training matrix (see ADR-0002).

Validation (ADR-0005): after the test holdout, the latest slice of each user's
remaining train rows is held out as validation. Tune on val only; refit the
chosen config on full train (train∪val); evaluate once on test.
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
       test; the remainder are the full-train pool. Require at least one
       full-train interaction.
    4. If ``val_fraction > 0``, from the full-train pool hold out the last
       ``max(1, floor(n_train * val_fraction))`` rows as validation; the
       remainder is fit-train. Require at least one fit-train row.

    Prevents within-user leakage; does not prevent cross-user / global
    temporal leakage (ADR-0002). Deterministic given the same ratings and config.
    """

    min_ratings: int = 5
    test_fraction: float = 0.2
    val_fraction: float = 0.0
    relevance_threshold: float = 4.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GlobalCutoffConfig:
    """Corpus-wide timestamp cutoff split (sanity check; ADR-0002 follow-up)."""

    timestamp_quantile: float = 0.8
    min_train_ratings: int = 5
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
    """Train(/val)/test frames plus the config used to produce them.

    ``train`` is the fit-train slice used for tuning. When ``val`` is present,
    ``full_train`` (= train ∪ val) is the matrix for the final refit before
    test evaluation. When ``val`` is None, ``full_train`` equals ``train``.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    config: SplitConfig
    n_users_kept: int
    n_users_dropped: int
    val: pd.DataFrame | None = None
    cold_start: ColdStartStats | None = None

    @property
    def full_train(self) -> pd.DataFrame:
        """Train matrix for final refit (train ∪ val when validation exists)."""
        if self.val is None or self.val.empty:
            return self.train
        return pd.concat([self.train, self.val], ignore_index=True)

    def summary(self) -> dict:
        out = {
            "n_train": int(len(self.train)),
            "n_test": int(len(self.test)),
            "n_users_kept": self.n_users_kept,
            "n_users_dropped": self.n_users_dropped,
            "n_train_users": int(self.train["user_id"].nunique()),
            "n_test_users": int(self.test["user_id"].nunique()),
            "n_train_items": int(self.train["item_id"].nunique()),
            "n_full_train": int(len(self.full_train)),
            "n_full_train_users": int(self.full_train["user_id"].nunique()),
            "n_full_train_items": int(self.full_train["item_id"].nunique()),
            "config": self.config.to_dict(),
        }
        if self.val is not None:
            out["n_val"] = int(len(self.val))
            out["n_val_users"] = int(self.val["user_id"].nunique())
        if self.cold_start is not None:
            out["cold_start"] = self.cold_start.to_dict()
        return out


@dataclass
class GlobalCutoffSplitResult:
    """Train/test frames from a corpus-wide timestamp cutoff."""

    train: pd.DataFrame
    test: pd.DataFrame
    config: GlobalCutoffConfig
    cutoff_timestamp: float
    n_users_before_filter: int
    n_items_before_filter: int
    n_test_interactions_before_filter: int
    n_users_kept: int
    n_items_kept: int
    n_test_interactions_kept: int
    cold_start: ColdStartStats | None = None

    def summary(self) -> dict:
        out = {
            "protocol": "global_time_cutoff",
            "cutoff_timestamp": float(self.cutoff_timestamp),
            "timestamp_quantile": self.config.timestamp_quantile,
            "n_train": int(len(self.train)),
            "n_test": int(len(self.test)),
            "n_users_before_filter": self.n_users_before_filter,
            "n_items_before_filter": self.n_items_before_filter,
            "n_test_interactions_before_filter": self.n_test_interactions_before_filter,
            "n_users_kept": self.n_users_kept,
            "n_items_kept": self.n_items_kept,
            "n_test_interactions_kept": self.n_test_interactions_kept,
            "n_train_users": int(self.train["user_id"].nunique()),
            "n_train_items": int(self.train["item_id"].nunique()),
            "config": self.config.to_dict(),
        }
        if self.cold_start is not None:
            out["cold_start"] = self.cold_start.to_dict()
        return out

    def as_split_result(self) -> SplitResult:
        """Adapt to :class:`SplitResult` for the shared eval harness."""
        return SplitResult(
            train=self.train,
            test=self.test,
            config=SplitConfig(
                min_ratings=self.config.min_train_ratings,
                test_fraction=0.0,
                val_fraction=0.0,
                relevance_threshold=self.config.relevance_threshold,
            ),
            n_users_kept=self.n_users_kept,
            n_users_dropped=0,
            cold_start=self.cold_start,
        )


def _holdout_tail(ordered: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``ordered`` into (head, tail) with tail = last max(1, floor(n*f))."""
    n = len(ordered)
    n_tail = max(1, int(n * fraction))
    if n_tail >= n:
        n_tail = n - 1
    if n_tail < 1:
        raise ValueError("Cannot hold out a non-empty tail while keeping a non-empty head")
    return ordered.iloc[:-n_tail], ordered.iloc[-n_tail:]


def time_based_split(ratings: pd.DataFrame, config: SplitConfig | None = None) -> SplitResult:
    """Per-user chronological holdout of the latest interactions.

    Prevents within-user leakage; does not prevent cross-user / global
    temporal leakage (see ADR-0002). With ``val_fraction > 0``, also holds out
    a validation slice from the train pool (ADR-0005).

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
    if config.val_fraction < 0 or config.val_fraction >= 1:
        raise ValueError("val_fraction must be in [0, 1)")
    if config.test_fraction <= 0 or config.test_fraction >= 1:
        raise ValueError("test_fraction must be in (0, 1)")

    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    kept = 0
    dropped = 0
    use_val = config.val_fraction > 0

    # Group in a deterministic user order.
    for _, group in ratings.groupby("user_id", sort=True):
        n = len(group)
        if n < config.min_ratings:
            dropped += 1
            continue

        # Stable sort by timestamp so ties keep relative order.
        ordered = group.sort_values("timestamp", kind="mergesort")
        try:
            full_train, test = _holdout_tail(ordered, config.test_fraction)
        except ValueError:
            dropped += 1
            continue

        if use_val:
            try:
                fit_train, val = _holdout_tail(full_train, config.val_fraction)
            except ValueError:
                dropped += 1
                continue
            train_parts.append(fit_train)
            val_parts.append(val)
        else:
            train_parts.append(full_train)

        test_parts.append(test)
        kept += 1

    if not train_parts:
        raise ValueError("No users remaining after split filters.")

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True) if use_val else None
    return SplitResult(
        train=train,
        test=test,
        val=val,
        config=config,
        n_users_kept=kept,
        n_users_dropped=dropped,
    )


def global_time_cutoff_split(
    ratings: pd.DataFrame,
    config: GlobalCutoffConfig | None = None,
) -> GlobalCutoffSplitResult:
    """Single corpus-wide timestamp cutoff (secondary sanity protocol).

    Train = all ratings with timestamp < cutoff; test = timestamp ≥ cutoff,
    where cutoff is the ``timestamp_quantile`` of all rating timestamps.
    Users with fewer than ``min_train_ratings`` train interactions are dropped.
    Test rows are restricted to users and items present in train (cold-start
    policy applied later by the eval harness for relevance).
    """
    if config is None:
        config = GlobalCutoffConfig()

    required = {"user_id", "item_id", "rating", "timestamp"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"ratings missing columns: {sorted(missing)}")
    if not 0.0 < config.timestamp_quantile < 1.0:
        raise ValueError("timestamp_quantile must be in (0, 1)")

    cutoff = float(ratings["timestamp"].quantile(config.timestamp_quantile))
    train_raw = ratings.loc[ratings["timestamp"] < cutoff].copy()
    test_raw = ratings.loc[ratings["timestamp"] >= cutoff].copy()

    n_users_before = int(test_raw["user_id"].nunique())
    n_items_before = int(test_raw["item_id"].nunique())
    n_test_before = int(len(test_raw))

    # Require enough train activity per user.
    train_counts = train_raw.groupby("user_id").size()
    eligible_users = set(train_counts[train_counts >= config.min_train_ratings].index.astype(int))
    train_items = set(train_raw["item_id"].astype(int))

    train = train_raw.loc[train_raw["user_id"].astype(int).isin(eligible_users)].reset_index(
        drop=True
    )
    # Keep test interactions only for train users; items stay for cold-start counts,
    # but cold items are dropped from relevance in apply_cold_start_policy.
    test = test_raw.loc[test_raw["user_id"].astype(int).isin(eligible_users)].reset_index(drop=True)

    return GlobalCutoffSplitResult(
        train=train,
        test=test,
        config=config,
        cutoff_timestamp=cutoff,
        n_users_before_filter=n_users_before,
        n_items_before_filter=n_items_before,
        n_test_interactions_before_filter=n_test_before,
        n_users_kept=int(train["user_id"].nunique()),
        n_items_kept=len(train_items & set(train["item_id"].astype(int))),
        n_test_interactions_kept=int(len(test)),
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

    For evaluation, ``train`` here should be the matrix the model was fit on
    (typically ``full_train`` when a validation slice exists).

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
