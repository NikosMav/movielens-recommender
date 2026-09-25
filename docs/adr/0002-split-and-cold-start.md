# ADR-0002: Time-based split and cold-start policy

## Status

Accepted (S1)

## Context

Random splits leak future interactions into training and inflate offline metrics. We also need an explicit policy for users/items that never appear in train.

## Decision

**Split (per user, deterministic):**

1. Drop users with fewer than `min_ratings` interactions (default **5**).
2. Sort that user’s interactions by `timestamp` ascending (stable mergesort).
3. Hold out the last `max(1, floor(n * test_fraction))` rows as test (`test_fraction=0.2`); remainder is train. Require ≥1 train row.

**Cleaning (before split):** drop nulls; require rating in `[0.5, 5.0]`; require positive ids and timestamps; for duplicate `(user_id, item_id)` keep the **latest** timestamp (then highest rating on ties).

**Cold-start policy:**

- **Cold users** (no train interactions): excluded from evaluation (they never appear after the split filters).
- **Cold items** (item id absent from train): removed from each user’s **relevant** test set before scoring. Users left with zero warm relevant items are excluded from ranking-metric averages; counts of dropped cold relevant items / users are recorded in results JSON.
- Models may only meaningfully recommend train-catalog items; eval still filters already-seen train items.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Global temporal cutoff | Starves late-joining users; harder to keep ≥1 train/test per user |
| Leave-one-out latest only | Too little test signal for NDCG@20 on sparse users |
| Keep cold items in relevance | Rewards recommending items the model never saw; confuses stage comparisons |
| Impute / random cold-start recs in S1–S2 | Out of scope; belongs in a dedicated cold-start stage write-up |

## Consequences

Metrics are leakage-safe and comparable across stages. Absolute scores are not comparable to papers that keep cold items or use random splits.
