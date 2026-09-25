# ADR-0002: Time-based split and cold-start policy

## Status

Accepted (S1)

## Context

Random splits put a user’s later interactions into training and inflate offline metrics. We also need an explicit policy for users/items that never appear in train. A **per-user** chronological holdout is easy to keep balanced (every kept user has train and test) but is not a global time barrier.

## Decision

**Split (per user, deterministic):**

1. Drop users with fewer than `min_ratings` interactions (default **5**).
2. Sort that user’s interactions by `timestamp` ascending (stable mergesort).
3. Hold out the last `max(1, floor(n * test_fraction))` rows as test (`test_fraction=0.2`); remainder is train. Require ≥1 train row.

This **prevents within-user leakage** (no user’s test interactions appear in that user’s train). It **does not prevent cross-user / global temporal leakage**: train still contains other users’ interactions from after a given user’s test period, so item–item neighbours, ALS factors, and popularity counts can see some “future” signal relative to that user.

**Cleaning (before split):** drop nulls; require rating in `[0.5, 5.0]`; require positive ids and timestamps; for duplicate `(user_id, item_id)` keep the **latest** timestamp (then highest rating on ties).

**Cold-start policy:**

- **Cold users** (no train interactions): excluded from evaluation (they never appear after the split filters).
- **Cold items** (item id absent from train): removed from each user’s **relevant** test set before scoring. Users left with zero warm relevant items are excluded from ranking-metric averages; counts of dropped cold relevant items / users are recorded in results JSON.
- Models may only meaningfully recommend train-catalog items; eval still filters already-seen train items.

## Alternatives considered

| Alternative | Why rejected (for S1–S2) |
| --- | --- |
| Global temporal cutoff | Starves late-joining users; harder to keep ≥1 train/test per user; deferred as an S3 sanity check |
| Leave-one-out latest only | Too little test signal for NDCG@20 on sparse users |
| Keep cold items in relevance | Rewards recommending items the model never saw; confuses stage comparisons |
| Impute / random cold-start recs in S1–S2 | Out of scope; belongs in a dedicated cold-start stage write-up |

## Consequences

- Protocol **prevents within-user leakage**; it **does not prevent cross-user / global temporal leakage**. Absolute scores are not comparable to papers that use a global cutoff or random splits.
- Metrics remain comparable across stages on **this same split**; stage ranking could still shift under a stricter global-time protocol.

## Follow-ups

1. **Validation split for tuning** — **done in S3a** (ADR-0005): per-user val holdout from the train pool (`val_fraction=0.1`); tune on val only; refit on full train; evaluate once on test.
2. **Global-time-cutoff sanity check on ml-1m** — **done in S3a** as a **secondary** table under `results/global_cutoff/ml-1m.json` (80th-percentile corpus timestamp cutoff; uses per-user-protocol tuned configs **without** re-tuning). Answers whether model ranking holds when cross-user future signal is removed; **not** the headline result.
