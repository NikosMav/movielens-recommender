# ADR-0005: Validation split and tuning protocol

## Status

Accepted (S3a)

## Context

Stages 1–2 evaluated fixed hyperparams on the test holdout. Later stages (and even baseline tuning) need a place to select hyperparameters **without** peeking at test. ADR-0002 already deferred a validation slice; this ADR defines it.

## Decision

**Validation holdout (per user, after the test split):**

1. Apply the S1–S2 test holdout first: last `max(1, floor(n * test_fraction))` interactions → test (`test_fraction=0.2`).
2. From the remaining **full-train** pool, hold out the latest `max(1, floor(n_train * val_fraction))` interactions as **validation** (`val_fraction=0.1`).
3. The earlier remainder is **fit-train**. Require ≥1 fit-train row (users who cannot satisfy this are dropped).

**Tuning protocol:**

1. Fit each grid trial on **fit-train only**.
2. Select by **validation NDCG@10** (point estimate; no bootstrap during search).
3. Refit the chosen config on **full-train** (= fit-train ∪ val).
4. Evaluate **once** on **test** (with the usual bootstrap CIs and cold-start policy).

**Never tune on test.** Test remains sealed until the final evaluation of a chosen config.

**Why `val_fraction=0.1`:** same rule shape as the test split (`max(1, floor(fraction · n))`), small enough that full-train ≈ S2 train (so default baselines stay comparable), large enough that every kept user has ≥1 val row for selection. A larger val slice (e.g. 0.2 of train) would shrink fit-train further without a clear gain for these small grids.

**Grids (committed under `results/tuning/`):**

| Model | Search space |
| --- | --- |
| ALS | `factors ∈ {32,64,128}`, `regularization ∈ {0.01,0.1}`, `alpha ∈ {20,40}`, `iterations=15` |
| item–item cosine | `k_neighbors ∈ {40,100,200}`, `shrinkage ∈ {0,100}`, `min_common=1` |

`most_popular` has no tunable hyperparameters.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Tune on test / report best test trial | Invalidates stage comparisons; silent leakage |
| Nested cross-validation | Heavier than needed for a small solo grid on ml-1m |
| Global validation cutoff for tuning | Couples tuning to a different protocol than the headline per-user split |
| `val_fraction=0.2` of train | More aggressive shrink of fit-train; 0.1 matches the “small holdout” intent |
| Random val rows | Breaks chronological consistency with the test holdout |

## Consequences

- Headline `results/*.json` reports both **S2 defaults** and **`_tuned`** models refit on full-train.
- Tuning artifacts live in `results/tuning/*.json` (grid, per-trial val scores, chosen configs).
- Absolute test scores for defaults should match S2 when full-train equals the old train matrix (same test holdout).
- Global-time-cutoff sanity check (ADR-0002) **reuses** these tuned configs and does **not** re-tune.
