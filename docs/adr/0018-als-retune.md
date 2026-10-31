# ADR-0018: Implicit ALS retune on ml-1m

## Status

Accepted. This is ADR-0018, the next number on main at `a1db6a0`, after ADR-0017 (beyond accuracy). The search space, the validation protocol, the budget rule, and the decision rule below are fixed before any retune validation score. A timing probe may shrink the grid before that search; it records no ranking metric. The outcome section is filled only after `results/als-retune/ml-1m.json` exists.

## Context

The committed ml-1m ALS model (`results/tuning/ml-1m.json`, 12 configs) has test NDCG@10 0.09062, which the README renders as 0.0906. Item–item cosine on the same split is 0.120135 (0.1201). EASE and the two-tower sit between them. The committed ALS winner is on three edges of that small grid: `factors=128` (high of `{32, 64, 128}`), `regularization=0.1` (high of `{0.01, 0.1}`), and `alpha=20` (low of `{20, 40}`). Iterations were fixed at 15. Confidence was only `1 + alpha * rating` on ratings of at least 4.

That leaves the gap ambiguous. It may be implicit ALS on this split, or it may be the search. This ADR re-tunes ALS under the same protocol as EASE and RP3beta (ADR-0005, ADR-0008). ADR-0015 changed top-n tie-breaking after the committed file was written, so the published 0.09062 is not a same-code number. The old hyperparameter config is fit again here and scored on the test split once, as the reference.

## Decision

**Data and split.** ml-1m, per-user time split, `test_fraction=0.2`, `val_fraction=0.1`, `min_ratings=5`, relevance 4.0, seed 42. Same split as `results/ml-1m.json`. No eval-user sample.

**What is fixed, not searched.**

- Positive mask: rating ≥ 4. Lower ratings are dropped in every scheme, as in the committed model. The mask is the relevance threshold. It is not a grid axis.
- `implicit` 0.7.3 `AlternatingLeastSquares` with the same arguments the committed code passed (`use_gpu=False`, other arguments at library defaults, including library `alpha=1`). The library multiplies the matrix by its `alpha` only when that value is not 1, so the confidence built below is the confidence ALS sees. `random_state=42`.
- Top-n ties break by lower item index (ADR-0015).

**Confidence schemes.** Applied to the positive ratings only.

| scheme | confidence |
| --- | --- |
| `linear` | `1 + alpha * r` (the committed formula) |
| `log` | `1 + alpha * ln(1 + r)` |
| `bm25` | `implicit.nearest_neighbours.bm25_weight` on the raw positive ratings, `K1=100`, `B=0.8` (the 0.7.3 defaults). `alpha` is not a parameter |
| `tfidf` | `implicit.nearest_neighbours.tfidf_weight` on the same raw ratings. `alpha` is not a parameter |

`linear` and `log` share one alpha grid. The two formulas put alpha on different scales; crossing both with the same alphas is the comparison.

**Base grid (144 configs).** Counted as 2 schemes × 3 × 3 × 2 × 3 alphas, plus 2 library schemes × 3 × 3 × 2.

| axis | values | why these |
| --- | --- | --- |
| factors | 64, 128, 256 | 128 won the committed grid. 256 is one factor of two past it. 32 lost in every committed cell and is dropped. 64 stays |
| regularization | 0.01, 0.1, 1.0 | 0.1 won. 1.0 is one factor of ten past it, the same ratio as 0.01 → 0.1 |
| alpha (`linear`, `log` only) | 10, 20, 40 | 20 won. 10 is one factor of two below it, the same ratio as 40 → 20. 40 stays |
| iterations | 15, 30 | 15 was fixed before. 30 is twice that |
| confidence | linear, log, bm25, tfidf | the four schemes above |

Grid order, which is also the tie-break order: confidence (`linear`, `log`, `bm25`, `tfidf`), then factors, regularization, iterations, and alpha, each ascending. `bm25` rows also store `bm25_k1=100` and `bm25_b=0.8`.

**Budget.** `results/tuning/ml-1m.json` holds 8 EASE configs and 125 RP3beta configs (the extended grids in ADR-0008). The original ALS search was 12. 144 is the plan: the same order as RP3beta's 125, and larger than EASE's 8.

A probe (`results/budget/als-retune-probe.json`) times two linear configs, factors 64 / 15 iterations and factors 256 / 30 iterations, including a validation-sized score pass. It stores wall time only (`ranking_metrics_recorded` is false). Time for a config is modeled as `a + b * factors * iterations`, fit from those two samples. If `a` or `b` is negative, time scales from the slower sample with no fixed overhead. The projected sum over the grid plus a 1200 s reserve (old-config refit, winner refit, item–item, and up to four edge steps) must fit in 3 hours (10800 s). If it does not, drop `factors=64` and recompute (96 configs). If it still does not, also drop `iterations=30` from the factorial (48 configs). The search uses that locked grid. No validation score is kept from the probe. If the sum of recorded trial fit and score seconds, plus the time since this process resumed the remaining trials, passes 9900 s, the search stops, selects among finished trials, skips any edge step not yet scored, and records `truncated`.

**One edge step, after the grid, validation only.** Ordered axes are factors, regularization, iterations, and alpha (alpha only for `linear` and `log`). A selected value equal to the min or max of the values scored on that axis is an edge. Weighting is a closed set of four schemes, not an edge. Factors and iterations step by 2; regularization by 10; alpha by 2. The low edge of iterations is flagged and not extended: 15 is the published iteration count, and fewer iterations is not a direction the committed result left open. Each edge that has a step adds one config, other hyperparameters held at the winner. At most four. They are appended after the base grid, so a tie keeps the earlier config. No second step. Test is still scored once, after this re-selection.

**Protocol.** Same as ADR-0005.

1. Fit each trial on fit-train only.
2. Select by validation NDCG@10, point estimate, no bootstrap. Tie: earliest grid config.
3. Refit the chosen config on full train.
4. Score test once, bootstrap 1000, alpha 0.05, seed 42.

The scorer used for selection returns the same lists as `ALSRecommender.recommend`. The winner's validation NDCG@10 is checked with `ndcg_point_estimate` before test.

**Same-code references, not candidates.**

- Old ALS: the committed winner, `factors=128`, `regularization=0.1`, `iterations=15`, `alpha=20`, `linear`. Its validation number is that grid cell. It is refit on full train and scored once on test.
- Item–item: the committed default, `k_neighbors=0`, `min_common=1`, `shrinkage=0`, the model behind 0.120135. Not re-tuned. Fit on full train, scored once on test.

**Paired differences.** Per-user test NDCG@10, retuned minus reference, on the users the harness keeps, `paired_bootstrap_delta_ci` (1000 resamples, alpha 0.05, seed 42). Two pairs: retuned minus old ALS, and retuned minus item–item.

**Decision.** Negative result if the selected config's test NDCG@10 is below the item–item test NDCG@10 from this same run, compared at the stored 6-decimal values. The paired interval is reported either way. A point estimate that trails is a negative result even when the interval contains 0. `results/ml-1m.json` is not rewritten. The headline table stays the committed one.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Search the rating ≥ 4 mask | Changes who is in the factor model. Out of scope for a confidence and capacity retune |
| Pass `alpha` into `AlternatingLeastSquares` as well as into the matrix | The library would scale confidence a second time. The committed path does not |
| Tune BM25 `K1` and `B` | The question is whether to use the library weighting, not a second grid inside it |
| Pick the winner on test | The same leakage ADR-0005 closed |
| Treat a categorical weighting choice as a grid edge | The four schemes are the whole set |
| Extend iterations below 15 | The committed run did not show that fewer iterations were the open direction |

## Outcome

**Negative.** The probe (`results/budget/als-retune-probe.json`, `ranking_metrics_recorded` false) projected 330.079 s for the full 144-config grid plus the 1200 s reserve, inside 10800 s, so the grid was not cut (`cut` null). The search scored all 144 and then the three edge steps (`truncated` false). Files: `results/tuning/als_retune_ml-1m.json` and `results/als-retune/ml-1m.json`. `results/ml-1m.json` is unchanged.

Validation selected TF-IDF confidence, factors 64, regularization 1.0, iterations 30. Validation NDCG@10 is 0.079694, and the harness check on a fresh fit is the same 0.079694. Best validation NDCG@10 by scheme on the base grid: linear 0.069351 (`alpha=10`, factors 128, regularization 0.1, iterations 30), log 0.07493 (`alpha=10`, factors 64, regularization 0.01, iterations 30), BM25 0.07456 (factors 64, regularization 0.01, iterations 30), TF-IDF 0.079694. The old linear cell (`alpha=20`, factors 128, regularization 0.1, iterations 15) reproduced its committed validation score, 0.064625.

On the base grid that winner sat on the low factors edge (64), the high regularization edge (1.0), and the high iterations edge (30). The one step past each scored lower: factors 32 at 0.077147, regularization 10.0 at 0.078915, iterations 60 at 0.079327. After those steps the selected config sits on no edge of the values scored (`grid_edges` empty, `on_grid_edge` false).

Test, scored once, 5958 users, bootstrap 1000, alpha 0.05, seed 42:

| model | test NDCG@10 | 95% CI |
| --- | --- | --- |
| old ALS, this code | 0.09062 | [0.087425, 0.09395] |
| retuned ALS | 0.113814 | [0.109856, 0.118148] |
| item–item cosine | 0.120135 | [0.115829, 0.124155] |

The same-code old ALS and item–item point estimates match the committed headline (0.09062 and 0.120135). Paired retuned-minus-old is 0.023195 [0.019684, 0.026683], excludes 0. Paired retuned-minus-item–item is −0.006321 [−0.01039, −0.002086], excludes 0. 0.113814 is below 0.120135 at the stored 6 decimals, so the pre-registered rule records a negative result. The search closed most of the gap to item–item; it did not close it.
