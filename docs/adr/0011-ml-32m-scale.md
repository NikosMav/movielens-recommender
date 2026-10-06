# ADR-0011: MovieLens 32M scale-up

## Status

Accepted (S3d). The protocol in this ADR is fixed from `results/budget/ml-32m.json` before any ml-32M test metric. The ranking paragraph is filled from `results/ml-32m.json` after that single test pass.

## Context

ml-1m has about 3.7k movies and is dense enough for a full item–item Gram and a full EASE inverse. MovieLens 32M is the catalog where a neural model is supposed to pull away from neighbour methods: about 32M ratings, 200k users, 87k movies. The question is whether the ml-1m order holds there, and whether the two-tower and LambdaRank gaps versus item–item and EASE grow, shrink, or flip.

The same rating-threshold protocol is used: per user, the last `max(1, floor(0.2 n))` ratings are test; from the rest, the last `max(1, floor(0.1 n_train))` are validation; relevant means rating ≥ 4.0. CI does not download this zip (`MOVIELENS_ALLOW_DOWNLOAD=0`). Tests use a tiny `ratings.csv` / `movies.csv` fixture.

## Decision

**Data.** Official GroupLens `ml-32m.zip`, SHA-256 `e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee`, CSV layout (`userId,movieId,rating,timestamp` and `movies.csv`). No `users.dat`, so ranker demographics stay `off`.

**What was measured** (`results/budget/ml-32m.json`, no ranking metric). The VM has `mem_total_mib` 16014.0, `swap_total_mib` 0.0, `commit_limit_mib` 8007.0, 4 CPUs, `overcommit` 0, `overcommit_ratio` 50. At probe start `mem_available_mib` was 4872.5 and `commit_headroom_mib` was 6189.7.

Loading and the per-user split took `load_sec` 44.116 and `split_sec` 19.86. Clean ratings: 32000204. Fit-train 23198846 rows, validation 2478684, test 6322674, full train 25677530. Train users 200948, fit-train items 66819, full-train items 71364, movie rows 87585. Users with a warm relevant test item: 196517. RSS after the split was 2127.3 MiB, with commit headroom 4124.6 MiB.

A dense users × items score matrix does not fit, so every model is scored on one seeded user sample. Training still uses every training interaction. Histories are not truncated. The sample is the same ids for validation metrics, ranker labels, and test metrics.

**Sample size 8000, seed 42.** ml-1m item–item evaluates `n_eval_users` 5958 (`results/ml-1m.json`). 8000 is that order of magnitude. Scoring all 196517 eligible users would make the ranker candidate matrices (users × K=200) too large to hold next to the training frames, and the models would no longer share one evaluation population if only the ranker were sampled. 8000 is the locked size.

**Item–item** stays sparse top-k. `k_neighbors=0` would allocate a 66819² Gram. Three blocks of the real fitter, on fit-train, took 3.24 s (rows 0:251), 0.874 s (33409:33660), and 0.745 s (66568:66819). Mean block 1.62 s, 267 blocks, extrapolated full fit 432.5 s. Head item ids are slower than the tail, so a full fit can land off that mean. The grid is three configs, plus the YAML default and one full-train refit.

**EASE** is the top 12000 items by fit interaction count (ties: smaller item id). Three float64 matrices is the peak the fitter holds (Gram, inverse, B). With the ratings frames resident, commit headroom was 4006.5 MiB. These sizes were skipped because three matrices exceeded 85% of that headroom:

| N | needed_mib | status |
| --- | --- | --- |
| 30000 | 20599.4 | skipped_over_commit_headroom |
| 25000 | 14305.1 | skipped_over_commit_headroom |
| 20000 | 9155.3 | skipped_over_commit_headroom |
| 16000 | 5859.4 | skipped_over_commit_headroom |
| 14000 | 4486.1 | skipped_over_commit_headroom |

N=12000 needed 3295.9 MiB, allocated, and inverted a diagonally dominant matrix in `inv_sec` 13.35. A real `EASERecommender` fit at `l2=5000`, `max_items=12000`, `cache_user_scores=false` took `fit_sec` 41.106 and fit 12000 of 66819 items. Peak RSS during the probe (`hwm_mib`) reached 6640.0. Items outside the head are not scored, so tail NDCG for EASE is limited by that restriction. The λ grid is `{500, 2000, 5000, 10000}`.

**RP3beta** uses the blocked top-k path. Three blocks extrapolated to 126.5 s for the walk, not counting the per-row prune. Grid: `(alpha, beta, top_k)` in `{(0.5, 0.5, 100), (1.0, 0.5, 100), (0.5, 0.5, 300), (1.0, 0.0, 100)}`.

**ALS** implicit, 2 iterations on fit-train, took `fit_sec` 17.285. Multiplying by 15/2 gives 129.6 s and overestimates a 15-iteration fit because setup is included. Grid: three configs in `configs/ml-32m.yaml` (factors 32/64, regularization 0.1/0.01, alpha 40/20, 15 iterations), plus the YAML default and one refit.

**Two-tower** reuses the ml-1m architecture (embedding 64, history 50, batch 1024, temperature 0.1, weight decay 1e-4). The first feature build stored one Python set per user and committed 8019.5 MiB against the 8007.0 MiB limit (`commit_headroom_mib` −12.5, RSS 6057.7). Seen ids are now int64 arrays. The rerun of that stage (`seen_storage=int64_arrays`) built features in 23.004 s, RSS 3636.7 MiB, commit headroom 2656.0 MiB. Twenty AdamW batches took 2.369 s (0.1445 s/batch, 11575 batches/epoch), which extrapolates to 1371.2 s per epoch. `recommend` on 100 users took 0.098 s, which extrapolates to 7.6 s for 8000 users. One seed only (`42`): each extra seed repeats the full-train refit. The grid is two learning rates, 0.001 and 0.003, `max_epochs` 6, `patience` 2. At the cap that is 2 × 6 × 1371.2 s of tuning plus up to 6 epochs of full-train refit plus up to 6 epochs of the ranker's fit-train refit. Early stopping can end sooner. That product is a cap from the extrapolation, not a measured full training run.

**LambdaRank** uses the ADR-0007 candidate rule (K=200; validation recall@200, then recall@100, then `union_balanced`, `item_item`, `two_tower`, `union_unbalanced`). Demographics `off`. Seeds `[42, 43, 44]`. The full-train two-tower top-k for the test sample is materialized first and that feature pack is dropped before the fit-train tower is built, so two packs are not live together.

**Global time cutoff is not run.** It is a second full training pass. One two-tower epoch already extrapolates to 1371.2 s.

**Cold-start `seen` maps** are built only for users in the evaluation frame. Stats still use the full train catalog. Contents for those users match a full groupby.

## Pre-registered comparison

ml-1m test NDCG@10 from `results/ml-1m.json` (demographics off; LambdaRank and two-tower are 3-seed means). Four decimals are the README rendering (`f"{value:.4f}"`):

| model | raw ndcg@10 | four decimals |
| --- | --- | --- |
| lambdarank | 0.127264 | 0.1273 |
| item_item_cosine (`k_neighbors=0`) | 0.120135 | 0.1201 |
| two_tower | 0.119246 | 0.1192 |
| item_item_cosine_tuned | 0.119235 | 0.1192 |
| ease | 0.11838 | 0.1184 |
| rp3beta | 0.113834 | 0.1138 |
| als_tuned | 0.09062 | 0.0906 |
| most_popular | 0.089508 | 0.0895 |
| als | 0.068617 | 0.0686 |

Raw order: lambdarank, item_item_cosine, two_tower, item_item_cosine_tuned, ease, rp3beta, als_tuned, most_popular, als.

Gaps of interest are differences of the four-decimal renderings:

- two_tower − item_item_cosine = 0.1192 − 0.1201 = −0.0009
- two_tower − ease = 0.1192 − 0.1184 = 0.0008
- lambdarank − item_item_cosine = 0.1273 − 0.1201 = 0.0072
- lambdarank − ease = 0.1273 − 0.1184 = 0.0089

On ml-32M the same four differences are taken from the README rendering of `results/ml-32m.json`. **Grow** means the absolute gap stays the same sign and gets larger. **Shrink** means it moves toward zero without changing sign. **Flip** means the sign changes. ml-32M item–item is sparse top-k, not the ml-1m all-neighbour model. ml-32M EASE does not score the tail catalog. Coverage@10 is the sample's lists, not the 196517-user population. ml-32M two-tower is one seed; ml-1m two-tower is a 3-seed mean. The ml-1m demographics-on LambdaRank row is not part of this comparison.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Evaluate every eligible user | The shared ranker matrices do not fit with the training frames; models would not share one population if only some were sampled |
| EASE on all 66819 items, or on 20k–30k | Three N×N matrices exceed the commit headroom measured above |
| Dense item–item (`k_neighbors=0`) | The Gram does not fit |
| Three two-tower seeds | Each extra seed repeats a full-train refit of up to 6 × 1371.2 s |
| Global time cutoff on ml-32M | A second full pass, dominated by the same two-tower training |
| Keep Python sets for seen items | The set-based feature pack committed past the 8007.0 MiB limit |

## Outcome

The test pass had not been run when the decisions above were locked. The paragraph below is copied from `results/ml-32m.json` after that pass and does not change the sample, the EASE head size, the grids, or the seed counts.

Pending: `results/ml-32m.json` is not in the tree yet.
