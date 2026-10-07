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

The sample is partly test-informed. `scale.py` picks eligible users with `warm_relevant_user_ids(full, split.test, ...)`: at least one test rating ≥ 4 whose item is in the train catalog. Validation rows are then restricted to that sample, so tuning and ranker training see users chosen partly by their test ratings. 196,517 of 200,948 train users (97.8%) are eligible. That is the same eligibility rule the harness uses for test metrics. The expected effect is small. Drawing the sample without test ratings is the cleaner alternative. That alternative was not run; it is a known limitation.

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

## Implementation note before test metrics

The first end-to-end process exited while retrieving validation candidates, before `results/ml-32m.json` existed. Sparse item–item `topk_with_scores` left a CSR product as a CSR matrix, and `np.asarray` refused it. No test metric was written. The scorer now densifies that product. The sample, EASE head size, grids, and seed counts are unchanged. The run was restarted with the saved tuning files.

After `results/ml-32m.json` was written, the ranker recreate command in that file was corrected from `configs/default.yaml` to `configs/ml-32m.yaml`. No metric changed.

## Outcome

The decisions above were not changed after the test file existed. Chosen configs that sit on a grid edge were left there: item–item `k_neighbors=50` (low edge of `{50, 100, 200}`) and `shrinkage=0` (natural bound), RP3beta `alpha=0.5` (low edge of `{0.5, 1.0}`), `beta=0.5` (high edge of `{0.0, 0.5}`), and `top_k=300` (high edge of `{100, 300}`), ALS `regularization=0.01` (low edge of `{0.1, 0.01}`), `factors=64` (high edge of `{32, 64}`), and `alpha=20` (low edge of `{40, 20}`), two-tower learning rate `0.001` (low edge of `{0.001, 0.003}`). EASE `l2=5000` is interior. They were left in place because of the locked compute budget: extending the grids would mean another multi-hour run after the test file existed. The full-softmax two-tower and the iALS re-tune are queued follow-ups that will revisit tuning.

`results/ml-32m.json` `runtime_sec` is 6560.575. That is the process started with the saved tuning files. It does not include the earlier grid search. The two-tower full-train refit in that file is `train_wall_time_sec` 2637.302 (5 epochs, one seed). The ranker block `runtime_sec` is 2841.798, which includes its own fit-train two-tower refit.

Test NDCG@10 below is the README four-decimal rendering of `results/ml-32m.json`. LambdaRank is the 3-seed mean. Two-tower is one seed.

| model | four-decimal NDCG@10 |
| --- | --- |
| two_tower | 0.1436 |
| lambdarank | 0.1423 |
| ease | 0.1243 |
| item_item_cosine (k=100, shrinkage 0) | 0.1113 |
| item_item_cosine_tuned (k=50, shrinkage 0) | 0.1093 |
| rp3beta | 0.0991 |
| als_tuned | 0.0983 |
| als | 0.0800 |
| most_popular | 0.0699 |

ml-1m order was lambdarank, item_item_cosine, two_tower, ease, rp3beta, als_tuned, most_popular. On ml-32M the point-estimate order is two_tower, lambdarank, ease, item_item_cosine, rp3beta, als_tuned, most_popular (top two tied). The ranking does not hold.

Gaps versus the pre-registered four-decimal differences:

- two_tower − item_item_cosine flips, from −0.0009 to +0.0323 (0.1436 − 0.1113).
- two_tower − ease grows, from +0.0008 to +0.0193 (0.1436 − 0.1243).
- lambdarank − item_item_cosine grows, from +0.0072 to +0.0310 (0.1423 − 0.1113).
- lambdarank − ease grows, from +0.0089 to +0.0180 (0.1423 − 0.1243).

Two-tower's point estimate is also above the LambdaRank mean (0.1436 vs 0.1423). On ml-1m LambdaRank led (0.1273 vs 0.1192), so that pair flips too. The intervals overlap: two-tower NDCG@10 CI [0.1398, 0.1479], LambdaRank primary-seed CI [0.1391, 0.1472]. `no_ranker` on the winning two-tower candidate list is 0.1436, the same point as the retriever, so LambdaRank did not raise NDCG@10 over that list.

Recall@200 is 0.6243 (two-tower), 0.5961 (EASE), 0.5541 (item–item). Coverage@10 is 0.0411, 0.0255, and 0.0308 on that same order. Tail NDCG@10 is 0.0006 (two-tower), 0.0000 (EASE), 0.0022 (item–item), 0.0049 (RP3beta). EASE's tail is zero because items outside the top 12000 are not scored. Two-tower's NDCG lead is on the head (head NDCG@10 0.1460 vs item–item 0.1114), not on the tail.

Caveats that were fixed before this file: ml-32M item–item is sparse top-k, not the ml-1m all-neighbour model; the evaluation is 8000 of 196517 users (`user_ids_sha256` `6b0d5cf6ce14cf5815e71f620b1dd5f37f7e6f09ac13ae5f3d138563b035c980`); coverage is that sample's lists; two-tower is one seed here and a 3-seed mean on ml-1m; there is no global time cutoff.
