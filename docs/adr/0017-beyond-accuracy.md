# ADR-0017: Beyond-accuracy metrics

## Status

Accepted. The definitions in this decision were written before any beyond-accuracy number was computed. The outcome section is filled only after `results/beyond-accuracy/ml-1m.json` exists, and it does not change the definitions.

## Context

NDCG@10 says whether the top of the list hits held-out likes. It does not say whether those hits are the same popular titles for every user. Coverage and mean popularity are already diagnostics (ADR-0003). They are not, by themselves, a beyond-accuracy panel: novelty and intra-list diversity are still missing, and the diagnostics are not shown next to a recomputed NDCG on the lists that current code actually emits.

Top-n ties now break by item index (ADR-0015). Rebuilding a model can therefore change a list relative to the one behind the committed NDCG, even when the hyperparameters and the seed are unchanged. The comparison has to say so.

No model artifact is committed. Rebuilding from the committed hyperparameters and seeds is the way to get lists. That rebuild is not a new model and not a new search.

## Decision

Score ml-1m only, at k=10, on the existing per-user time split and on the same warm-relevant test users as `results/ml-1m.json`. Fit on full train. Do not retune. Do not change a hyperparameter. ml-32M is future work and is not scored here.

The seven models, and the committed block each one is rebuilt from:

| Panel name | Committed source | What is fit |
| --- | --- | --- |
| `most_popular` | `hyperparameters.most_popular` | Interaction counts on full train |
| `item_item_cosine` | S2 defaults: `k_neighbors` 0, `shrinkage` 0, `min_common` 1 | Full train. This is the untuned item–item model, not `item_item_cosine_tuned` |
| `als_tuned` | factors 128, regularization 0.1, alpha 20, iterations 15, seed 42 | Full train |
| `ease` | `l2` 5000 | Full train |
| `rp3beta` | alpha 0.4, beta 0.6, top_k 2000 | Full train |
| `two_tower` | Validation winner, refit for `best_epoch` 6, seed 42 | Full train. One seed, not the three-seed mean |
| `lambdarank` | LightGBM LambdaRank on two-tower candidates, K=200, seed 42, demographics off | Ranker trained on the validation window with fit-train features, then the test lists come from the full-train retrievers. Same early-stop-then-refit procedure as S4 |

`item_item_cosine` is the item–item model in this panel. "ALS (tuned)" is `als_tuned` because the untuned ALS is a different, weaker model. The ranker is the S4 LambdaRank whose candidate set is the two-tower, which is the committed winner. Its feature set has no demographic columns. `configs/ml-1m.yaml` now defaults demographics to `both`; this run does not use that default.

Lists are the harness lists: already-seen train items removed, top 10 of what remains. Ties use `top_n_indices` (ADR-0015). Two-tower and LambdaRank lists are the primary seed only (the config seed, 42).

### Metrics

All of them use the train matrix the model was fit on (full train).

**Catalog coverage@10.** The existing ratio, unique recommended items over the train catalog. **Point estimate only.** The harness bootstrap is `bootstrap_mean_ci`, a percentile bootstrap of a mean. Coverage is a set-union. Resampling users with replacement shrinks the union, so that interval sits below the point estimate (ADR-0003). This panel does not attach that interval.

**Novelty@10.** For an item with train interaction count `c(i) > 0` and `N` the number of train interactions,

`self-information(i) = -log2(c(i) / N)`.

A user's score is the mean self-information of the top 10. Items with no positive train count are left out of that mean. A list with none left scores 0. The reported novelty is the mean of the user scores.

**Intra-list diversity@10.** One minus the mean pairwise cosine similarity of the top 10, then averaged over users.

The item vector is the genre multi-hot from `movies.dat`, the same vector `build_item_side_features` builds for the two-tower: one coordinate per entry of `GENRES` in `movies.py` (19 genres, including `(no genres listed)`). The ml-1m label `Children's` maps to `Children`. An item missing from `movies.dat`, or one whose genres are empty or unknown, gets a single 1 on `(no genres listed)`. Cosine uses these raw 0/1 vectors, not a separately normalized copy. Pairs are positions `i < j` in the top-k prefix. A list with fewer than two items scores 0. A zero-norm vector has cosine 0 with every item; the movies.dat builder does not emit an all-zero vector.

**Popularity bias.** The existing `mean_popularity@10`: mean over users of the mean train interaction count in the top 10. Unknown items contribute 0. This panel does not add a second name for the same number.

**Long-tail share@10.** Mean over users of the fraction of the top 10 that falls in the train tail. The tail is the complement of the existing head: `n_head = max(1, int(n_items * 0.2))`, items ordered by interaction count descending and item id ascending (`head_tail_items`). What does not fit in that head is the tail. When the catalog size is divisible by 5, that tail is the bottom 80% of items. The integer cut keeps any leftover item in the tail. An empty list scores 0.

**NDCG@10.** The existing binary NDCG, relevance at rating ≥ 4, recomputed on these lists. It is not copied out of `results/ml-1m.json`.

### Uncertainty

User-level means (novelty, intra-list diversity, mean popularity, long-tail share, NDCG) get the harness percentile bootstrap: 1000 resamples, alpha 0.05, seed 42, via `bootstrap_mean_ci`. Each metric is resampled on its own with that seed, which is what the ranking harness already does. The user order is the order `collect_recommendations` inserts, which follows `relevant` from `apply_cold_start_policy`. Every reported interval has to contain its point estimate.

### Committed NDCG

For the five models with one committed test number, the comparison target is `metrics.<name>.ndcg@10` in `results/ml-1m.json`.

For two-tower and LambdaRank, the lists are seed 42, so the comparison target is that seed's test NDCG@10 (`two_tower.per_seed` and `ranker.per_seed`). The headline `metrics.two_tower.ndcg@10` and `metrics.lambdarank.ndcg@10` are means over three seeds. Those headline numbers are stored beside the seed-42 target. A match is equality of the values printed to 4 decimal places. A mismatch is reported as a mismatch.

The results file records the host block from ADR-0015 (OS, CPU count, Python, NumPy SIMD features).

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| User-bootstrap CI for coverage | Invalid for a set-union; ADR-0003 already dropped it |
| A new popularity metric next to `mean_popularity@10` | Same quantity under a second name |
| Intra-list diversity on embedding cosine | The genre vector is fixed, inspectable, and already in the repo. An embedding would move when the tower is rebuilt |
| `item_item_cosine_tuned` instead of the S2 item–item | The request names item–item and, separately, ALS (tuned). The tuned item–item is a different model |
| Average the three two-tower seeds into one list | A mean is not a list. The panel scores the primary seed and says so |
| Score ml-32M in this change | Far larger refit, and no committed request to retune or rebuild it |
| Copy the old NDCG into the new file | The lists may have moved under ADR-0015. The NDCG has to be the one for these lists |

## Outcome

Scored on the committed ml-1m split, 5958 eval users, full train of 802,553 interactions and 3,667 items. The tail cut keeps 733 head items and 2,934 tail items (80.0109% of the catalog; 3,667 is not divisible by 5). Runtime 143.335 seconds. Host: Linux 6.12.94+, x86_64, 4 CPUs, Python 3.12.3, NumPy SIMD AVX2, AVX512F, AVX512_ICL, AVX512_SKX, FMA3, SSE42. Library pins match `results/ml-1m.json` (NumPy 2.2.6, implicit 0.7.3, Torch 2.6.0+cpu, LightGBM 4.6.0). Definitions above were not changed after these numbers.

Point estimates at k=10. Intervals are the 1,000-resample user bootstrap except coverage, which stays a point estimate. Full precision is in `results/beyond-accuracy/ml-1m.json`.

| model | NDCG@10 | Coverage | Novelty | Intra-list diversity | Mean popularity | Long-tail share |
| --- | --- | --- | --- | --- | --- | --- |
| most_popular | 0.0895 [0.0857, 0.0935] | 0.0325 | 8.4476 [8.4436, 8.4519] | 0.6321 [0.6299, 0.6344] | 2325.9361 [2319.9340, 2331.5737] | 0.0000 [0.0000, 0.0000] |
| item_item_cosine | 0.1201 [0.1158, 0.1242] | 0.1274 | 8.8996 [8.8910, 8.9077] | 0.6288 [0.6257, 0.6321] | 1794.5672 [1786.4853, 1803.2589] | 0.0031 [0.0022, 0.0040] |
| als_tuned | 0.0906 [0.0874, 0.0940] | 0.4541 | 9.8391 [9.8212, 9.8573] | 0.6518 [0.6482, 0.6552] | 1118.0085 [1106.0058, 1129.8557] | 0.1117 [0.1076, 0.1156] |
| ease | 0.1184 [0.1142, 0.1226] | 0.2100 | 9.1409 [9.1306, 9.1524] | 0.6301 [0.6264, 0.6336] | 1575.2565 [1565.4193, 1584.1931] | 0.0059 [0.0049, 0.0069] |
| rp3beta | 0.1138 [0.1095, 0.1178] | 0.1756 | 8.9648 [8.9537, 8.9758] | 0.6119 [0.6084, 0.6157] | 1750.1566 [1740.9753, 1759.7112] | 0.0072 [0.0057, 0.0088] |
| two_tower | 0.1191 [0.1150, 0.1230] | 0.4669 | 9.8603 [9.8386, 9.8841] | 0.6160 [0.6119, 0.6197] | 1175.1585 [1164.3297, 1186.3043] | 0.1040 [0.0999, 0.1084] |
| lambdarank | 0.1293 [0.1248, 0.1333] | 0.3870 | 9.5047 [9.4908, 9.5193] | 0.6424 [0.6385, 0.6461] | 1315.4293 [1305.6593, 1325.1133] | 0.0459 [0.0430, 0.0489] |

`lambdarank` leads NDCG@10. `two_tower` has the highest novelty point estimate, and its interval overlaps `als_tuned`. Its NDCG@10 is lower, so it is the accuracy-for-novelty trade at the top of this panel. `als_tuned` has the highest long-tail share, with NDCG@10 near most-popular. `most_popular` is the popularity pole: mean popularity about 2326 and a long-tail share of 0. ml-32M is future work.

Recomputed NDCG@10 matches the committed list target to 4 decimals for six models, and to 6 decimals for those six (`most_popular` 0.089508, `item_item_cosine` 0.120135, `als_tuned` 0.090620, `ease` 0.118380, `rp3beta` 0.113834, `two_tower` 0.119078). Their coverage and mean popularity also match the committed figures, except `two_tower`, whose committed headline coverage and mean popularity are the three-seed means. This list's mean popularity matches the seed-42 interval mean in `results/ml-1m.json` (1175.158459).

`lambdarank` does not match. This run's NDCG@10 is 0.129253 against the committed seed-42 value 0.128786 (0.1293 versus 0.1288 at 4 decimals). Early stopping picked 126 trees; the committed primary seed picked 95. The likely cause is a different early-stop result on this host, together with the ADR-0015 top-n tie-break changing the two-tower candidate lists; this was not isolated. The hyperparameters, candidate set, demographic mode, and early-stop-then-refit procedure were not changed. The headline three-seed mean is 0.127264, so the list also misses the headline at 4 decimals (0.1293 versus 0.1273). `two_tower` matches its seed-42 target and misses the headline three-seed mean 0.119246 at 4 decimals (0.1191 versus 0.1192). That headline gap is expected: the panel scores one seed.
