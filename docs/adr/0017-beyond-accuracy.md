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

Pending the ml-1m run. Definitions above stay as written.
