# ADR-0008: EASE and RP3beta baselines

## Status

Accepted (S3c) — **negative result** on the ml-1m NDCG@10 comparison against item–item cosine and the S4 LambdaRank ranker.

## Context

Reproducibility studies (Dacrema et al. 2019; Anelli et al. 2022) found that a tuned linear model (EASE^R, Steck 2019) and a tuned graph model (RP3beta; Christoffel et al. 2015, Paudel et al. 2017) often beat ItemKNN and neural recommenders on MovieLens 1M. Our strongest simple baseline on this repo's per-user time split is item–item cosine. After S4, the learned ranker is also ahead of that baseline on NDCG@10. S3c checks whether those two classic models clear the same bars under ADR-0005 (tune on validation only, refit on full train, evaluate once on test). The S4 ranker is not changed.

## Decision

**Both models are numpy/scipy only.** The interaction matrix is binary: any observed train rating is an edge. Relevance (rating ≥ 4.0) stays an evaluation rule. Already-seen train items are removed inside `recommend`, as the other baselines do.

**EASE^R.** With `X` the binary user–item matrix and `λ` (`l2`):

- `P = (XᵀX + λI)⁻¹`
- `B = I − P · diag(1 / diag(P))`, which sets `diag(B) = 0`
- scores are `X B`

**RP3beta.** Row-normalize the user→item matrix and the boolean item→user matrix, raise both elementwise to `alpha`, and form the item–item walk `S = P_iu P_ui` (item → user → item). Scoring with `X S` is the third step, from the user. Divide each destination column by `item_degree^beta`, zero the diagonal, and keep at most `top_k` neighbors per row. Dacrema's optional extra L1 renormalization of `S` is not applied. `alpha=0` sets stored transition weights to 1 and leaves structural zeros at zero.

**Grids** (validation NDCG@10, point estimate, no bootstrap). Trials are in `results/tuning/<dataset>.json`.

The first test run used this grid:

| Model | First grid |
| --- | --- |
| EASE | `λ ∈ {10, 50, 100, 500, 1000}` |
| RP3beta | `alpha ∈ {0.6, 1.0}`, `beta ∈ {0.0, 0.3, 0.6}`, `top_k ∈ {50, 200}` (12 configs) |

On that run every chosen ml-1m hyperparameter sat on a grid edge (`λ=1000`, `alpha=0.6`, `beta=0.6`, `top_k=200`), and on ml-latest-small `alpha=0.6` and `top_k=200` did too. The grid was extended after that first test run because the winners were at grid edges. The new points were scored on validation only. Test labels were not used to pick them.

| Model | Extended grid |
| --- | --- |
| EASE | `λ ∈ {10, 50, 100, 500, 1000, 2000, 5000, 10000}` |
| RP3beta | `alpha ∈ {0.2, 0.4, 0.6, 0.8, 1.0}`, `beta ∈ {0.0, 0.3, 0.6, 0.8, 1.0}`, `top_k ∈ {50, 200, 500, 1000}` (100 configs) |

Where that extended-grid winner was still on a non-natural edge, the axis was extended one step, once. Natural edges that stop the search are `alpha=1`, `beta=0`, and `top_k` covering every fit-train item. Anything else at the end of an axis is non-natural.

- ml-1m: `top_k=2000` (fit-train catalog 3655 items). `λ`, `alpha`, and `beta` were interior, so those axes stayed put.
- ml-latest-small: `alpha=0.0` and `top_k=2000` (fit-train catalog 7667 items).

No axis was extended a second time. `top_k=2000` is in the shared `RP3BETA_TOPKS` tuple. The `alpha=0` trials are only in `results/tuning/ml-latest-small.json`.

Chosen configs are refit on full train. The ml-1m global-time cutoff reuses them and does not re-tune.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Edges only where rating ≥ 4 | Drops users whose fit rows are all below the threshold; item–item still scores those users from raw ratings |
| Rating-weighted `X` | The published EASE / RP3beta setups that beat ItemKNN are interaction graphs, not a new explicit-feedback variant |
| L1-normalize RP3beta `S` after pruning | Optional Dacrema post-step; not part of the walk above, and it would add another grid axis |
| SLIM in this PR | More than the check this stage is for |
| Feed EASE / RP3beta into the S4 ranker now | Changes the ranker; out of scope |
| A second step past `top_k=2000` | The rule for this extension was one step past a non-natural edge, then stop |

## Outcome

Test metrics below are four-decimal renderings of `results/ml-1m.json`, `results/ml-latest-small.json`, and `results/global_cutoff/ml-1m.json` (the same rounding as `scripts/make_results_table.py`). Validation scores are the six-decimal values in `results/tuning/*.json` (`0.08083` is stored for ml-1m EASE and prints as 0.080830). LambdaRank NDCG@10 is the 3-seed mean already stored on that metrics row. Other LambdaRank columns are dashes: recall, coverage, and tail for that model stay in the generated README table (headline tail NDCG@10 is the primary seed, not the 3-seed mean).

### Chosen hyperparameters

| Dataset | EASE | val NDCG@10 | RP3beta | val NDCG@10 |
| --- | --- | --- | --- | --- |
| ml-latest-small | `l2=500` | 0.088075 | `alpha=0.2`, `beta=0.3`, `top_k=2000` | 0.079885 |
| ml-1m | `l2=5000` | 0.080830 | `alpha=0.4`, `beta=0.6`, `top_k=2000` | 0.080956 |

**Edges after the one step.** EASE is interior on both datasets: ml-1m validation NDCG@10 is 0.079999 at `λ=2000`, 0.080830 at 5000, and 0.079879 at 10000; ml-latest-small is 0.085410 at 100, 0.088075 at 500, and 0.087937 at 1000. RP3beta `alpha` and `beta` are interior on both datasets. On ml-latest-small, `alpha=0.2` sits between the one-step value 0.0 (0.078953 at `beta=0.3`, `top_k=2000`) and 0.4 (0.078949). `top_k=2000` is still a non-natural edge on both datasets. At the winning `alpha` and `beta`, validation NDCG@10 goes from 0.080553 at `top_k=1000` to 0.080956 at 2000 on ml-1m, and from 0.079841 to 0.079885 on ml-latest-small. Both fit-train catalogs are larger than 2000 (3655 and 7667). The search stopped there.

### ml-1m test (the comparison that matters here)

| Model | NDCG@10 | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | Tail NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| item_item_cosine | 0.1201 [0.1158, 0.1242] | 0.0786 | 0.3471 | 0.4989 | 0.1274 | 0.0321 [0.0292, 0.0354] |
| lambdarank (3-seed mean) | 0.1273 | — | — | — | — | — |
| ease (`l2=5000`) | 0.1184 [0.1142, 0.1226] | 0.0861 | 0.4314 | 0.6003 | 0.2100 | 0.0743 [0.0699, 0.0788] |
| rp3beta (`alpha=0.4`, `beta=0.6`, `top_k=2000`) | 0.1138 [0.1095, 0.1178] | 0.0782 | 0.3723 | 0.5258 | 0.1756 | 0.0391 [0.0357, 0.0428] |

**Gate: negative.** Neither point estimate beats item–item cosine **0.1201** or the LambdaRank 3-seed mean **0.1273**. EASE's interval [0.1142, 0.1226] overlaps the item–item interval [0.1158, 0.1242], and its high end 0.1226 is above the item–item point estimate. The EASE point estimate 0.1184 is still below 0.1201. RP3beta's interval tops out at 0.1178, under that point. Both sit under the ranker mean.

They are stronger than item–item on several other columns of this split. EASE Recall@100 / @200 is 0.4314 / 0.6003, next to two-tower's 0.4354 / 0.6037 and above item–item's 0.3471 / 0.4989. Tail NDCG@10 is 0.0743 (EASE) and 0.0391 (RP3beta) against item–item 0.0321. Coverage@10 is 0.2100 and 0.1756 against 0.1274. The miss is top-10 ranking.

**Global-time cutoff** (same hyperparameters, not re-tuned), NDCG@10: EASE 0.2259 [0.2116, 0.2407], RP3beta 0.2217 [0.2064, 0.2364], item–item 0.2319 [0.2162, 0.2473], LambdaRank 0.2323 [0.2193, 0.2465]. The same ordering holds.

### ml-latest-small test

| Model | NDCG@10 | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | Tail NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| item_item_cosine | 0.0899 [0.0784, 0.1011] | 0.0771 | 0.3306 | 0.4507 | 0.0565 | 0.0105 [0.0051, 0.0179] |
| lambdarank (3-seed mean) | 0.0990 | — | — | — | — | — |
| ease (`l2=500`) | 0.1022 [0.0898, 0.1151] | 0.0855 | 0.4033 | 0.5426 | 0.0618 | 0.0303 [0.0177, 0.0441] |
| rp3beta (`alpha=0.2`, `beta=0.3`, `top_k=2000`) | 0.0858 [0.0735, 0.0985] | 0.0674 | 0.3133 | 0.4432 | 0.0220 | 0.0000 [0.0000, 0.0000] |

EASE's point estimate is above both bars on this smaller panel. Its interval overlaps them. RP3beta's point estimate 0.0858 is below item–item 0.0899, and its tail NDCG@10 is 0.0000. This does not reverse the ml-1m result.

**Why the literature win did not show up on ml-1m.** Our item–item model uses raw ratings, not the binary ItemKNN those papers tune. The split is a per-user time holdout, not the random or global splits in those studies. After the extension, EASE's `λ` is interior and the point estimate is still under both bars. RP3beta's `top_k=2000` is still short of the catalog, and validation NDCG@10 was still higher there than at 1000. The recorded result is this search, including that one extra step.

## Consequences

- `ease` and `rp3beta` are part of the tune-and-eval path when `tune=true`, including the ml-1m global cutoff.
- A later PR could try them as extra ranker candidates or features. EASE's ml-1m Recall@100 is in the same range as the two-tower retriever. This PR does not change the S4 ranker.
- The NDCG@10 bar for a later stage is unchanged: item–item cosine, and the ranker mean above it.
