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

**RP3beta.** Row-normalize the user→item matrix and the boolean item→user matrix, raise both elementwise to `alpha`, and form the item–item walk `S = P_iu P_ui` (item → user → item). Scoring with `X S` is the third step, from the user. Divide each destination column by `item_degree^beta`, zero the diagonal, and keep at most `top_k` neighbors per row. Dacrema's optional extra L1 renormalization of `S` is not applied.

**Grids** (validation NDCG@10, point estimate, no bootstrap). Trials are in `results/tuning/<dataset>.json`.

| Model | Search space |
| --- | --- |
| EASE | `λ ∈ {10, 50, 100, 500, 1000}` |
| RP3beta | `alpha ∈ {0.6, 1.0}`, `beta ∈ {0.0, 0.3, 0.6}`, `top_k ∈ {50, 200}` (12 configs) |

Chosen configs are refit on full train. The ml-1m global-time cutoff reuses them and does not re-tune.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Edges only where rating ≥ 4 | Drops users whose fit rows are all below the threshold; item–item still scores those users from raw ratings |
| Rating-weighted `X` | The published EASE / RP3beta setups that beat ItemKNN are interaction graphs, not a new explicit-feedback variant |
| L1-normalize RP3beta `S` after pruning | Optional Dacrema post-step; not part of the walk above, and it would add another grid axis |
| SLIM, or a larger grid, in this PR | More than the small check this stage is for |
| Feed EASE / RP3beta into the S4 ranker now | Changes the ranker; out of scope |

## Outcome

Test metrics below are four-decimal renderings of `results/ml-1m.json`, `results/ml-latest-small.json`, and `results/global_cutoff/ml-1m.json`. Validation scores are the six-decimal values in `results/tuning/*.json`. LambdaRank NDCG@10 is the 3-seed mean already stored on that metrics row. Other LambdaRank columns are dashes: recall, coverage, and tail for that model stay in the generated README table (headline tail NDCG@10 is the primary seed, not the 3-seed mean).

### Chosen hyperparameters

| Dataset | EASE | val NDCG@10 | RP3beta | val NDCG@10 |
| --- | --- | --- | --- | --- |
| ml-latest-small | `l2=500` | 0.088075 | `alpha=0.6`, `beta=0.3`, `top_k=200` | 0.074587 |
| ml-1m | `l2=1000` | 0.079069 | `alpha=0.6`, `beta=0.6`, `top_k=200` | 0.077278 |

On ml-1m the EASE winner is the largest `λ` in the grid (val 0.078371 at 500, 0.079069 at 1000). RP3beta's winner is the low `alpha`, high `beta`, and high `top_k` corner. The grid was not extended after test.

### ml-1m test (the comparison that matters here)

| Model | NDCG@10 | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | Tail NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| item_item_cosine | 0.1201 [0.1158, 0.1242] | 0.0786 | 0.3471 | 0.4989 | 0.1274 | 0.0321 [0.0292, 0.0354] |
| lambdarank (3-seed mean) | 0.1273 | — | — | — | — | — |
| ease (`l2=1000`) | 0.1134 [0.1095, 0.1175] | 0.0870 | 0.4345 | 0.5958 | 0.2896 | 0.0803 [0.0758, 0.0849] |
| rp3beta | 0.1135 [0.1091, 0.1176] | 0.0774 | 0.3827 | 0.5493 | 0.2004 | 0.0555 [0.0519, 0.0596] |

**Gate: negative.** Neither point estimate beats item–item cosine **0.1201** or the LambdaRank 3-seed mean **0.1273**. Both new intervals top out under the item–item point estimate (0.1175 and 0.1176 vs 0.1201) and still overlap the item–item interval, whose low is 0.1158. Both sit under the ranker mean.

They are not worse at everything else on this split. EASE Recall@100 / @200 is 0.4345 / 0.5958, next to two-tower's 0.4354 / 0.6037 and above item–item's 0.3471 / 0.4989. Tail NDCG@10 is 0.0803 (EASE) and 0.0555 (RP3beta) against item–item 0.0321. Coverage@10 is 0.2896 and 0.2004 against 0.1274. The miss is top-10 ranking, not whether the models can see past the head.

**Global-time cutoff** (same hyperparameters, not re-tuned), NDCG@10: EASE 0.2122 [0.1989, 0.2259], RP3beta 0.2250 [0.2102, 0.2397], item–item 0.2319 [0.2162, 0.2473], LambdaRank 0.2323 [0.2193, 0.2465]. The same ordering holds.

### ml-latest-small test

| Model | NDCG@10 | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | Tail NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- |
| item_item_cosine | 0.0899 [0.0784, 0.1011] | 0.0771 | 0.3306 | 0.4507 | 0.0565 | 0.0105 [0.0051, 0.0179] |
| lambdarank (3-seed mean) | 0.0990 | — | — | — | — | — |
| ease (`l2=500`) | 0.1022 [0.0898, 0.1151] | 0.0855 | 0.4033 | 0.5426 | 0.0618 | 0.0303 [0.0177, 0.0441] |
| rp3beta | 0.0894 [0.0771, 0.1018] | 0.0686 | 0.3434 | 0.4822 | 0.0269 | 0.0132 [0.0054, 0.0224] |

EASE's point estimate is above both bars on this smaller panel. Its interval overlaps them. RP3beta does not clear item–item. This does not reverse the ml-1m result.

**Why the literature win did not show up on ml-1m.** Our item–item model uses raw ratings, not the binary ItemKNN those papers tune. The split is a per-user time holdout, not the random or global splits in those studies. The grids are small and, on ml-1m, sit on a corner. That is enough to record a negative result for this protocol. It is not a claim that no EASE or RP3beta configuration can rank higher.

## Consequences

- `ease` and `rp3beta` are part of the tune-and-eval path when `tune=true`, including the ml-1m global cutoff.
- A later PR could try them as extra ranker candidates or features. EASE's ml-1m Recall@100 is in the same range as the two-tower retriever. This PR does not change the S4 ranker.
- The NDCG@10 bar for a later stage is unchanged: item–item cosine, and the ranker mean above it.
