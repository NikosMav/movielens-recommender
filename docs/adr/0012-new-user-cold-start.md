# ADR-0012: New-user profile (cold start)

## Status

Accepted (S5c). The protocol below was fixed before any held-out test metric was computed. Outcome numbers are copied from `results/cold-start/ml-1m.json` after that run and are not inputs to the choices.

## Context

The production path is two-tower candidates (K=200) re-ranked by LightGBM LambdaRank with `demographics=both` (ADR-0007, ADR-0009). The Streamlit page in ADR-0010 only serves users who already have a learned user-id embedding. A person who has never appeared in MovieLens cannot use that embedding, and the page must not ask for gender, age, occupation, or ZIP.

S4b already reported a simulated cold start at N=5 and N=10. That check keeps the user in every training matrix, truncates the query, and still uses the learned id embedding. It is not a measurement of a brand-new person. This ADR adds that measurement and a session-only profile in the app. The headline two-tower, the headline ranker, and `results/ml-1m.json`, `results/ml-latest-small.json`, and `results/demographics/ml-1m.json` stay as they are.

## Decision

**Who is held out.** On ml-1m, take users with at least `min_ratings` rows (the count only). Shuffle those ids with seed 42 and hold out `round(0.1 * n)` of them, clamped to `[1, n - 1]`. Rating values and which rows would fall in a test tail are not inputs. Every row of a held-out user is removed before any fit. The remaining users get the usual per-user chronological split. Their per-user test tails are unused, matching the repo's full-train refit. Held-out users contribute no training row.

**What the model sees at test time.** For each held-out user and each N in {1, 3, 5, 10}, the profile is the first N ratings in timestamp order (item id breaks ties). Targets are later ratings with rating at least 4 whose item is in the refit catalog. Users with N or fewer ratings, and users with no warm relevant later item, are counted and dropped from that N. Profile items are excluded from the list. Cold items are dropped from relevance, not treated as misses inside the catalog.

**Metrics.** NDCG@10 with the harness user bootstrap (1000 resamples, alpha 0.05, seed 42). Recall@10 is a point estimate. Coverage@10 is a point estimate and has no user bootstrap (ADR-0003). Models on the same table: most-popular, item-item fold-in, EASE fold-in, the history-capable two-tower, and the full new-user pipeline (winning candidates, then LambdaRank). The paired interval is pipeline minus the simple baseline with the highest NDCG@10 point on that same table. Ties prefer most-popular, then item-item fold-in, then EASE fold-in. That choice is descriptive. It was not a single pre-registered comparison against one named baseline.

**Representation, chosen on validation only.** Train four user towers on fit-train of the 90%, with the ADR-0006 hyperparameters (embedding size, learning rate, temperature, and the rest of `results/tuning/two_tower_ml-1m.json`). They are not searched again. The variants are history-only (no id term) and user-id dropout with p in {0.25, 0.5, 0.75}. Early stopping uses validation NDCG@10 with the id embedding removed. The winner is higher validation NDCG@10, then history-only, then lower p. A dropout winner at 0.25 or 0.75 is a grid edge and is recorded. The final tower is refit on the 90% full train for that many epochs, with no validation pass.

**Candidates, chosen on validation only.** Score the fit-train profile of each validation user. Compare history two-tower (id removed), item-item (ratings, tuned `k_neighbors=200`, `shrinkage=100`, `min_common=1`), and EASE fold-in (binary, tuned `l2=5000`). The winner is validation recall@200, then recall@100, then item-item fold-in, EASE fold-in, history two-tower. This query is the full fit-train profile, not an N-shot profile.

**Ranker, chosen on validation only.** One LightGBM seed (42), hyperparameters from `configs/ml-1m.yaml`. Features are the S4 columns plus `ease_score` and `ease_rank`. Compare two variants on the early-stop slice, both scored with demographic and group columns missing:

- `demographics_off`: those columns are absent.
- `demographics_nan`: the model is trained with the columns filled, and early-stopped on a copy of the slice with the columns set to NaN. New users are scored the same way.

Higher early-stop NDCG@10 wins. An exact tie selects `demographics_off`. If LightGBM rejects all-NaN categorical columns, that variant is retried with numeric splits and the JSON records `nan_variant_categoricals: false`. The winner is refit on all validation labels for `best_iteration` rounds. The refit does not early-stop again.

**App.** New user sits next to the existing-user picker. Title search, a 1–5 star rating, at least 3 ratings, about 5 suggested. Get recommendations returns a top 10 that excludes rated movies. Each row has "Why this result?" from the existing templates, plus "Because you rated {title} highly" when item-item similarity is positive and the rating is at least 4. Demographic sentences and `demo_*` / `group_*` names are not shown. Nothing is written to disk. The snapshot is the same 90% model that was measured. Latency is one warmed `recommend_new_user` call: the five most common training items, ties by item id, each rated 5, top 10, after one warmup call.

**Difference from S4b.** S4b keeps each evaluated user in the training matrices, truncates the query to the earliest N full-train ratings (N of 5 and 10), and scores the original per-user test split. The user-id embedding is the one learned for that user, and item-item similarities include that user's later train ratings. Here the user is absent from every training row. The model sees only the first N chronological ratings, N in {1, 3, 5, 10}, and is scored on the later ratings with relevance at least 4. There is no user-id embedding at score time, and demographic features are not used. The S4b N=5 rows are copied into the JSON for contrast and are not the S5c result. Known-user headline numbers are copied from `results/ml-1m.json` and are not recomputed.

## Outcome

Copied from `results/cold-start/ml-1m.json`. These numbers were not available when the choices above were fixed.

604 of 6040 eligible users were held out (seed 42). All 604 have a later warm relevant item at every N in {1, 3, 5, 10}.

**Validation choices.**

| choice | winner | validation number |
| --- | --- | --- |
| User representation | `dropout_0.25` (id embedding zeroed with probability 0.25; not history-only) | NDCG@10 0.071919 at epoch 2 |
| history-only | not selected | NDCG@10 0.069339 at epoch 1 |
| dropout 0.5 | not selected | NDCG@10 0.070440 at epoch 2 |
| dropout 0.75 | not selected | NDCG@10 0.069066 at epoch 1 |
| Candidates (K=200) | `ease_fold_in` | recall@200 0.661893, recall@100 0.488925 |
| history two-tower | not selected | recall@200 0.596463, recall@100 0.423225 |
| item-item fold-in | not selected | recall@200 0.591635, recall@100 0.435251 |
| Ranker | `demographics_off` | early-stop NDCG@10 0.093165, 32 trees |
| demographics set to NaN | not selected | early-stop NDCG@10 0.088365, 11 trees |

LightGBM accepted the NaN demographic columns as categoricals (`nan_variant_categoricals` is true). The off variant still won, so the served ranker has no demographic columns. The app does not ask for them.

**Grid edge.** The best dropout probability is 0.25, the low end of {0.25, 0.5, 0.75}, and that model was selected. `dropout_best_at_edge` and `selected_at_grid_edge` are both true. The grid was not extended below 0.25. Epoch 2 is inside the fixed cap of 20.

**Held-out NDCG@10** (user bootstrap, 1000 resamples, alpha 0.05, seed 42). The pipeline is the EASE fold-in top 200, re-ranked by the demographics-off LambdaRank model.

| N | pipeline | most-popular | item-item fold-in | EASE fold-in | history two-tower |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.134591 [0.118512, 0.151437] | 0.394351 [0.373228, 0.414514] | 0.214050 [0.190999, 0.236713] | 0.203732 [0.182888, 0.225252] | 0.244827 [0.222933, 0.265197] |
| 3 | 0.244204 [0.225151, 0.263132] | 0.386298 [0.364612, 0.406805] | 0.308793 [0.286262, 0.330203] | 0.292558 [0.270564, 0.314322] | 0.325211 [0.302732, 0.347518] |
| 5 | 0.282941 [0.263355, 0.303226] | 0.376660 [0.355133, 0.396619] | 0.335085 [0.312266, 0.358069] | 0.323677 [0.301472, 0.346462] | 0.358266 [0.336460, 0.379757] |
| 10 | 0.338091 [0.317881, 0.358199] | 0.346903 [0.325275, 0.367036] | 0.377477 [0.355073, 0.399054] | 0.358806 [0.338956, 0.380023] | 0.371606 [0.349003, 0.393841] |

Recall@10 and Coverage@10 are in the JSON and the generated README panel. Coverage is a point estimate.

Paired NDCG@10, pipeline minus the best simple baseline on that same table (the interval excludes zero at every N):

| N | baseline | difference |
| --- | --- | --- |
| 1 | most-popular | -0.259760 [-0.280871, -0.238800] |
| 3 | most-popular | -0.142094 [-0.162014, -0.124142] |
| 5 | most-popular | -0.093719 [-0.110429, -0.076215] |
| 10 | item-item fold-in | -0.039387 [-0.053201, -0.026206] |

The new-user pipeline does not beat popularity, and it does not beat the unranked fold-in list it reorders. At N=5 its NDCG@10 is 0.282941, against 0.376660 for most-popular on the same users. A known user on the headline split, who was in training, has LambdaRank NDCG@10 0.127264 (primary-seed interval [0.124690, 0.132934]) and most-popular 0.089508 ([0.085661, 0.093487]). Those known-user numbers are copied from `results/ml-1m.json`. They are not on the same user set or the same target slice, so they are context, not a paired test. The cold-start most-popular number is higher because a new profile has not yet consumed the popular titles, and the targets are the rest of that person's ratings rather than a short test tail.

**Latency.** One warmed top-10 call, five most-common training titles rated 5, took 0.084710 seconds.

**What this does not change.** The candidate source and the ranker stay the validation winners. The held-out gap is the reason the validation query (full fit-train profile) is listed as a limitation, not a reason to pick a different model after seeing the test table.

## Round 2

The round 1 pipeline loses to most-popular at every N. This section was written before the held-out 604 were scored again. Round 1 stays in the JSON under `round1` and is not recomputed. The choices below are locked on validation. The held-out ratings are read only after that, once.

**Dropout grid.** The round 1 grid was {0.25, 0.5, 0.75}, and 0.25 was the low edge. Round 2 adds p=0.1 and p=0.0. p=0.0 is the same training loop with the id mask never applied. Scoring still zeros the id embedding. History-only stays in the comparison. The winner is still higher validation NDCG@10, then history-only, then lower p. An edge is now the first or last dropout value, 0.0 or 0.75. The grid is not extended after the held-out scores exist.

**Who the ranker is validated on.** From the 90% full-train users only, hold out `early_stop_fraction` (0.2) with `select_held_out_user_ids`, seed 42. Those ranker-validation users are disjoint from the 604. Retrievers used for the selection metric are fit on the other ranker-train users' full train, so the validation users are absent. A further early-stop split, same fraction and seed, is taken from the ranker-train users only. Ranker-train users' later full-train ratings can still sit in the item similarity. That can bias the fit. It does not enter the selection metric.

**Simulated profiles.** For each ranker-train user and each N in {1, 3, 5, 10}, the profile is the first N ratings of that user's full chronology. Every history feature (activity, genre affinity, and the retriever scores) is recomputed from that prefix. Labels are later ratings with relevance at least 4, in the ranker-train catalog. N is a feature. Item popularity is a feature. Demographics stay off.

**Candidates and K.** The pool is the union of EASE fold-in, most-popular, and the history two-tower, in that order, duplicates kept once. Each source contributes a flag. Item-item fold-in does not add candidates. It is a feature when the item is in its top K, and it is a serving option. K is chosen from {50, 100, 200} by the mean validation NDCG@10 of this ranker across N. A tie keeps the smaller K. Lists are retrieved once at 200 and truncated, so a rank is the rank inside that prefix.

**What the page serves.** For each N, serve whichever of {cold-start ranker, most-popular, item-item fold-in, history two-tower} has the higher validation NDCG@10. A tie prefers most-popular, then item-item fold-in, then the history two-tower, then the ranker. A live profile of length n uses the largest grid N that is at most n. The ranker, when it is the one served, still sees the actual profile length as `profile_n`. The page says in plain words which method it used. A popularity reason reads "Popular with many viewers". It never reads a demographic reason. After the choices are frozen, retrievers and the ranker are refit on the whole 90% for `best_iteration` trees, with no second early stop.

**Held-out score, once.** The primary target is unchanged: all later ratings, relevance at least 4. The sensitivity view, added after the first results, keeps only each held-out user's harness tail (the last `max(1, int(n_ratings * 0.2))` ratings, leaving at least one head row) where that tail is after the first N. The serving rule is not re-chosen on the sensitivity view, and pipeline v1 is not invented on that target. Paired intervals compare the served list, and the new ranker, with the best simple baseline on that same table, and again with most-popular. If the served method does not beat popularity at a small N, the page still serves the validation winner. Serving popularity there is a legitimate outcome.

**Why cold-start popularity looks higher than the known-user number.** The cold-start target is every later rating, a long tail, and the short profile has not consumed the popular titles. The known-user most-popular number is a short per-user test tail after a long history. Those are not a paired test.

### Round 2 outcome

Filled from `results/cold-start/ml-1m.json` after the single held-out pass. Not an input to the choices above.

## Limitations

- Round 1's candidate and ranker selection used each validation user's full fit-train profile, while the reported test was N-shot. That pipeline is the negative result under `round1`. Round 2 selects the new ranker and the per-N rule on truncated profiles.
- Ranker-train users' later full-train ratings can sit in the item similarity used to fit the ranker. Ranker-validation users are absent from that fit.
- Two-tower, item-item, and EASE hyperparameters were not searched again.
- The ranker is one seed.
- The app model was trained without the held-out 10%. It is the measured model, not a second fit on every ml-1m user.
- Coverage@10 has no confidence interval.
- The paired baseline is chosen on the table it is compared against.
- EASE fold-in is binary. A 5-star and a 1-star rating both contribute 1.
- "Because you rated … highly" needs a stored item-item similarity above 0 and a rating of at least 4.
- The sensitivity tail was added after the first held-out table. It does not choose the serving rule.
