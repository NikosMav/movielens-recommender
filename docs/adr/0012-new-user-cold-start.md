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

Filled from `results/cold-start/ml-1m.json` after the run. Until that file exists, this section does not quote a metric.

## Limitations

- Validation candidate and ranker selection uses each validation user's full fit-train profile. The reported test is N-shot. Those are different queries.
- Two-tower, item-item, and EASE hyperparameters were not searched again.
- The ranker is one seed.
- The app model was trained without the held-out 10%. It is the measured model, not a second fit on every ml-1m user.
- Coverage@10 has no confidence interval.
- The paired baseline is chosen on the table it is compared against.
- EASE fold-in is binary. A 5-star and a 1-star rating both contribute 1.
- "Because you rated … highly" needs a stored item-item similarity above 0 and a rating of at least 4.
