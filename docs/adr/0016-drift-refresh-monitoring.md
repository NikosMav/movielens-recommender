# ADR-0016: Drift, refresh, and serving monitoring

## Status

Accepted (S6). The replay protocol, the drift signals, the refresh policies, and the threshold rule below were written before any replay metric was computed. Monthly rating, user, and item counts were looked at to choose the periods. No model was fit for that. The outcome section is filled only after `results/ops/ml-1m.json` exists.

## Context

Every earlier stage splits each user's history at one point in time and scores once. A deployed recommender is trained at one time and served afterwards, while new titles arrive, tastes move, and the snapshot ages. S5b serves a snapshot with no sign of its age and no request telemetry. S6 asks two questions:

1. How fast does a model trained at one time get worse, and does a cheap, data-only drift signal say when retraining matters?
2. What should a running API report, so that a person can see that it is healthy and when its inputs drift?

MovieLens is a fixed historical dataset, so time has to be replayed. ml-1m covers April 2000 to February 2003, but 95% of its ratings fall in April–December 2000, and no new users arrive after December 2000. Later months hold 100–550 returning users each.

Compute is kept small on purpose. The study uses the three cheap models, whose fits take seconds. The two-tower and the LightGBM ranker are not refit here.

## Decision

### Replay

**Periods.** Monthly from August to December 2000, then quarterly from 2001 Q1 to 2003 Q1 (January–February 2003): 14 periods. The first model is trained on every rating before 2000-08-01. Periods use UTC calendar boundaries on the rating timestamp.

**Models.** Most popular, item–item cosine (`k_neighbors` 200, `shrinkage` 100, `min_common` 1), and EASE (`l2` 5000). The hyperparameters are the ml-1m validation winners in `results/tuning/ml-1m.json` and are not searched again. Those were tuned on a split that saw later data. That is a limitation, and it is the same for every policy.

**Who is scored in period P.** Users with at least 5 ratings before P starts and at least one rating of 4 or more in P. Their query profile is every rating they made before P starts. The target is the set of items they rated 4 or more in P. The top 10 excludes the profile. Item–item and EASE fold in the current profile through the similarity of the model in use, so a stale model still sees the user's latest history; what is stale is the similarity and the popularity. Target items the model has never seen stay in the target. Missing them is part of the cost of staleness. Users whose first rating falls inside P are not scored: they belong to the cold-start problem (ADR-0012). Their share of P's ratings is reported as a drift signal.

**Metric.** NDCG@10 per user, averaged per period, with the 1000-resample user bootstrap from config (alpha 0.05, seed 42). Model comparisons are paired over the same users.

### Drift signals

Computed at the start of each period from data before it only. "Last period" is the period just before P (July 2000 for the first).

- **Item divergence.** Jensen–Shannon divergence, base 2 (0 to 1), between the item distribution of 2,000 ratings sampled from the last period and 2,000 sampled from the model's training data, without replacement, seed 42. Equal sample sizes keep small quarters comparable with large months. Sampling noise alone gives a floor above 0, which is reported.
- **New-item share.** Share of the last period's ratings on items with fewer than 5 ratings in the model's training data.
- **New-user share.** Share of the last period's ratings from users whose first rating is in that period.

### Refresh policies

For each model:

- **frozen:** trained once on everything before 2000-08-01.
- **periodic:** retrained at the start of every period on everything before it.
- **drift(τ):** at the start of each period, retrained on everything before it if the item divergence between the last period and the current model's training data is above τ. τ ∈ {0.10, 0.20, 0.30, 0.40, 0.50, 0.60}.

A retrain at a given start uses the same data as the periodic model at that start, so those fits are shared.

**Choosing τ.** The tuning periods are the first three (August–October 2000); the other eleven are the test. τ is chosen on item–item and then applied to all three models. Item–item and EASE are level on the main ml-1m table (test NDCG@10 0.1192 and 0.1184, popularity 0.0895); item–item is the one the production ranker already uses as a feature and for its reasons. Rule: among the τ whose mean tuning NDCG@10 is at least the periodic mean minus 0.005, choose the one with the fewest tuning retrains. A tie goes to the larger τ. If none qualifies, use the smallest τ. The test periods do not choose anything.

**Reported per model and policy.** Per-period NDCG@10 with intervals, the mean over test periods, and the number of retrains. Per period, the paired difference periodic minus frozen. Across the 14 periods, the Spearman correlation between each drift signal and that paired difference. This says whether the signal tracks when retraining matters.

### Serving monitoring

The FastAPI service (ADR-0014) gains:

- **`GET /metrics`** in Prometheus text format, written without a new dependency: request counts by route template and status code, a latency histogram by route template, the number of requests in flight, whether each snapshot loaded, and the snapshot age in seconds (from the manifest's `created_at`).
- **New-user input drift.** For the last 500 new-user requests: the mean training-popularity percentile of the rated items and the share of rated items in the training tail (outside the top 20% by count). The reference values from the training ratings are exported next to them, so the comparison needs no extra state.
- **One JSON log line per request** on the `movielens_recommender.api` logger: route template, method, status, latency in milliseconds, and `n` when the request has one. `/metrics` scrapes are not counted or logged. No ratings, item ids, or raw paths are logged.

Route templates (`/v1/users/{user_id}/recommendations`) are used instead of raw paths, so user ids do not become metric labels.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Refit the two-tower and the ranker at every refresh | Hours of all-core training per run; the cheap models answer the decay question |
| Population stability index on rating values | Ratings are 1–5 and barely move; the item mix is what drifts |
| JS divergence on full period samples | Small quarters would look drifted from sampling noise alone |
| `prometheus_client` | A new dependency for a few counters and one histogram |
| Logging user ids or raw paths | Turns the log into a per-user history |
| A reload endpoint that swaps the snapshot | The API has no authentication (ADR-0014); refresh is rebuild and restart |

## Outcome

Numbers from `results/ops/ml-1m.json` (runtime 2 min 37 s at 4 threads). The generated README panel has the per-period table.

**No measurable decay.** Mean test NDCG@10 over the eleven test periods:

| model | frozen | periodic | drift (τ=0.6) | retrains (periodic / drift) |
| --- | --- | --- | --- | --- |
| most popular | 0.0896 | 0.0891 | 0.0896 | 13 / 0 |
| item–item | 0.1013 | 0.1009 | 0.1013 | 13 / 0 |
| EASE | 0.0997 | 0.1015 | 0.0997 | 13 / 0 |

A model trained once on April–July 2000 scores as well as one retrained before every period, for all three models, up to two and a half years later. Of the 39 paired periodic-minus-frozen intervals (13 periods after the first, three models), 3 exclude 0, and they point both ways: most popular +0.0044 in October 2000, item–item -0.0083 in November 2000, EASE +0.0190 in 2002 Q3. Three of 39 is about what chance gives at alpha 0.05.

**Why.** ml-1m's catalog stops at release year 2000. Items first rated on or after 2000-08-01 are 387 of 3,706 but carry 16,343 of 1,000,209 ratings (1.6%), and 1.2–6.9% of each period's targets are items the frozen model never saw. The user side changes a lot (in each month from July to December 2000, 75–98% of the ratings come from users who joined that month), but item–item and EASE fold in a user's current history at scoring time, so new users do not need a retrain. On this data, what goes stale is almost nothing.

**The divergence signal is noise here.** Its sampling floor (two independent samples from the frozen training data) is 0.369. Period values run from 0.369 to 0.451, so most of the signal is sampling noise. Every τ from 0.1 to 0.6 was within 0.005 of periodic on the tuning periods, and τ=0.6 had the fewest retrains (none), so the rule chose never to retrain. Spearman correlation with the retraining gain is inconsistent in sign across models: item divergence -0.62 (most popular), +0.04 (item–item), +0.26 (EASE); new-item share -0.60, +0.08, +0.40. None of the three signals says when a retrain helps.

**What this means for operations.**

- For these models on this data, retraining is not what keeps quality up. A rebuild schedule should follow catalog changes (new titles to recommend) and data fixes, not a fixed clock. The snapshot age gauge makes the age visible; it is not an alert by itself.
- A drift signal has to beat its own sampling floor before it can trigger anything. On a stream this size, item-mix divergence at 2,000 ratings does not.
- This is one dataset with almost no new items. On a catalog that keeps growing, such as ml-32M's, the result could differ. That replay was not run, to keep compute small.
- The two-tower and the LightGBM ranker were not replayed. A learned user-id embedding cannot fold in a new user's history the way item–item can, so the neural model may decay faster than these three.

**Monitoring.** `/metrics`, the request log, and the new-user input gauges are covered by tests on synthetic snapshots. The reference values (training-popularity percentile and tail share of training ratings) come from the loaded production snapshot.
