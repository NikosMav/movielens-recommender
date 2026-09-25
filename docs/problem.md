# Problem statement

## Task

Build a **top-N recommender** that, for each user, ranks **movies the user has not already interacted with** and returns a short list (N = 10 / 20). Relevance is defined from explicit MovieLens ratings.

| Entity | Meaning |
| --- | --- |
| Users | MovieLens `userId` values |
| Items | MovieLens `movieId` values (movies) |
| Interaction | A rating with a timestamp |
| Relevant item | Rating **≥ 4.0** (binary relevance) |

Primary use of the model: surface unseen movies the user is likely to rate highly.

## Metrics

| Role | Metric | Notes |
| --- | --- | --- |
| **Primary** | **NDCG@10** | Ranking quality with binary gains; main gate for stage progression |
| Secondary | NDCG@20, Precision@k, Recall@k (k ∈ {10, 20}) | Completeness / precision trade-offs |
| Diagnostics | Catalog coverage@k, mean recommended-item popularity | Diversity / popularity bias (not optimization targets) |

All ranking metrics are averaged over eligible test users (see cold-start policy in the README / ADR-0002). Bootstrap 95% CIs over users are reported for ranking metrics and diagnostics.

## Baselines later stages must beat

Stages 1–2 establish three collaborative-filtering baselines on a fixed harness:

1. **most_popular** — global interaction-count ranking  
2. **item_item_cosine** — item–item cosine CF  
3. **als** — matrix factorization via `implicit` ALS  

Committed numbers live in `results/*.json`. **Later model stages (retrieval, ranker) must beat the best prior stage on the same split and primary metric (NDCG@10), or document a negative result** with analysis—no silent regressions.

## Planned lifecycle

| Stage | Focus |
| --- | --- |
| **S1** | Framing + data (download, clean, split, EDA, eval harness) |
| **S2** | Classic CF baselines |
| **S3** | Two-tower retrieval (candidate generation) |
| **S4** | Learned ranker (re-rank retrieved candidates) |
| **S5** | Serving (inference path, latency budget) |
| **S6** | Operations (monitoring, refresh, drift) |

Each modeling stage reuses this harness. If a stage does not improve NDCG@10 (with CI context), write it up as a negative result rather than claiming progress.
