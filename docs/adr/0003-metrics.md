# ADR-0003: Metrics and uncertainty

## Status

Accepted (S1); amended for coverage CI policy

## Context

We need a single primary metric for stage gates, secondary ranking metrics, and lightweight diagnostics for popularity collapse—without building a full experiment platform. An early implementation also attached user-bootstrap CIs to catalog coverage; those intervals systematically excluded the point estimate.

## Decision

- **Primary:** NDCG@10 (binary relevance, rating ≥ 4.0).
- **Secondary:** NDCG@20, Precision@k, Recall@k for k ∈ {10, 20}.
- **Diagnostics:** catalog coverage@k (unique recommended items / \|train catalog\|) and mean train popularity of recommended items (interaction count).
- **Uncertainty:**
  - **User-level means** (precision / recall / NDCG / mean popularity): percentile bootstrap over users (default 1000 resamples, seed from config, 95% CI). Every reported CI must contain its point estimate.
  - **Catalog coverage:** **point estimate only** — no user-bootstrap CI.
- Segment breakdowns (activity terciles, head/tail items) deferred in S1–S2; **added in S3a** (see `segments` in `results/*.json` and ADR-0005 / README).

### Why coverage has no user-bootstrap CI

Coverage@k = `|⋃_u recs_u[:k]| / |catalog|` is a **set-union** over the evaluated user panel, not a mean of per-user scores. Resampling users **with replacement** yields fewer *unique* users than the full panel, so the union of their top-k lists is smaller in expectation and coverage is **biased downward**. The full-sample point estimate then sits **above** the bootstrap percentile interval (observed for every model in early `results/*.json`).

Valid alternatives (item jackknife, design-based survey estimators, etc.) add complexity out of scope for S1–S2. Reporting the diagnostic as a point estimate is honest and avoids a misleading interval.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| MAP / MRR as primary | Less standard for graded top-N MovieLens comparisons we care about |
| Sampled metrics (e.g. 100 negatives) | Faster but not comparable to full-catalog offline protocol used here |
| Jackknife / analytic SE only | Bootstrap is simple for user-level means |
| User-bootstrap CI for coverage | Invalid for a set-union; systematically misses the point estimate |
| Item-level bootstrap for coverage | Extra complexity for a non-gate diagnostic |
| Full fairness / calibration suite | Gold-plating for a solo S1–S2 repo |

## Consequences

Stage progression is judged on NDCG@10 with CI context. Coverage and popularity remain visible diagnostics; only popularity carries a user-bootstrap CI. Results JSON records `uncertainty_policy` so the omission is explicit.
