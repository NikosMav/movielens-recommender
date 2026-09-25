# ADR-0003: Metrics and uncertainty

## Status

Accepted (S1)

## Context

We need a single primary metric for stage gates, secondary ranking metrics, and lightweight diagnostics for popularity collapse—without building a full experiment platform.

## Decision

- **Primary:** NDCG@10 (binary relevance, rating ≥ 4.0).
- **Secondary:** NDCG@20, Precision@k, Recall@k for k ∈ {10, 20}.
- **Diagnostics:** catalog coverage@k (unique recommended items / \|train catalog\|) and mean train popularity of recommended items (interaction count).
- **Uncertainty:** percentile bootstrap over users (default 1000 resamples, seed from config, 95% CI) for per-user ranking metrics and for diagnostics recomputed on resampled user recommendation lists.
- Segment breakdowns (activity buckets, head/tail) deferred to keep S1–S2 small.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| MAP / MRR as primary | Less standard for graded top-N MovieLens comparisons we care about |
| Sampled metrics (e.g. 100 negatives) | Faster but not comparable to full-catalog offline protocol used here |
| Jackknife / analytic SE only | Bootstrap is simple and matches “CI over users” wording |
| Full fairness / calibration suite | Gold-plating for a solo S1–S2 repo |

## Consequences

Stage progression is judged on NDCG@10 with CI context. Coverage and popularity are reported so a “win” that only recommends blockbusters is visible.
