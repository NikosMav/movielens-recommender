# movielens-recommender

A small, standalone movie recommender built from scratch on [MovieLens](https://grouplens.org/datasets/movielens/) ratings. Stages 1–2 cover framing, data, evaluation, and classic collaborative-filtering baselines. Later stages add retrieval, ranking, serving, and operations.

**MovieLens data is not redistributed here.** Download it yourself from GroupLens and respect their [license and terms of use](https://grouplens.org/datasets/movielens/). Raw and derived rating files live under a gitignored `data/` directory and must never be committed. Aggregate EDA stats/figures under `docs/eda/` are fine to commit.

See [`docs/problem.md`](docs/problem.md) for the task definition, primary metric, and stage-gate rule. Design decisions live in [`docs/adr/`](docs/adr/).

## Roadmap

| Stage | Focus |
| --- | --- |
| **S1** | Framing + data (download, clean, split, EDA, eval harness) |
| **S2** | Classic CF baselines |
| **S3** | Two-tower retrieval |
| **S4** | Learned ranker |
| **S5** | Serving |
| **S6** | Operations |

Each modeling stage must beat the previous best on **NDCG@10** (same harness) or be written up as a negative result.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Download data

Archives are fetched from official GroupLens URLs and verified against **pinned SHA-256 checksums** (see ADR-0001).

```bash
movielens-recommender download --dataset ml-latest-small   # default
movielens-recommender download --dataset ml-1m
```

| Dataset | SHA-256 |
| --- | --- |
| ml-latest-small | `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436` |
| ml-1m | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |

### Cleaning rules

1. Drop nulls in `user_id`, `item_id`, `rating`, `timestamp`.
2. Keep ratings in `[0.5, 5.0]`.
3. Require positive ids and timestamps.
4. Deduplicate `(user_id, item_id)` keeping the latest timestamp (then higher rating).

## Split and cold-start

For each user (deterministic):

1. Drop if fewer than **`min_ratings=5`** interactions.
2. Sort by **`timestamp` ascending** (stable).
3. Hold out the last **`max(1, floor(n * 0.2))`** interactions as test; rest train (≥1 train required).

**Relevance:** rating **`≥ 4.0`**.

**Cold-start policy:** cold items (absent from train) are removed from relevant test sets; users with no remaining warm relevant items are excluded from ranking-metric averages (counts recorded in results JSON). Already-seen train items are filtered from recommendations.

## Metrics

| Role | Metric |
| --- | --- |
| Primary | **NDCG@10** |
| Secondary | NDCG@20, Precision@k, Recall@k (k=10,20) |
| Diagnostics | Catalog coverage@k, mean train popularity of recommended items |

Averaged over eligible test users. **95% bootstrap CIs** over users (1000 resamples, seed from config) for ranking metrics and mean popularity. **Catalog coverage is a point estimate only** — a user-level bootstrap is invalid for a set-union statistic (see ADR-0003).

## Config-driven run

```bash
movielens-recommender run --config configs/default.yaml
movielens-recommender run --config configs/ml-1m.yaml
```

Writes `results/<dataset>.json` (committed experiment log: metrics, CIs, checksum, split, config, library versions). Regenerate the README table (never hand-edit numbers):

```bash
python scripts/make_results_table.py
```

## EDA

```bash
python scripts/run_eda.py --dataset ml-latest-small --download
```

Figures and aggregate stats land in `docs/eda/` (committed). Raw data does not.

## Tests and CI

```bash
pytest -q
ruff check src tests scripts
```

GitHub Actions runs ruff + pytest on push/PR and does **not** download MovieLens.

## Baselines (S2)

| Model | Notes |
| --- | --- |
| `most_popular` | Train interaction counts (ties by item id). |
| `item_item_cosine` | Item–item cosine CF. |
| `als` | `implicit` ALS (factors=64, iterations=15, α=40, seed=42) — **not tuned**. |

## Results

<!-- BEGIN RESULTS TABLE -->

### `ml-1m` (from `results/ml-1m.json`)

Pinned version: `ml-1m@sha256:a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.

Split: min_ratings=5, test_fraction=0.2, relevance_threshold=4.0, seed=42, ks=[10, 20], bootstrap=1000 @ alpha=0.05. Primary metric: **ndcg@10**. Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003).

| model | ndcg@10 | precision@10 | recall@10 | ndcg@20 | precision@20 | recall@20 | coverage@10 | mean_popularity@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0.0686 | 0.0564 | 0.0598 | 0.0847 | 0.0542 | 0.1049 | 0.5198 | 956.1292 |
| item_item_cosine | 0.1201 | 0.0980 | 0.0786 | 0.1284 | 0.0836 | 0.1277 | 0.1274 | 1794.5672 |
| most_popular | 0.0895 | 0.0787 | 0.0466 | 0.0951 | 0.0698 | 0.0874 | 0.0325 | 2325.9361 |

95% bootstrap CIs (NDCG@10):

| model | ndcg@10 CI |
| --- | --- |
| als | [0.0658, 0.0714] |
| item_item_cosine | [0.1158, 0.1242] |
| most_popular | [0.0857, 0.0935] |

### `ml-latest-small` (from `results/ml-latest-small.json`)

Pinned version: `ml-latest-small@sha256:696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`.

Split: min_ratings=5, test_fraction=0.2, relevance_threshold=4.0, seed=42, ks=[10, 20], bootstrap=1000 @ alpha=0.05. Primary metric: **ndcg@10**. Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003).

| model | ndcg@10 | precision@10 | recall@10 | ndcg@20 | precision@20 | recall@20 | coverage@10 | mean_popularity@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0.0795 | 0.0622 | 0.0693 | 0.0948 | 0.0531 | 0.1268 | 0.0965 | 108.2937 |
| item_item_cosine | 0.0899 | 0.0707 | 0.0771 | 0.1017 | 0.0583 | 0.1266 | 0.0565 | 122.4519 |
| most_popular | 0.0743 | 0.0563 | 0.0514 | 0.0802 | 0.0466 | 0.0859 | 0.0119 | 216.7525 |

95% bootstrap CIs (NDCG@10):

| model | ndcg@10 CI |
| --- | --- |
| als | [0.0692, 0.0902] |
| item_item_cosine | [0.0784, 0.1011] |
| most_popular | [0.0636, 0.0860] |

<!-- END RESULTS TABLE -->
