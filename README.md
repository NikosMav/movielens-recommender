# movielens-recommender

A small, standalone movie recommender built from scratch on [MovieLens](https://grouplens.org/datasets/movielens/) ratings. Stages 1–2 cover framing, data, evaluation, and classic collaborative-filtering baselines. Stage 3a adds a validation split for tuning, segment breakdowns, and a global-time-cutoff sanity check. Later stages add retrieval, ranking, serving, and operations.

**MovieLens data is not redistributed here.** Download it yourself from GroupLens and respect their [license and terms of use](https://grouplens.org/datasets/movielens/). Raw and derived rating files live under a gitignored `data/` directory and must never be committed. Aggregate EDA stats/figures under `docs/eda/` are fine to commit.

See [`docs/problem.md`](docs/problem.md) for the task definition, primary metric, and stage-gate rule. Design decisions live in [`docs/adr/`](docs/adr/).

## Roadmap

| Stage | Focus |
| --- | --- |
| **S1** | Framing + data (download, clean, split, EDA, eval harness) |
| **S2** | Classic CF baselines |
| **S3a** | Validation split + tuned baselines + segments + global-time-cutoff sanity check |
| **S3b** | Two-tower retrieval (optional torch extra; tune on val only) |
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

Optional **two-tower** extras (S3b; PyTorch CPU wheel):

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[deep]"
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
3. Hold out the last **`max(1, floor(n * 0.2))`** interactions as **test**.
4. From the remaining train pool, hold out the last **`max(1, floor(n_train * 0.1))`** as **validation**; earlier rows are **fit-train** (ADR-0005).

**Tune** on validation only → **refit** chosen configs on **full-train** (= fit-train ∪ val) → **evaluate once** on test. **Never tune on test.**

This **prevents within-user leakage** but **does not prevent cross-user / global temporal leakage** (other users’ later ratings can still appear in train — see ADR-0002). A **global-time-cutoff** sanity check on ml-1m is reported as a secondary table under `results/global_cutoff/`.

**Relevance:** rating **`≥ 4.0`**.

**Cold-start policy:** cold items (absent from train) are removed from relevant test sets; users with no remaining warm relevant items are excluded from ranking-metric averages (counts recorded in results JSON). Already-seen train items are filtered from recommendations.

**Segments (S3a):** NDCG@10 by user-activity terciles (train rating count) and head/tail items (head = top 20% of train items by popularity). Item-segment metrics restrict relevant **and** recommended items to the segment.

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
python scripts/run_eda.py --dataset ml-1m --download
```

Committed aggregate stats and figures for **both** datasets live under [`docs/eda/`](docs/eda/) (rating distribution, user activity, item popularity, ratings over time). Raw data is not committed.

## Tests and CI

```bash
pytest -q
ruff check src tests scripts
```

GitHub Actions installs the CPU PyTorch wheel, runs ruff + pytest (including two-tower unit tests on synthetic data), and does **not** download MovieLens.

## Baselines (S2 defaults + S3a tuned) and S3b two-tower

| Model | Notes |
| --- | --- |
| `most_popular` | Train interaction counts (ties by item id). |
| `item_item_cosine` | Item–item cosine CF (S2 defaults: all neighbours, no shrinkage). |
| `item_item_cosine_tuned` | Same model; `k_neighbors` / `shrinkage` chosen on validation NDCG@10. |
| `als` | `implicit` ALS (factors=64, iterations=15, α=40, seed=42) — S2 defaults. |
| `als_tuned` | ALS with factors/regularization/α chosen on validation NDCG@10. |
| `two_tower` | Optional PyTorch two-tower retrieval (ADR-0006); tuned on val NDCG@10 with early stopping; test over 3 seeds. |

Tuning grids and per-trial validation scores: `results/tuning/*.json` and `results/tuning/two_tower_*.json`.

**S3b gate (ml-1m):** two-tower must beat **both** item–item default and tuned on test NDCG@10 with CIs taken into account, or be written up as a negative result (S4 then uses item–item as the retriever).

## Results

<!-- BEGIN RESULTS TABLE -->

### `ml-1m` (from `results/ml-1m.json`)

Pinned version: `ml-1m@sha256:a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.

Split: min_ratings=5, test_fraction=0.2, val_fraction=0.1, relevance_threshold=4.0, seed=42, ks=[10, 20], bootstrap=1000 @ alpha=0.05. Primary metric: **ndcg@10**. Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003). Names marked **(tuned)** used validation-selected hyperparameters (ADR-0005); others are S2 YAML defaults.

| model | ndcg@10 | precision@10 | recall@10 | ndcg@20 | precision@20 | recall@20 | coverage@10 | mean_popularity@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0.0686 | 0.0564 | 0.0598 | 0.0847 | 0.0542 | 0.1049 | 0.5198 | 956.1292 |
| als_tuned | 0.0906 | 0.0740 | 0.0715 | 0.1072 | 0.0677 | 0.1257 | 0.4541 | 1118.0085 |
| item_item_cosine | 0.1201 | 0.0980 | 0.0786 | 0.1284 | 0.0836 | 0.1277 | 0.1274 | 1794.5672 |
| item_item_cosine_tuned | 0.1192 | 0.0983 | 0.0808 | 0.1313 | 0.0861 | 0.1384 | 0.1721 | 1634.2176 |
| most_popular | 0.0895 | 0.0787 | 0.0466 | 0.0951 | 0.0698 | 0.0874 | 0.0325 | 2325.9361 |

95% bootstrap CIs (NDCG@10):

| model | ndcg@10 CI |
| --- | --- |
| als | [0.0658, 0.0714] |
| als_tuned | [0.0874, 0.0940] |
| item_item_cosine | [0.1158, 0.1242] |
| item_item_cosine_tuned | [0.1147, 0.1233] |
| most_popular | [0.0857, 0.0935] |

#### Segment NDCG@10

User activity = train rating-count terciles (low/mid/high). Item head = top 20% of train items by popularity; tail = rest. Item-segment metrics restrict **relevant and recommended** items to the segment (users with no relevant items in-segment are excluded).

| model | activity low | activity mid | activity high | item head | item tail |
| --- | --- | --- | --- | --- | --- |
| als | 0.0747 [0.0689, 0.0803] | 0.0591 [0.0549, 0.0635] | 0.0721 [0.0675, 0.0770] | 0.0807 [0.0772, 0.0839] | 0.0314 [0.0287, 0.0341] |
| als_tuned | 0.0893 [0.0823, 0.0960] | 0.0800 [0.0748, 0.0849] | 0.1026 [0.0971, 0.1080] | 0.1030 [0.0989, 0.1070] | 0.0329 [0.0300, 0.0360] |
| item_item_cosine | 0.0928 [0.0856, 0.1001] | 0.0943 [0.0881, 0.1002] | 0.1734 [0.1654, 0.1817] | 0.1311 [0.1265, 0.1357] | 0.0014 [0.0007, 0.0023] |
| item_item_cosine_tuned | 0.0917 [0.0844, 0.0988] | 0.0920 [0.0860, 0.0975] | 0.1740 [0.1655, 0.1826] | 0.1307 [0.1259, 0.1352] | 0.0029 [0.0019, 0.0042] |
| most_popular | 0.0395 [0.0351, 0.0440] | 0.0677 [0.0622, 0.0731] | 0.1614 [0.1530, 0.1710] | 0.0946 [0.0908, 0.0988] | 0.0000 [0.0000, 0.0000] |

Tuning log: [`results/tuning/ml-1m.json`](results/tuning/ml-1m.json). Chosen ALS={'alpha': 20.0, 'factors': 128, 'iterations': 15, 'regularization': 0.1} (val NDCG@10=0.0646); item–item={'k_neighbors': 200, 'min_common': 1, 'shrinkage': 100.0} (val NDCG@10=0.0810).

### `ml-latest-small` (from `results/ml-latest-small.json`)

Pinned version: `ml-latest-small@sha256:696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`.

Split: min_ratings=5, test_fraction=0.2, val_fraction=0.1, relevance_threshold=4.0, seed=42, ks=[10, 20], bootstrap=1000 @ alpha=0.05. Primary metric: **ndcg@10**. Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003). Names marked **(tuned)** used validation-selected hyperparameters (ADR-0005); others are S2 YAML defaults.

| model | ndcg@10 | precision@10 | recall@10 | ndcg@20 | precision@20 | recall@20 | coverage@10 | mean_popularity@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0.0795 | 0.0622 | 0.0693 | 0.0948 | 0.0531 | 0.1268 | 0.0965 | 108.2937 |
| als_tuned | 0.0875 | 0.0653 | 0.0764 | 0.1013 | 0.0553 | 0.1284 | 0.1119 | 96.1963 |
| item_item_cosine | 0.0899 | 0.0707 | 0.0771 | 0.1017 | 0.0583 | 0.1266 | 0.0565 | 122.4519 |
| item_item_cosine_tuned | 0.0997 | 0.0763 | 0.0774 | 0.1150 | 0.0674 | 0.1354 | 0.0485 | 160.4564 |
| most_popular | 0.0743 | 0.0563 | 0.0514 | 0.0802 | 0.0466 | 0.0859 | 0.0119 | 216.7525 |

95% bootstrap CIs (NDCG@10):

| model | ndcg@10 CI |
| --- | --- |
| als | [0.0692, 0.0902] |
| als_tuned | [0.0768, 0.0982] |
| item_item_cosine | [0.0784, 0.1011] |
| item_item_cosine_tuned | [0.0867, 0.1130] |
| most_popular | [0.0636, 0.0860] |

#### Segment NDCG@10

User activity = train rating-count terciles (low/mid/high). Item head = top 20% of train items by popularity; tail = rest. Item-segment metrics restrict **relevant and recommended** items to the segment (users with no relevant items in-segment are excluded).

| model | activity low | activity mid | activity high | item head | item tail |
| --- | --- | --- | --- | --- | --- |
| als | 0.0690 [0.0498, 0.0882] | 0.0770 [0.0611, 0.0936] | 0.0924 [0.0748, 0.1131] | 0.0840 [0.0734, 0.0947] | 0.0003 [0.0000, 0.0010] |
| als_tuned | 0.0841 [0.0632, 0.1046] | 0.0744 [0.0587, 0.0919] | 0.1040 [0.0856, 0.1235] | 0.0936 [0.0825, 0.1049] | 0.0008 [0.0000, 0.0021] |
| item_item_cosine | 0.0815 [0.0601, 0.1030] | 0.0756 [0.0600, 0.0921] | 0.1126 [0.0920, 0.1356] | 0.0961 [0.0839, 0.1077] | 0.0004 [0.0000, 0.0012] |
| item_item_cosine_tuned | 0.0890 [0.0660, 0.1133] | 0.0756 [0.0603, 0.0924] | 0.1343 [0.1093, 0.1598] | 0.1056 [0.0925, 0.1195] | 0.0000 [0.0000, 0.0000] |
| most_popular | 0.0588 [0.0401, 0.0795] | 0.0610 [0.0461, 0.0779] | 0.1030 [0.0793, 0.1246] | 0.0783 [0.0673, 0.0914] | 0.0000 [0.0000, 0.0000] |

Tuning log: [`results/tuning/ml-latest-small.json`](results/tuning/ml-latest-small.json). Chosen ALS={'alpha': 20.0, 'factors': 128, 'iterations': 15, 'regularization': 0.1} (val NDCG@10=0.0750); item–item={'k_neighbors': 40, 'min_common': 1, 'shrinkage': 100.0} (val NDCG@10=0.0873).

### Global-time-cutoff sanity check (`ml-1m`, secondary)

From `results/global_cutoff/ml-1m.json`. Secondary sanity check: whether model ranking holds when cross-user future signal is removed (ADR-0002). Not the headline table. Hyperparams: Per-user-protocol tuned configs (validation NDCG@10); not re-tuned for the global cutoff. most_popular has no hyperparameters. **Not re-tuned** for this protocol (`retuned=False`).

Cutoff: timestamp quantile=0.8 (cutoff_timestamp=975768738.0). Surviving: train users=5392, train items=3662, test interactions (after user filter)=103937, eval users (warm relevant)=1114.

| model | ndcg@10 | ndcg@10 CI | precision@10 | recall@10 |
| --- | --- | --- | --- | --- |
| als | 0.1553 | [0.1449, 0.1669] | 0.1479 | 0.0427 |
| item_item_cosine | 0.2319 | [0.2162, 0.2473] | 0.2104 | 0.0575 |
| most_popular | 0.2136 | [0.1990, 0.2286] | 0.1987 | 0.0490 |

<!-- END RESULTS TABLE -->
