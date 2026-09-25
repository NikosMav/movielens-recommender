# movielens-recommender

A small, standalone movie recommender built from scratch on [MovieLens](https://grouplens.org/datasets/movielens/) ratings. Stages 1–2 cover data/split/evaluation and classic collaborative-filtering baselines. Later stages will add two-tower retrieval and a learned ranker.

**MovieLens data is not redistributed here.** Download it yourself from GroupLens and respect their [license and terms of use](https://grouplens.org/datasets/movielens/). All raw and derived data live under a gitignored `data/` directory and must never be committed.

## Roadmap

1. **Data, split, evaluation** — download, time-based holdout, precision/recall/NDCG harness
2. **Baselines** — most-popular, item-item cosine, ALS matrix factorization
3. **Two-tower retrieval** — learned candidate generation
4. **Ranking model** — re-rank retrieved candidates
5. **Final results** — end-to-end comparison and analysis

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Download data

```bash
movielens-recommender download --dataset ml-latest-small   # default, fast
movielens-recommender download --dataset ml-1m             # optional larger set
```

Official GroupLens URLs are used (`files.grouplens.org`). Files land in `data/` (gitignored).

## Split rule

For each user:

1. Drop the user if they have fewer than **`min_ratings=5`** interactions.
2. Sort interactions by **`timestamp` ascending** (stable sort).
3. Hold out the last **`max(1, floor(n * test_fraction))`** interactions as test with **`test_fraction=0.2`**; the rest are train. Users that would have an empty train set are dropped.

The split is deterministic. **Relevance threshold:** ratings **`>= 4.0`** count as relevant for metrics (binary relevance).

## Metrics

Averaged over test users who have at least one relevant test item. Already-seen train items are excluded from recommendations before scoring.

| Metric | Definition (binary relevance) |
| --- | --- |
| Precision@k | `\|recs[:k] ∩ relevant\| / k` |
| Recall@k | `\|recs[:k] ∩ relevant\| / \|relevant\|` |
| NDCG@k | DCG@k / IDCG@k with gain 1 for relevant items and discount `1/log2(rank+1)` |

Reported at **k = 10 and 20**.

## Run the full pipeline

One command downloads (if needed), splits, trains all baselines, evaluates, and writes metrics JSON:

```bash
movielens-recommender run --dataset ml-latest-small
```

Output: `results/ml-latest-small.json` (committed). The JSON includes dataset name, split config, relevance threshold, k values, seed, library versions, hyperparameters, and metrics.

Regenerate the README results table from committed JSON (do not hand-edit numbers):

```bash
python scripts/make_results_table.py
```

## Tests

```bash
pytest -q
```

CI runs **ruff** + **pytest** and does **not** download MovieLens.

## Baselines

| Model | Notes |
| --- | --- |
| `most_popular` | Rank items by training interaction count (ties broken by item id). |
| `item_item_cosine` | Item-item CF; cosine similarity on rating vectors; score = S @ user_ratings. |
| `als` | Alternating Least Squares via the [`implicit`](https://github.com/benfred/implicit) library (factors=64, iterations=15, alpha=40, seed=42). |

Hyperparameters are intentionally simple defaults — not tuned.

## Results

<!-- BEGIN RESULTS TABLE -->

### `ml-1m` (from `results/ml-1m.json`)

Split: min_ratings=5, test_fraction=0.2, relevance_threshold=4.0, seed=42, ks=[10, 20].

| model | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 |
| --- | --- | --- | --- | --- | --- | --- |
| als | 0.0572 | 0.0603 | 0.0692 | 0.0542 | 0.1060 | 0.0853 |
| item_item_cosine | 0.0996 | 0.0804 | 0.1219 | 0.0850 | 0.1307 | 0.1306 |
| most_popular | 0.0790 | 0.0468 | 0.0897 | 0.0707 | 0.0895 | 0.0961 |

### `ml-latest-small` (from `results/ml-latest-small.json`)

Split: min_ratings=5, test_fraction=0.2, relevance_threshold=4.0, seed=42, ks=[10, 20].

| model | precision@10 | recall@10 | ndcg@10 | precision@20 | recall@20 | ndcg@20 |
| --- | --- | --- | --- | --- | --- | --- |
| als | 0.0621 | 0.0682 | 0.0790 | 0.0530 | 0.1250 | 0.0941 |
| item_item_cosine | 0.0706 | 0.0761 | 0.0895 | 0.0582 | 0.1248 | 0.1009 |
| most_popular | 0.0562 | 0.0505 | 0.0740 | 0.0465 | 0.0844 | 0.0796 |

<!-- END RESULTS TABLE -->
