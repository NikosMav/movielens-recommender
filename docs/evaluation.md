# Evaluation and models

How the data is prepared, how every model is measured, which models exist, and how to rerun the experiments. The numbers themselves are in [results.md](results.md).

Back to the [README](../README.md).

## Data

Archives come from the official GroupLens URLs and are checked against pinned SHA-256 checksums (ADR-0001). Raw and derived rating files live under the gitignored `data/` directory and are never committed. CI never downloads MovieLens.

```bash
movielens-recommender download --dataset ml-latest-small   # default
movielens-recommender download --dataset ml-1m
movielens-recommender download --dataset ml-32m
```

| Dataset | SHA-256 |
| --- | --- |
| ml-latest-small | `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436` |
| ml-1m | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |
| ml-32m | `e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee` |

Only ml-1m has user demographics (`users.dat`), so `configs/default.yaml` and `configs/ml-32m.yaml` keep demographic features off.

**Cleaning rules**

1. Drop nulls in `user_id`, `item_id`, `rating`, `timestamp`.
2. Keep ratings in `[0.5, 5.0]`.
3. Require positive ids and timestamps.
4. Deduplicate `(user_id, item_id)`, keeping the latest timestamp (then the higher rating).

## Evaluation protocol

The full definition is in [`docs/problem.md`](problem.md).

**Split.** For each user, deterministically:

1. Drop the user if they have fewer than `min_ratings=5` ratings.
2. Sort by `timestamp` ascending (stable).
3. Hold out the last `max(1, floor(n * 0.2))` ratings as **test**.
4. From the rest, hold out the last `max(1, floor(n_train * 0.1))` as **validation**. Earlier rows are **fit-train** (ADR-0005).

Tune on validation only, refit the chosen config on full train (fit-train plus validation), then evaluate once on test. Test is never used for tuning. New ml-32M runs draw the 8,000 evaluation users from full train only (`user_sample_mode: train_only`, ADR-0029). Reproducing `results/ml-32m.json` keeps the historical warm-test sample.

This prevents within-user leakage but not cross-user, global temporal leakage: other users' later ratings can still appear in train (ADR-0002). A global-time-cutoff check on ml-1m is reported as a secondary table.

**Relevance.** A rating of 4.0 or higher.

**Cold items.** Items absent from train are removed from relevant test sets. Users with no remaining warm relevant items are left out of metric averages, and the counts are recorded in the results JSON. Already-seen train items are filtered from recommendations.

**Metrics.**

| Role | Metric |
| --- | --- |
| Primary | **NDCG@10** |
| Secondary | NDCG@20, Precision@k, Recall@k (k = 10, 20) |
| Diagnostics | Catalog coverage@k, mean train popularity of recommended items |
| Segments | NDCG@10 by user-activity tercile, and by head/tail item (head = top 20% of train items by popularity) |

Metrics are averaged over eligible test users, with 95% bootstrap intervals over users (1000 resamples, seed from config). Catalog coverage is a point estimate only, because a user-level bootstrap is not valid for a set-union statistic (ADR-0003). Item-segment metrics restrict both relevant and recommended items to the segment. Novelty, intra-list diversity, and long-tail share are defined in [ADR-0017](adr/0017-beyond-accuracy.md) and reported for ml-1m in the Results section.

## Models

| Model | Notes |
| --- | --- |
| `most_popular` | Train interaction counts, ties broken by item id |
| `item_item_cosine` | Item–item cosine CF with S2 defaults: all neighbours, no shrinkage |
| `item_item_cosine_tuned` | Same model; `k_neighbors` and `shrinkage` chosen on validation NDCG@10 |
| `als` | `implicit` ALS with S2 defaults (factors 64, 15 iterations, α 40, seed 42) |
| `als_tuned` | ALS with factors, regularization, and α chosen on validation NDCG@10 |
| `ease` | EASE^R (Steck 2019); λ chosen on validation NDCG@10 (ADR-0008) |
| `ease_dan` | EASE-DAN (Park et al. 2025 / Steck 2020); closed-form linear autoencoder with item-degree dropout and adaptive normalization (ADR-0028) |
| `rp3beta` | RP3beta random-walk item similarity; `(alpha, beta, top_k)` chosen on validation (ADR-0008) |
| `two_tower` | PyTorch two-tower retrieval with an in-batch sampled-softmax loss; tuned on validation with early stopping; tested over 3 seeds (ADR-0006) |
| `sasrec` | Causal self-attentive sequence recommender trained with full-catalog softmax cross-entropy (ADR-0019) |
| `esasrec` | SASRec with LiGR residual blocks and a SwiGLU feed-forward (eSASRec, RecSys 2025; ADR-0030) |
| `pctm` | Pairwise transition model with Dirichlet smoothing and recency log-pooling (PCTM, RecSys 2026; ADR-0030) |
| `lambdarank` | LightGBM LambdaRank re-ranking the validation-chosen candidate set (ADR-0007) |
| `seq_off` | The served ranker: LambdaRank over PCTM and two-tower candidates, with PCTM score and rank as features and no demographics (ADR-0033) |

Tuning grids and per-trial validation scores are in `results/tuning/`.

**Two-tower gate (S3b).** On ml-1m the two-tower must beat both item–item default and tuned on test NDCG@10, with intervals taken into account, or be written up as a negative result. It was written up as one.

**Ranker (S4).** Candidate sets (tuned item–item, two-tower, a balanced union, and an unbalanced union) are compared on validation Recall@100/200, and the winner is the default. The balanced sets use budget K; the unbalanced union can be larger. The ranker trains on validation-window labels with fit-train features, early-stops on held-out validation users, then refits for `best_iteration` rounds. Before test scoring, retrievers and features are rebuilt on full train. All three stored seed means exceed the historical default item-item point. On ml-latest-small the ranker remains below EASE and tuned item-item; on ml-32M it remains below the two-tower point. A fixed-baseline pass therefore does not establish an overall stage improvement. New runs also record a primary-seed paired interval against the unranked list. Details are in [ADR-0007](adr/0007-ranker.md) and [ADR-0021](adr/0021-results-and-reproducibility.md).

`run` also writes a gitignored refit ranker to `models/<dataset>/` (`ranker.txt`, `ranker_meta.json`). `explain_candidates` returns per-candidate LightGBM `pred_contrib` values (SHAP values plus bias) by feature name, with the retriever that supplied each candidate and its score and rank. The Streamlit page turns those into short sentences (ADR-0010).

## Reproducing the results

To rerun a published headline protocol, use its saved result record. This reads the recorded configuration and selected hyperparameters, including demographics **off** for the S4 ml-1m headline, rather than today's serving defaults:

```bash
movielens-recommender reproduce --from-results results/ml-latest-small.json
movielens-recommender reproduce --from-results results/ml-1m.json
movielens-recommender reproduce --from-results results/ml-32m.json
```

Each command reserves a unique directory under `artifacts/runs/`, freezes its inputs, and writes results and model files there. It reuses the recorded validation selections, then refits and evaluates with the current implementation. It does not overwrite published results or serving snapshots. Existing data is required unless `--download` is supplied. Add `--prepare-only` to inspect the config without training. `--run-id` supplies a label; an existing label is refused. See [the reproduction guide](reproduction.md) for environment limits and the other experiments.

The original `run` command performs a new experiment using the current YAML and writes to its configured output paths:

```bash
movielens-recommender run --config configs/default.yaml   # ml-latest-small
movielens-recommender run --config configs/ml-1m.yaml
movielens-recommender run --config configs/ml-32m.yaml    # long; budget in ADR-0011
movielens-recommender drift-study --config configs/ml-1m.yaml   # S6 replay, about 3 min
python scripts/run_ranker_pctm.py                         # ADR-0033, about 8 min
python scripts/run_beyond_accuracy.py                     # ADR-0017, ml-1m only
python scripts/run_als_retune.py --stage all              # ADR-0018, ml-1m ALS retune
python scripts/run_sasrec.py --stage all                  # ADR-0019, ml-1m SASRec
python scripts/run_ease_dan.py --stage all                # ADR-0028, ml-1m EASE-DAN
python scripts/run_sequence_modern.py --stage all         # ADR-0030, eSASRec and PCTM
python scripts/run_ranker_ease_dan.py                     # ADR-0031, third retriever
python scripts/run_slider_eval.py                         # ADR-0032, discovery slider
python scripts/make_results_table.py                      # regenerate the Results section
```

Each run writes `results/<dataset>.json`: metrics, intervals, the data checksum, the split, the config, and library versions. The tables in [results.md](results.md) are generated from those files; never edit its numbers by hand.

New multi-seed output separates `seed_mean_metrics`, `primary_seed_metrics`, and `per_seed_metrics`. Top-level scalar metric aliases are seed means; their `confidence_intervals` block is empty because no interval for that mean was computed. Individual user-bootstrap intervals and segment diagnostics belong to their named seed. Historical JSON records retain their original schema and measurements; the table generator reads both formats.

EDA (aggregate stats and figures only, committed under [`docs/eda/`](eda/)):

```bash
python scripts/run_eda.py --dataset ml-latest-small --download
python scripts/run_eda.py --dataset ml-1m --download
```

## Stages

Model improvements are assessed on **NDCG@10** on the same harness. The historical S4 gate compares with default item-item cosine only. Passing it does not mean beating every prior model. The generated results also compare the ranker with the strongest recorded prior point and its unranked candidate list; these descriptive comparisons do not choose training settings or establish significance.

| Stage | Status | Focus | ADR |
| --- | --- | --- | --- |
| S1 | Done | Framing and data: download, cleaning, split, EDA, evaluation harness | [0001](adr/0001-dataset-version.md)–[0004](adr/0004-tooling-and-tracking.md) |
| S2 | Done | Classic collaborative-filtering baselines | |
| S3a | Done | Validation split, tuned baselines, segments, global-time-cutoff check | [0005](adr/0005-validation-and-tuning.md) |
| S3b | Done | Two-tower retrieval | [0006](adr/0006-two-tower-retrieval.md) |
| S3c | Done | EASE^R and RP3beta baselines | [0008](adr/0008-ease-rp3beta.md) |
| S3d | Done | Scale-up to MovieLens 32M | [0011](adr/0011-ml-32m-scale.md) |
| S3e | Done | Full-softmax and sampled-softmax two-tower | [0013](adr/0013-full-softmax-two-tower.md) |
| S4 | Done | LightGBM LambdaRank over a validation-chosen candidate set | [0007](adr/0007-ranker.md) |
| S4b | Done | User demographic features (ml-1m) | [0009](adr/0009-user-demographics.md) |
| S5a | Done | Streamlit explainable UI | [0010](adr/0010-streamlit-ui.md) |
| S5c | Done | New-user profile (cold start) | [0012](adr/0012-new-user-cold-start.md) |
| S5b | Done | Batch recommendations, FastAPI service, Dockerfile | [0014](adr/0014-batch-api-docker.md) |
| S6 | Done | Operations: monitoring, refresh, drift | [0016](adr/0016-drift-refresh-monitoring.md) |
| S7a | Done | Beyond-accuracy evaluation & ALS retune | [0017](adr/0017-beyond-accuracy.md)–[0018](adr/0018-als-retune.md) |
| S7b | Done | Sequential self-attention (SASRec) & shuffled-order check | [0019](adr/0019-sasrec.md) |
| S7c | Done | Unified public showcase UI, discovery controls, & RSS budget | [0020](adr/0020-huggingface-space.md)–[0027](adr/0027-readable-benchmarks-and-pipeline.md) |
| S8 | Done | Data-Adaptive Normalized EASE (`EASE-DAN`, SIGIR 2025) | [0028](adr/0028-ease-dan.md) |
| S8b | Done | Train-only evaluation-user sampling for new ml-32M runs | [0029](adr/0029-train-only-user-sampling.md) |
| S9 | Done | eSASRec vs pairwise transitions (PCTM) | [0030](adr/0030-esasrec-and-pctm.md) |
| S10 | Done | EASE-DAN as a third LambdaRank retriever | [0031](adr/0031-ranker-ease-dan.md) |
| S11 | Done | Offline evaluation of the discovery slider | [0032](adr/0032-discovery-slider-evaluation.md) |
| S12 | Done | Pairwise transitions in the ranker, and the served model | [0033](adr/0033-pctm-in-the-ranker.md) |
| S13 | Done (negative) | A larger Discover catalogue from ml-32M: fold-in, then a popularity blend | [0034](adr/0034-extended-discover-catalogue.md)–[0035](adr/0035-extended-catalogue-popularity-blend.md) |

## Engineering process

Work landed in stages. Each decision has an ADR in `docs/adr/`. Experiments were registered before they ran. Reported numbers come from the committed `results/*.json` files, and the tables in [results.md](results.md) are generated from those files. GitHub Actions runs the tests before a change merges.

The [demo review](reviews/2026-11-11-demo-review.md) records comparisons with
other movie discovery tools, two interface review rounds, and a saved-snapshot
probe of what the adventurous slider changes. Those behavioral checks are
separate from the offline ranking metrics.
