# ADR-0007: LambdaRank over a validation-chosen candidate set

## Status

Accepted (S4). The headline gate is test NDCG@10 against item–item cosine (the best S3 model). The pass/fail line is the `ranker.gate` object in `results/*.json`, written by the pipeline. This ADR does not invent that number.

## Context

S3a set the per-user time split, validation tuning, segments, and the global-time-cutoff check (ADR-0005, ADR-0002). S3b added a two-tower retriever that missed the NDCG@10 gate and led item–item on Recall@100/200 (ADR-0006). S4 re-ranks a candidate set. The candidate set is chosen on validation recall, not on test NDCG.

The ranker has to stay small, CPU-trainable on ml-1m, and free of test labels. MovieLens files and model binaries stay out of git.

## Decision

**Ranker:** LightGBM LambdaRank (`objective=lambdarank`, NDCG@10 early stopping). It lives in an optional extra, `pip install -e ".[rank]"`, with **lightgbm pinned to 4.6.0**. Three seeds are reported. They share the candidate sets and the retriever models. Only the LightGBM seed changes.

**Candidate sets, chosen on validation only**, at a fixed budget `K` (200):

| Set | Construction |
| --- | --- |
| `item_item` | Top K from **tuned** item–item cosine |
| `two_tower` | Top K from the tuned two-tower (fixed epoch count = validation `best_epoch`) |
| `union_balanced` | Top K/2 from each source (integer division), de-duplicated, then backfill alternating item–item then two-tower until length K |
| `union_unbalanced` | Top K from each source, alternating and de-duplicated. Not truncated to K. Mean size is reported |

Validation reports are candidate Recall@100 and Recall@200 (point estimates, same warm-relevant rule as the harness). The default set is the winner by Recall@200, then Recall@100, then the order `union_balanced`, `item_item`, `two_tower`, `union_unbalanced`. Item–item is source A in every alternation.

**Features** (fit only on the matrix named below):

- item–item score and rank, two-tower score and rank (NaN when that retriever did not return the item in its top K)
- `in_both`
- item popularity and recency (max timestamp) from the allowed matrix
- genres and release year from the movies file
- user activity: rating count, mean rating, rating std
- user–genre affinity: mean genre multi-hot over the user's allowed interactions

**Labels:** the validation window only. A pair is relevant when its validation rating is ≥ 4.0. Candidates the user did not rate in that window are 0. Test-window labels never enter training, early stopping, or candidate-set selection.

**Early stopping:** a held-out slice of validation users (`early_stop_fraction=0.2`, split by the experiment seed, not the ranker seed). `best_iteration` is recorded. The refit ranker is then trained on **all** validation-window users for `num_boost_round = best_iteration` with no validation callback. Features for that fit are still fit-train features.

### Refit semantics

Before scoring test, both retrievers are retrained on full train (fit-train union validation) and every feature is rebuilt from that matrix and those retrievers. The ranker is trained only on validation-window labels with features from the train part (fit-train). A held-out slice of validation users is used only to early-stop and record best_iteration. The refit ranker is then trained on all validation-window users for num_boost_round equal to best_iteration, still with fit-train features, and with no validation callback. Test-window labels never enter ranker training, early stopping, or candidate-set selection. If the ranker is retrained on a later window, num_boost_round is fixed to that best_iteration (primary seed): no new early stopping and no new candidate-set choice. On the global-time cutoff, the later window is the pre-cutoff train. Its chronological tail (val_fraction) supplies labels; features and retrievers for that training come from the head only. Before scoring the post-cutoff test, retrievers and features are rebuilt on the full pre-cutoff train. Post-cutoff labels are never used.

### Leakage controls

- Ranker-training retrievers are fit on fit-train (train minus validation).
- Feature context for that fit reads only fit-train timestamps, counts, and ratings.
- The full-train rebuild, used only at test scoring, reads fit-train ∪ validation and does not read test timestamps.
- The same head/tail rule applies inside the global-cutoff train. The post-cutoff test is not a label source.
- Unit tests assert item recency and user counts match the allowed frame and do not pick up a later forbidden timestamp.

### Ablations (test, primary seed unless noted)

- LambdaRank over `item_item`, `two_tower`, and `union_balanced` (and `union_unbalanced` when that set wins)
- `no_ranker`: the winning set in its original retriever order, after the full-train retriever refit
- `lambdarank_drop_retriever_features`: same winning set, dropping item–item and two-tower score and rank columns (`in_both` stays)

The headline `lambdarank` row is the mean over the three ranker seeds on the winning set. Feature gain importance is from the primary-seed refit booster.

### Explainability hook

`explain_candidates` returns per-candidate LightGBM `pred_contrib=True` contributions (feature SHAP values plus bias) mapped to feature names, the raw score, and which retriever(s) supplied the candidate with their scores and ranks. The refit ranker is reloadable from a gitignored `models/<dataset>/` directory (`ranker.txt`, `ranker_meta.json`, schema version 1) via `movielens-recommender run --config configs/<dataset>.yaml`. A unit test checks that contributions plus bias sum to the raw score. There is no UI in this stage.

### Gate

The bar is `metrics["item_item_cosine"]["ndcg@10"]` from the same results JSON (S3 default item–item cosine). It is not hard-coded. `negative_result` is true when the 3-seed mean does not strictly exceed that point estimate. Seed-CI separation versus the bar CI is recorded and is not a substitute for the point gate.

### Future work

A small torch MLP ranker, reusing the existing `[deep]` extra, is not in this stage.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Torch MLP ranker now | Heavier than LightGBM for a first ranker; recorded as future work on the `[deep]` extra |
| Hand-tuned ranker hyperparameters on test | Forbidden by ADR-0005 |
| Training the ranker on test-window labels | Leakage |
| Rebuilding ranker features from validation or test rows | Leakage; features stay on the allowed train matrix |
| One shared seed that also refits the retrievers | The brief asks for shared retrievers and candidate sets, with only the ranker seed changing |
| Unbalanced union as the only union | Its size is not K, so Recall@200 would mix budget with ordering. It is reported, and balanced union is the fixed-budget union |
| ANN / another retrieval stack | Catalog size does not need it (ADR-0006) |

## Outcome

Every figure below is read from the committed results JSON. Metric columns use 4 decimal places; validation candidate recall and mean size use 6, matching `scripts/make_results_table.py`. The README results tables are generated from the same files.

### ml-1m

Source: `results/ml-1m.json`. Pipeline runtime 668.3570s. Ranker stage runtime 335.6810s. lightgbm 4.6.0.

Validation candidate recall at K=200 (n_eval_users=5619). Winner: `two_tower`.

| candidate set | recall@100 | recall@200 | mean size |
| --- | --- | --- | --- |
| item_item | 0.435321 | 0.593272 | 200.000000 |
| two_tower | 0.489363 | 0.655227 | 200.000000 |
| union_balanced | 0.485470 | 0.653053 | 200.000000 |
| union_unbalanced | 0.485470 | 0.651810 | 280.720057 |

Test NDCG@10 over the three ranker seeds (shared retrievers and candidate sets):

| seed | best_iteration | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 42 | 95 | 0.1288 | [0.1247, 0.1329] | 0.0943 | 0.3763 |
| 43 | 94 | 0.1274 | [0.1232, 0.1316] | 0.0917 | 0.3919 |
| 44 | 43 | 0.1256 | [0.1213, 0.1301] | 0.0925 | 0.4003 |

Across seeds: mean=0.1273, std=0.0013, min=0.1256, max=0.1288.

Gate: ranker mean NDCG@10=0.1273 against item_item_cosine 0.1201 [0.1158, 0.1242] from `metrics['item_item_cosine']['ndcg@10']`. `mean_exceeds_bar_point=true`. `negative_result=false`. `all_seed_ci_low_above_bar_ci_high=false`.

Item–item cosine in the same file: NDCG@10=0.1201, Recall@10=0.0786, Coverage@10=0.1274, tail NDCG@10=0.0321 [0.0292, 0.0354].

Ablations. On the `lambdarank` row, NDCG@10, Recall@10, and Coverage@10 are 3-seed means; the NDCG@10 CI and tail NDCG@10 are the primary seed.

| ablation | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 | tail ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| lambdarank | 0.1273 | [0.1247, 0.1329] | 0.0928 | 0.3895 | 0.0915 [0.0868, 0.0969] |
| lambdarank_drop_retriever_features | 0.1000 | [0.0964, 0.1037] | 0.0622 | 0.3736 | 0.0732 [0.0691, 0.0777] |
| lambdarank_item_item | 0.1264 | [0.1225, 0.1306] | 0.0909 | 0.3270 | 0.0492 [0.0456, 0.0533] |
| lambdarank_two_tower | 0.1288 | [0.1247, 0.1329] | 0.0943 | 0.3763 | 0.0915 [0.0868, 0.0969] |
| lambdarank_union_balanced | 0.1245 | [0.1201, 0.1286] | 0.0921 | 0.3744 | 0.0907 [0.0860, 0.0960] |
| no_ranker | 0.1191 | [0.1150, 0.1230] | 0.0897 | 0.4669 | 0.0825 [0.0782, 0.0873] |

Top feature gains (primary-seed refit booster):

| feature | gain |
| --- | --- |
| two_tower_rank | 12083.8040 |
| item_item_rank | 5382.0176 |
| item_item_score | 4988.1077 |
| item_popularity | 4535.2138 |
| user_n_ratings | 4488.3243 |
| two_tower_score | 3534.1110 |
| item_recency | 2220.1169 |
| affinity_romance | 2140.1855 |
| user_mean_rating | 1925.5218 |
| affinity_comedy | 1850.7273 |

### ml-latest-small

Source: `results/ml-latest-small.json`. Pipeline runtime 129.4080s. Ranker stage runtime 41.7760s. lightgbm 4.6.0.

Validation candidate recall at K=200 (n_eval_users=536). Winner: `union_unbalanced`.

| candidate set | recall@100 | recall@200 | mean size |
| --- | --- | --- | --- |
| item_item | 0.415189 | 0.532968 | 200.000000 |
| two_tower | 0.403221 | 0.526670 | 200.000000 |
| union_balanced | 0.418921 | 0.553063 | 200.000000 |
| union_unbalanced | 0.418921 | 0.553083 | 309.819030 |

Test NDCG@10 over the three ranker seeds (shared retrievers and candidate sets):

| seed | best_iteration | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 42 | 1 | 0.0852 | [0.0734, 0.0968] | 0.0692 | 0.0999 |
| 43 | 9 | 0.0987 | [0.0867, 0.1102] | 0.0861 | 0.0931 |
| 44 | 72 | 0.1131 | [0.0990, 0.1257] | 0.0961 | 0.0985 |

Across seeds: mean=0.0990, std=0.0114, min=0.0852, max=0.1131.

Gate: ranker mean NDCG@10=0.0990 against item_item_cosine 0.0899 [0.0784, 0.1011] from `metrics['item_item_cosine']['ndcg@10']`. `mean_exceeds_bar_point=true`. `negative_result=false`. `all_seed_ci_low_above_bar_ci_high=false`.

Item–item cosine in the same file: NDCG@10=0.0899, Recall@10=0.0771, Coverage@10=0.0565, tail NDCG@10=0.0105 [0.0051, 0.0179].

Ablations. On the `lambdarank` row, NDCG@10, Recall@10, and Coverage@10 are 3-seed means; the NDCG@10 CI and tail NDCG@10 are the primary seed.

| ablation | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 | tail ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| lambdarank | 0.0990 | [0.0734, 0.0968] | 0.0838 | 0.0972 | 0.0119 [0.0062, 0.0185] |
| lambdarank_drop_retriever_features | 0.0838 | [0.0722, 0.0960] | 0.0647 | 0.1157 | 0.0088 [0.0051, 0.0131] |
| lambdarank_item_item | 0.1071 | [0.0940, 0.1199] | 0.0899 | 0.0924 | 0.0015 [0.0000, 0.0041] |
| lambdarank_two_tower | 0.0933 | [0.0813, 0.1057] | 0.0766 | 0.1003 | 0.0105 [0.0061, 0.0157] |
| lambdarank_union_balanced | 0.1218 | [0.1078, 0.1354] | 0.0975 | 0.0878 | 0.0227 [0.0134, 0.0340] |
| lambdarank_union_unbalanced | 0.0852 | [0.0734, 0.0968] | 0.0692 | 0.0999 | 0.0119 [0.0062, 0.0185] |
| no_ranker | 0.0983 | [0.0860, 0.1110] | 0.0803 | 0.1054 | 0.0114 [0.0062, 0.0175] |

Top feature gains (primary-seed refit booster):

| feature | gain |
| --- | --- |
| in_both | 275.5560 |
| two_tower_rank | 85.9516 |
| item_recency | 47.2973 |
| user_n_ratings | 42.1854 |
| affinity_drama | 41.1328 |
| item_item_rank | 32.6754 |
| affinity_comedy | 28.1487 |
| affinity_western | 24.4671 |
| item_item_score | 23.9401 |
| affinity_musical | 21.2151 |

### Global-time cutoff (ml-1m, secondary)

Source: `results/global_cutoff/ml-1m.json`. `retuned=false`. Ranker protocol: {"candidate_set": "two_tower", "early_stopping": false, "labels": "chronological tail of pre-cutoff train", "num_boost_round": 95, "scoring_features": "full pre-cutoff train", "test_labels_used": false, "training_features": "head of pre-cutoff train"}. Item–item hyperparameters reused on this split: {"k_neighbors": 200, "min_common": 1, "shrinkage": 100.0}.

| model | ndcg@10 | ndcg@10 CI | recall@10 | recall@200 |
| --- | --- | --- | --- | --- |
| lambdarank | 0.2323 | [0.2193, 0.2465] | 0.0720 | 0.4801 |
| no_ranker | 0.2140 | [0.2007, 0.2277] | 0.0655 | 0.4801 |
| item_item_cosine | 0.2319 | [0.2162, 0.2473] | 0.0575 | 0.4171 |
| two_tower | 0.2140 | [0.2007, 0.2277] | 0.0655 | 0.4801 |
| als | 0.1553 | [0.1449, 0.1669] | 0.0427 | 0.3875 |
| most_popular | 0.2136 | [0.1990, 0.2286] | 0.0490 | 0.3866 |

### Reading

The pre-registered gate compares the 3-seed mean test NDCG@10 with `metrics["item_item_cosine"]["ndcg@10"]` in the same file. On ml-1m that mean is above the item–item point estimate (0.1201 at 4 decimals), so `negative_result` is false. This is a point-estimate win. `all_seed_ci_low_above_bar_ci_high` is false: the seed intervals overlap the bar interval. Each seed's own NDCG@10 point estimate is still above the bar point.

On ml-1m the no-ranker line (winning two-tower order after the full-train refit) and the drop-retriever-features ablation both sit under that bar. The recorded lift is LambdaRank using the retriever score and rank features. Primary-seed tail NDCG@10 is higher for the ranker than for item–item cosine and higher than the no-ranker tail. Coverage@10 of the ranker mean is higher than item–item cosine and lower than the no-ranker line. Gain is led by `two_tower_rank`, `item_item_rank`, `item_item_score`, `item_popularity`, and `user_n_ratings`.

On ml-latest-small, validation Recall@200 selects `union_unbalanced` over `union_balanced` by a margin that shows up at 6 decimals, with mean size above K. The 3-seed mean exceeds that dataset's item–item point estimate, so `negative_result` is false there too. The primary seed's `best_iteration` is 1, and that seed's test NDCG@10 is under the item–item point and under the no-ranker line. The primary-seed CI does not cover the 3-seed mean. Seed std is an order of magnitude larger than on ml-1m. The small-dataset point win is the mean, and it is unstable.

The global cutoff keeps the ml-1m winner (`two_tower`) and trains for a fixed `num_boost_round` equal to the headline primary `best_iteration` (95). `test_labels_used` is false. On that split the ranker NDCG@10 point is just above the reused tuned item–item row, and the no-ranker row matches two-tower. Intervals on that check overlap. It is a secondary sanity check, not the headline gate.

## Consequences

- Headline `results/*.json` gains `lambdarank`, the ablations, and a `ranker` block.
- `results/global_cutoff/ml-1m.json` includes the fixed-round ranker when the global cutoff is enabled. It is not re-tuned.
- `models/` is gitignored. The recreate command is in `ranker_meta.json` and the README.
- CI installs the `[rank]` extra and runs synthetic tests only. It does not download MovieLens and does not run ml-1m.
