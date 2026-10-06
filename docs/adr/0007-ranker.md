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

Filled from `results/*.json` after the pipeline run. See the `ranker` object (validation candidate recall, seed NDCG@10, ablations, feature gains, gate, runtime). README tables are generated from that JSON.

## Consequences

- Headline `results/*.json` gains `lambdarank`, the ablations, and a `ranker` block.
- `results/global_cutoff/ml-1m.json` includes the fixed-round ranker when the global cutoff is enabled. It is not re-tuned.
- `models/` is gitignored. The recreate command is in `ranker_meta.json` and the README.
- CI installs the `[rank]` extra and runs synthetic tests only. It does not download MovieLens and does not run ml-1m.
