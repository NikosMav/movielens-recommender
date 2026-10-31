# ADR-0019: SASRec on ml-1m

## Status

Accepted. This is ADR-0019, the next number on main at `3096324`, after ADR-0018. The model, the grid, the budget cut, the shuffled-order check, and the paired comparisons below are fixed before any training run and before any ranking metric. A timing probe may shrink the grid. It records wall time only (`ranking_metrics_recorded` is false) and its weights are discarded. The outcome section is filled only after `results/sasrec/ml-1m.json` exists, and it does not change the decisions above it.

## Context

The committed ml-1m models score a user from a bag of train interactions. Item–item, EASE, RP3beta, ALS, and the two-tower either ignore order or mean-pool a truncated history (ADR-0006, `max_history` 50). The headline two-tower test NDCG@10 is 0.119246 (primary seed 0.119078, interval [0.114975, 0.123048]). The headline LambdaRank, on two-tower candidates, is 0.127264 (primary seed 0.128786, interval [0.12469, 0.132934]). Both numbers are the committed `results/ml-1m.json` figures. This stage asks whether a causal sequence model on the same split beats a copy of itself whose sequences are shuffled, and where that model sits relative to those two headlines.

The comparison is only informative if the architecture, the loss, the search, and the shuffle are fixed before the validation scores exist. `results/ml-1m.json` is not rewritten.

## Decision

**Model.** A SASRec encoder written in this repository, in PyTorch, which is already the two-tower dependency. No external recommender framework is imported.

The user is the sequence of their allowed interactions, not an id embedding. There are no genre or year features. Those would mix this comparison with the two-tower's side features.

| Piece | Value |
| --- | --- |
| Tokens | Item ids in the allowed slice, timestamp ascending. Ties keep the original row order (`kind="mergesort"`), the same stable order as the split |
| Window | The most recent `max_seq_len` interactions. Shorter histories are left-padded with index 0 |
| Embedding | Item embedding of width `d_model`, row 0 reserved for padding and held at 0. Learned positional embedding of length `max_seq_len`. The rightmost slot is always the latest kept item, so it always receives the last position id |
| Blocks | `n_layers` pre-norm residual blocks. Each block is LayerNorm, causal multi-head self-attention, dropout, residual add, then LayerNorm, a pointwise feed-forward, dropout, residual add |
| Attention | `n_heads` heads, `d_model` divisible by `n_heads`. A position attends to itself and to earlier positions, and to neither future positions nor padding. The mask is boolean: future entries are blocked |
| Feed-forward | `Linear(d_model, d_model)`, ReLU, `Linear(d_model, d_model)`. Inner width equals `d_model`, as in the SASRec pointwise network, not a 4× transformer expansion |
| Dropout | One rate on the sum of the item and position embeddings, on the attention weights, and on both residual branches |
| Output | Tied item embedding, pad row excluded. The logit for an item is the dot product of the hidden state with that row. No output bias |
| Inference state | Hidden state at the last slot. That slot has attended to every real token in the window |
| Rank | Exact scores over the fit catalog. Seen items in the allowed slice are set to −∞, then `top_n_indices` (ADR-0015). The harness filters seen items again |
| Init | PyTorch defaults, after the seed is set. `Embedding(padding_idx=0)` keeps the pad row at 0 |
| Optimizer | Adam, learning rate 0.001, default betas, `weight_decay` 0. Device is CPU |

`d_model` is 64, the committed two-tower width. `n_heads` is 1, so each position has one causal attention distribution of width `d_model`. Neither is searched. Batch size is 128. Users are shuffled each epoch with a torch generator seeded from the model seed and the epoch index. `drop_last` is false. Data-loader workers are 0.

**What the window contains.** Every interaction in the allowed slice, at any rating. A rating below the relevance threshold still happened, and dropping it would rewrite the order of the remaining items. The loss does not treat that rating as a positive.

**Loss: full softmax.** At each position whose target is the next item in the window, the training target is that next item and the denominator is every catalog item except the pad row. Cross-entropy is averaged over positions whose target rating is at least 4.0. Positions with a lower target rating, and pad positions, contribute nothing. The mean may be accumulated in chunks so the logit tensor stays small. The chunk size is not a hyperparameter and does not change the mean.

Binary cross-entropy with sampled negatives is the published SASRec objective. It is rejected here. The ml-1m catalog on this split is 3,667 train items, and ADR-0013 already measured a full-softmax logit batch at 14.3 MiB on that catalog. One sampled negative per position is a high-variance stand-in for the ranking the harness computes, and the sampling distribution would be a second unregistered hyperparameter. Full softmax trains the same catalog distribution that inference sorts. It is fixed, not a grid axis. Seen items are not removed inside the training denominator. They are removed only when the list is cut.

**Sequences and leakage.** The allowed slice follows the existing per-user split (ADR-0005): `test_fraction=0.2`, `val_fraction=0.1`, `min_ratings=5`, relevance 4.0.

- While a config is scored on validation, the slice is fit-train (`split.train`). Validation rows and test rows are not tokens and are not targets.
- The test model is a refit on full train (fit-train ∪ validation) for `best_epoch` epochs. Its tokens are full-train interactions. Test rows are not tokens and are not targets.
- Validation rows are training rows of that refit. They are not available to the model that validation selects. That is the same boundary as the two-tower history (ADR-0006), not a second look at the test set.

A unit test builds a synthetic split and checks three claims: a fit-train window is disjoint from that user's validation and test item ids; a full-train window is disjoint from that user's test item ids; every training target lies in the same allowed slice as the window. The causal-mask test checks that a future token cannot change an earlier hidden state, and that the mask blocks `j > i` and allows `j ≤ i`.

**Training and selection.** Same protocol as ADR-0005 and ADR-0006.

1. Fit each trial on fit-train only. Seed 42.
2. After each epoch, score validation NDCG@10 with `ndcg_point_estimate` (point estimate, no bootstrap). Early stop when the score does not exceed the best by more than 1e-6 for `patience` 3 epochs. `max_epochs` is 30 unless the budget cut lowers it.
3. The trial score is that best validation NDCG@10, stored at 6 decimals. The winner is the strict maximum. A tie keeps the earlier config.
4. Refit the winner on full train for exactly `best_epoch` epochs, with no validation pass. Score test once.

`best_epoch` is 1-indexed. The validation interval reported below is computed after this choice, on the early-stopped fit-train checkpoint, and is not an input to it.

**Grid.** Eight configs, the same count as the ADR-0006 two-tower grid and the EASE grid. Smaller than RP3beta (125) and the ADR-0018 ALS search (144), which are closed form. Order is the tie-break order: `max_seq_len`, then `n_layers`, then `dropout`, each ascending.

| axis | values |
| --- | --- |
| `max_seq_len` | 50, 200 |
| `n_layers` | 1, 2 |
| `dropout` | 0.2, 0.5 |

50 matches the two-tower history cap. 200 is the length the SASRec paper used for this dataset. One and two layers, and dropout 0.2 and 0.5, are the paper's small search. Fixed beside the grid: `d_model=64`, `n_heads=1`, `learning_rate=0.001`, `batch_size=128`, `weight_decay=0`, `max_epochs=30`, `patience=3`, full softmax.

**Budget.** Cap 10800 s (3 h), from the start of the probe through the shuffled refit and the two comparison refits. Worst case is every trial running `max_epochs` with no early stop.

The probe (`results/budget/sasrec-probe.json`) times two fit-train epochs, seed 42, and discards the weights. Each measurement is one training epoch plus one forward encode of the validation users. It does not compute NDCG, Recall, or any other ranking metric.

| measurement | config |
| --- | --- |
| `t200` | `max_seq_len` 200, `n_layers` 2, `dropout` 0.5 |
| `t50` | `max_seq_len` 50, `n_layers` 2, `dropout` 0.5 |

Projected seconds for a config are `max_epochs * t50 * (n_layers / 2)` when `max_seq_len` is 50, and `max_epochs * t200 * (n_layers / 2)` when it is 200. Dropout does not change the projection. If the 200-length probe raises a memory error, `t200` is treated as larger than the cap, which forces the first cut below, and no ranking metric is read off the failed run.

Tail, always reserved, using the slowest config still in the locked grid and the worst-case edge steps:

- Edge steps that the rule below can still add, each at `max_epochs`.
- Test refits: one per test seed, each at `max_epochs`.
- Shuffled refit: one, at `max_epochs`.
- Comparison refits: 1800 s for the committed two-tower (6 epochs, seed 42) and the committed LambdaRank primary seed.

The locked grid is the first cut whose projected total fits in 10800 s.

| order | cut | what changes |
| --- | --- | --- |
| 0 | none | 8 configs, `max_epochs` 30, test seeds 42, 43, 44 |
| 1 | drop `max_seq_len=200` | 4 configs, all length 50. 200 is not added back as an edge |
| 2 | `max_epochs=15` | patience stays 3 |
| 3 | one test seed | seeds `[42]`. The shuffled refit still runs |
| 4 | drop `n_layers=2` | 2 configs, length 50, one layer. A one-value layer axis has no edge step |

If cut 4 still does not fit, the search does not start. The ADR is amended and pushed before any validation NDCG is computed.

During the search, a trial that has not started is skipped when the elapsed search wall, plus that trial's projection, plus the tail still owed, would pass 10800 s. Finished trials are kept. `truncated` is true. Unscored edge steps are skipped. Selection uses finished trials only. A finished trial is not trained again if the process is resumed. The probe epoch is not reused as a trial.

**One edge step, validation only, after the base grid.** An edge is a selected value equal to the minimum or the maximum of the values scored on that axis. A one-value axis is not an edge. `d_model` and `n_heads` are fixed, so they are not edges.

| axis | low edge | high edge |
| --- | --- | --- |
| `max_seq_len` | not extended. The step toward a longer window is 200, already in the grid, unless cut 1 removed it | not extended. 200 is the paper length, and the next doubling is outside this budget. If cut 1 already dropped 200, it is not reintroduced |
| `n_layers` | not extended. Zero layers is not this model | one step to 3, other hyperparameters held |
| `dropout` | not extended. The gap between 0.2 and 0.5 is 0.3, and 0.2 − 0.3 is negative | one step to 0.8 (0.5 + 0.3) |

At most two extra configs, appended after the base grid, so a tie keeps the earlier one. A step replaces the winner only when its validation NDCG@10 is strictly greater at 6 decimals. No second step. Test is scored once, after this re-selection.

**Test.** Seeds `[42, 43, 44]` unless cut 3 locked `[42]`. Each seed refits on full train for `best_epoch` epochs. The reported test NDCG@10 point is the mean across seeds. The 95% interval is the primary seed (42), `bootstrap_mean_ci`, 1000 resamples, alpha 0.05, seed 42, on the harness user order. The other harness metrics (precision, recall, coverage, mean popularity, at 10 and 20) are stored for that primary seed. Coverage stays a point estimate (ADR-0003).

After the winner is chosen, and before the test refit, the early-stopped fit-train checkpoint is scored once on validation with the same bootstrap. That interval is reported. It does not change the winner, the epoch, or the edge step.

**Shuffled-order check.** One more SASRec, identical hyperparameters, including `best_epoch`, `d_model`, `n_heads`, dropout, `max_seq_len`, and the optimizer. Weight seed 42. No validation selection and no early stop: it trains for `best_epoch` epochs on full train.

The only change is order. For each user, build the same window the real model uses (most recent `max_seq_len` interactions, timestamp order), then permute that window. The set of items in the window does not change. Ratings stay on their items, so the loss mask is unchanged. One `numpy.random.default_rng(42)` stream, re-seeded at the start of each sequence build, users in ascending user id, `permutation` of that user's window. The stream is not the torch seed. Fit-train is not used for this refit. Inference uses the permuted full-train window. Targets for NDCG stay the real test items.

The pre-registered comparison is paired bootstrap NDCG@10, real minus shuffled, on test, primary seed against this run, `paired_bootstrap_delta_ci`, 1000 resamples, alpha 0.05, seed 42. Users are the harness `relevant` order from `apply_cold_start_policy` on full train. `order_helps` is true only when the stored interval's low end is above 0. If the interval includes 0, or lies entirely at or below 0, order does not help on this split. That is a negative result (`negative_result` true). The interval is stored at 6 decimals and the flag is computed from the stored ends.

**Comparisons.** Both use the existing harness (`evaluate_recommender` and `paired_bootstrap_delta_ci`) on the same full-train / test split and the same users. `results/ml-1m.json` is read and not written.

- Two-tower: the committed headline, not the ADR-0013 losses. `hyperparameters.two_tower` from `results/ml-1m.json` (dim 64, learning rate 0.003, temperature 0.1, in-batch sampled softmax), refit on full train for 6 epochs, seed 42, through `fit_two_tower_recommender`.
- LambdaRank: the committed headline, not the demographics default now in `configs/ml-1m.yaml`. Demographics `off`. Candidate set `two_tower`, K=200, the committed LightGBM hyperparameters, seed 42. Candidate-set selection is not redone. The ranker is trained by the existing ranker code: early stop on the validation-user slice, then refit for that iteration count. A tree count other than the committed 95 is recorded and is still the model in the paired test. The host block is stored so a mismatch can be read against ADR-0015.

Paired differences, primary seed, SASRec minus that refit: `sasrec_minus_two_tower` and `sasrec_minus_lambdarank`. The published headline means (0.119246 and 0.127264) and the published primary-seed intervals are copied into the results file by the script. They are not a paired comparison. The paired interval is against the refit, which is the run that has per-user scores.

**Files.**

- `results/budget/sasrec-probe.json` — wall times, `ranking_metrics_recorded` false.
- `results/tuning/sasrec_ml-1m.json` — grid, cut, trials, edge steps, winner.
- `results/sasrec/ml-1m.json` — host block from ADR-0015, library versions, seeds, validation and test metrics, shuffled result, both paired comparisons, published headlines, `order_helps`, `negative_result`, grid edges.

The README panel is rendered by `scripts/make_results_table.py` from `results/sasrec/ml-1m.json` only. It is a short reading of two or three sentences plus the table. Every number in the panel is read from that JSON. The panel is empty when the file is absent, so the rest of the README stays byte-identical. Running the script twice leaves the README byte-identical.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| BCE with one sampled negative, the paper objective | Extra sampling hyperparameter, and a poor match to a 3,667-item catalog softmax the harness already sorts. ADR-0013 is the local evidence |
| Search `d_model` and `n_heads` | Doubles or quadruples a neural grid. 64 and 1 head are fixed to the two-tower width and the paper's ml-1m head count. The edge rule does not pretend a fixed value was searched |
| Inner feed-forward width 4× `d_model` | A generic transformer block. The SASRec pointwise network keeps the inner width at `d_model` |
| Mean-pool the sequence, or add a user id | That is the two-tower history path. This model has to be able to use order |
| Permute the full history, then truncate | That changes which items the model sees. The control keeps the window and permutes it |
| Early-stop the shuffled model on validation | A second selection. The check holds hyperparameters, including `best_epoch`, fixed |
| Pair against the published aggregate without refitting | `results/ml-1m.json` has no per-user scores. A paired interval needs lists from this run |
| Demographics `both` for the LambdaRank comparison | That is the later default (ADR-0009), not the headline row |
| Tune on test, or pick the best test seed | Closed by ADR-0005 |
| Extend `max_seq_len` past 200, or reintroduce 200 after dropping it for compute | The budget rule is the cap. A later step would be a new decision |

## Outcome

Filled after `results/sasrec/ml-1m.json` exists. The decisions above stay as written.
