# ADR-0013: Full-softmax two-tower

## Status

Accepted (S3e). The protocol and the compute budget in this ADR are fixed from `results/budget/two-tower-v2.json` before any test metric. That file was built from a timing probe (`results/budget/two-tower-v2-probe.json`) that records no ranking metric (`ranking_metrics_used` is false). The outcome section is filled only after `results/two-tower-v2/` exists.

## Context

The production two-tower loss is the in-batch sampled softmax with a log-q correction (ADR-0006). On ml-1m its test NDCG@10 ties item–item. On ml-32M (ADR-0011) it ties LambdaRank and both beat item–item, and the ranker adds no lift over the two-tower candidate list. The same run left the two-tower learning rate at 0.001, the low edge of `{0.001, 0.003}`. Tail NDCG@10 for that two-tower is below item–item.

This stage asks whether a different training objective improves retrieval, and whether LambdaRank on the new candidates then gains. The default config stays the in-batch loss, so a re-run of the existing pipeline is unchanged. The Streamlit app is not switched in this change.

## Decision

**Losses.** `models.two_tower.loss` is one of `in_batch`, `full_softmax`, `sampled_softmax`. Omitted means `in_batch`.

- `in_batch` is the ADR-0006 loss: scores inside the batch, subtract log q, cross-entropy on the diagonal. q is proportional to positive counts.
- `full_softmax` is cross-entropy over every catalog item. There is no log-q term. The denominator is the catalog.
- `sampled_softmax` draws 256 negatives without replacement from q, shared by the batch. The corrected logit is score minus log q. A negative that equals that row's positive is masked. 256 is fixed, not tuned.

Temperature stays a tuned float. It is not a learned parameter. Batch size 1024, weight decay 0.0001, and max history 50 stay fixed.

**Data and split.** Same per-user time split as ADR-0005. ml-32M uses the ADR-0011 sample: 8000 users, seed 42, `user_ids_sha256` `6b0d5cf6ce14cf5815e71f620b1dd5f37f7e6f09ac13ae5f3d138563b035c980` (`results/budget/two-tower-v2.json`). Training still uses every training interaction. The reference hyperparameters and `best_epoch` are read from `results/tuning/two_tower_ml-1m.json` (epoch 6) and `results/tuning/two_tower_ml-32m.json` (epoch 5). They are not searched again.

**Probe** (`results/budget/two-tower-v2-probe.json`). Five warmup batches, then 20 timed batches, seed 42, dim 64, temperature 0.1, learning rate 0.001, batch 1024. No NDCG. Host: 4 CPUs, `MemTotal` 16014.0 MiB, `MemAvailable` 7147.7 MiB, `CommitLimit` 8007.0 MiB, `SwapTotal` 0.0.

| dataset | items | positives | batches/epoch | in_batch s/batch | full_softmax s/batch | sampled_softmax s/batch | val score s/epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ml-1m | 3655 | 430387 | 420 | 0.016 | 0.0216 | 0.0179 | 2.4 |
| ml-32m | 66819 | 11852789 | 11574 | 0.1134 | 0.5921 | 0.1022 | 7.7 |

Extrapolated epoch train seconds: ml-1m 6.7 / 9.1 / 7.5, ml-32m 1312.0 / 6852.6 / 1182.6 (in-batch / full softmax / sampled softmax). A full-softmax logit batch is 14.3 MiB on ml-1m and 261.0 MiB on ml-32m. Feature build was 0.374 s on ml-1m and 22.658 s on ml-32m. `recommend` on 100 users was 0.039 s and 0.097 s. Peak RSS in the ml-32m probe block was 3257.3 MiB.

ADR-0011 measured a 5-epoch in-batch full-train refit at `train_wall_time_sec` 2637.302 (about 527 s/epoch) and extrapolated 1371.2 s/epoch from a 20-batch probe. This probe's in-batch extrapolation is 1312.0 s/epoch on fit-train. The budget uses this probe, not the earlier 527 s figure.

**Tuning budget.** Worst case is every trial running `max_epochs` with no early stop, plus the extrapolated validation score. ml-1m cap 21600 s (6 h). ml-32m cap 75600 s (21 h). The 21 h cap is the smallest hour-multiple that holds a grid in which full softmax still trains for 4 epochs. An 8 h cap cannot do that: one full-softmax epoch is 6852.6 s.

ml-1m keeps the full factorial: both new losses, dims `{32, 64}`, learning rates `{0.0003, 0.001, 0.003}`, temperatures `{0.05, 0.1, 0.2}`, `max_epochs` 20, patience 3. 36 trials, cap 7704.0 s, inside the limit.

ml-32m rejected, in order, a 36-trial factorial (869464.8 s), a 12-trial one-factor grid (289821.6 s), full softmax cut to 5 points (248659.8 s), three full-softmax learning rates with sampled softmax dropping temperature 0.2 (159194.4 s), and a balanced grid at 4 epochs for both losses (78688.4 s). The locked grid is the next one, cap 72736.9 s, inside 75600 s:

| loss | points | max_epochs | patience |
| --- | --- | --- | --- |
| full_softmax | dim 64, temperature 0.1, learning rate 0.0003 and 0.001 | 4 | 2 |
| sampled_softmax | (64, 0.0003, 0.1), (64, 0.001, 0.1), (64, 0.003, 0.1), (64, 0.001, 0.05), (32, 0.001, 0.1) | 3 | 1 |

On ml-32m, full softmax does not search temperature or embedding dim. Both of its learning rates are edges of `{0.0003, 0.001}`. Sampled softmax does not include temperature 0.2. A selected value on the edge of its own slice is recorded after validation and is not extended inside this budget. Early stopping can finish under the cap. The cap is the worst case.

**Test budget, fixed before any test metric.** A loss is scored on test only after validation selection.

- ml-1m: seeds `[42, 43, 44]`. `three_seed_cap_sec` 666.6 against a 14400 s limit. Both new losses are tested (`test_both_cap_sec` 1116.6 against 21600 s), plus the reference in-batch refit for the published 6 epochs.
- ml-32m: seed `[42]` only. `three_seed_cap_sec` 143026.8 exceeds 14400 s. Only the validation winner is tested, plus the reference refit for the published 5 epochs. `test_both_cap_sec` 54771.2 exceeds 21600 s. The test-cap arithmetic uses 6 epochs, not the shrunk epoch counts, so it is conservative.

Headline NDCG@10 is the seed mean. The bootstrap CI, head NDCG@10, and tail NDCG@10 are the primary seed (42), same as the main pipeline. The paired interval is candidate minus the reference rerun, 1000 resamples, alpha 0.05, seed 42. Reported point metrics: NDCG@10, Recall@10, Recall@100, Recall@200, Coverage@10, head NDCG@10, tail NDCG@10.

**Ranker.** LambdaRank runs only if the selected new loss beats the published reference validation NDCG@10 (strict). Candidates are that tower's top 200. K=200. ml-1m demographics `both`, as in `configs/ml-1m.yaml`. ml-32m demographics `off`. The comparison ranker is retrained in this run on the reference tower with the same demographics. It is not an unpaired look at the published aggregate. The result says whether the new ranker adds lift over `no_ranker`.

**App.** This change does not switch the Streamlit model. Switch only if all three hold: the best new tower beats the reference on validation; the ranker on its candidates has a test NDCG@10 mean strictly above the reference-tower ranker from this run; and the paired interval of (new ranker minus that reference ranker) has a low end above 0. Otherwise the app stays on the current model.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Learned temperature | The existing temperature grid would no longer be the control |
| Tune `n_negatives` | 256 is the scalable default; another axis does not fit the ml-32m cap |
| ml-32m full factorial, or 4 epochs of full softmax on every sampled point | Caps 869464.8 s and 78688.4 s, both over 75600 s |
| Three ml-32m test seeds, or testing both new losses there | Caps 143026.8 s and 54771.2 s, over 14400 s and 21600 s |
| Change `configs/ml-1m.yaml` or the Streamlit snapshot in this change | The default loss must keep reproducing the in-batch model; the app rule above is not yet evaluable |

## Outcome

ml-32M has no test file yet. The numbers below are ml-1m only, copied from `results/tuning/two_tower_v2_ml-1m.json` and `results/two-tower-v2/ml-1m.json`. The locked grid was not changed after those files existed.

**ml-1m validation.** 36 trials, `tune_wall_sec` 7315.623, under the 7704.0 s cap. Reference validation NDCG@10 is 0.082601.

| loss | val NDCG@10 | best epoch | chosen point | edges |
| --- | --- | --- | --- | --- |
| full_softmax | 0.08548 | 9 | dim 64, lr 0.003, temperature 0.2 | high edge on dim, learning rate, and temperature |
| sampled_softmax | 0.08556 | 20 | dim 64, lr 0.003, temperature 0.2, 256 negatives | same three high edges, and `best_epoch` equals `max_epochs` 20 |

The validation winner is sampled softmax. It beats the reference (0.08556 > 0.082601). Learning rate 0.0003 was in the grid and did not win. The chosen learning rate is 0.003, which was already the high edge of the ADR-0006 grid. Temperature 0.2 and embedding dim 64 are high edges of this grid. They were not extended. The sampled-softmax run was still improving at epoch 20.

**ml-1m test.** Seeds 42, 43, 44. NDCG@10 below is the 3-seed mean. The 95% interval is the primary seed (42), 1000 resamples, alpha 0.05. `test_train_sec` 706.29. `test_wall_sec` 1194.074.

| model | NDCG@10 mean | primary 95% CI | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | head | tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in_batch | 0.119246 | [0.114975, 0.123048] | 0.090056 | 0.435363 | 0.603669 | 0.472321 | 0.136369 | 0.082537 |
| full_softmax | 0.128563 | [0.124716, 0.134048] | 0.091163 | 0.438279 | 0.610096 | 0.33097 | 0.142905 | 0.086697 |
| sampled_softmax | 0.13049 | [0.125398, 0.134676] | 0.089073 | 0.42799 | 0.600377 | 0.300245 | 0.143228 | 0.081597 |

The in-batch mean matches the published `results/ml-1m.json` two-tower mean 0.119246. Head and tail in the table are the primary seed, same as the main pipeline.

Paired difference, primary seed, candidate minus the reference rerun:

| comparison | mean | low | high | excludes 0 | users |
| --- | --- | --- | --- | --- | --- |
| full_softmax NDCG@10 | 0.01046 | 0.00775 | 0.013008 | yes | 5958 |
| full_softmax head | 0.006536 | 0.003868 | 0.00924 | yes | 5796 |
| full_softmax tail | 0.004159 | 0.000769 | 0.007368 | yes | 4850 |
| sampled_softmax NDCG@10 | 0.010969 | 0.008056 | 0.013778 | yes | 5958 |
| sampled_softmax head | 0.006859 | 0.004028 | 0.009806 | yes | 5796 |
| sampled_softmax tail | -0.00094 | -0.004762 | 0.002264 | no | 4850 |

Full softmax helped on ml-1m. Sampled softmax helped by more on NDCG@10. Published item–item tail is 0.032052, so the reference tail 0.082537 was already ahead of item–item. There was no ml-1m tail gap to fix. Full softmax's tail is higher than the reference and the paired interval excludes 0. Sampled softmax's tail is not. Coverage@10 falls from 0.472321 (in-batch) to 0.33097 (full softmax) and 0.300245 (sampled softmax).

**ml-1m ranker.** Demographics `both`, K=200, candidates forced to the tower's top 200. `ranker_wall_sec` 325.702. NDCG@10 is the 3-seed mean.

| ranker | NDCG@10 mean | primary 95% CI | primary `no_ranker` |
| --- | --- | --- | --- |
| reference tower, demographics both | 0.13341 | [0.128167, 0.136643] | 0.119078 |
| sampled-softmax tower | 0.1301 | [0.126979, 0.135586] | 0.130047 |

The reference-tower ranker mean matches `results/demographics/ml-1m.json` `variants.both.summary.ndcg@10_mean` 0.13341. Paired, primary seed, new ranker minus the reference-tower ranker: mean -0.001341, low -0.004345, high 0.001913, includes 0, 5958 users. New ranker minus `no_ranker`: mean 0.00099, low -0.002499, high 0.004415, includes 0. The ranker does not gain. It does not beat the reference-tower ranker, and it does not add a lift over its candidate list whose interval excludes 0.

The app stays on the current model. The new tower beat the reference on validation, and the other two conditions fail: the new ranker's test mean 0.1301 is below the reference-tower ranker 0.13341, and the paired interval's low end is -0.004345.
