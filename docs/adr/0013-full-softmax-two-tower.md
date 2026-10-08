# ADR-0013: Full-softmax two-tower

## Status

Accepted (S3e). The ml-1m protocol and compute budget are fixed from `results/budget/two-tower-v2.json` before any test metric. That file was built from a timing probe (`results/budget/two-tower-v2-probe.json`) that records no ranking metric (`ranking_metrics_used` is false). The ml-32m grid in that file was abandoned for compute. The replacement is the amendment below and `results/budget/two-tower-v2-ml32m-reduced.json`, fixed before any ml-32m validation or test metric. A later amendment, matched epoch budget and the ml-1m edge check, was added after those test files existed and before that round was scored. The outcome section is filled only after the corresponding `results/two-tower-v2/` file exists.

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

## Amendment: reduced ml-32m budget

The first ml-32m grid in `results/budget/two-tower-v2.json` (7 trials, projected cap 72736.9 s, limit 75600 s) was abandoned for compute. It was started and stopped during trial 1, full softmax, dim 64, learning rate 0.0003, temperature 0.1, `max_epochs` 4. No epoch finished. `results/tuning/two_tower_v2_ml-32m.json` was not written. No ml-32m validation score and no ml-32m test metric exist. This amendment is the pre-registration for the replacement.

The replacement file is `results/budget/two-tower-v2-ml32m-reduced.json`, from `reduced_ml32m_budget` on the same probe times. Sampled softmax is 1182.6 s/epoch, full softmax is 6852.6 s/epoch, and the validation score is 7.7 s/epoch. The cap is 14400 s (4 h). It covers tuning plus test training. It does not cover a second reference refit. `ranking_metrics_used` is false.

| loss | points | max_epochs | patience |
| --- | --- | --- | --- |
| sampled_softmax | (64, 0.003, 0.2), (64, 0.001, 0.1), (64, 0.0003, 0.1); 256 negatives | 3 | 1 |

Projected tuning is 10712.7 s. Projected test training is 3547.8 s. Total 14260.5 s, inside 14400 s. One full-softmax epoch, including the validation score, is 6860.3 s. Adding it makes 21120.8 s, over the cap, so full softmax is skipped for compute. It is not scored.

The three points are the ml-1m validation winner, the published ml-32m point with the new loss, and learning rate 0.0003 (below the ADR-0011 edge). In this slice, embedding dim 64 is the only value. Learning rate 0.0003 is the low edge and 0.003 is the high edge of `{0.0003, 0.001, 0.003}`. Temperature 0.1 is the low edge and 0.2 is the high edge of `{0.1, 0.2}`. A selected edge is recorded after validation and is not extended. If a trial runs slower than the probe, later trials are shortened or skipped so the reserved test refit stays inside the cap.

The test uses one seed, `[42]`. The reference two-tower is the published `results/ml-32m.json` result. It is not refit. The eval sample stays the ADR-0011 sample, `user_ids_sha256` `6b0d5cf6ce14cf5815e71f620b1dd5f37f7e6f09ac13ae5f3d138563b035c980`. The published file has no per-user scores, so a paired interval against that two-tower, and against the published LambdaRank, will not be computed. The result will carry the published point and CI and the unpaired difference.

LambdaRank on ml-32m runs only if the selected config beats the published validation NDCG@10 (0.100781 in `results/tuning/two_tower_ml-32m.json`) and the time left under 14400 s covers another sampled-softmax refit of `best_epoch` epochs plus the ADR-0011 item–item extrapolation of 432.5 s (`results/budget/ml-32m.json`). Otherwise the ranker is not run. Demographics stay `off`. Candidates would be that tower's top 200. A paired interval versus `no_ranker` is available only if the ranker runs. The reference tower is still not refit.

The app stays on the current model. The app rule needs a paired interval against a reference-tower ranker trained in this run, and this amendment does not train that ranker.

## Amendment: matched epoch budget and ml-1m edge check

This amendment was written after `results/two-tower-v2/ml-1m.json` and `results/two-tower-v2/ml-32m.json` existed, and before either check below was scored. The first ml-32m sampled-softmax trials all stopped at `max_epochs` 3 while still improving. The published reference had `max_epochs` 6, patience 2, and stopped at epoch 5. That comparison is not matched. The ml-1m sampled-softmax winner sat on three high edges, and its `best_epoch` equalled `max_epochs` 20.

**ml-32m.** One retrain of the first-round validation winner: sampled softmax, dim 64, learning rate 0.001, temperature 0.1, 256 negatives. Epoch budget matches the published reference: `max_epochs` 6, patience 2, seed 42. Same split and the same 8,000-user sample. The file is `results/budget/two-tower-v2-ml32m-matched.json`. Cap 10800 s (3 h), from the start of this round.

Rates used only to decide whether a later stage fits, taken from the finished 3-epoch run and the probe: sampled validation 700.326 s/epoch (2100.978 s over 3 epochs), sampled test training 800.073 s/epoch (2400.219 s over 3 epochs), in-batch probe 1312.0 s/epoch, item–item extrapolation 432.5 s. A 6-epoch validation is 4202.0 s. A 6-epoch test refit is 4800.4 s. Together those fit (`validation_plus_test_within_cap` is true). A 6-epoch ranker projection is 4634.5 s and the reference refit is 5 × 1312.0 = 6560.0 s. All four stages at 6 epochs do not fit.

Order, fixed here:

1. Validation always runs.
2. Full-train test scoring runs only if validation NDCG@10 is strictly above 0.100781 and the time left covers `best_epoch` × 800.073 s.
3. LambdaRank runs only if test was scored and the time left covers `best_epoch` × 700.326 s plus 432.5 s. Demographics `off`, K=200, seeds from `configs/ml-32m.yaml`.
4. The in-batch reference is refit for seed 42 and the published 5 epochs only if the time left covers 6560.0 s. That refit is what makes a paired interval possible. Otherwise the comparison stays unpaired.

If validation does not beat 0.100781 and `best_epoch` is below 6, the result is: sampled softmax does not beat the in-batch reference on ml-32M at matched epoch budget. If `best_epoch` is still 6, the run is truncated and is not called a negative result. If validation wins but the test is unscored, or the test is scored without a paired interval, the comparison is inconclusive. A paired interval that excludes 0 decides a loss or a win on test. This round does not switch the Streamlit app. The original three switch conditions still apply.

**ml-1m.** Validation only, sampled softmax only, after the test file existed. The grid does not use the test metrics to choose its axes. It extends the high edges: dim `{64, 128}`, learning rate `{0.003, 0.01}`, temperature `{0.2, 0.5}`, `max_epochs` 40, patience 3 (the ml-1m factorial patience). Eight trials. The file is `results/budget/two-tower-v2-ml1m-edges.json`. The prior winner is dim 64, learning rate 0.003, temperature 0.2, validation NDCG@10 0.08556. It is one of the eight points and is trained again at the longer budget. The winner changes only when another point is strictly higher. A tie keeps the prior point. Test (seeds 42, 43, 44) and LambdaRank are re-scored only when the winner changes. Otherwise the edges were checked and the winner held.

## Outcome

ml-1m was scored before the ml-32m amendment. Its grid was not changed after `results/two-tower-v2/ml-1m.json` existed. The ml-1m numbers are copied from `results/tuning/two_tower_v2_ml-1m.json` and `results/two-tower-v2/ml-1m.json`. The ml-32m numbers are copied from `results/tuning/two_tower_v2_ml-32m.json` and `results/two-tower-v2/ml-32m.json`, both written after the reduced-budget amendment. The matched-budget round is `results/two-tower-v2/ml-32m-matched.json`. The ml-1m edge check is `results/two-tower-v2/ml-1m-edges.json`. Both were scored after the amendment above.

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

**ml-32m validation.** Three sampled-softmax trials, `max_epochs` 3, patience 1, seed 42. `tune_wall_sec` 7176.076, under the 14400 s cap. Full softmax was not run. Reference validation NDCG@10 is 0.100781.

| learning rate | temperature | val NDCG@10 | best epoch | epochs trained | wall s |
| --- | --- | --- | --- | --- | --- |
| 0.003 | 0.2 | 0.078218 | 3 | 3 | 3062.695 |
| 0.001 | 0.1 | 0.096709 | 3 | 3 | 2100.978 |
| 0.0003 | 0.1 | 0.094712 | 3 | 3 | 2012.403 |

The winner is learning rate 0.001, temperature 0.1, dim 64, 256 negatives. It does not beat the reference (0.096709 < 0.100781). Every trial's `best_epoch` equals `max_epochs` 3, so each was still improving when the epoch cap stopped it. That cap was not extended. Embedding dim 64 is the only value in the slice. Temperature 0.1 is the low edge of `{0.1, 0.2}`. Learning rate 0.001 is interior: 0.0003 and 0.003 were both tried and lost.

**ml-32m test.** One seed, 42, as fixed in the amendment. The published two-tower was not refit. `test_train_sec` 2400.219. `test_wall_sec` 2457.849. NDCG@10 and the 95% interval are that one seed, 1000 resamples, alpha 0.05. The eval sample hash is `6b0d5cf6ce14cf5815e71f620b1dd5f37f7e6f09ac13ae5f3d138563b035c980`.

| model | NDCG@10 | 95% CI | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | head | tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| published in_batch | 0.14361 | [0.139788, 0.147949] | 0.113759 | 0.474196 | 0.624256 | 0.041099 | 0.145979 | 0.000577 |
| sampled_softmax | 0.137912 | [0.13429, 0.142417] | 0.110554 | 0.465346 | 0.614399 | 0.037344 | 0.138332 | 0.003451 |

Sampled-softmax head NDCG@10 is 0.138332, interval [0.134506, 0.142554]. Tail NDCG@10 is 0.003451, interval [0.00081, 0.006749], 579 users. Published item–item tail is 0.002161 and published item–item head is 0.111443.

A paired interval versus the published two-tower was not computed. The published file has no per-user scores, and this run did not refit that tower. The unpaired point difference, sampled softmax minus the published two-tower, is NDCG@10 -0.005698, head -0.007647, tail 0.002874. The tail point is above the published two-tower and above item–item. That is not a paired result.

**ml-32m ranker.** Not run. The new two-tower did not beat the reference on validation. A paired interval versus the published LambdaRank was not computed. That published ranker is NDCG@10 0.142276, interval [0.13908, 0.147192], demographics off, and its `no_ranker` point is 0.14361.

Full softmax did not help on ml-32m because it was skipped for compute. One full-softmax epoch is 6860.3 s, and the sampled-softmax plan already uses 14260.5 s of the 14400 s cap.

Tune plus test training is 7176.076 + 2400.219 = 9576.295 s, inside 14400 s. The ranker added none.

The app stays on the current model. The new tower did not beat the reference on validation, and the ranker was not run, so the other two switch conditions were not met.

**ml-32m matched epoch budget.** One retrain of the first-round sampled-softmax winner: dim 64, learning rate 0.001, temperature 0.1, 256 negatives, seed 42, `max_epochs` 6, patience 2. Same split and the same 8,000-user sample (`user_ids_sha256` `6b0d5cf6ce14cf5815e71f620b1dd5f37f7e6f09ac13ae5f3d138563b035c980`). Numbers from `results/two-tower-v2/ml-32m-matched.json` and `results/tuning/two_tower_v2_ml-32m-matched.json`.

Validation NDCG@10 by epoch: 0.086161, 0.090905, 0.096709, 0.098672, 0.097158, 0.097207. Epochs 1–3 match the first-round trial at the recorded values (epoch 3 is 0.096709). The best epoch is 4. Training continued through epoch 6 because patience is 2. `best_epoch` 4 is below 6, so the run is not truncated. Validation NDCG@10 is 0.098672, below the published reference 0.100781. `beats_reference_validation` is false. `validation_wall_sec` is 4177.048. Elapsed wall clock is 4284.071 s of the 10800 s cap, on 4 CPUs.

sampled softmax does not beat the in-batch reference on ml-32M at matched epoch budget.

That sentence is a validation comparison against the published 0.100781. Test was not re-scored. LambdaRank was not run. The in-batch reference was not refit, so the comparison stays unpaired. A paired interval was not computed. Verdict `loses_at_matched_budget`.

This point was not a new search. Dim, learning rate, and temperature were held at the first-round winner. The epoch cap that stopped the first round is not what stopped this run.

The app stays. This change does not switch the Streamlit model. The new tower did not beat the reference on validation, so the other two switch conditions were not met.

**ml-1m edge extension.** This grid was chosen after `results/two-tower-v2/ml-1m.json` existed. It does not use test metrics to pick its axes. Sampled softmax only: dim `{64, 128}`, learning rate `{0.003, 0.01}`, temperature `{0.2, 0.5}`, `max_epochs` 40, patience 3. Eight trials. `tune_wall_sec` 1883.764. The prior point was trained again. A later point replaces it only when its validation NDCG@10 is strictly higher.

| dim | learning rate | temperature | val NDCG@10 | best epoch | epochs trained | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | 0.003 | 0.2 | 0.08556 | 20 | 23 | 195.466 |
| 64 | 0.003 | 0.5 | 0.070415 | 15 | 18 | 263.814 |
| 64 | 0.01 | 0.2 | 0.084248 | 14 | 17 | 165.469 |
| 64 | 0.01 | 0.5 | 0.069401 | 15 | 18 | 349.405 |
| 128 | 0.003 | 0.2 | 0.082919 | 6 | 9 | 81.175 |
| 128 | 0.003 | 0.5 | 0.066883 | 6 | 9 | 81.498 |
| 128 | 0.01 | 0.2 | 0.08601 | 17 | 20 | 260.67 |
| 128 | 0.01 | 0.5 | 0.070934 | 17 | 20 | 486.267 |

The re-run of dim 64, learning rate 0.003, temperature 0.2 reproduced 0.08556 at epoch 20 and stopped at epoch 23. Dim 128, learning rate 0.01, temperature 0.2 scored 0.08601 at epoch 17. That is strictly higher, so the winner changed. `best_epoch` 17 is below `max_epochs` 40, and training stopped at epoch 20. The new winner is the high edge of dim `{64, 128}` and of learning rate `{0.003, 0.01}`, and the low edge of temperature `{0.2, 0.5}`. Temperature 0.5 lost at every dim and learning rate in this grid. Dim 128 and learning rate 0.01 were not extended further.

Test was re-scored because the winner changed. Seeds 42, 43, 44. NDCG@10 below is the 3-seed mean. The 95% interval is the primary seed (42), 1000 resamples, alpha 0.05. Train wall by seed: 176.991 s, 163.55 s, 195.41 s.

| model | NDCG@10 mean | primary 95% CI | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | head | tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sampled_softmax, dim 128, lr 0.01, temperature 0.2 | 0.130866 | [0.125522, 0.134583] | 0.088634 | 0.430728 | 0.602566 | 0.300245 | 0.143847 | 0.08168 |

Head NDCG@10 interval is [0.139211, 0.148855], 5796 users. Tail NDCG@10 interval is [0.077197, 0.086531], 4850 users. The primary-seed point inside the NDCG@10 interval is 0.130181.

Paired, primary seed, this tower minus the reference refit (the saved in-batch checkpoint, published hyperparameters, 6 epochs, seed 42):

| comparison | mean | low | high | excludes 0 | users |
| --- | --- | --- | --- | --- | --- |
| NDCG@10 | 0.011103 | 0.008232 | 0.014117 | yes | 5958 |
| head | 0.007478 | 0.004668 | 0.01052 | yes | 5796 |
| tail | -0.000858 | -0.004654 | 0.002778 | no | 4850 |

The re-scored tower beats the reference refit on NDCG@10. The tail interval includes 0.

**ml-1m edge ranker.** Demographics `both`, K=200. `runtime_sec` 343.103. NDCG@10 is the 3-seed mean. The interval is the primary seed.

| ranker | NDCG@10 mean | primary 95% CI | primary `no_ranker` |
| --- | --- | --- | --- |
| reference tower, demographics both | 0.13341 | [0.128167, 0.136643] | 0.119078 |
| new sampled-softmax tower | 0.127688 | [0.122043, 0.130358] | 0.130181 |

Paired, primary seed, new ranker minus the reference-tower ranker: mean -0.006272, low -0.009728, high -0.002685, excludes 0, 5958 users. New ranker minus `no_ranker`: mean -0.004074, low -0.007344, high -0.000816, excludes 0, 5958 users. The ranker does not gain. It is below its candidate list, and it is below the reference-tower ranker.

The first ml-1m test still stands: both new losses beat the reference two-tower, and the ranker does not gain. The app stays. The re-scored tower beats the reference on validation (0.08601 > 0.082601) and on the paired test interval above, but the ranker mean 0.127688 is below the reference-tower ranker 0.13341, and the paired interval's low end is -0.009728. This change does not switch the Streamlit model.
