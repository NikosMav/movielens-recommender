# ADR-0009: User demographic features on the ml-1m ranker

## Status

Accepted (S4b). The pre-registered adoption rule passed. `configs/ml-1m.yaml` sets `models.ranker.demographics` to `both`. That yaml change was applied after the test run, as the rule, and the experiment was not re-run. The dataclass default and `configs/default.yaml` stay `off`. `results/ml-1m.json` is unchanged: its LambdaRank row is still the S4 feature set. S4b numbers live in `results/demographics/ml-1m.json`.

## Context

Published work on MovieLens usually finds that gender, age, and occupation add little once a user has interaction history, and that they can help when that history is short. This repo had not used `users.dat` at all. ml-1m ships gender, age bucket, occupation, and zip code. ml-latest-small does not, so this experiment is ml-1m only and does not change any ml-latest-small result.

The question is whether those features improve the S4 LightGBM LambdaRank ranker on the existing candidate sets, overall and for users with little history. Sensitive attributes are used here for that research comparison. Raw zip code is not a feature.

## Decision

**Reuse S4.** Same per-user temporal split, validation labels, candidate budget K=200, candidate-set rule (validation recall@200, then recall@100, then `union_balanced`, `item_item`, `two_tower`, `union_unbalanced`), retrievers, and ranker seeds 42, 43, 44. Seeds share the retrievers and the candidate sets. Only the LightGBM seed changes. The winner on this run is `two_tower`, the same winner as S4. The four validation recall@100 and recall@200 values equal the four values under `ranker.candidates` in `results/ml-1m.json`.

**Raw features** (categorical, non-negative codes for LightGBM):

| Feature | Codes |
| --- | --- |
| `demo_gender` | M=0, F=1, unknown=2 |
| `demo_age` | GroupLens age code (1, 18, 25, 35, 45, 50, 56); unknown=0 |
| `demo_occupation` | 0–20; unknown=21 |
| `demo_region` | first ZIP digit 0–9; otherwise 10 |

**Group-affinity features**, computed from fit-train interactions only, including when other features are rebuilt on full train at test scoring. For group \(g\) (age bucket, gender, or occupation), item \(i\), prior strength \(m = 20\), global positive rate \(p\), and item global share \(s_i\):

- positive rate \((p_{ig} + m p) / (n_{ig} + m)\). An item with \(n_{ig} = 0\) takes the global positive rate.
- popularity share \((n_{ig} + m s_i) / (n_g + m)\).

Global rates use the whole allowed frame. Group counts use only users present in `users.dat`. The six columns are `group_{age,gender,occupation}_{pos_rate,pop_share}`.

**Variants.** `off` is the S4 set (51 columns). `raw` adds 4 (55). `affinity` adds 6 (57). `both` adds 10 (61). Categorical columns are passed to LightGBM only when that list is non-empty, so the baseline call shape stays the S4 one. Training labels stay the validation window. Test rows never enter features or labels.

**Adoption rule, fixed before the test run.** Keep demographics as the ml-1m ranker default only when the +both 3-seed mean test NDCG@10 is strictly greater than the baseline mean and the primary-seed paired bootstrap CI of (both − baseline) has low > 0. Otherwise leave the flag at `off`.

**Low history.** (a) The existing user-activity terciles on the primary seed. (b) A simulated cold start: each evaluated user's query profile is their earliest N full-train ratings (timestamp, then item id), for N = 5 and N = 10. Test targets stay the same. Candidate generation and user-history ranker features use only that prefix. The seen filter stays the full train history. `most_popular` is fit on full train. Demographic-group most-popular is popularity inside the user's age × gender group, fit on fit-train only, with fallback to that frame's global ranking. Item–item uses the tuned similarity fit on full train and the truncated query. The two rankers are the primary-seed refit boosters from this experiment.

**Cold-start caveats.** `most_popular` and `group_most_popular` ignore the truncated profile, so their rows repeat at N=5 and N=10. The two-tower user-id embedding is still the one learned at fit time; only the history pool and the seen mask are truncated. Item–item similarities are fit on full train, so a user's later train ratings sit in the gram matrix; the query vector does not.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Raw ZIP as a feature | Fine-grained and unnecessary; the first digit is the region code |
| Group statistics from full train | Would leak the validation window into features used while training on validation labels |
| A new candidate generator | The question is the feature set, on the S4 candidates |
| Enabling demographics on ml-latest-small | That dataset has no `users.dat`; the loader raises |
| Re-running the headline `results/ml-1m.json` pipeline after the yaml flip | The adoption comparison is this experiment. A later full `run` would rewrite the S4 headline row |

## Outcome

Four-decimal tables below are the same rounding as `scripts/make_results_table.py`. Stored fields that the 4-decimal table rounds are quoted from `results/demographics/ml-1m.json` in the notes. Population std is the stored `std` (ddof=0). NDCG@10 mean and std are over seeds 42, 43, 44. The NDCG@10 CI, tail NDCG@10, activity slices, fairness slices, and the paired difference are the primary seed (42), 1000 bootstrap resamples, alpha 0.05.

### Reproduction of the S4 baseline

| Source | NDCG@10 mean | std | candidate winner |
| --- | --- | --- | --- |
| `results/ml-1m.json` LambdaRank | 0.127264 | 0.001287 | `two_tower` |
| this run, `baseline` | 0.126654 | 0.000791 | `two_tower` |

`reproduction.abs_diff_mean` is 0.00061. `matches_s4_at_4_decimals` is false (0.1267 vs 0.1273). `candidate_set_matches_s4` is true, and the validation recall figures match as stated above. Primary-seed `best_iteration` is 75 here and 95 on the S4 `lambdarank_two_tower` row (S4 seeds 95 / 94 / 43; this baseline 75 / 26 / 32). The adoption comparison is the within-run baseline, not the older headline mean.

### Ranker variants (test)

| variant | ndcg@10 mean | std | ndcg@10 CI | recall@10 | coverage@10 | tail ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.1267 | 0.0008 | [0.1234, 0.1318] | 0.0918 | 0.3842 | 0.0905 [0.0858, 0.0959] |
| raw | 0.1249 | 0.0027 | [0.1217, 0.1298] | 0.0906 | 0.3811 | 0.0908 [0.0861, 0.0959] |
| affinity | 0.1317 | 0.0008 | [0.1287, 0.1371] | 0.0969 | 0.4009 | 0.0917 [0.0873, 0.0970] |
| both | 0.1334 | 0.0009 | [0.1282, 0.1366] | 0.0983 | 0.4012 | 0.0922 [0.0876, 0.0973] |

Paired bootstrap of NDCG@10 (`both_minus_baseline`, seed 42, n=5958): stored mean 0.004658, CI [0.001464, 0.007981], `excludes_zero` true.

**Raw demographics alone are a negative result.** The raw mean (stored 0.12489) is below the baseline mean. Affinity alone is above baseline (stored 0.131667). +both is the highest mean (stored 0.13341). There is no paired CI of +both minus affinity, so this ADR does not treat the extra raw columns, on top of affinity, as a separate significant gain.

The rule passed: stored +both mean 0.13341 is greater than stored baseline mean 0.126654, and the paired CI low is above 0. `decision.keep_as_default` is true. `decision.negative_result` is false.

### Gain importance (+both primary-seed refit booster)

Rank is among all 61 features. The highest gain in that booster is `two_tower_rank` at 12552.400821. Demographic rows:

| feature | gain | rank |
| --- | --- | --- |
| demo_occupation | 3922.3723 | 2 |
| group_gender_pos_rate | 2584.2826 | 7 |
| group_age_pop_share | 2435.5404 | 8 |
| group_gender_pop_share | 1673.7744 | 10 |
| group_occupation_pop_share | 1341.6961 | 12 |
| group_occupation_pos_rate | 1333.0279 | 13 |
| group_age_pos_rate | 1324.1932 | 14 |
| demo_region | 760.1644 | 27 |
| demo_age | 275.3523 | 37 |
| demo_gender | 69.9087 | 41 |

`demo_occupation` is rank 2. That does not overturn the raw-only ablation: gain rank is not a lift over the baseline.

### Activity terciles (primary seed, n=1986 each)

| variant | activity low | activity mid | activity high |
| --- | --- | --- | --- |
| baseline | 0.1162 [0.1083, 0.1233] | 0.0937 [0.0881, 0.0991] | 0.1732 [0.1647, 0.1813] |
| raw | 0.1096 [0.1019, 0.1173] | 0.0945 [0.0889, 0.0999] | 0.1730 [0.1652, 0.1813] |
| affinity | 0.1181 [0.1106, 0.1253] | 0.1032 [0.0975, 0.1091] | 0.1772 [0.1690, 0.1852] |
| both | 0.1173 [0.1096, 0.1247] | 0.1025 [0.0969, 0.1083] | 0.1773 [0.1687, 0.1850] |

The low-activity intervals for baseline and +both overlap. The mid-activity point estimate moves further than the low-activity one, and those intervals overlap too. The gain on the full test set is not concentrated in the short-history tercile. Raw is below baseline on the low tercile.

### Simulated cold start

`n_eval_users` is 5958 at both N. `n_users_truncated` is 5958. `n_users_shorter_than_n` is 0: every evaluated user had more than 10 train ratings. No paired CI was computed for this table.

| N | model | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 5 | most_popular | 0.0895 | [0.0857, 0.0935] | 0.0466 | 0.0325 |
| 5 | group_most_popular | 0.0960 | [0.0923, 0.1001] | 0.0510 | 0.0614 |
| 5 | item_item | 0.0815 | [0.0775, 0.0850] | 0.0421 | 0.2345 |
| 5 | ranker_baseline | 0.0982 | [0.0944, 0.1017] | 0.0696 | 0.4920 |
| 5 | ranker_both | 0.1024 | [0.0986, 0.1062] | 0.0715 | 0.4688 |
| 10 | most_popular | 0.0895 | [0.0857, 0.0935] | 0.0466 | 0.0325 |
| 10 | group_most_popular | 0.0960 | [0.0923, 0.1001] | 0.0510 | 0.0614 |
| 10 | item_item | 0.0961 | [0.0921, 0.1002] | 0.0560 | 0.1800 |
| 10 | ranker_baseline | 0.1039 | [0.1002, 0.1077] | 0.0748 | 0.4328 |
| 10 | ranker_both | 0.1077 | [0.1039, 0.1117] | 0.0770 | 0.4265 |

`group_most_popular` (fit-train age × gender counts) has a higher point estimate than `most_popular` (full train). The ranker with demographics has the highest point estimate at both N. Those ranker intervals still overlap the ranker-without-demographics intervals. Item–item at N=5 is below global most-popular; at N=10 it is close to group most-popular. Read this table with the cold-start caveats above.

### Fairness (primary seed)

`n_eval_users_missing_demographics` is 0. Delta is +both minus baseline.

| group | n | baseline ndcg@10 | both ndcg@10 | delta | delta CI | excludes 0 |
| --- | --- | --- | --- | --- | --- | --- |
| gender F | 1689 | 0.1167 | 0.1164 | -0.0003 | [-0.0061, 0.0055] | false |
| gender M | 4269 | 0.1321 | 0.1387 | 0.0066 | [0.0032, 0.0102] | true |
| age 1 (Under 18) | 220 | 0.1278 | 0.1302 | 0.0024 | [-0.0150, 0.0214] | false |
| age 18 (18-24) | 1085 | 0.1385 | 0.1426 | 0.0041 | [-0.0028, 0.0115] | false |
| age 25 (25-34) | 2070 | 0.1348 | 0.1376 | 0.0028 | [-0.0020, 0.0081] | false |
| age 35 (35-44) | 1180 | 0.1259 | 0.1351 | 0.0092 | [0.0017, 0.0171] | true |
| age 45 (45-49) | 543 | 0.1150 | 0.1142 | -0.0009 | [-0.0114, 0.0090] | false |
| age 50 (50-55) | 486 | 0.1091 | 0.1204 | 0.0113 | [0.0006, 0.0214] | true |
| age 56 (56+) | 374 | 0.1060 | 0.1088 | 0.0028 | [-0.0132, 0.0183] | false |

The gains are uneven. The gender interval that excludes 0 is men; women's point estimate is slightly down and the interval includes 0. Age buckets whose intervals exclude 0 are 35–44 and 50–55, both positive. The 45–49 point estimate is slightly negative and the interval includes 0. The other age intervals include 0.

## What is kept

Demographics stay in the ml-1m ranker (`both`). They stay off for ml-latest-small. S5 (the explainable UI) is not built. When it is, it should not present gender, age, occupation, or ZIP region as user-facing reasons. Runtime of this experiment is the stored `runtime_sec` 400.071.
