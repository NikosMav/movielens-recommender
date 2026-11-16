# ADR-0033: Pairwise transitions as a retriever and ranker feature, and what the demo serves (`ml-1m`)

## Status

Accepted. The candidate sets, variants, comparisons, and the serving rule below are fixed before any validation or test number for them is computed. `results/ml-1m.json`, `results/demographics/ml-1m.json`, and `results/sequence-modern/ml-1m.json` stay read-only.

## Context

ADR-0030 scored a pairwise transition probe (`PCTM`) at test NDCG@10 `0.1787` on the ml-1m headline split. The served known-user model is LambdaRank over two-tower candidates with demographic features (ADR-0007, ADR-0009), at a three-seed mean of `0.1327` in `results/ranker-ease-dan/ml-1m.json`. Those two numbers come from different files and were never paired. The strongest recorded model is therefore not the one behind Explore a MovieLens user.

PCTM counts which film a viewer rated within `W` steps after another film, weights the viewer's recent history, and adds no learned parameters. It is refit from the ratings in a few seconds, so serving it needs no new parameter file.

Two questions are open. Does a ranker add anything on top of the PCTM list? And which list should the demo serve?

## Decision

**Retrievers.** Item–item and two-tower keep the hyperparameters in `results/ml-1m.json`. PCTM keeps the validation-selected hyperparameters in `results/sequence-modern/ml-1m.json` (`decay=harmonic`, `max_distance=20`, `smoothing_tau=50`, `pop_beta=-0.5`, `max_history=200`). Nothing is retuned. Each retriever is fit on `split.train` for validation lists and ranker labels, and on `split.full_train` for the one test scoring.

**Candidate sets, budget K=200.** `two_tower` (the served set), `pctm`, `pctm_two_tower` (balanced union of PCTM and two-tower), and `pctm_tri` (balanced three-way union of item–item, two-tower, and PCTM). Validation Recall@200 chooses one sequence-aware set from the last three. Ties break on Recall@100, then on the order `pctm`, `pctm_two_tower`, `pctm_tri`.

**Ranker features.** `pctm_ranker_feature_names` adds `pctm_score` and `pctm_rank` to the S4 columns. Both are NaN for a candidate outside PCTM's top 200. LightGBM settings, the early-stop user split, and ranker seeds `42, 43, 44` are the ADR-0007 ones.

**Variants.**

| name | candidates | features |
| --- | --- | --- |
| `bi_both` | `two_tower` | S4b, demographics on. The served configuration. |
| `pctm_only` | PCTM top list, in PCTM order | none |
| `seq_off` | the selected sequence-aware set | S4 plus PCTM score and rank |
| `seq_both` | the selected sequence-aware set | S4b plus PCTM score and rank |

**Paired comparisons.** Primary-seed per-user test NDCG@10, 1,000 user-bootstrap resamples, alpha `0.05`, seed `42`: `pctm_only − bi_both`, `seq_off − pctm_only`, `seq_both − pctm_only`, `seq_both − seq_off`, `seq_off − bi_both`, `seq_both − bi_both`. A comparison is a win when the interval's low end is above 0.

**Serving rule.** Applied in this order, from the stored flags:

1. If `seq_both − seq_off` is a win, the ranker candidate is `seq_both`. Otherwise it is `seq_off`, which has fewer features and no demographic columns.
2. If the ranker candidate beats `pctm_only` (paired win), serve the ranker candidate.
3. Otherwise, if `pctm_only − bi_both` is a win, serve the PCTM list as it is.
4. Otherwise keep `bi_both`.

Rule 2 changes the ranker file, so it needs a new parameter archive. Rule 3 needs none.

**Not in scope.** The new-user path (Discover) is unchanged. A new profile has no rating order, so a transition model has nothing to read there. ml-32M is not rerun.

Output: `results/ranker-pctm/ml-1m.json`, written by `scripts/run_ranker_pctm.py`.

## Outcome

Sealed run (`results/ranker-pctm/ml-1m.json`, 5,958 warm-relevant users, 1,000 user-bootstrap resamples, primary seed 42, 474 s on a 16-thread Windows laptop capped at 4 threads). The other results files are unchanged.

Validation Recall@200: `two_tower` `0.655227`, `pctm` `0.694187`, `pctm_two_tower` `0.725132`, `pctm_tri` `0.718931`. The selected sequence-aware set is `pctm_two_tower`.

| variant | candidates | features | test NDCG@10 | primary-seed 95% CI | Coverage@10 |
| --- | --- | --- | --- | --- | --- |
| `bi_both` | `two_tower` | 61 | 0.130861 | `[0.125127, 0.133625]` | 0.396964 |
| `pctm_only` | `pctm` | 0 | 0.178666 | `[0.173683, 0.183685]` | 0.711208 |
| `seq_off` | `pctm_two_tower` | 53 | **0.202149** | `[0.196142, 0.206814]` | 0.509954 |
| `seq_both` | `pctm_two_tower` | 63 | 0.201948 | `[0.196606, 0.206673]` | 0.497410 |

Paired primary-seed NDCG@10 differences:

1. **PCTM alone beats the served ranker.** `pctm_only − bi_both` is `+0.049236` (`[+0.043479, +0.054955]`).
2. **A ranker adds to PCTM.** `seq_off − pctm_only` is `+0.022756` (`[+0.020327, +0.025301]`), and `seq_both − pctm_only` is `+0.022889` (`[+0.020519, +0.025462]`).
3. **Demographic features add nothing here.** `seq_both − seq_off` is `+0.000133` (`[-0.001251, +0.001816]`, includes 0).
4. **Against the served ranker.** `seq_off − bi_both` is `+0.071992` (`[+0.066750, +0.077553]`).

Serving rule 1 makes `seq_off` the ranker candidate, because demographics did not win. Rule 2 then serves it: `serve=seq_off`. `pctm_rank` is the largest gain in that ranker, followed by `two_tower_rank` and `pctm_score`.

`bi_both` scores `0.130861` on this host against `0.132735` in `results/ranker-ease-dan/ml-1m.json`, which was run on Linux. The two-tower refit differs across hosts (ADR-0015). Every comparison above is within this file.

Coverage falls from `0.711` for PCTM alone to `0.510` with the ranker. That is still above the earlier ranker's `0.397`.

### What changed in serving

- `build-artifacts` reads the recorded choice. For ml-1m it trains the ranker on `pctm_two_tower` candidates with the 53 `seq_off` columns and no demographic features. Datasets without a record build as before.
- The snapshot stores no PCTM file. Its manifest has a `pctm` block, and the model is counted again from the snapshot's training rows at load, in user chunks, which adds about 8 s to a load and 27 MB resident.
- A recommendation that the transition signal lifted names the rated film behind it. Explore a MovieLens user no longer uses age, gender, or occupation.
- The demo's parameter archive changes, because the ranker file changes. The new-user parameters in it are byte-identical to the previous archive.

A rebuild from the exported parameter files, done the way the demo starts, returned the same top 10 as the full snapshot for 8 of 8 checked viewers.

`results/serving/ml-1m.json` was measured on the earlier ranker and is not rewritten here. On this host a warm explained request for one viewer took about 36 ms with the new snapshot.

The demo installs parameters by checksum. An extracted tree now records the archive it came from, so a moved pin replaces it on the next start, and an archive host that is down leaves the previous tree serving.
