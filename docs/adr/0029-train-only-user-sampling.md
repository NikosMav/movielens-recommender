# ADR-0029: Train-Only Evaluation User Sampling (`user_sample_mode`)

## Status

Accepted. Corrects the evaluation-user sampling pipeline for new scale runs (`configs/ml-32m.yaml`) while preserving exact historical reproduction of `results/ml-32m.json`. Does not overwrite `results/ml-32m.json`.

## Context

`ADR-0011` and `ADR-0021` noted a subtle limitation in `apply_eval_user_sample` (`src/movielens_recommender/scale.py`): when `eval.user_sample_size > 0` (used on `ml-32M` to sample 8,000 users out of ~200k), eligible users were drawn via `warm_relevant_user_ids(full_train, split.test, relevance_threshold)`. Because that sample is shared across validation tuning, LambdaRank label construction, and final test evaluation, conditioning the sample pool on having at least one warm positive in `split.test` made the validation user pool subtly test-informed.

At the same time, `movielens-recommender reproduce --from-results results/ml-32m.json` (`ADR-0021`) must continue to reconstruct the exact historical `warm_test` user sample (`user_ids_sha256 = 6b0d5cf6ce14cf5815f37f7e6f09ac13ae5f3d138563b0385...`) recorded in `results/ml-32m.json`.

## Decision

1. Add `user_sample_mode` (`"train_only"` or `"warm_test"`) to `EvalYAML` (`src/movielens_recommender/config.py`), defaulting to `"train_only"` for all new runs and in `configs/ml-32m.yaml`.
2. In `apply_eval_user_sample(..., mode="train_only")`:
   - **`"train_only"` (new default):** Draw the seeded sample strictly from `np.unique(split.full_train["user_id"])` without inspecting `split.test` ratings, items, or relevance. Validation and test evaluation harnesses still apply their normal warm-relevant eligibility filter independently on their respective slices, so users in the sample with no warm test positives are naturally excluded from test averages (and reported in `cold_start.n_users_excluded_no_warm_relevant`) without any test-to-validation selection leakage.
   - **`"warm_test"` (historical mode):** Retains the original `warm_relevant_user_ids(full, split.test, relevance_threshold)` pool.
3. In `prepare_reproduction` (`src/movielens_recommender/experiments.py`), if a historical result record has `eval.user_sample_size > 0` and its embedded `config.eval` omits `user_sample_mode` (as in `results/ml-32m.json`), explicitly freeze `user_sample_mode: "warm_test"` in the materialized reproduction config so historical reproductions keep the exact recorded user sample.

## Consequences

- New `ml-32M` runs (`movielens-recommender run --config configs/ml-32m.yaml`) have zero test-informed user-sample selection bias.
- Historical reproductions (`movielens-recommender reproduce --from-results results/ml-32m.json`) preserve the exact historical protocol.
- Unit tests verify that perturbing `split.test` ratings or items cannot change the sampled user IDs or `user_ids_sha256` under `mode="train_only"`.
