# ADR-0031: `EASE-DAN` as a 3rd Retriever in `LambdaRank` (`ml-1m`)

## Status

Accepted. Candidate-set definitions, feature layout, validation selection rule, and paired bootstrap gates below are locked before any validation or test ranking metric is computed. Existing committed results (`results/ml-1m.json`, `results/demographics/ml-1m.json`) remain read-only.

## Context

In `S4` (`ADR-0007`) and `S4b` (`ADR-0009`), `LambdaRank` re-ranks candidates retrieved from two sources: `item_item_cosine_tuned` and `two_tower`, using 51 base features (`demographics="off"`) or 61 features (`demographics="both"`). Meanwhile, the cold-start ranker (`ADR-0012`) already incorporates `ease_score` and `ease_rank` via `cold_start_feature_names(mode)`, and `ADR-0028` showed that closed-form `EASE-DAN` (`l2=5000, dropout_p=0.6, alpha=0.0, beta=0.0`) beats `EASE^R` on `ml-1m` (`0.1195` vs `0.1184`) in a 2.4-second solve.

Because `EASE-DAN` solves global conditional precision weights `(XᵀX + Λ)⁻¹` across all items simultaneously — unlike pairwise `item_item_cosine` (`XᵀX` marginal cosine) and neural `two_tower` (low-rank dot-product embeddings) — adding `EASE-DAN` as a 3rd retriever provides complementary conditional collaborative signal to the gradient-boosted decision trees.

## Decision

1. **Tri-Retriever Candidate Sets (`src/movielens_recommender/ranker/candidates.py`):**
   - Add `balanced_three_way_union(a, b, c, k)` (takes top `k // 3` round-robin from `item_item`, `two_tower`, and `ease_dan`, then alternates backfill to budget `k = 200`) and `unbalanced_three_way_union(a, b, c, k)` (round-robin union of all three top-`k` lists).
   - On `split.val` (using retrievers fit strictly on `split.train`), compare validation `Recall@200` (tie-break `Recall@100`) between `two_tower` (`K=200`), `union_balanced` (2-way), `tri_union_balanced` (`K=200`), and `tri_union_unbalanced`. Select the candidate set with the highest validation `Recall@200` among the `K=200` budget sets (`two_tower`, `union_balanced`, `tri_union_balanced`), and also record `tri_union_unbalanced` validation recall for diagnostic comparison.

2. **Features (`ease_dan` mapped into `cold_start_feature_names(mode)` + `in_all_three`):**
   - Extend `src/movielens_recommender/ranker/features.py` with `tri_ranker_feature_names(mode)` which adds `ease_dan_score`, `ease_dan_rank`, and `in_all_three` (`1.0` when all three retrievers retrieved the item in their top-`K`, else `0.0`), while keeping every existing `feature_names(mode)` column untouched for backwards compatibility.
   - Add `topk_with_scores(n, user_ids=None)` to `EASEDANRecommender` (`src/movielens_recommender/baselines/ease_dan.py`) for fast batched scoring.

3. **Evaluation Protocol (`scripts/run_ranker_ease_dan.py`):**
   - Re-use the frozen retriever hyperparameters from `results/ml-1m.json` (`item_item_cosine_tuned`, `two_tower`) and `results/ease-dan/ml-1m.json` (`ease_dan`: `l2=5000.0, dropout_p=0.6, alpha=0.0, beta=0.0`), selected strictly on validation in their respective stages.
   - Compare 2-retriever `LambdaRank` (`item_item + two_tower`) vs. 3-retriever `LambdaRank` (`item_item + two_tower + ease_dan`) under **both** `demographics="off"` (S4 protocol) and `demographics="both"` (S4b production protocol), across all 3 ranker seeds (`42, 43, 44`), using identical fit-train features, validation labels, held-out validation early stopping (`early_stop_fraction=0.2`), and full-train retriever/feature rebuild before test scoring.
   - Pre-registered paired bootstrap checks (primary seed `42`, `1000` resamples, `alpha=0.05`):
     - `tri_off_minus_bi_off`: Does adding `EASE-DAN` lift `S4` (`demographics="off"`) `LambdaRank` test `NDCG@10` (`low > 0.0`)?
     - `tri_both_minus_bi_both`: Does adding `EASE-DAN` lift `S4b` (`demographics="both"`, the strongest model in the repository at `0.1334`) even further (`low > 0.0`)?
   - Output: `results/ranker-ease-dan/ml-1m.json`.

## Outcome

Sealed run in `437.432 s` (`results/ranker-ease-dan/ml-1m.json`, 5,958 test users, 1,000 bootstrap resamples, seeds `42, 43, 44`). `results/ml-1m.json` and `results/demographics/ml-1m.json` are unchanged.

Validation Recall@200 on fit-train retrievers, K=200 unless noted:

| candidate set | Recall@100 | Recall@200 | mean size |
| --- | --- | --- | --- |
| `union_balanced` | 0.485470 | 0.653053 | 200 |
| `two_tower` | 0.489363 | 0.655227 | 200 |
| `ease_dan` | 0.486654 | 0.660237 | 200 |
| `tri_union_balanced` | **0.494255** | **0.660693** | 200 |
| `tri_union_unbalanced` | 0.493231 | 0.659804 | 301.19 |

The pre-registered K=200 rule selects `tri_union_balanced`.

| variant | candidate set | 3-seed NDCG@10 | primary 95% CI | Coverage@10 |
| --- | --- | --- | --- | --- |
| 2 retrievers, demographics off | `two_tower` | 0.128809 | `[0.124798, 0.133291]` | 0.389692 |
| 3 retrievers, demographics off | `tri_union_balanced` | 0.127324 | `[0.1251, 0.133568]` | 0.368966 |
| 2 retrievers, demographics on | `two_tower` | **0.132741** | `[0.12979, 0.138106]` | **0.401327** |
| 3 retrievers, demographics on | `tri_union_balanced` | 0.129124 | `[0.12653, 0.134762]` | 0.381511 |
| 3 retrievers, unbalanced, demographics on | `tri_union_unbalanced` | 0.132502 | `[0.128819, 0.137232]` | 0.391601 |

Pre-registered gates, all negative:

1. **`tri_off_lifts_bi_off: false`.** The three-seed mean is lower (`0.127324` vs `0.128809`). The primary-seed paired difference is `+0.000265` (`[-0.002584, +0.003475]`, includes 0).
2. **`tri_both_lifts_bi_both: false`.** The three-seed mean is lower (`0.129124` vs `0.132741`). The primary-seed paired difference is **`-0.003217` (`[-0.006088, -0.000329]`, excludes 0)**.
3. **`tri_unbalanced_both_lifts_bi_both: false`.** Growing the pool to a mean of 301 candidates recovers most of the demographics-on score (`0.132502` vs `0.132741`). The paired difference is `-0.000954` (`[-0.003837, +0.002041]`, includes 0).

`ease_dan_score` and `ease_dan_rank` are the second and third gain features in the demographics-on three-retriever model, behind `two_tower_rank`. The trees use the new signal. Spending part of a fixed 200-candidate budget on EASE-DAN still removes two-tower candidates the ranker was using, and test NDCG@10 falls. The within-run two-retriever demographics-on mean (`0.132741`) sits near the published S4b point (`0.133410`); that cross-file gap is not a paired test.
