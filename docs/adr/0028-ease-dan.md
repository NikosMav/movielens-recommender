# ADR-0028: Data-Adaptive Normalized Linear Autoencoder (`EASE-DAN`) on ml-1m

## Status

Accepted. This is ADR-0028, after ADR-0027. The model formulation, search space, validation protocol, budget rule, edge-extension rule, and pre-registered gate below are fixed before any validation or test ranking metric is computed. A timing probe may shrink the grid before tuning; it records wall time only (`ranking_metrics_recorded` is false). The outcome section is filled only after `results/ease-dan/ml-1m.json` exists.

Note: this ADR, the implementation, and the results land on one feature branch; the protocol and grids below were specified before running validation tuning or test evaluation.

## Context

In the committed `ml-1m` benchmark (`results/ml-1m.json`), `EASE^R` (Steck, 2019; ADR-0008) reaches test NDCG@10 of 0.118380 (`[0.114173, 0.122587]`), trailing untuned `item_item_cosine` at 0.120135 (`[0.115829, 0.124155]`) and the full `lambdarank` pipeline at 0.127264 (primary seed 0.128786 `[0.124690, 0.132934]`). Furthermore, in the beyond-accuracy evaluation (`results/beyond-accuracy/ml-1m.json`, ADR-0017), `EASE^R` exhibits strong popularity bias: Coverage@10 is 0.2100 and Long-tail share@10 is 0.0059 (versus 0.1040 for `two_tower` and 0.0459 for `lambdarank`).

Recent collaborative filtering literature (Steck, NeurIPS 2020 *Autoencoders that don't overfit towards the Identity* — EDLAE; Moon et al., SIGIR 2023 *It's Enough: Relaxing Diagonal Constraints in Linear Autoencoders for Recommendation* — RLAE/RDLAE; Park et al., SIGIR 2025 *Why is Normalization Necessary for Linear Recommenders?* — DAN / `EASE-DAN`) identifies two structural biases in `EASE^R`:

1. **Popularity bias:** Uniform L2 regularization `λI` under-regularizes popular items relative to rare items, and raw Gram inner products `XᵀX` let high-degree target items dominate column weights.
2. **Neighborhood bias:** High-activity users contribute quadratically (`d_u²` item pairs) to `XᵀX`, causing local co-occurrence patterns of heavy users to crowd out global collaborative signals.

Park et al. (SIGIR 2025) combine dropout-equivalent item degree regularization (`p / (1 - p) · D_I`) with user degree weighting (`D_U^{-β}`) and asymmetric source/target item degree similarity transformation (`D_I^α B̂ D_I^{-α}`), yielding a closed-form linear autoencoder (`EASE-DAN`) that requires a single `n_items × n_items` matrix inversion, zero neural training epochs, and exact reduction to standard `EASE^R` when `α = 0`, `β = 0`, and `p = 0`.

This stage tests two questions on the exact `ml-1m` harness without modifying `results/ml-1m.json` or the served snapshots:

1. Does `EASE-DAN` improve test NDCG@10 over `EASE^R` on the same split, and can a single closed-form linear model close the gap to `item_item_cosine` and `lambdarank`?
2. How does `EASE-DAN` shift tail NDCG@10 and the ADR-0017 beyond-accuracy metrics (coverage, novelty, intra-list diversity, and long-tail share)?

## Decision

**Formulation.** Let `X ∈ {0, 1}^{m × n}` be the binary user–item interaction matrix on the allowed training slice (any observed train rating is 1, matching `EASERecommender` in ADR-0008). Let `d_U = X 1_n` (user degrees, clamped below at 1) and `d_I = Xᵀ 1_m` (item degrees, clamped below at 1). Given hyperparameters `l2 > 0`, `dropout_p ∈ [0, 1)`, `alpha ≥ 0`, and `beta ≥ 0`:

1. **User-adaptive Gram matrix:** `G = Xᵀ diag(d_U^{-β}) X`.
2. **Denoising + L2 diagonal regularization:** `Λ_ii = l2 + (dropout_p / (1 - dropout_p)) · (d_I)_i`. Invert `P = (G + diag(Λ))⁻¹`.
3. **Zero-diagonal constraint:** `B̃ = I - P · diag(1 / diag(P))`, with `diag(B̃) = 0`.
4. **Item-adaptive similarity transformation:** `B = diag(d_I^α) · B̃ · diag(d_I^{-α})`, with `diag(B) = 0`.

Inference scores for user row `x_u` are `x_u B`; seen training items are masked to `-∞` and ranked via `top_n_indices` (ADR-0015). When `alpha = 0`, `beta = 0`, and `dropout_p = 0`, `B` is algebraically and numerically identical to `EASERecommender(l2=l2)`.

**Data and split.** `ml-1m`, per-user chronological split (`min_ratings=5`, `test_fraction=0.2`, `val_fraction=0.1`, `relevance_threshold=4.0`, `seed=42`), identical to `results/ml-1m.json`. `results/ml-1m.json` is read-only.

**Base grid (108 configs).** Comparable in size to RP3beta (125) and the ADR-0018 ALS retune (144):

| axis | values | rationale |
| --- | --- | --- |
| `l2` | `10.0, 100.0, 500.0, 2000.0` | Spans low to high base ridge penalty; user weighting `d_U^{-β}` shrinks Gram entries by roughly 10–100× compared with unnormalized `XᵀX`, shifting optimal `l2` downward from unnormalized EASE's 5000 |
| `dropout_p` | `0.0, 0.3, 0.6` | `0.0` disables EDLAE degree-proportional regularization; `0.3` and `0.6` match the range reported in Steck (2020) and Park et al. (2025) |
| `alpha` | `0.0, 0.2, 0.4` | `0.0` disables item target popularity debiasing; `0.2` and `0.4` span the MovieLens range in Park et al. (2025) (`[0, 0.5]`) |
| `beta` | `0.0, 0.3, 0.6` | `0.0` disables user degree normalization; `0.3` and `0.6` suppress heavy-user neighborhood dominance without full random-walk over-penalization (`β=1`) |

Grid order (also the deterministic tie-break order): `l2`, `dropout_p`, `alpha`, `beta`, each ascending.

**Budget.** Cap `3600 s` (1 hour), with a `600 s` tail reserve for edge extensions, test refits, paired comparisons, and beyond-accuracy scoring (`SEARCH_WALL_SEC = 3000 s`).
A timing probe (`results/budget/ease-dan-probe.json`) fits and scores one validation-sized pass (without computing NDCG or any ranking metric; `ranking_metrics_recorded` is `false`). Projected search time is `n_configs * t_probe`. If `108 * t_probe + 600 > 3600`, drop `l2 = 2000.0` (81 configs); if still over budget, drop `dropout_p = 0.6` (54 configs).

**One edge step, validation only.** Ordered axes are `l2`, `dropout_p`, `alpha`, and `beta`. Zero (`0.0`) on `dropout_p`, `alpha`, and `beta` is a natural boundary (feature off) and is not extended downward. High edges step once:
- `l2` low edge (`10.0`) → `2.0`; high edge (`2000.0`) → `5000.0`.
- `dropout_p` high edge (`0.6`) → `0.8`.
- `alpha` high edge (`0.4`) → `0.6`.
- `beta` high edge (`0.6`) → `0.9`.

At most four extra configs are appended after the base grid and scored on validation `ndcg_point_estimate` only. A tie keeps the earlier config. No second step. Test is scored once after final validation selection.

**Same-code references and comparisons.** On `full_train` (`fit-train ∪ val`), fit and evaluate once on `split.test` (with segment breakdowns and ADR-0017 beyond-accuracy metrics):
1. Selected `ease_dan` (`tuned`).
2. Committed `ease` (`l2=5000.0`, from `results/ml-1m.json`).
3. Committed `item_item_cosine` (`k_neighbors=0, min_common=1, shrinkage=0.0`, from `results/ml-1m.json`).

Paired user-bootstrap differences (`paired_bootstrap_delta_ci`, 1000 resamples, `alpha=0.05`, `seed=42`) on per-user test NDCG@10:
- `ease_dan_minus_ease`
- `ease_dan_minus_item_item`

We also copy the published `two_tower` and `lambdarank` headline points/CIs from `results/ml-1m.json` for descriptive context (without modifying `results/ml-1m.json`).

**Pre-registered gate.** `beats_ease` is `true` only when `ease_dan_minus_ease` paired 95% bootstrap interval low end is strictly greater than `0.0` (`low > 0.0`). `beats_item_item` is `true` when `ease_dan_minus_item_item` paired 95% bootstrap interval low end is strictly greater than `0.0`. If `beats_ease` is `false`, the stage is recorded as a negative result (`negative_result: true`).

**Files.**
- `src/movielens_recommender/baselines/ease_dan.py` — `EASEDANRecommender` implementation (`fit`, `recommend`, `topk_for_profiles`, `hyperparams`) and grid/edge helpers.
- `scripts/run_ease_dan.py` — `probe`, `tune`, and `test` runner.
- `results/budget/ease-dan-probe.json` — timing probe (`ranking_metrics_recorded: false`).
- `results/tuning/ease_dan_ml-1m.json` — base grid, edge extensions, and validation winner.
- `results/ease-dan/ml-1m.json` — test metrics, segment head/tail NDCG@10, ADR-0017 beyond-accuracy metrics, and paired bootstrap intervals.
- `scripts/make_results_table.py` and `space/quality.py` / `space/pages.py` — render the committed `results/ease-dan/ml-1m.json` panel in both `README.md` and the Streamlit "How good is it?" view without recomputing metrics or altering existing headline JSONs.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Overwrite `results/ml-1m.json` or retrain `space-weights.tar.gz` | ADR-0021 and ADR-0024 freeze the historical S4 record and the pinned demo recommendation IDs; additive stages live in their own `results/<stage>/ml-1m.json` files (like ADR-0018 and ADR-0019) |
| HSTU (Zhai et al., ICML 2024) as the primary new benchmark | On ml-1m (`3,667` items, bag-of-ratings history), ADR-0019 already showed causal sequence order adds only `+0.0037` NDCG@10 (`0.0924`), whereas linear item-item models (`0.1184`–`0.1201`) are competitive with neural retrieval (`0.1192`) at 100× lower compute |
| Post-hoc popularity penalty on `EASE^R` scores | Park et al. (2025, Appendix A.3) prove post-hoc column scaling neither denoises source co-occurrences nor suppresses heavy-user neighborhood bias (`β`) |
| Tune on test or run multiple edge steps | Forbidden by ADR-0005 and ADR-0018 |

## Outcome

The probe (`results/budget/ease-dan-probe.json`, `ranking_metrics_recorded` false) measured `2.509 s` per trial (`fit_sec=2.328 s`, `score_sec=0.181 s` on 5,619 validation users) and projected `270.951 s` across all 108 base configs, well within the `3600 s` cap (`cut: null`, `truncated: false`). All 108 base configs and 2 one-step edge extensions were scored. `results/ml-1m.json` is byte-identical.

1. **Validation selection reveals pure EDLAE denoising beats degree re-scaling on per-user time splits.**
   - On the 108-config base grid, the winner was `l2=2000.0, dropout_p=0.6, alpha=0.0, beta=0.0` (`val NDCG@10 = 0.080355`), sitting on the high edges of `l2` (`2000.0`) and `dropout_p` (`0.6`).
   - Under the pre-registered one-step rule, stepping `l2` to `5000.0` (`dropout_p=0.6, alpha=0.0, beta=0.0`) improved validation NDCG@10 to **`0.081247`** (exceeding committed `EASE^R` validation `0.080830`), whereas stepping `dropout_p` to `0.8` (`l2=2000.0`) scored `0.080365`.
   - Non-zero `alpha` (target popularity suppression) and `beta` (active-user down-weighting) consistently reduced overall validation NDCG@10 on `ml-1m` (best `alpha=0.2, beta=0.3` cell reached `0.079338`). Because `ml-1m` has high item Gini skewness and chronological per-user test tails still favor popular head films, validation NDCG@10 selects `alpha=0.0, beta=0.0` and retains only the item-degree dropout regularizer `Λ_ii = 5000 + 1.5 · d_{I,i}` (i.e., Steck's 2020 **EDLAE** special case of `EASE-DAN`). The selected config sits on the high `l2` edge (`5000.0`), with no second step allowed by the protocol.

2. **Test evaluation (5,958 users, 1,000 user-bootstrap resamples, `seed=42`):**

| model | validation NDCG@10 | test NDCG@10 | 95% CI | Recall@10 | Coverage@10 | fit time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `ease` (`l2=5000`) | 0.0808 | 0.1184 (`0.118380`) | `[0.1142, 0.1226]` | 0.0861 | 0.2100 | 2.39 |
| `ease_dan` (`l2=5000, p=0.6, α=0, β=0`) | **0.0812** | **0.1195** (`0.119454`) | `[0.1151, 0.1238]` | **0.0871** | 0.2053 | 2.44 |
| `item_item_cosine` | — | 0.1201 (`0.120135`) | `[0.1158, 0.1242]` | 0.0786 | 0.1274 | 1.34 |

   - **Pre-registered gate against `EASE^R`: WIN (`beats_ease: true`, `negative_result: false`).** Paired `ease_dan_minus_ease` test NDCG@10 is **`+0.001073` (`[+0.000432, +0.001739]`, excludes 0)**.
   - **Comparison with `item_item_cosine` and neural/ranker models:** Paired `ease_dan_minus_item_item` is **`-0.000681` (`[-0.003535, +0.002177]`, includes 0; `beats_item_item: false`)**. `EASE-DAN` (`0.1195`) statistically closes the gap to `item_item_cosine` (`0.1201`) while retaining `61%` higher catalog coverage (`0.2053` vs `0.1274`) and higher `NDCG@20` (`0.1324` vs `0.1284`) and `Recall@100` (`0.4301` vs `0.3471`). It also slightly edges the published 3-seed `two_tower` point (`0.1192`) in a `2.44 s` closed-form solve, while trailing the full `lambdarank` pipeline (`0.1273` S4 / `0.1334` with demographics).
   - **Beyond-accuracy & head/tail trade-off:** Adding `1.5 · d_I` to `5000 I` strengthens shrinkage across popular co-occurrences, lifting head `NDCG@10` (`0.1324` vs `0.1314`) and all three user-activity terciles (`0.1028 / 0.0910 / 0.1645` vs `0.1017 / 0.0904 / 0.1630`), while slightly lowering tail `NDCG@10` (`0.0724` vs `0.0743`), Novelty@10 (`9.1199` vs `9.1409`), and Long-tail share@10 (`0.0056` vs `0.0059`). Improving tail metrics with `EASE-DAN` requires setting `alpha > 0` explicitly (trading off head accuracy, just as Park et al. report on high-Gini MovieLens splits) rather than selecting hyperparameters solely by overall `NDCG@10`.
