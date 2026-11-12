# ADR-0030: Higher-Order vs. Pairwise Sequential Recommendation (`eSASRec` vs. `PCTM`) on ml-1m

## Status

Accepted. The architectures, grids, validation protocol, and paired bootstrap decision rules below are locked before any validation or test ranking metric is computed. `results/ml-1m.json` and `results/sasrec/ml-1m.json` remain read-only.

## Context

`ADR-0019` trained causal `SASRec` on `ml-1m` with full-catalog softmax cross-entropy and showed that while keeping real chronological order beats a shuffled-window copy (`0.0924` vs `0.0892`, paired `+0.0037 [+0.0024, +0.0052]`), `SASRec` had severe popularity collapse (`coverage@10 = 0.0327`, near `most_popular`'s `0.0325`) and trailed `two_tower` (`0.1191`) and `lambdarank` (`0.1293`).

Two recent sequential recommendation papers directly address both sides of this phenomenon:

1. **Modular Enhanced Transformer (`eSASRec`, Tikhonovich et al., RecSys 2025 — *Enhancing Transformer-based Recommendations in a Modular Fashion*):** Replaces vanilla SASRec's ReLU residual blocks with **LiGR** sigmoid-gated residual blocks (`h + Dropout(F(LN(h))) * σ(W_g LN(h))`, Borisyuk et al., 2025) and **SwiGLU** feed-forward networks (`W_2 (SiLU(W_1 x) * W_v x)`), improving feature selectivity and Pareto accuracy–coverage across benchmarks.
2. **Probabilistic Collaborative Transition Model (`PCTM`, Petrov et al., RecSys 2026 — *Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?*):** A non-neural, zero-embedding capacity probe that estimates Dirichlet-smoothed distance-decayed forward transition probabilities `P̂(a | b)` within window `W` and combines a user's `L_max` recent history items as log-linear experts weighted by a recency kernel (`E(r)` or `H(k, ρ, r)`), plus a popularity debiasing term `λ_pop log p_pop(a)`. Petrov et al. show that if a Transformer fails to beat `PCTM`, the sequential signal in that dataset is pairwise-decomposable rather than requiring higher-order sequence representations.

## Decision

1. **`eSASRec` (`LiGRBlock` in `src/movielens_recommender/sasrec/model.py`):**
   - Add `block_type="ligr"` (`SASRecModel(..., block_type="ligr", ff_mult=2)`).
   - Each `LiGRBlock` applies pre-LayerNorm, multi-head causal attention gated elementwise by `σ(Linear(normed))`, followed by pre-LayerNorm `SwiGLU` feed-forward (`inner_dim = ff_mult * d_model`) gated elementwise by `σ(Linear(normed))`.
   - Same allowed sequence slices (`build_sequence_data`), full-catalog softmax cross-entropy over positive next-item targets (`rating >= 4.0`), `d_model=64`, `n_heads=2`, `batch_size=128`, `learning_rate=0.001`, `max_epochs=30`, `patience=3`.
   - Validation grid (8 configs, matching ADR-0019's 8-config budget): `max_seq_len ∈ {50, 200} × n_layers ∈ {2, 3} × dropout ∈ {0.2, 0.5}`.

2. **`PCTM` (`PCTMRecommender` in `src/movielens_recommender/sasrec/pctm.py`):**
   - Built strictly on the allowed slice (`split.train` during validation, `split.full_train` for the single test evaluation), sorted stably by `(user_id, timestamp)`.
   - Forward distance-weighted causal counts `C_{b,a} = ∑_{d=1}^W γ_d 1[x_{t}=b, x_{t+d}=a, r_{t+d} ≥ 4.0]` with `γ_d = 1 / d` (harmonic) or `0.7^{d-1}` (exponential). Notice only positive targets (`r ≥ 4.0`) accumulate target transition mass, matching `SASRec`/`eSASRec`'s positive target mask while keeping all history items as valid context sources `b`.
   - Log-expert scoring over the user's latest `L = min(m, L_max)` interactions (`j = 1` newest):
     `s(a | h) = ∑_{j=1}^L w_j [log(C_{b_j, a} + τ / |I|) - log(C_{b_j, ·} + τ)] + λ_pop log p_pop(a)`
     Using the exact sparse identity `log(C_{b,a} + τ/|I|) - log(τ/|I|) = log(1 + |I| C_{b,a} / τ)`, candidate-independent terms `(−log(C_{b,·} + τ) + log(τ/|I|))` cancel across `a ∈ I`, so scoring is exact over all catalog items via a single sparse matrix-vector product per user!
   - Validation grid (36 configs, closed-form sparse counts):
     `max_distance W ∈ {10, 20} × decay ∈ {"harmonic", "exp_0.7"} × smoothing_tau ∈ {50.0, 150.0, 500.0} × pop_beta λ_pop ∈ {0.0, -0.25, -0.5}`, with Petrov et al.'s MovieLens head-tail history kernel `H(k=7, ρ=0.8, r=0.6)` and `L_max = 200`.

3. **Protocol & Pre-Registered Comparisons (`scripts/run_sequence_modern.py`):**
   - Tune both `eSASRec` (seed 42) and `PCTM` on `split.train → split.val` (`ndcg_point_estimate@10`).
   - Refit `eSASRec` across 3 test seeds (`42, 43, 44`) for `best_epoch` epochs on `split.full_train`, and fit `PCTM` once on `split.full_train`.
   - Re-evaluate the committed `SASRec` primary-seed architecture (`max_seq_len=50, n_layers=3, dropout=0.5, n_heads=1, epochs=30, seed=42`) on the same split so exact aligned per-user paired bootstrap intervals (`1000` resamples, `alpha=0.05`, `seed=42`) are computed for:
     - `esasrec_minus_sasrec`: Does modular `LiGR + SwiGLU` improve over vanilla `SASRec`?
     - `pctm_minus_sasrec`: Does pairwise transition log-pooling beat vanilla `SASRec`?
     - `esasrec_minus_pctm`: Does higher-order Transformer sequence modeling exceed the `PCTM` pairwise envelope on `ml-1m`?
   - Also compute ADR-0017 beyond-accuracy metrics (`coverage@10`, `novelty@10`, `intra_list_diversity@10`, `long_tail_share@10`) for all three sequence models.
   - Output: `results/sequence-modern/ml-1m.json` and `results/tuning/sequence_modern_ml-1m.json`.

## Outcome

Sealed test (`results/sequence-modern/ml-1m.json`, 5,958 warm-relevant users, 1,000 user-bootstrap resamples, primary seed 42). `results/sasrec/ml-1m.json` and `results/ml-1m.json` are unchanged. The SASRec reference refit on this host scores test NDCG@10 `0.092941`, matching the committed primary-seed point.

Validation selected PCTM `decay=harmonic`, `max_distance=20`, `smoothing_tau=50`, `pop_beta=-0.5` (validation NDCG@10 `0.186311`, 36 configs) and eSASRec `max_seq_len=50`, `n_layers=2`, `dropout=0.2`, best epoch `13` (validation NDCG@10 `0.050854`, 8 configs).

| model | val NDCG@10 | test NDCG@10 | 95% CI | Coverage@10 | Long-tail share@10 |
| --- | --- | --- | --- | --- | --- |
| PCTM | 0.186311 | **0.178666** | `[0.173683, 0.183685]` | **0.711208** | **0.175915** |
| SASRec reference | 0.050455 | 0.092941 | `[0.088943, 0.096719]` | 0.032724 | 0.0 |
| eSASRec | 0.050854 | 0.083568 mean (primary 0.079545) | `[0.076070, 0.082741]` | 0.034088 | 0.0 |

Pre-registered paired primary-seed NDCG@10 differences:

1. **PCTM beats SASRec (`pctm_beats_sasrec: true`).** `+0.085725` (`[+0.079596, +0.091731]`, excludes 0).
2. **eSASRec loses to SASRec (`esasrec_beats_sasrec: false`).** `-0.013396` (`[-0.015777, -0.011014]`, excludes 0). Seeds `42 / 43 / 44` score `0.079545 / 0.084798 / 0.086360`.
3. **Higher-order sequence modeling stays below the pairwise probe (`higher_order_beats_pairwise_pctm: false`).** eSASRec minus PCTM is `-0.099121` (`[-0.104959, -0.093624]`, excludes 0).

PCTM also covers `71.1%` of the catalog at 10, against `3.3%` for the SASRec reference. That coverage comes with lower intra-list genre diversity (`0.511069` vs `0.700464`). The published two-tower point (`0.119246`) and LambdaRank points (`0.127264` without demographics, `0.133410` with demographics) sit below the PCTM point. Those comparisons are unpaired across files and are not a gate.
