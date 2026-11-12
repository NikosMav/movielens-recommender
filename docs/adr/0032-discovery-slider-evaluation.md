# ADR-0032: Offline Multi-Objective Evaluation of the Discovery Slider (`ml-1m`)

## Status

Accepted. Evaluates the existing `Familiar ↔ Adventurous` maximal marginal relevance (MMR) + popularity-novelty discovery control (`space/value.py:rerank_candidates`) offline on the held-out cold-start benchmark (`ADR-0012`) without changing any model weights, candidate pool, or default position-0 recommendations.

## Context

`ADR-0025` and `ADR-0026` added a 5-position slider (`Familiar = 0.0`, `Closer = 0.25`, `In between = 0.5`, `Further = 0.75`, `Adventurous = 1.0`) to the Streamlit Discover view. Whenever a visitor moved the slider above `Familiar`, the UI displayed a disclaimer: *"This re-ranking has not been evaluated offline. Only the default position is the measured model."* `scripts/check_demo_controls.py` checked only four illustrative profiles without computing relevance (`NDCG@10`) or formal `ADR-0017` beyond-accuracy metrics (`Coverage@10`, `Novelty@10`, `Intra-list diversity@10`, `Long-tail share@10`).

Evaluating all five slider positions across the 604 held-out cold-start users (`N ∈ {3, 5, 10}`, where the Discover UI enables recommendations at `N ≥ 3`) quantifies the exact empirical Pareto frontier between ranking accuracy (`NDCG@10`) and discovery (`Coverage@10`, `Novelty@10`, `Intra-list diversity@10`, `Long-tail share@10`).

## Decision

1. **Exact Deployed Cold-Start Snapshot & Held-Out Split (`scripts/run_slider_eval.py`):**
   - Load the exact materialized cold-start serving bundle (`space/.runtime/cold_start`, reconstructed from the pinned `WEIGHTS_SHA256` archive and `ml-1m`) and evaluate the exact 604 held-out users from `results/cold-start/ml-1m.json` (`select_held_out_user_ids(eligible, fraction=0.1, seed=42)` — users completely absent from every retriever and ranker in that bundle).
   - For each profile size `N ∈ {3, 5, 10}` (matching the Discover UI's `≥ 3` rating requirement) and each slider position `w ∈ {0.0, 0.25, 0.5, 0.75, 1.0}`:
     - Run the exact `recommend_new_user` candidate scoring path once per user, then apply `rerank_candidates(ranked_ids, ranked_scores, adventurous=w, embeddings=..., popularity=..., n=10)`.
     - Verify at `w = 0.0` (`Familiar`) that `NDCG@10` matches `results/cold-start/ml-1m.json` (`0.4089` at `N=3`, `0.4087` at `N=5`, `0.4076` at `N=10`).
     - Score `NDCG@10`, `Recall@10`, `Coverage@10`, `Novelty@10`, `Intra-list diversity@10`, `Mean popularity@10`, and `Long-tail share@10` (with 1,000-resample user-bootstrap 95% CIs, `seed=42`), plus paired `NDCG@10` deltas relative to `Familiar` (`w = 0.0`).
   - Also compute the same 5-position trade-off curve on the sensitivity harness-tail target (`last 20%` of each held-out user's ratings).

2. **Truthful UI & Quality-View Update (`space/value.py`, `space/quality.py`, `space/pages.py`):**
   - Update `unevaluated_note` in `space/value.py` so that when `results/slider-eval/ml-1m.json` is present (or via concise factual wording), the Discover caption accurately states the measured trade-off (that moving toward *Adventurous* trades held-out `NDCG@10` for higher catalog coverage, novelty, and genre diversity, while *Familiar* maximizes held-out `NDCG@10`), keeping default `Familiar` (`w=0.0`) untouched.
   - Surface the measured 5-position Pareto table on the **How good is it?** page (`space/pages.py`) and in `README.md` (`scripts/make_results_table.py`).
   - Output: `results/slider-eval/ml-1m.json`.

## Outcome

Evaluated in `200.986 s` (`results/slider-eval/ml-1m.json`) against the deployed cold-start serving snapshot (`WEIGHTS_SHA256 = f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd`) on the 604 held-out cold-start users (`results/cold-start/ml-1m.json` untouched). The slight ~`0.001` difference at `Familiar` vs `results/cold-start/ml-1m.json` (`0.4095` vs `0.4087` at `N=5`) reflects the documented `ADR-0020` complement refit in `space-weights.tar.gz` and host BLAS (`ADR-0015`).

At `N=5` (604 held-out users, all later relevant ratings):

| Slider position (`w`) | NDCG@10 | 95% CI | Coverage@10 | Novelty@10 | Diversity@10 | Long-tail share@10 |
| --- | --- | --- | --- | --- | --- | --- |
| `Familiar` (`0.00`) | **0.4095** | `[0.3879, 0.4296]` | 0.0194 | 8.4396 | 0.6889 | 0.0000 |
| `Closer` (`0.25`) | 0.4033 | `[0.3814, 0.4236]` | 0.0271 | 8.5080 | 0.7298 | 0.0012 |
| `In between` (`0.50`) | 0.2710 | `[0.2542, 0.2869]` | 0.2114 | 9.8298 | 0.8220 | 0.1995 |
| `Further` (`0.75`) | 0.0905 | `[0.0812, 0.1002]` | 0.3678 | 11.0415 | **0.8721** | 0.3983 |
| `Adventurous` (`1.00`) | 0.0519 | `[0.0450, 0.0588]` | **0.3820** | **11.3095** | 0.8676 | **0.4611** |

Key takeaways:
1. **`Closer` (`w = 0.25`) is a nearly lossless diversity nudge:** At `N=5`, `NDCG@10` drops by only `-0.0062` (`0.4095 → 0.4033`; and on the `N=5` harness-tail sensitivity check it slightly rises from `0.0444 → 0.0458`), while catalog coverage rises `+40%` (`0.0194 → 0.0271`) and intra-list genre diversity rises from `0.6889 → 0.7298`.
2. **`In between` (`w = 0.50`) unlocks genuine long-tail discovery (`11×` coverage):** Catalog coverage jumps from `1.9%` to **`21.1%`** and long-tail share from `0.0%` to **`20.0%`**, while retaining two-thirds of offline `NDCG@10` (`0.2710` on all-later targets; `0.0330` vs `0.0444` on the harness-tail sensitivity check).
3. **`Further` / `Adventurous` (`w ≥ 0.75`) prioritize niche exploration (`38.2%` coverage, `46.1%` tail share) at steep offline `NDCG@10` cost (`0.0519`),** because retrospective MovieLens ratings are heavily concentrated on blockbuster head titles. Keeping `Familiar` (`w=0.0`) as the default while exposing the measured trade-off gives visitors an honest, empirically quantified control.
