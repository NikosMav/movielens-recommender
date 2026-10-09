# ADR-0015: Deterministic top-n selection

## Status

Accepted. A bug fix, not a modeling stage. It changes no hyperparameter, grid, or selection rule. It makes one rule that the code already documented actually hold.

## Context

Rebuilding the S5c new-user snapshot on a second machine (Windows 11, 16 CPUs, AVX2 without AVX-512) did not reproduce the committed `results/cold-start/ml-1m.json`, which came from a 4-CPU Linux cloud host. Popularity, item–item fold-in on the test table, and every two-tower number matched exactly. The short-profile LightGBM ranker did not: early stopping picked 179 trees instead of 88. Every served and ranker metric moved with it, and the N=1 verdict "beats most popular" flipped from false to true.

LightGBM already runs with `deterministic=True` and `force_row_wise=True`, so its inputs were the suspect.

**Root cause.** Every top-n in the package used the same pattern:

```python
part = np.argpartition(-scores, n - 1)[:n]
order = part[np.argsort(-scores[part], kind="mergesort")]
```

The docstrings say ties break by item index. The mergesort only orders the n entries `argpartition` already chose. Which tied entries make the cut is up to `argpartition`, and NumPy 2.x picks its partition code by CPU: separate AVX-512, AVX2, and baseline paths that order ties differently. On the second machine, the old helper returned the highest-index tied items at the cut (`3795, 3794` where the documented rule gives `100, 101`).

Ties at the cut are common on short profiles. One rating gives most catalog items a fold-in score of exactly zero. So the item–item and EASE candidate lists for simulated short profiles differed by CPU, the ranker trained on different rows, and early stopping landed somewhere else. With long histories ties are rare. Across all 6,040 ml-1m users in the full-train item–item model, the top-10, top-20, and top-200 cuts had no ties. The neighbour cut at k=200 had a positive tie in 7 of 3,667 item rows.

**Evidence.** Same machine, same code, NumPy's AVX2 path switched off with `NPY_DISABLE_CPU_FEATURES`:

| run | top-n code | dispatch | K chosen | trees | N=1 paired vs most-popular |
| --- | --- | --- | --- | --- | --- |
| committed | old | cloud host | 50 | 88 | 0.009141 [-0.000148, 0.018419] |
| second machine | old | AVX2 | 50 | 179 | 0.009825 [0.000409, 0.019409] |
| second machine | old | baseline | 100 | 135 | 0.008996 [-0.000609, 0.018809] |
| second machine | new | AVX2 | 50 | 142 | 0.008838 [-0.000726, 0.018216] |
| second machine | new | baseline | 50 | 142 | 0.008838 [-0.000726, 0.018216] |

With the old code, switching the CPU path changed 248 values in the results file, including which K the protocol selected. With the new code, the two paths give the same file, apart from wall-clock fields.

**A second, environmental cause.** The fixed code still gives a different ranker on Linux (the project's Docker image) than on Windows, on the same CPU. Everything outside the ranker matches exactly: popularity, item–item fold-in, EASE fold-in, and two-tower metrics. LightGBM is not the cause either: the same LambdaRank training on fixed synthetic data gives byte-identical trees on both. The ranker's inputs differ. EASE fold-in scores for the same profile come out in the same order on both platforms, but 399 of 400 differ in the last bits (largest difference 1.3e-15). Inside one Docker image, switching OpenBLAS between its Haswell and Sandy Bridge kernels (`OPENBLAS_CORETYPE`) does the same (5.3e-16). Those scores are ranker features. A last-bit change can move a value across a LightGBM histogram bin edge, and early stopping then lands on a different tree count. This comes from the BLAS and LAPACK builds, not from code in this repository. Bit-identical ranker results need the same OS build and the same CPU family.

## Decision

One helper, `baselines.common.top_n_indices(scores, n)`, returns the indices of the n largest finite scores, best first, with ties going to the lower index. It equals `np.argsort(-scores, kind="stable")[:n]` over the finite entries. It still partitions first, so the cost stays linear: it reads the n-th largest value, which is the same whichever tied entry `argpartition` puts there, keeps everything above it, and fills the rest with the lowest-index entries equal to it.

All twelve call sites use it: top-n in item–item, ALS, EASE, RP3beta, the two-tower recommender, and `recommend_from_scores`, plus the k-neighbour cuts in item–item and RP3beta. Neighbour lists are stored in column order.

Tests check the helper against a full stable sort on tie-heavy inputs, check the cut case above, and run the same probe in a subprocess with NumPy's SIMD paths disabled to confirm the output does not change.

**Recording the host.** Results files written by `run`, `demographics`, and `cold-start` now carry a `host` block: OS, release, machine, processor, CPU count, Python, and the NumPy SIMD features in use. The next mismatch can then be traced to its platform from the file alone.

**Results files.**

- `results/cold-start/ml-1m.json` is regenerated with the fixed code inside the project's Docker image, so anyone can repeat the exact run:

  ```bash
  docker build -t movielens-recommender .
  docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/out:/out" movielens-recommender movielens-recommender cold-start --config configs/ml-1m.yaml --results-dir /out/results --no-download
  ```

  (copy `results/` into `out/` first; the command reads `results/tuning/`). It is the file the bug moved, and its N=1 verdict sits on the line.
- `results/ml-1m.json`, `results/ml-latest-small.json`, `results/ml-32m.json`, `results/demographics/`, `results/global_cutoff/`, `results/two-tower-v2/`, and `results/serving/` are not regenerated. They were produced by the old code on their original hosts. For known users, ties at the top-n cut did not occur on ml-1m, and a tie at the item–item neighbour cut swaps one neighbour for another with the same similarity in 7 of 3,667 rows. Re-running them costs hours (ml-32M far more) for an effect that size. If one of them is regenerated for another reason, it will pick up the fixed rule.

## Outcome

Five distinct results of the cold-start protocol: the committed one, two with the old code on the second machine, and two with the new code. The two new-code runs on Windows, one per CPU path, were identical and share a row. Held-out NDCG@10, served list minus most-popular, paired, primary protocol, 604 users.

| run | K | trees | N=1 | N=3 | N=5 | N=10 |
| --- | --- | --- | --- | --- | --- | --- |
| committed, old code, cloud Linux | 50 | 88 | 0.0091 [-0.0001, 0.0184] | 0.0218 [0.0115, 0.0319] | 0.0309 [0.0204, 0.0416] | 0.0577 [0.0424, 0.0740] |
| old code, Windows, AVX2 | 50 | 179 | 0.0098 [0.0004, 0.0194] | 0.0240 [0.0138, 0.0346] | 0.0308 [0.0200, 0.0418] | 0.0612 [0.0460, 0.0784] |
| old code, Windows, baseline | 100 | 135 | 0.0090 [-0.0006, 0.0188] | 0.0235 [0.0135, 0.0338] | 0.0293 [0.0186, 0.0407] | 0.0567 [0.0422, 0.0727] |
| new code, Windows, either path | 50 | 142 | 0.0088 [-0.0007, 0.0182] | 0.0238 [0.0140, 0.0343] | 0.0310 [0.0201, 0.0426] | 0.0583 [0.0432, 0.0755] |
| new code, Docker (Linux), committed | 200 | 134 | 0.0062 [-0.0033, 0.0157] | 0.0226 [0.0124, 0.0333] | 0.0321 [0.0211, 0.0427] | 0.0607 [0.0465, 0.0769] |

What holds in every run: the served method is the short-profile ranker at every N, and it beats most-popular at N=3, 5, and 10 with intervals well clear of 0. What does not: the K the protocol picks (the three options are within about 0.002 on validation), the tree count, and the N=1 verdict, whose interval's low end ranges from -0.0033 to +0.0004. At N=1 the served list is not shown to beat popularity. Read that as "no detectable difference", not as a firm win or loss.

The sensitivity view (harness tail only) moves the same way. Its N=3 "beats most-popular" is true in the committed run and false in the new Docker run.

`results/cold-start/ml-1m.json` now holds the Docker run. ADR-0012 keeps its original numbers and gains an amendment pointing here.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| `np.argsort(..., kind="stable")` on the whole array | Same answer, but a full sort per user per call; the catalog is 66,819 items on ml-32M |
| Pin NumPy's CPU dispatch with `NPY_DISABLE_CPU_FEATURES` | Hides the bug on one setup, slows every sort, and does not fix the documented rule |
| Add a tiny index-based jitter to scores | Changes scores, and still depends on float rounding |
| Keep the committed cold-start file | It can no longer be produced by the code in the repo |
