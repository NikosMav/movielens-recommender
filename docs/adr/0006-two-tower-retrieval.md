# ADR-0006: Two-tower candidate retrieval

## Status

Accepted (S3b) — **negative result** on the headline ml-1m NDCG@10 gate.

## Context

S3a set the tuned bar on the per-user time split. On ml-1m test NDCG@10 the strongest baselines are item–item cosine (default **0.1201** [0.1158, 0.1242], tuned **0.1192** [0.1147, 0.1233]). S3b needs a **candidate-retrieval** model that can later feed a ranker (S4), trained and selected under ADR-0005 (tune on validation only; refit on full-train; evaluate once on test).

The model must stay small enough for CPU iteration on ml-1m, avoid new heavy serving deps, and not leak validation/test interactions into user-history features.

## Decision

**Architecture (two-tower, exact top-k):**

| Tower | Inputs | Combination |
| --- | --- | --- |
| User | User-id embedding + mean-pooled embeddings of that user’s **allowed** train items | Sum → LayerNorm → L2-normalize |
| Movie | Movie-id embedding + linear map of genre multi-hot + linear map of normalized release year | Sum → LayerNorm → L2-normalize |

- **Score:** temperature-scaled dot product.
- **Train loss:** in-batch sampled softmax with **log-q** popularity correction (`q(i) ∝` positive-interaction count in the fit matrix).
- **Retrieval:** exact brute-force top-k over the train catalog (no ANN). Catalog size (~3.7k on ml-1m) does not justify FAISS.
- **History features:** built only from interactions allowed by the protocol — **fit-train** while tuning / early-stopping on val; **full-train** for test and for the global-cutoff check. The positive item is excluded from that example’s pooled history during training.
- **Positives:** ratings ≥ relevance threshold (same 4.0 gate as ALS).
- **Dependency:** PyTorch in optional extras `pip install .[deep]` so the baseline install stays light. Pinned; CI installs the CPU wheel.

**Tuning (validation NDCG@10 only):**

Small grid over `embedding_dim ∈ {32, 64}`, `learning_rate ∈ {1e-3, 3e-3}`, `temperature ∈ {0.05, 0.1}`; fixed `batch_size=1024`, `weight_decay=1e-4`, `max_epochs=20`, early stopping patience **3** on val NDCG@10 (point estimate, no bootstrap). Chosen config is **refit on full-train** for the early-stopped epoch count (best epoch), then evaluated once on test.

**Reporting:**

- Test metrics over **3 seeds** (mean and spread) plus the usual user-bootstrap CI **per seed**.
- **Recall@100 / Recall@200** for two-tower and the baselines (retrieval role for S4).
- Included in segment breakdowns and the ml-1m global-time-cutoff table **without re-tuning** (same HPs as the per-user protocol).

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| BPR-MF | Strong pairwise baseline, but not a retrieval tower with side features; overlaps S2 ALS conceptually |
| Pure ID two-tower (no genres/year/history) | Weaker inductive bias on MovieLens metadata we already have; history pooling is cheap |
| TensorFlow Recommenders | Heavy TF stack for one model; PyTorch optional extra is enough |
| ANN / FAISS | Unnecessary at this catalog size; adds a native dep and approximate recall |
| Tune on test / pick best test seed | Forbidden by ADR-0005; would fake a stage win |
| Large grids / deep MLPs / transformers | Gold-plating; CPU runtime on ml-1m would dominate the stage |

## Outcome (ml-1m test)

Committed numbers: `results/ml-1m.json`, `results/tuning/two_tower_ml-1m.json`.

| Model | NDCG@10 | Notes |
| --- | --- | --- |
| item_item_cosine (default) | 0.1201 [0.1158, 0.1242] | Bar |
| item_item_cosine_tuned | 0.1192 [0.1147, 0.1233] | Bar |
| **two_tower** (3-seed mean) | **0.1192 ± 0.0001** | Per-seed CIs overlap both bars |

Chosen HPs: `embedding_dim=64`, `learning_rate=3e-3`, `temperature=0.1` (plus fixed batch/weight-decay/history); **best_epoch=6** (refit epoch count). Val NDCG@10 ≈ 0.0826.

**Gate: negative.** Mean NDCG@10 does not exceed the default item–item bar, and seed CIs do not clear either bar’s CI high. We do **not** claim a stage win.

**Where two-tower still helps (not the gate):**

- Recall@100 / @200 ≈ **0.435 / 0.604** vs item–item tuned **0.383 / 0.541** (and default 0.347 / 0.499) — better candidate coverage for a later ranker.
- Tail-item NDCG@10 ≈ **0.083** vs item–item ≈ **0.001–0.003** — much less head-collapsed.
- Global-cutoff NDCG@10 ≈ 0.214 vs item–item 0.232 (still behind; not re-tuned).

**Plausible reasons for the NDCG@10 miss:** neighborhood CF is a very strong inductive bias on dense MovieLens co-occurrence; in-batch softmax with a short early-stopped run (6 epochs) optimizes retrieval likelihood more than top-10 ranking; temperature/log-q help calibration but do not invent neighbour structure. Protocol was not bent to chase a win.

**S4 retriever:** use **item–item cosine** (default or tuned) as the candidate generator for the learned ranker, unless a later change reverses the NDCG@10 gate. Two-tower remains available as an optional diverse-retrieval baseline (stronger Recall@100/200 / tail).

## Consequences

- Headline comparison uses the same harness as S3a.
- Numbers are produced only by code-written JSON — never hand-edited.
- Optional `[deep]` install keeps the baseline path light; CI covers two-tower unit tests on synthetic data without downloading MovieLens.
