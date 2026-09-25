# ADR-0006: Two-tower candidate retrieval

## Status

Accepted (S3b) — **negative result** on the headline ml-1m gate (see Consequences).

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
- **Recall@100 / Recall@200** for two-tower and the tuned baselines (retrieval role for S4).
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

## Consequences

- Headline comparison uses the same harness as S3a. **To claim a win, two-tower must beat both item–item default and tuned on ml-1m test NDCG@10 with CIs taken into account.**
- If it does not, this ADR and the README record a **negative result**; S4 should use the stronger retriever (item–item cosine) as the candidate generator unless a later change reverses the gate.
- Tuning logs: `results/tuning/two_tower_*.json`. Metrics: `results/*.json` and `results/global_cutoff/ml-1m.json`.
- Numbers are produced only by code-written JSON — never hand-edited.
