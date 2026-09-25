# ADR-0004: Tooling and experiment tracking

## Status

Accepted (S1)

## Context

Need reproducible runs and lightweight tracking without operating an ML platform.

## Decision

- **Config-driven runs:** YAML under `configs/` (dataset, split, eval, model hyperparams, seed).
- **Seeds:** single run `seed` drives ALS and bootstrap resampling.
- **Dependencies:** `pyproject.toml` plus pinned `requirements.txt` for reproducible installs.
- **Tracking:** committed `results/*.json` (metrics, CIs, split summary, checksum, library versions, config snapshot)—**no MLflow / W&B**.
- **README table:** generated from JSON via `scripts/make_results_table.py` (never hand-edit numbers).
- **CI:** ruff + pytest; no MovieLens download.
- **Package layout:** `src/movielens_recommender/` with console script entry point.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| MLflow / Weights & Biases | Extra account/server surface; JSON-in-git is enough for S1–S2 |
| Poetry/PDM lock only | `requirements.txt` pins are enough and CI-friendly |
| Notebooks as the only entrypoint | Harder to test and CI; keep a scripted EDA instead |
| Hydra | Heavier than one YAML loader for this scale |

## Consequences

Re-running with the same config+seed+checksum should regenerate the same JSON (ALS/float noise aside). Experiment history is the git history of `results/`.
