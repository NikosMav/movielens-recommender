# Reproducing a named experiment

## Headline comparisons

Use a published record to recover the protocol it actually used:

| Experiment | Command |
| --- | --- |
| Small panel, S4 | `movielens-recommender reproduce --from-results results/ml-latest-small.json` |
| 1M, S4 without demographics | `movielens-recommender reproduce --from-results results/ml-1m.json` |
| 32M scale panel | `movielens-recommender reproduce --from-results results/ml-32m.json` |

Run from the repository root with the required package extras installed. Data is read from the record's `data_dir`, resolved from the working directory. If the data lives elsewhere, prepare the run, edit its explicit data path, and treat that edited configuration as a new experiment using `run`; the recorded config hash belongs to the original preparation.

Each command creates `artifacts/runs/<run-id>/` containing `source-results.json`, the materialized `config.yaml`, `run.json`, derived validation selections under `results/tuning/`, new results, and any ranker model files. Published records and serving snapshots are not rewritten. `--output-root` changes the parent directory. Reusing an existing `--run-id` fails instead of overwriting it.

To inspect inputs without fitting models:

```bash
movielens-recommender reproduce --from-results results/ml-1m.json --prepare-only
```

To run the experiment, issue the command without `--prepare-only`; it reserves a new directory. Preparation is an inspection step, not a resumable job. Add `--download` only when data must be fetched. The 32M rerun can take hours and retains its original 8,000-user, head-catalog, and seed-count constraints.

This command refits using settings selected in the source record. It does not repeat the original hyperparameter search. `run --config ...` remains available for a new search and writes to the paths in that config. In particular, `configs/ml-1m.yaml` is the demographics-on serving configuration, so it does not recreate the earlier S4 headline feature set.

New code writes explicit seed summaries. Original records keep their original schema. Check the source revision and environment when investigating numerical differences: deterministic top-n was corrected after several headline records, and BLAS/LAPACK differences still affect the cold-start ranker. See ADR-0015 and ADR-0021.

## Other published experiments

The generic reproduction command intentionally accepts headline records with embedded configs. The following stage-specific commands have different input dependencies and may write their own snapshots. Use a separate checkout with its own `data/`, `results/`, and `artifacts/` when reproducing these stages so their outputs cannot replace the current demonstration or historical records.

| Experiment | Stage-specific entry point | Protocol and qualifications |
| --- | --- | --- |
| 1M demographic comparison | `movielens-recommender demographics --config configs/ml-1m.yaml --no-download` | ADR-0009; compare variants within the same run, rather than subtracting unrelated headline runs |
| New-user experiment | `movielens-recommender cold-start --config configs/ml-1m.yaml --no-download` | ADR-0012 and ADR-0015; the second round reused the held-out people; Docker environment matters |
| Alternative tower losses | `python scripts/run_two_tower_v2.py --help` | Follow the staged probe, budget, tuning, and test commands in ADR-0013; later extensions are not fresh independent tests |
| Serving latency | `python scripts/measure_serving.py --artifacts artifacts/ml-1m` | ADR-0014; identify both snapshots first; host and sequential workload affect timing |
| Drift replay | `movielens-recommender drift-study --config configs/ml-1m.yaml --no-download` | ADR-0016; only popularity, item-item, and EASE were replayed |

Historical revision references are `c82b6da` for demographics, `7227d01` for the corrected Docker cold-start record, `9e34701` for the alternative-loss work, `d6ed85f` for serving measurements, and `c4d96fe` for the drift replay. Later code can add corrections, so record the revision actually used instead of assuming these experiments share one implementation.

## Serving identity

`GET /v1/snapshot` now reports both loaded models under `snapshots`. New snapshots have a `snapshot_id`, a config hash, environment information, and verified artifact checksums. Older snapshots remain usable with `artifact_integrity: legacy_unverified`. An absent timestamp or revision is not filled with an invented historical value. `benchmark_record: null` means no association was supplied; an operational recommendation response does not prove benchmark equivalence.

Build new artifacts intentionally with `build-artifacts --out-dir <new-directory>` if replacing the demonstration is not the goal. For cold-start, the stage-specific command still writes its default snapshot location, which is another reason to use a separate checkout for reproduction. Inspect identities before comparing service outputs.

For a container built from an identified source revision, pass its revision as `--build-arg VCS_REF=<revision>` when building the image. The image exports this as `MOVIELENS_BUILD_SHA` for snapshot creation without Git. The default is explicitly `unknown`; the build argument must describe the actual source contents.
