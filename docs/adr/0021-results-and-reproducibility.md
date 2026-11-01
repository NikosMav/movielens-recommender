# ADR-0021: Result identities, comparison scope, and isolated reproductions

## Status

Accepted for implementation. This corrects reporting and provenance; it does not rerun or replace committed experiment measurements. It does not change the model selection protocol, learned weights, or local snapshots.

## Problems

- The historical S4 gate compared with default item-item cosine, although that model is not the strongest prior point on every dataset.
- Multi-seed result objects combined seed-mean points with primary-seed intervals and diagnostics under the same top-level keys.
- The cold-start snapshot lacked creation time and complete environment/configuration metadata. The API exposed only production provenance.
- Today's 1M YAML enables demographics, while the S4 headline record describes demographics off.
- Checkpoint time was saved before the final disk write and returned afterwards. Slow writes could make a resumed cumulative time appear to decrease.

## Decisions

### Comparisons

Keep the historical fixed item-item gate and its existing `negative_result` semantics for compatibility. Add `scope: fixed_item_item_baseline` and separate descriptive test comparisons. The prior-model families are fixed in `ranker/gate.py`: popularity, default/tuned item-item, default/tuned ALS, EASE, RP3beta, and two-tower. Ranker ablations are not prior models. Missing comparisons remain unknown, and exact ties do not count as improvements.

For new ranker runs, compare the seed mean with the strongest available prior point and the primary-seed unranked candidate list. These are descriptive points, not validation selection or significance tests. Report a primary-seed paired user-bootstrap interval for ranker minus unranked using aligned users. Prior-model per-user scores are not retained, so their paired interval is explicitly unavailable. Do not infer equivalence or significance from overlapping individual intervals.

For historical records the table generator computes the descriptive point comparisons without modifying JSON files or reclassifying their historical fixed-baseline gates.

### Seed summaries

New summaries contain:

- `seed_mean_metrics`: mean scalar metrics across the recorded seeds.
- `seed_summary`: mean, population standard deviation, minimum, maximum, and values by seed for each scalar metric.
- `primary_seed` and `primary_seed_metrics`: the complete named run, including its intervals and segments.
- `per_seed_metrics`: complete metrics keyed by seed.
- Top-level scalar aliases for existing consumers, an explicit `aggregation`, and an empty top-level `confidence_intervals` block. No across-seed-mean interval is claimed.

All scalar metric cutoffs are averaged consistently. Summaries reject mismatched metric sets and unequal eligible-user counts. Equal counts alone do not prove user identity; the main pipeline supplies shared evaluation frames. Readers use `primary_metrics` to support both new summaries and historical records. README interval tables display the point for the run that owns each interval separately from the seed mean.

### Snapshot identity

New snapshot writers record creation time, input run configuration and its hash when supplied, selected model settings, saving-host information, package versions, and SHA-256 hashes of sibling artifact files. Selected settings can override the input YAML, so both are retained. A `snapshot_id` binds that metadata to those hashes. Nested snapshots and batch subdirectories have their own scope. New loaders verify metadata and file hashes before loading models.

These hashes detect mismatch or corruption; they are not a signature from an independent authority. A snapshot's saving environment is recorded at serialization. Benchmark correspondence is not inferred merely from matching settings.

Old manifests remain loadable and the API labels their artifact integrity `legacy_unverified`. Missing metadata remains missing. The API retains its existing production fields and adds `snapshots.production` and `snapshots.new_user`, with separate identities, settings, timestamps, environment, and metadata gaps. `benchmark_record: null` means no association was supplied. New cold-start evaluations also record their snapshot ID in the results.

Git identity now detects staged and untracked changes as well as unstaged ones. When Git is unavailable, `MOVIELENS_BUILD_SHA` can supply a build identifier; otherwise the revision remains `unknown`. Container builds can set this through the `VCS_REF` build argument. Existing artifacts are not silently relabeled or rebuilt.

### Reproduction

`reproduce --from-results` supports the three headline records. It freezes their embedded configuration and selected hyperparameters in a unique directory under `artifacts/runs/`, with separate results and models. Selection files are reconstructed from the recorded choices, not copied from mutable tuning files. The source record and its hash, effective config hash, code revision, host, versions, and run status are recorded. Existing run IDs are refused. `--prepare-only` does not train. Downloads require an explicit flag.

This reruns the recorded protocol using current code. Historical result formats and the deterministic top-n correction can differ from the original run. BLAS/LAPACK and OS differences can affect ranker inputs. Bit-identical reproduction requires the original implementation and compatible environment; the command does not promise it. Other stage-specific experiments keep their own documented protocols.

### Checkpoint timing

For checkpointed training, saved and returned cumulative elapsed time ends immediately before the last completed epoch's checkpoint write. The final write and loading a completed checkpoint add no training time; earlier checkpoint writes in the same session remain included. This makes timing resumable and removes dependence on final disk-write speed. A fake clock with simulated slow writes tests the accounting without sleeps. Resume remains explicitly non-bit-exact because complete random state is not restored.

## Validation and remaining research

Regression tests cover slow checkpoint writes, fixed-gate wins below stronger comparators, ties and missing comparisons, seed/interval separation, a complete synthetic ranker stage and paired comparison, manifest corruption, legacy loading, both API snapshot identities, staged/untracked Git state, isolated reproduction, a real synthetic baseline rerun, and failure status.

The original result and snapshot files are preserved. The 32M test-informed sample, reused cold-start holdout, and lack of a neural/ranker drift replay remain research limitations. They require new experiments and are not corrected by relabeling existing results.
