# ADR-0001: Dataset version and integrity

## Status

Accepted (S1)

## Context

We need a public, redistributable-by-license-terms ratings corpus that is small enough for clone-and-run iteration and large enough for believable CF baselines. MovieLens must not be committed to git; downloads must be verifiable.

## Decision

- Default dataset: **MovieLens `ml-latest-small`**, identified by the **SHA-256 of the official GroupLens zip** (not by a mutable “latest” label alone).
- Optional larger set: **MovieLens 1M (`ml-1m`)**, also SHA-256 pinned.
- Download only from official `files.grouplens.org` URLs; verify checksum before extract; refuse on mismatch.
- Never commit raw or derived rating matrices under `data/`.

Pinned checksums (computed from the official archives at pin time):

| Dataset | SHA-256 |
| --- | --- |
| ml-latest-small | `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436` |
| ml-1m | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` |

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Trust “latest” without checksum | Silent corpus drift breaks reproducibility |
| Vendor / commit the zip | Violates GroupLens redistribution norms; bloats the repo |
| Only ml-1m | Slower local loop; small set is enough for S1–S2 harness work |
| Mirror on S3 / HF | Extra moving parts for a solo public repo |

## Consequences

CI never downloads data. Local/pipeline runs fail closed if GroupLens rotates `ml-latest-small` without a checksum update (intentional: update the pin deliberately in a follow-up).
