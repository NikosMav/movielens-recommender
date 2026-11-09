# ADR-0023: Server RSS budget for the live demo

## Status

Accepted. This decision records the measurement and the budget. It does not rerun experiments, rewrite `results/`, or commit MovieLens files or an API key. Code that brings the running server under the budget follows this record. It must not change a recommendation or a stored metric.

## Context

The public demo is the Streamlit Community Cloud app at `https://nikosmav-movielens-recommender.streamlit.app/`, entrypoint `space/app.py` (ADR-0020, ADR-0022). Community Cloud gives this app 1 GB of RAM. ADR-0022 set the budget for a fresh process: peak `ru_maxrss` under about 850 MiB, which is 870,400 KiB. That check passed. On this VM a fresh process that imported Streamlit, rebuilt both snapshots, and kept both models resident peaked at 776,972 KiB (`ru_maxrss`) and 777,388 kB (`VmHWM`) after ADR-0022's title-query and theme changes.

The process that actually serves the page is not that fresh process. It is the long-lived `streamlit run space/app.py` process. In the verification for the change that became ADR-0022's amendment, a browser walked the three views and the same process recorded a higher high-water mark: `VmHWM` 1,064,276 kB in the light session and 1,011,920 kB in the dark session. Steady RSS afterwards was 832,584 kB and 781,388 kB. 1,064,276 kB is over 1 GiB (1,048,576 kB). Community Cloud restarts the app when the process is killed for memory. The fresh-process number does not catch that.

`VmHWM` is the peak resident set of the process. It does not go down when memory is freed. A temporary copy that is released after a click still counts. The budget below is that peak, read from `/proc/<pid>/status` for the Streamlit server process only. Browser processes are not included. 850 MiB is 870,400 KiB, the same conversion as ADR-0022. The server passes when its `VmHWM` across the walk below stays under 870,400 kB.

## Decision

**Budget.** The long-lived Streamlit process that serves `space/app.py` stays under 850 MiB peak resident memory on the walk in this decision. The recorded value is `VmHWM` in kB. The line is 870,400 kB. A fresh-process `ru_maxrss` is still reported, and ADR-0022's fresh-process budget still applies. The number that gates this decision is the server `VmHWM`.

**What must not change.** Top-10 item ids for a fixed set of existing ml-1m users, and for one fixed new-user rating list, stay the ids produced by `main` at `268c75a`. Stored metrics and `results/` stay byte-identical. The ranker, the retrievers, and the feature values that are scored are not retrained and not re-rounded. Mode switches still open under a spinner (ADR-0022). Both modes stay available without a second-long blank page.

**Measurement.** `scripts/measure_server_rss.py` is the procedure. It is repeatable on this VM. It does not commit data and it does not write under `results/`.

Preparation, before the server process starts:

- The pinned weight archive is the file already checked in ADR-0020, SHA-256 `f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd`. If it is not beside `space/app.py`, the script downloads it to a temporary directory and links it there. The link is gitignored.
- ml-1m is the GroupLens zip pinned in `DATASET_SHA256`. The script extracts it under the runtime directory's `data/` path when that tree is missing. `MOVIELENS_SPACE_RUNTIME` points at a temporary directory, so the rebuild is not written into the git checkout.
- The server is not given `MOVIELENS_ARTIFACTS`. It rebuilds both snapshots from the archive and ml-1m, which is what Community Cloud does on a cold start.

Server:

- `streamlit run space/app.py`, headless, bound to `127.0.0.1`, usage stats off, file watcher off. The watcher is off so a file write during the walk cannot restart the process and reset the high-water mark.
- No `TMDB_API_KEY`. Poster lookups return the placeholder. No image bytes are fetched.
- The script reads `VmHWM` and `VmRSS` from `/proc/<pid>/status` for that Python process.

Walk, with two browser sessions open at the same time. Headless Chrome drives the page. One session is not closed before the other has started.

1. Session A opens Existing user and waits until the production model and its top 10 are on the page.
2. Session B opens and waits until Existing user is on the page. From here the two sessions overlap.
3. Session A switches to New user, selects Comedy, rates the first five films on that shelf 5, 4, 5, 3, and 4, and waits until Get recommendations has returned a top 10.
4. Session A opens How good is it?, then Existing user, then New user, then How good is it? again.
5. Session B opens How good is it? and then New user.
6. Both browsers close. The script waits two seconds and reads `VmRSS` (steady) and `VmHWM` (peak over the whole process, including startup).

Fresh process, after the server has exited, in a new interpreter: import Streamlit, rebuild both snapshots from the archive and ml-1m already on disk, keep both models resident, build one genre shelf, render one recommendation card, read the quality JSON, and call the poster lookup with an empty key. Print `ru_maxrss` and `VmHWM`. That is the ADR-0022 shape. It is not the gate for this decision.

The script prints `peak_vmhwm_kb`, `steady_vmrss_kb`, `fresh_ru_maxrss_kib`, and `fresh_vmhwm_kb`. Those are the numbers to compare before and after. They are measurements, not targets copied into the source.

## Consequences

- A change to the demo that raises the server `VmHWM` on this walk to 870,400 kB or above is over budget, even when a fresh process still passes ADR-0022.
- The walk includes startup, so a one-time allocation during rebuild counts. Freeing it afterwards does not lower `VmHWM`.
- `results/` is not an input to this measurement and must not change.
- The live app picks up `space/` from this repo. It does not pick up `src/` until the pin in `space/requirements.txt` moves (ADR-0020). A reduction that the running Community Cloud process needs has to live in `space/` or the pin has to move.
