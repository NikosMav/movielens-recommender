# ADR-0014: Batch recommendations, HTTP API, and Docker image

## Status

Accepted (S5b). The latency and batch budgets below were written before any serving latency was measured. The outcome section is filled only after `results/serving/ml-1m.json` exists.

## Context

S5a (ADR-0010) and S5c (ADR-0012) serve the ml-1m production ranker and the new-user path inside a Streamlit page. The page calls `recommend_for_user` and `recommend_new_user` on snapshots that `build-artifacts` and `cold-start` write to `artifacts/ml-1m/`. Nothing outside Python can call those functions, there is no way to score every user at once, and there is no packaged runtime.

Three helpers on the request path scan whole tables on every call. `known_user_ids` takes `unique()` over every training rating and sorts it, and `recommend_for_user` does that once per request. `user_history_items` filters every training rating to find one user. `movie_row` filters the movie table once per history row and once per card. On ml-1m that is 802,553 training rows scanned twice, plus a few hundred scans of 3,883 movies, for each request.

## Decision

**No model change.** S5b serves the snapshots that already exist. It does not retrain, re-tune, or change any metric in `results/*.json`. The recommended lists are the same lists the Streamlit page shows for the same snapshot.

**Indexes on load.** The serving bundles build three lookups once, the first time they are needed: the set of known user ids, each user's row positions in the history table, and an item id to movie row map. The three helpers use them. Their outputs do not change, and a test checks that `recommend_for_user` returns the same items before and after the change.

**Ranking without explanations.** `rank_for_user` returns the production top n with ranker scores and stops there. `recommend_for_user` calls it and then adds explanations. Batch scoring calls only `rank_for_user`, because `pred_contrib` and the reason text are the expensive part and a batch file has no reader for them.

**Batch.** `movielens-recommender batch-recommend --artifacts artifacts/ml-1m` writes, under `artifacts/ml-1m/batch/` by default:

- `recommendations.csv.gz`: one row per (user, rank) with `user_id`, `rank`, `item_id`, `score`, `title`, `year`
- `batch_manifest.json`: the snapshot's dataset SHA-256, git SHA, and creation time; `n`; the number of users; wall time; users per second

The output is derived from MovieLens ratings, so it stays under the gitignored `artifacts/` directory and is never committed. `--users` limits the run to a comma-separated list of ids. `--n` defaults to 10.

**HTTP API.** FastAPI, in the new optional extra `api` (`fastapi==0.115.12`, `uvicorn==0.34.2`, `httpx==0.28.1` for the test client). The app factory takes the production and cold-start snapshot directories and loads each once at startup. Environment variables match the Streamlit page: `MOVIELENS_ARTIFACTS` (default `artifacts/ml-1m`) and `MOVIELENS_COLD_ARTIFACTS` (default `<production>/cold_start`).

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | `ok` with what is loaded, or 503 with the build command when the production snapshot is missing |
| GET | `/v1/snapshot` | Dataset, dataset SHA-256, git SHA, created time, candidate set, and whether the new-user snapshot is loaded |
| GET | `/v1/users/{user_id}/recommendations` | Production top `n` (1–100, default 10). `explain=true` (default) adds the plain-language reasons. 404 for an unknown user |
| POST | `/v1/recommendations/new-user` | Top `n` for `{"ratings": [{"item_id", "rating"}], "n"}`. 1–200 ratings, each 0.5–5.0. 422 for an item id that is not in the catalog. 503 when the new-user snapshot is missing |
| GET | `/v1/movies/search?q=` | Up to `limit` (1–50, default 20) titles containing `q`, case-insensitive, so a client can find item ids for the new-user call |

The reason text and the no-demographics rule are the ones in ADR-0010 and ADR-0012. The API does not accept or return gender, age, occupation, or ZIP. It does not store profiles.

Model calls in one process run under a single lock. The two-tower, LightGBM `predict`, and `pred_contrib` paths were written for one caller at a time, and this keeps them that way. Throughput comes from running more worker processes, not threads.

The API has no authentication, rate limiting, or CORS. It is a local and demo service, and is not meant to be exposed to the internet as is.

**Docker.** One `Dockerfile` on `python:3.12-slim`: `libgomp1` for LightGBM, the CPU PyTorch 2.6.0 wheel, then the package with `[deep,rank,api]`. It runs as a non-root user and serves on port 8000 with `uvicorn --factory`. A `HEALTHCHECK` calls `/health`.

The image contains no MovieLens data and no snapshot. The GroupLens terms do not allow redistributing the data, and a snapshot holds per-user rating histories. Snapshots are mounted at run time (`-v ./artifacts:/app/artifacts:ro`). The same image can build them, because it carries the CLI: mount `data/` and `artifacts/` read-write and run `movielens-recommender build-artifacts`.

CI builds the image and checks that a container with no snapshot answers `/health` with 503 and the build command. CI still downloads no MovieLens data.

**Latency and batch budget, fixed before measurement.** Measured on ml-1m with `scripts/measure_serving.py`, in one process, through the FastAPI test client, so JSON encoding and validation are included and the network is not. Each call is warmed once first.

| Measure | Sample | Budget |
| --- | --- | --- |
| Known user, top 10 with explanations | 500 known users drawn with seed 42 | p95 ≤ 300 ms |
| Known user, top 10 without explanations | the same 500 users | p95 ≤ 100 ms |
| New user, five ratings, top 10 | 200 profiles: five distinct items drawn with seed 42 from the 500 most-rated training items, ratings drawn from {3, 4, 5} | p95 ≤ 300 ms |
| Batch, every known user, top 10 | all 6,040 ml-1m users | ≤ 600 s wall |

The file records p50, p95, p99, and max for each, the host's CPU count and memory, and library versions. The same script also times the known-user path with the indexes turned off, so the effect of the indexes is a measured number. A missed budget is reported as missed. The budget is not moved after the measurement.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Parquet for the batch file | Needs `pyarrow`, a large new dependency for one file; gzip CSV is readable everywhere |
| Explanations in the batch file | `pred_contrib` per user is the slow part, and a file with no reader for reasons does not need them |
| Snapshot baked into the image | Redistributes MovieLens-derived rating histories, which the GroupLens terms do not allow |
| Async endpoints with a thread pool around the models | The models run on CPU and hold the GIL for most of a call; worker processes scale more predictably |
| Serving ml-32M | No ml-32M serving snapshot exists; `build-artifacts` targets ml-1m (ADR-0010) |

## Outcome

To be filled from `results/serving/ml-1m.json`.
