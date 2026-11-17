# Serving and deployment

How the public demo is deployed, and how to run the same models behind an HTTP API, as a batch job, or in Docker.

Back to the [README](../README.md).

## The public demo on Streamlit Community Cloud

The [public demo](https://nikosmav-movielens-recommender.streamlit.app/) runs on [Streamlit Community Cloud](https://share.streamlit.io/). Posters use the Streamlit secret `TMDB_API_KEY` when it is set; without it, each card uses a placeholder. The page follows Streamlit's System, Light, and Dark setting.

The entrypoint is `space/app.py`. Community Cloud runs `streamlit run` with the repository root as the working directory and looks for dependencies in the entrypoint's directory first. `space/requirements.txt` is that file: pinned versions, Python 3.12, and CPU PyTorch from `--extra-index-url https://download.pytorch.org/whl/cpu`. `packages.txt` at the repo root installs `libgomp1` for LightGBM. In Advanced settings, select Python 3.12.

On [share.streamlit.io](https://share.streamlit.io/), create an app from `NikosMav/movielens-recommender`, branch `main`, main file path `space/app.py`.

The app looks first for `space/space-weights.tar.gz`. That file is gitignored. If it is absent, the app downloads `https://github.com/NikosMav/movielens-recommender/releases/download/weights-v2/space-weights.tar.gz` into `space/.runtime/cache/` and checks `WEIGHTS_SHA256` (`19ac85cf113f9de8e6962dfadab365bfcf9c35eaf0acb89fccc66961085031c2`). A `WEIGHTS_URL` environment variable overrides that address, and so does a Streamlit secret named `WEIGHTS_URL` when the environment variable is unset. If the file is missing or the checksum fails, the page shows `make space-snapshot` and does not download MovieLens. On success it downloads ml-1m from GroupLens and does not train. The page does not show gender, age, occupation, ZIP, or region as reasons.

A scheduled GitHub Actions workflow opens the demo in a headless browser every four hours (`scripts/keep_awake.py`) so Community Cloud does not put it to sleep. GitHub pauses scheduled workflows in a repository with no commits for 60 days; re-enable it from the Actions tab.

## Serve over HTTP, batch, and Docker

S5b puts the same two snapshots behind an HTTP API, scores every user to a file, and packages the API as a Docker image (ADR-0014). Build the snapshots first, as in [Build the models locally](#build-the-models-locally).

```bash
uvicorn --factory movielens_recommender.serving.api:create_app_from_env --port 8000
```

| Method | Path | What it does |
| --- | --- | --- |
| GET | `/health` | `ok`, or 503 with the build command when no snapshot is mounted |
| GET | `/v1/snapshot` | Dataset checksum, git SHA, and creation time of the loaded snapshot |
| GET | `/v1/users/{user_id}/recommendations?n=10&explain=true` | Production top n for a known ml-1m user, with plain-language reasons |
| GET | `/v1/users/{user_id}/comparison?n=10` | Recent training ratings and ranked lists from the production ranker, item–item, and two-tower models, plus the pairwise-transition list when the snapshot has it |
| POST | `/v1/recommendations/new-user` | Top n for `{"ratings": [{"item_id": 2571, "rating": 5}], "n": 10}` |
| GET | `/v1/movies/search?q=matrix` | Find item ids by title |
| GET | `/metrics` | Prometheus metrics: requests and latency by route, snapshot age, new-user input drift |

Interactive docs are at `/docs`. The API takes ratings only: demographic fields are rejected. It has no authentication, so keep it local.

`/metrics` labels requests by route template (`/v1/users/{user_id}/recommendations`), never by raw path, so user ids do not leak into labels. For the last 500 new-user requests it reports how popular the rated titles are, next to the same figures for the training ratings, so a shift in who is using the new-user path shows up without extra tooling. Each request also writes one JSON line to the `movielens_recommender.api` logger, with no item ids or ratings (ADR-0016).

```bash
curl "http://127.0.0.1:8000/v1/users/155/recommendations?n=5"
```

Batch scoring writes the top 10 for every snapshot user to `artifacts/ml-1m/batch/recommendations.csv.gz`, with a manifest that records the snapshot it came from. The file is rating-derived data and stays gitignored.

```bash
movielens-recommender batch-recommend --artifacts artifacts/ml-1m
```

The Docker image holds the code, configs, and committed results, but no MovieLens data and no snapshot. Mount the snapshots read-only:

```bash
docker build -t movielens-recommender .
docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" movielens-recommender
```

The same image can build the snapshots, with `data/` and `artifacts/` mounted writable:

```bash
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" movielens-recommender movielens-recommender build-artifacts --config configs/ml-1m.yaml
```

Measured latency and batch time are in the [serving panel](results.md#serving-latency-and-batch-ml-1m-s5b) of the results.

## The React frontend

For the local React and Astryx frontend, follow the [frontend setup guide](../frontend/README.md). It uses FastAPI for movie search, personalized picks, and the Model Lab comparisons.

## Build the models locally

The public app downloads ready-made weights. To build the serving snapshots from scratch and run the local research page:

```bash
movielens-recommender build-artifacts --config configs/ml-1m.yaml
movielens-recommender cold-start --config configs/ml-1m.yaml
streamlit run app/streamlit_app.py
```

`build-artifacts` downloads ml-1m if `data/` does not have it, reads the tuned hyperparameters from `results/tuning/`, and writes a serving snapshot to `artifacts/ml-1m/`. `cold-start` writes the new-user snapshot to `artifacts/ml-1m/cold_start/` and the measured table to `results/cold-start/ml-1m.json`. Nothing under `artifacts/` is committed.

The local research page (`app/streamlit_app.py`) has two modes:

- **Existing user.** Pick an ml-1m user. The list is the production pipeline: the ADR-0033 candidate set (pairwise transitions and two-tower) re-ranked by LambdaRank, with no demographic features. Each title opens a plain-language explanation, and a checkbox shows the item–item and two-tower lists side by side (ADR-0010).
- **New user.** Search titles and rate at least three (about five is a good start). The page says which method produced the list for that profile size. It does not ask for gender, age, occupation, or ZIP, and it does not save the profile (ADR-0012).

The public Community Cloud app above does the same jobs in one navigation (ADR-0024).
