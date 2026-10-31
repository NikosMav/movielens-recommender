# ADR-0020: Public Hugging Face Space for the ml-1m demo

## Status

Accepted. The Space is not created by this change. No Hugging Face API is called, and no token is stored. Pushing the Space is a later manual step, outside this repository's CI.

## Context

The Streamlit page in `app/streamlit_app.py` (ADR-0010, ADR-0012) loads two gitignored snapshots, `artifacts/ml-1m/` and `artifacts/ml-1m/cold_start/`. It does not train. Existing-user mode serves the production two-tower candidates and the LightGBM LambdaRank refit. New-user mode serves the cold-start ranker. A public demo should be that same page, on a free CPU Hugging Face Space named `NikosMav/movielens-recommender`.

A serving snapshot is not a set of model parameters. `save_bundle` and `save_cold_start_bundle` also write `movies.csv` (titles), `histories.npz` (rating rows), `item_item_user_item.npz` and `item_item_index.npz` (the user–item matrix and user ids), `two_tower_features.npz` (per-user histories), and `feature_context.npz` (per-user statistics and demographic codes). Those files are derived from MovieLens. The GroupLens terms decide whether any of them may be committed or uploaded.

The ml-1m archive used by this repo is the zip whose SHA-256 is pinned in `DATASET_SHA256` (`a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`). Its `ml-1m/README` states the terms. The relevant block, decoded as latin-1 from that zip, is:

```
Neither the University of Minnesota nor any of the researchers
involved can guarantee the correctness of the data, its suitability
for any particular purpose, or the validity of results based on the
use of the data set.  The data set may be used for any research
purposes under the following conditions:

     * The user may not state or imply any endorsement from the
       University of Minnesota or the GroupLens Research Group.

     * The user must acknowledge the use of the data set in
       publications resulting from the use of the data set
       (see below for citation information).

     * The user may not redistribute the data without separate
       permission.

     * The user may not use this information for any commercial or
       revenue-bearing purposes without first obtaining permission
       from a faculty member of the GroupLens Research Project at the
       University of Minnesota.
```

The citation in that README is F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19 (December 2015), 19 pages. DOI `http://dx.doi.org/10.1145/2827872`.

The same README says the archive holds 1,000,209 anonymous ratings of approximately 3,900 movies by 6,040 MovieLens users. That sentence is GroupLens's description of the archive. It is not a measured size of a model file.

## Decision

**Do not bundle MovieLens rows or titles.** Redistribution needs separate permission, and titles plus rating rows are the dataset. The Space does not commit `ratings.dat`, `movies.dat`, `users.dat`, `movies.csv`, rating histories, the user–item matrix, user ids, or per-user feature tables. It does not commit `popular_items.npy`. The GitHub tree ignores `space/weights/` and `space/data/`.

**Download ml-1m at startup.** When weights are present and `MOVIELENS_ARTIFACTS` is not already a full snapshot, the Space calls the same `download_dataset` path as `movielens-recommender download`. The zip is checked against the pinned SHA-256. Histories, the movie table, the user–item CSR, two-tower feature tables, and the ranker feature context are rebuilt from that download and the same deterministic split the snapshot was trained on. Rebuilding those tables is not training.

**Ship only small model-parameter files, and only in the manual Space push.** `make space-snapshot` writes them under `space/weights/`, which this GitHub repo ignores. The operator includes that directory when pushing `space/` as the Space repo root. The files are:

- item–item similarity (`item_item_similarity.npz`, `item_item_meta.json`, `item_ids.npy`)
- two-tower state dict and constructor metadata (`two_tower_model.pt`, `two_tower_meta.json`)
- LightGBM trees (`ranker.txt`, `ranker_meta.json`)
- for the new-user model, EASE item similarity (`ease.npz`, `ease_meta.json`)

`item_ids.npy` is the column order of the similarity matrix. It is not a user id list and not a rating row. Similarity matrices are derived from ratings. Refitting them inside the Space would be training, so they are copied and the user–item CSR is rebuilt in the saved item-id order. SciPy CSR sums duplicate coordinates, which is what `ItemItemCosineRecommender.fit` does. The Space does not call `ItemItemCosineRecommender.fit`, `EASERecommender.fit`, `fit_two_tower_recommender`, or `fit_lambdarank`. A popularity ranking, when `serving_by_n` is set, is a count via `MostPopularRecommender.fit` (or the same group-by). That count is computed at startup and is not uploaded.

**No training in the Space.** The production ranker and the new-user ranker are the ones `make space-snapshot` fits, with hyperparameters read from the committed results. The Space only loads those parameters. If `space/weights/` is absent, the page starts, prints `make space-snapshot`, and does not download and does not fit.

**One folder is the Space repo.** `space/` contains `app.py`, `runtime.py`, pinned `requirements.txt`, `README.md` with Hugging Face front matter (`sdk: streamlit`, `sdk_version: 1.65.0`, `app_file: app.py`, `python_version: "3.12"`), and `packages.txt` (`libgomp1` for LightGBM). Requirements install CPU `torch==2.6.0+cpu`, `lightgbm==4.6.0`, `streamlit==1.65.0`, and this package from GitHub at `23ab9515a3cbc0a18471d0fd36d03ae56b96a641`. Hydration and the page live in `space/` and call only APIs that exist at that commit. Pushing the contents of `space/` as the Space root does not need the rest of this repo. `space/.gitignore` ignores `data/`, `.runtime/`, and `__pycache__/`, and does not ignore `weights/`, so a fresh `git init` inside `space/` after `make space-snapshot` includes the parameters and excludes a local download.

**Hardware.** The free CPU size is `cpu-basic` (2 vCPU, 16 GB RAM), chosen in the Space creation form. A `hardware` key in the README front matter is ignored by Hugging Face, so this repo does not set one.

**Manual push, no token script.** Deploy happens outside this repo and no Hugging Face token is available here. There is no script that reads `HF_TOKEN`. After `make space-snapshot`, create a public Streamlit Space named `NikosMav/movielens-recommender` on free CPU in the browser, then from inside `space/`:

```bash
git init
git add .
git commit -m "MovieLens recommender Space"
git remote add origin https://huggingface.co/spaces/NikosMav/movielens-recommender
git push -u origin main
```

**Reasons.** The page never shows gender, age, occupation, ZIP, region, or a `demo_*` / `group_*` feature name. That filter is applied when the text is drawn. The explanation payload is unchanged, so the API and the existing reason tests keep the same card fields. The generic sentence “Popular with viewers similar to you” stays. The page states that GroupLens and the University of Minnesota do not endorse the demo, and it shows the Harper and Konstan citation.

**Cold-start train matrix.** The new-user weights are fit on the S5c complement: eligible users, hold out `round(0.1 * n)` with seed 42 (`HOLDOUT_FRACTION`), drop those users, then `time_based_split` on the remainder. Retrievers are fit on `split.full_train`. The ranker refit uses `simulated_queries` on the complement (`remain`), because that frame still has each user's test tail as labels. Startup rebuilds feature tables from `full_train` only. `make space-snapshot` reads `round2` from `results/cold-start/ml-1m.json` and does not call `finish_round2`, which would rewrite that file. The script hashes `results/**/*.json` before and after and aborts if any byte changes.

**What was measured, and what was not.**

The official ml-1m zip at the pinned SHA-256 is 5,917,549 bytes. Uncompressed members: `ratings.dat` 24,594,131 bytes, `movies.dat` 171,308 bytes, `users.dat` 134,368 bytes, `README` 5,577 bytes. These are archive sizes. They are not model-artifact sizes.

The byte size of `space/weights/` was not measured. `make space-snapshot` prints those sizes when it is run. This change does not run it, and the GitHub tree does not contain the fitted files, so no weight-file size is stated here.

Process start time on a `cpu-basic` Space was not measured. The Space does not exist yet. `results/cold-start/ml-1m.json` field `serving.latency_sec` is 0.152284. That file's `serving` object records a warmed second call of new-user scoring (`warmed` is true). It is not a process start, and it was not measured on `cpu-basic`. Request latencies in `results/serving/ml-1m.json` are also not startup times.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Commit the full `artifacts/ml-1m` snapshot | It contains titles, rating rows, user ids, and per-user histories. The README quoted above forbids redistribution without permission |
| Commit titles but not ratings | A title list is still the MovieLens movie file. The safe default is to download it |
| Refit item–item and EASE inside the Space | That is training on the free CPU, and it is slow. Copying the similarity matrices avoids it. They contain no rating rows and no titles |
| A deploy script that reads `HF_TOKEN` | The push happens outside this environment. No token is available, and CI must not need one |
| `sdk: docker` as the committed front matter | The Space is specified as Streamlit. `space/Dockerfile` is only the fallback if the Hub rejects `sdk: streamlit` (the built-in Streamlit SDK was deprecated on 2025-04-30; current Hub docs list `gradio`, `docker`, and `static`). The committed front matter stays `sdk: streamlit`. The Docker build is not added to CI; CI already builds the API image |
| Bake the download into the Space repo | That redistributes the dataset |

## Consequences

- A push of the committed `space/` tree, without running `make space-snapshot`, starts the app and shows the build command. It cannot recommend until the operator includes `space/weights/`.
- The first request after a real deploy spends time on the GroupLens download (about 6 MB compressed, about 25 MB of `ratings.dat`) plus the table rebuild. That wall time is not measured here.
- `space/requirements.txt` pins the library at `23ab9515a3cbc0a18471d0fd36d03ae56b96a641`, the commit that already contains `download_dataset` and the serving loaders. Later library changes are not picked up until that pin moves.
- The GitHub README links `https://huggingface.co/spaces/NikosMav/movielens-recommender` and marks the badge pending. The URL 404s until the manual push.
- Demographic feature names can still appear in the in-memory explanation object. They are dropped only at display time.
