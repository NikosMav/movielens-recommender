# ADR-0020: Public ml-1m demo on Streamlit Community Cloud

## Status

Accepted. The public app is not created by this change. No token is stored. Deploy is a later manual step on Streamlit Community Cloud, outside this repository's CI.

Hugging Face Docker Spaces require a PRO account. This repo does not use that path. The entrypoint is `space/app.py` in this GitHub repository.

## Context

The Streamlit page in `app/streamlit_app.py` (ADR-0010, ADR-0012) loads two gitignored snapshots, `artifacts/ml-1m/` and `artifacts/ml-1m/cold_start/`. It does not train. Existing-user mode serves the production two-tower candidates and the LightGBM LambdaRank refit. New-user mode serves the cold-start ranker. A public demo should be that same page. The first plan was a free CPU Hugging Face Space named `NikosMav/movielens-recommender`. Docker Spaces now require PRO, so the demo runs on Streamlit Community Cloud instead, from this repository, with a 1 GB RAM budget.

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

**Download ml-1m at startup.** After the parameter archive is in place, and when `MOVIELENS_ARTIFACTS` is not already a full snapshot, the Space calls the same `download_dataset` path as `movielens-recommender download`. The zip is checked against the pinned SHA-256. Histories, the movie table, the user–item CSR, two-tower feature tables, and the ranker feature context are rebuilt from that download and the same deterministic split the snapshot was trained on. Rebuilding those tables is not training.

**Parameter files are not committed to this GitHub repo.** `make space-snapshot` writes them under `space/weights/`. Both this repo and `space/.gitignore` ignore that directory. The Space stores one gzip, `space-weights.tar.gz`, next to `app.py`. This repo gitignores that archive. `space/.gitignore` does not, and `space/.gitattributes` marks `*.tar.gz` for Git LFS, so the Space push can add the file. The person who pushes copies the archive in. They do not need MovieLens data and do not need the `rank`, `deep`, or `ui` extras. The files inside the archive are:

- item–item similarity (`item_item_similarity.npz`, `item_item_meta.json`, `item_ids.npy`)
- two-tower state dict and constructor metadata (`two_tower_model.pt`, `two_tower_meta.json`)
- LightGBM trees (`ranker.txt`, `ranker_meta.json`)
- for the new-user model, EASE item similarity (`ease.npz`, `ease_meta.json`)

`item_ids.npy` is the column order of the similarity matrix. It is not a user id list and not a rating row. The checked tree has no `user_id` or `user_ids` arrays and no rating arrays. Similarity matrices are derived from ratings. Refitting them inside the Space would be training, so they are copied and the user–item CSR is rebuilt in the saved item-id order. SciPy CSR sums duplicate coordinates, which is what `ItemItemCosineRecommender.fit` does. The Space does not call `ItemItemCosineRecommender.fit`, `EASERecommender.fit`, `fit_two_tower_recommender`, or `fit_lambdarank`. A popularity ranking, when `serving_by_n` is set, is a count via `MostPopularRecommender.fit` (or the same group-by). That count is computed at startup and is not in the archive.

**Read those parameters from a local archive, or from one release URL.** The app looks first for `space-weights.tar.gz` next to `app.py`. If that file is absent, it downloads `https://github.com/NikosMav/movielens-recommender/releases/download/space-weights-v1/space-weights.tar.gz` into `space/.runtime/cache/` and refuses the bytes unless the SHA-256 matches `WEIGHTS_SHA256` in `space/runtime.py`. It then extracts into `weights/`. If the file is missing or the checksum fails, the page shows `make space-snapshot` and does not download MovieLens and does not fit. This change does not create that GitHub Release. The archive built with the sizes below has SHA-256 `f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd` and is 119,917,467 bytes. The release tag is `space-weights-v1`. The asset name is `space-weights.tar.gz`.

**No training in the Space.** The production ranker and the new-user ranker are the ones `make space-snapshot` fits, with hyperparameters read from the committed results. The Space only loads those parameters.

**Community Cloud runs `space/app.py` from this repo.** `space/requirements.txt` sits next to that entrypoint, so Community Cloud installs it instead of the root `requirements.txt` or `pyproject.toml`. It pins CPU `torch==2.6.0+cpu` via `--extra-index-url https://download.pytorch.org/whl/cpu`, `lightgbm==4.6.0`, `streamlit==1.65.0`, and this package from GitHub at `23ab9515a3cbc0a18471d0fd36d03ae56b96a641`. Python is 3.12, selected in Advanced settings (Community Cloud's default). `packages.txt` at the repository root installs `libgomp1`. Hydration and the page live in `space/` and call only APIs that exist at that pin. The library loader densifies item–item similarity; `space/runtime.py` keeps that matrix sparse so the process fits in 1 GB. EASE stays float64. The archive bytes are unchanged.

**Hardware.** Community Cloud's RAM budget for this app is 1 GB. There is no `hardware` key to set.

**Manual deploy, no token script.** Deploy happens outside this repo. There is no script that reads a token. On [share.streamlit.io](https://share.streamlit.io/), create an app from `NikosMav/movielens-recommender`, branch `main`, main file path `space/app.py`, Python 3.12. The GitHub README badge stays pending until that app exists.

**Reasons.** The page never shows gender, age, occupation, ZIP, region, or a `demo_*` / `group_*` feature name. That filter is applied when the text is drawn. The explanation payload is unchanged, so the API and the existing reason tests keep the same card fields. The generic sentence “Popular with viewers similar to you” stays. The page states that GroupLens and the University of Minnesota do not endorse the demo, and it shows the Harper and Konstan citation.

**Cold-start train matrix.** The new-user weights are fit on the S5c complement: eligible users, hold out `round(0.1 * n)` with seed 42 (`HOLDOUT_FRACTION`), drop those users, then `time_based_split` on the remainder. Retrievers are fit on `split.full_train`. The ranker refit uses `simulated_queries` on the complement (`remain`), because that frame still has each user's test tail as labels. Startup rebuilds feature tables from `full_train` only. `make space-snapshot` reads `round2` from `results/cold-start/ml-1m.json` and does not call `finish_round2`, which would rewrite that file. The script hashes `results/**/*.json` before and after and aborts if any byte changes.

**What was measured, and what was not.**

The official ml-1m zip at the pinned SHA-256 is 5,917,549 bytes. Uncompressed members: `ratings.dat` 24,594,131 bytes, `movies.dat` 171,308 bytes, `users.dat` 134,368 bytes, `README` 5,577 bytes. These are archive sizes. They are not model-artifact sizes.

`make space-snapshot` was run on this VM. The parameter files, in bytes, are:

| File | Bytes |
| --- | --- |
| `cold_start/ease.npz` | 102,711,367 |
| `cold_start/ease_meta.json` | 19 |
| `cold_start/item_ids.npy` | 29,384 |
| `cold_start/item_item_meta.json` | 66 |
| `cold_start/item_item_similarity.npz` | 6,214,039 |
| `cold_start/manifest.json` | 2,441 |
| `cold_start/ranker.txt` | 488,876 |
| `cold_start/ranker_meta.json` | 1,497 |
| `cold_start/two_tower_meta.json` | 585 |
| `cold_start/two_tower_model.pt` | 2,337,568 |
| `production/item_ids.npy` | 29,464 |
| `production/item_item_meta.json` | 66 |
| `production/item_item_similarity.npz` | 6,233,196 |
| `production/manifest.json` | 3,565 |
| `production/ranker.txt` | 108,750 |
| `production/ranker_meta.json` | 1,625 |
| `production/two_tower_meta.json` | 498 |
| `production/two_tower_model.pt` | 2,494,752 |

The files sum to 120,657,758 bytes. They are not in git. The gzip archive of `production/` and `cold_start/` is 119,917,467 bytes, SHA-256 `f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd`. `cold_start/ease.npz` arrays are `similarity` and `item_ids`. The item–item files are SciPy sparse components (`indices`, `indptr`, `format`, `shape`, `data`). None of those names is a user id or a rating row.

Process start on Community Cloud was not measured. On this VM (4 CPUs, Intel Xeon, `MemTotal` 16,387,280 kB), a fresh Python process that imported Streamlit, rebuilt both snapshots from the local weight files and the already extracted ml-1m tree, kept both loaded models resident, and returned ten existing-user and ten new-user recommendations peaked at 781,632 KiB RSS (`ru_maxrss`; 800,391,168 bytes). `/proc` `VmHWM` at the end was 781,824 kB. Elapsed wall time was 13.909 seconds. That is VM time, not a Community Cloud measurement. Item–item similarity stayed sparse. An earlier run in the same process shape, which densified item–item to float64, peaked at 926,872 KiB and was over the about-900 MB line, so the demo does not densify that matrix. The archive SHA-256 is unchanged. An earlier cold process, which also downloaded ml-1m into an empty directory, was ready in 15.696 seconds of wall time (in-process clock 15.293 seconds). `results/cold-start/ml-1m.json` field `serving.latency_sec` is 0.152284. That file's `serving` object records a warmed second call of new-user scoring (`warmed` is true). It is not a process start and not a RAM measurement. Request latencies in `results/serving/ml-1m.json` are also not startup times.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Commit the full `artifacts/ml-1m` snapshot | It contains titles, rating rows, user ids, and per-user histories. The README quoted above forbids redistribution without permission |
| Commit titles but not ratings | A title list is still the MovieLens movie file. The safe default is to download it |
| Refit item–item and EASE inside the Space | That is training on the free CPU, and it is slow. Copying the similarity matrices avoids it. They contain no rating rows and no titles |
| A deploy script that reads `HF_TOKEN` | The push happens outside this environment. No token is available, and CI must not need one |
| Hugging Face Docker Space | Docker Spaces require a PRO account. The demo runs on Streamlit Community Cloud from this GitHub repo |
| `sdk: streamlit` on Hugging Face | The built-in Streamlit SDK was deprecated on 2025-04-30, and the deploy is no longer a Hugging Face Space |
| Commit `space/weights/` or the gzip archive to this GitHub repo | The archive is 119,917,467 bytes. This repo gitignores it. The Space repo stores that one file with Git LFS |
| Download the archive from a GitHub Release URL | The release was never published. The file lives in the Space repo, beside `app.py` |
| Bake the MovieLens download into the Space repo | That redistributes the dataset |

## Consequences

- A push of this GitHub repo does not include `space-weights.tar.gz`. The app can recommend after a local copy of that file is present, or after release `space-weights-v1` asset `space-weights.tar.gz` is served and its SHA-256 is `f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd`. That release is not created here. A missing file or a failed checksum shows `make space-snapshot`.
- On this VM the process that started, served one existing user, and served one new user peaked at 781,632 KiB RSS with item–item kept sparse. That number is not a Community Cloud measurement. The GroupLens zip is about 6 MB and `ratings.dat` is 24,594,131 bytes; the parameter archive is the larger download.
- `space/requirements.txt` pins the library at `23ab9515a3cbc0a18471d0fd36d03ae56b96a641`, the commit that already contains `download_dataset` and the serving loaders. Later library changes are not picked up until that pin moves.
- The GitHub README badge says Streamlit Community Cloud is pending. The app URL does not exist until the manual deploy.
- Demographic feature names can still appear in the in-memory explanation object. They are dropped only at display time.
