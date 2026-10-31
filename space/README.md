---
title: MovieLens recommender
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
license: mit
short_description: ml-1m ranker and new-user recommendations
---

# MovieLens recommender

Existing-user and new-user recommendations for MovieLens ml-1m. The page loads prebuilt model parameters. It does not train.

**Existing user.** Pick an ml-1m user. Candidates come from the validation-chosen retriever and LightGBM LambdaRank re-ranks them.

**New user.** Rate a few films. The list is the cold-start ranker. The page does not ask for gender, age, occupation, or ZIP, and it does not show those attributes as reasons.

MovieLens ml-1m is downloaded from [GroupLens](https://grouplens.org/datasets/movielens/) when the Space starts. This repository does not redistribute the ratings, the movie titles, or the user file. GroupLens and the University of Minnesota do not endorse this demo. Cite F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19. DOI [10.1145/2827872](https://doi.org/10.1145/2827872).

The GroupLens ml-1m README says the dataset may be used for research under these conditions: the user may not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group; the user must acknowledge the dataset; the user may not redistribute the data without separate permission; the user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.

## Weights

Model parameters live in `space-weights.tar.gz` next to `app.py`. At startup the app checks `WEIGHTS_SHA256` in `runtime.py` and extracts the archive into `weights/`. The archive holds item–item and EASE similarity, the two-tower state dict, LightGBM trees, and the item-id column order. It has no rating rows and no user ids. The page then downloads ml-1m from GroupLens and rebuilds the tables those parameters need. It does not train.

If that file is missing or the checksum fails, the page shows `make space-snapshot` and stops. It does not download MovieLens in that case. When the file is absent, the app may also try `hf_hub_download` from this public Space (`repo_type=space`) with no token.

`make space-snapshot`, from a checkout of [NikosMav/movielens-recommender](https://github.com/NikosMav/movielens-recommender) with the `rank`, `deep`, and `ui` extras, writes gitignored `space/weights/` and does not rewrite `results/`. Pack `production/` and `cold_start/` into `space-weights.tar.gz`. The person who pushes this folder copies that archive in. They do not need the extras and they do not need MovieLens data.

## Push this folder

Create a public Docker Space named `NikosMav/movielens-recommender` in the browser and choose the free CPU (`cpu-basic`). Hardware is set there. A `hardware` key in this file is ignored. The built-in Streamlit SDK was deprecated on 2025-04-30, so this Space uses `sdk: docker` and `app_port: 8501`. The `Dockerfile` listens on 8501, copies `space-weights.tar.gz`, and does not copy `weights/`.

From the GitHub repo root, copy the archive into this directory, then push **this directory** as the Space repository root. `space/.gitattributes` stores `*.tar.gz` in Git LFS. Authenticate with your Hugging Face account when Git asks. There is no deploy script and no token in this repo.

```bash
cp space-weights.tar.gz space/space-weights.tar.gz
cd space
git lfs install
git init -b main
git add .
git commit -m "MovieLens recommender Space"
git remote add origin https://huggingface.co/spaces/NikosMav/movielens-recommender
git push -u origin main
```

`space/.gitignore` ignores a local download (`data/`), rebuilt runtime tables (`.runtime/`), and the extracted `weights/` directory. It does not ignore `space-weights.tar.gz`. The GitHub repo ignores that archive so the file is not stored there. If the Hub already created a starter commit, pull that commit into this repo before the push.

The link stays pending until that push: https://huggingface.co/spaces/NikosMav/movielens-recommender

`requirements.txt` installs the library from GitHub at `23ab9515a3cbc0a18471d0fd36d03ae56b96a641`, plus CPU PyTorch 2.6.0, LightGBM 4.6.0, and Streamlit 1.65.0. The image installs `libgomp1` for LightGBM.
