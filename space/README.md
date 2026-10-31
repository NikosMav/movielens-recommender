---
title: MovieLens recommender
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.65.0
app_file: app.py
python_version: "3.12"
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

`weights/` is not in the GitHub tree. From a checkout of [NikosMav/movielens-recommender](https://github.com/NikosMav/movielens-recommender), with the `rank`, `deep`, and `ui` extras installed:

```bash
make space-snapshot
```

That writes `space/weights/production/` and `space/weights/cold_start/`. The files are model parameters (similarity matrices, a torch state dict, LightGBM trees) plus the item-id column order. They do not include rating rows or titles. The command prints each file's size in bytes. It does not rewrite `results/`.

Without those files the app starts and shows the build command. It does not download MovieLens and it does not fit a model.

## Push this folder

Create a public Streamlit Space named `NikosMav/movielens-recommender` in the browser and choose the free CPU (`cpu-basic`). Hardware is set there. A `hardware` key in this file is ignored.

Then push **this directory** as that Space's repository root, after `make space-snapshot` so `weights/` is included. Authenticate with your Hugging Face account when Git asks. There is no deploy script and no token in this repo.

```bash
cd space
git init -b main
git add .
git commit -m "MovieLens recommender Space"
git remote add origin https://huggingface.co/spaces/NikosMav/movielens-recommender
git push -u origin main
```

`space/.gitignore` ignores a local download (`data/`) and the rebuilt runtime tables (`.runtime/`). It does not ignore `weights/`. If the Hub already created a starter commit, pull that commit into this repo before the push.

The link stays pending until that push: https://huggingface.co/spaces/NikosMav/movielens-recommender

Hugging Face deprecated the built-in Streamlit SDK on 2025-04-30. This file still sets `sdk: streamlit`, which is the SDK this demo is built for. If the Hub rejects it, the fallback is `sdk: docker` with `app_port: 8501` and the `Dockerfile` in this folder. That image listens on 8501 and its build fails if `weights/` is absent. Do not switch the committed front matter unless the Streamlit SDK is refused.

`requirements.txt` installs the library from GitHub at `23ab9515a3cbc0a18471d0fd36d03ae56b96a641`, plus CPU PyTorch 2.6.0, LightGBM 4.6.0, and Streamlit 1.65.0. `packages.txt` installs `libgomp1` for LightGBM.
