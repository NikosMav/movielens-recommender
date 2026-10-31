# MovieLens recommender

Existing-user and new-user recommendations for MovieLens ml-1m. The page loads prebuilt model parameters. It does not train.

**Existing user.** Pick an ml-1m user. Candidates come from the validation-chosen retriever and LightGBM LambdaRank re-ranks them.

**New user.** Rate a few films. The list is the cold-start ranker. The page does not ask for gender, age, occupation, or ZIP, and it does not show those attributes as reasons.

MovieLens ml-1m is downloaded from [GroupLens](https://grouplens.org/datasets/movielens/) when the app starts. This repository does not redistribute the ratings, the movie titles, or the user file. GroupLens and the University of Minnesota do not endorse this demo. Cite F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5, 4, Article 19. DOI [10.1145/2827872](https://doi.org/10.1145/2827872).

The GroupLens ml-1m README says the dataset may be used for research under these conditions: the user may not state or imply any endorsement from the University of Minnesota or the GroupLens Research Group; the user must acknowledge the dataset; the user may not redistribute the data without separate permission; the user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from a faculty member of the GroupLens Research Project at the University of Minnesota.

## Weights

The app looks first for `space-weights.tar.gz` next to `app.py`. If that file is absent, it downloads `WEIGHTS_URL` in `runtime.py` into `.runtime/cache/` and checks `WEIGHTS_SHA256`. The archive holds item–item and EASE similarity, the two-tower state dict, LightGBM trees, and the item-id column order. It has no rating rows and no user ids. The page then downloads ml-1m from GroupLens and rebuilds the tables those parameters need. It does not train.

If the file is missing or the checksum fails, the page shows `make space-snapshot` and stops. It does not download MovieLens in that case.

`make space-snapshot`, from a checkout of [NikosMav/movielens-recommender](https://github.com/NikosMav/movielens-recommender) with the `rank`, `deep`, and `ui` extras, writes gitignored `space/weights/` and does not rewrite `results/`. Pack `production/` and `cold_start/` into `space-weights.tar.gz`. This GitHub repo does not commit that archive. The release tag is `space-weights-v1` and the asset name is `space-weights.tar.gz`.

## Streamlit Community Cloud

Hugging Face Docker Spaces require a PRO account, so this app is deployed from this GitHub repo on Streamlit Community Cloud. The entrypoint is `space/app.py`. Community Cloud reads `requirements.txt` in this directory (it takes precedence over the repo root) and `packages.txt` at the repo root (`libgomp1` for LightGBM). Choose Python 3.12. The steps are in the repository README. The public link stays pending until that deploy.
