# ADR-0010: Streamlit UI and explanations

## Status

Accepted (S5a).

## Context

S4 re-ranks a validation-chosen candidate set with LightGBM LambdaRank (ADR-0007). On ml-1m that set is two-tower. S4b kept demographic features because the pre-registered rule passed (`results/demographics/ml-1m.json`: +both 3-seed mean test NDCG@10 0.1334, within-run S4 features 0.1267, item–item cosine 0.1201 in `results/ml-1m.json`). ADR-0009 says the explainable UI must not present gender, age, occupation, or ZIP region as a user-facing reason. EASE and RP3beta do not beat the ranker on ml-1m NDCG@10 (ADR-0008), so they are not part of the demo.

The demo has to call the existing retrievers, feature builder, and `pred_contrib` path. It must not train, and it must not commit model binaries.

## Decision

**One Streamlit page** at `app/streamlit_app.py`, optional extra `pip install -e ".[ui]"` (`streamlit==1.65.0`). Together with the existing extras the install is `pip install -e ".[rank,deep,ui]"` plus the CPU PyTorch wheel. The page does not fit a model.

**Snapshot.** `movielens-recommender build-artifacts --config configs/ml-1m.yaml` fits the production pipeline and writes a gitignored `artifacts/<dataset>/` directory:

- manifest (`schema_version`, dataset SHA-256, config, git SHA, created time, candidate set, `best_iteration`)
- full-train item–item model and two-tower checkpoint
- refit LightGBM booster
- feature context (group-affinity from fit-train only, when demographics are on)
- movie metadata and the full-train rating histories

Hyperparameters are read from `results/tuning/<dataset>.json` and `results/tuning/two_tower_<dataset>.json`. They are not searched again. The candidate set and `best_iteration` are computed on the validation window, primary seed only. Ablations, bootstrap intervals, and the global-time cutoff are not part of this command. Test labels are not used. The app loads the directory once with `st.cache_resource`.

**Who can be recommended.** Existing users in the snapshot only. The quick profile was dropped.

The two-tower user tower is a user-id embedding plus the mean of history item embeddings (ADR-0006). `encode_user` returns nothing for an id that was not in the fit. Inventing an id vector, or dropping the id term and pooling the new ratings alone, would be a different model from the one in the results. The other path in the brief — item–item candidates plus the ranker with demographic columns removed — is also a different model: `configs/ml-1m.yaml` sets `demographics: "both"`, and the booster was trained with those columns. Shipping a second ranker just for a typed-in profile is more machinery than this page needs.

The page says that a new profile is not supported because the neural retriever has no embedding for a user it was not trained on.

**Recommendations.** Top 10 from the production path: the validation-chosen candidate set (K from config), scored by the refit ranker with `build_feature_matrix`. A checkbox shows the same user's top 10 from item–item cosine and from two-tower alone, with no ranker. History on the page is the full-train profile. The per-user test holdout is not shown and not fed to the model.

**Reasons.** For each production title, LightGBM `pred_contrib` is mapped to at most three sentences. Columns that share a sentence are summed. Only positive contributions count. Item–item score and rank become “You rated {title} {stars} and this is one of its nearest neighbours”. The named title is the history row with the largest `similarity × rating` on the row of `S` the item–item score uses (`S @ r`), so a weakly rated neighbour does not crowd out a film the user actually scored highly. Affinity for a genre becomes “Matches your liking for {Genre}”, and names a well-rated history title in that genre when one exists. Two-tower score and rank become “Ranked highly by the neural retriever for your taste profile”. Every `demo_*` and `group_*` column becomes the single sentence “Popular with viewers similar to you”. The main view does not show raw feature names and does not say SHAP. A collapsed Details block lists the largest raw contributions and the bias.

**Limits.** `pred_contrib` explains the ranker's score for that candidate. It is not a causal account of why the person would like the film, and the sentences are a fixed mapping of those numbers. A large user-activity contribution moves every candidate's score and still only appears when it is one of the top positive groups. Neighbour text needs a positive stored similarity; after neighbourhood truncation that can be absent, and the sentence then does not name a film.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Quick profile by pooling new ratings in the user tower | The tower adds a learned user-id embedding. There is no row for a new id. Omitting it changes the model |
| Quick profile via item–item plus a ranker with demographics stripped | That booster is not the production `both` model |
| Precompute every user's top 10 into the snapshot | Slower build, larger files, and a single user is cheap once the models are loaded |
| Show EASE or RP3beta in the comparison | ADR-0008: they do not beat the ranker on ml-1m NDCG@10 |
| Put gender, age, occupation, or region in the reason text | ADR-0009. Gain rank is not a reason to surface a sensitive attribute |
| Commit `artifacts/` | Same rule as `data/` and `models/`: binaries stay local |

## Consequences

- S5a is the Streamlit page, the `build-artifacts` command, and the reason templates.
- S5b (batch recommendations, FastAPI, Dockerfile) is still planned. S6 is still planned.
- CI installs the `ui` extra and runs the synthetic snapshot test plus an AppTest smoke test. It does not download MovieLens.
- The demo's own NDCG is not reported. Metrics stay in the committed results JSON.
