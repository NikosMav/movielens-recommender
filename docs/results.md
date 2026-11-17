# Results

What each experiment found, then every table behind it. The headline model comparison is on MovieLens 1M (ml-1m); the same harness also runs on ml-latest-small and MovieLens 32M. How the numbers are measured is in [evaluation.md](evaluation.md).

Back to the [README](../README.md).

## Findings

Every number behind these sentences is in the [tables below](#tables), generated from `results/*.json`.

- **ml-1m.** Tuned item–item kNN is a strong baseline. The two-tower model does not beat it on NDCG@10 (a recorded negative result), but it leads on Recall@100/200, catalog coverage, and the long tail. LambdaRank over two-tower candidates clears the item–item gate.
- **The served model (ml-1m).** A LambdaRank ranker over the union of pairwise-transition (PCTM) and two-tower candidates, with PCTM score and rank as features, scores NDCG@10 `0.2021`. The ranker served before it scores `0.1309` on the same run (paired `+0.0720 [+0.0668, +0.0776]`), and PCTM alone scores `0.1787` (paired `+0.0228 [+0.0203, +0.0253]`). Demographic features add nothing on top (`+0.0001 [-0.0013, +0.0018]`), so the served ranker has none. The pre-registered rule chose it, and Explore a MovieLens user now runs it (ADR-0033).
- **ml-32M.** The two-tower model and LambdaRank tie at the top and both beat item–item (ADR-0011).
- **Baselines.** EASE^R and RP3beta are tuned on the same validation split and compared in the results tables (ADR-0008). Adding Data-Adaptive Normalization / dropout-regularized EASE (`EASE-DAN`, Park et al. 2025 / Steck 2020) beats `EASE^R` on ml-1m (`0.1195` vs `0.1184`, paired 95% CI `[+0.0004, +0.0017]`) and closes the statistical gap to item–item cosine (`0.1201`) in a 2.4s closed-form solve (ADR-0028).
- **Sequence modeling (ml-1m).** Causal SASRec with full-catalog softmax beats a shuffled-order control (`0.0924` vs `0.0892`, paired `[+0.0024, +0.0052]`), confirming sequential order carries a small signal (+4% relative), while trailing bag-of-history retrieval and LambdaRank on MovieLens ratings (ADR-0019). Replacing the blocks with LiGR and SwiGLU (`eSASRec`, RecSys 2025) scores below that SASRec reference (paired primary-seed `-0.0134`, interval excludes 0). A pairwise transition probe (`PCTM`, RecSys 2026) scores `0.1787` against the refit SASRec `0.0929` (paired `+0.0857 [+0.0796, +0.0917]`) and covers `71.1%` of the catalog at 10, against `3.3%` for SASRec. The enhanced transformer stays below that pairwise probe (ADR-0030).
- **Discovery slider.** On the 604 held-out cold-start users with 5 ratings, Familiar keeps the highest NDCG@10 (`0.4095`). Closer costs about `0.006` NDCG@10 and raises coverage from `1.9%` to `2.7%`. In between raises coverage to `21.1%` and long-tail share to `20.0%` (ADR-0032). Familiar stays the default.
- **Third retriever (ml-1m).** A balanced union of item–item, two-tower, and EASE-DAN wins validation Recall@200 (`0.6607` vs `0.6552` for two-tower alone). Inside that same K=200 budget the demographics-on ranker then scores lower: three-seed NDCG@10 `0.1291` vs `0.1327`, paired primary-seed `-0.0032 [-0.0061, -0.0003]` (excludes 0). Demographics off, the paired interval includes 0 and the three-seed mean is lower. Letting the union grow past 200 candidates (mean size 301) returns the demographics-on mean to `0.1325`, and that paired interval includes 0. All three pre-registered lifts fail. EASE-DAN score and rank are still among the strongest split features (ADR-0031).
- **A larger Discover catalogue (ml-32M, not shipped).** 8,769 films through 2018 can be served from MovieLens 32M ratings with titles from the 1 MB `ml-latest-small` download. Two registered attempts at a model small enough for the 1 GB server both failed their ship rule. Item–item fold-in loses to most-popular at every profile size (N=5: `0.2898` vs `0.3149`, paired `-0.0251 [-0.0326, -0.0176]`; ADR-0034). Blending similarity with popularity, on new test users, wins at 5 and 10 ratings (`+0.0134 [+0.0077, +0.0198]` and `+0.0231 [+0.0176, +0.0291]`) but not at 3 (`+0.0033 [-0.0028, +0.0096]`), and the rule needed all three (ADR-0035). Discover still serves ml-1m.
- **Demographics (ml-1m).** User age, gender, and occupation features passed their pre-registered rule, so `configs/ml-1m.yaml` turns them on for the two-tower ranker (ADR-0009). The headline LambdaRank row in `results/ml-1m.json` is still the S4 feature set. The ranker served since ADR-0033 does not use them: with watch order in the model they no longer help.
- **New users.** Someone with no MovieLens id rates a few films and gets an explained top 10. The cold-start panel reports quality after 1, 3, 5, and 10 ratings, next to known users and popularity (ADR-0012). On the main protocol the list beats popularity from 3 ratings on; after a single rating it cannot be told apart from it.
- **Reproducibility.** Top-n selection used to break ties differently on different CPUs, which moved the new-user results between machines; ties now always go to the lower item index. Small floating-point differences between math-library builds still change the new-user ranker's details across platforms, so its results are regenerated in the Docker image and record the host they ran on (ADR-0015).
- **Other two-tower losses (S3e).** On ml-1m, full softmax and sampled softmax both beat the in-batch two-tower on test NDCG@10. LambdaRank on the validation-chosen new tower is worse than LambdaRank on the current two-tower, and worse than the new tower's own list (both paired intervals exclude 0). On ml-32M, sampled softmax does not beat the in-batch two-tower on validation, even at the same epoch budget, and full softmax was skipped for compute. The app keeps the current model (ADR-0013).
- **Serving (S5b).** The same snapshots run behind a FastAPI service, a batch scorer, and a Docker image. Every latency and batch budget fixed in advance was met on the measuring host, and indexing the snapshot tables at load time more than halved the explained known-user latency (ADR-0014).
- **Drift and refresh (S6).** Replaying ml-1m month by month, a popularity, item–item, or EASE model trained once in mid-2000 scores as well as one retrained every period, for two and a half years. ml-1m adds almost no new titles after 2000, and the item-mix drift signal never rises clearly above its own sampling noise, so the pre-registered rule chose never to retrain (ADR-0016).

## Tables

The tables are generated by `python scripts/make_results_table.py` from the committed `results/*.json` files. Never edit their numbers by hand.

<!-- BEGIN RESULTS TABLE -->

### `ml-1m` (from `results/ml-1m.json`)

Pinned version: `ml-1m@sha256:a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.

Split: min_ratings=5, test_fraction=0.2, val_fraction=0.1, relevance_threshold=4.0, seed=42, ks=[10, 20], retrieval_ks=[100, 200], bootstrap=1000 @ alpha=0.05. Primary metric: **ndcg@10**. Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003). Names marked **(tuned)** used validation-selected hyperparameters (ADR-0005 / ADR-0006 / ADR-0007); others are S2 YAML defaults.

| model | ndcg@10 | precision@10 | recall@10 | ndcg@20 | precision@20 | recall@20 | coverage@10 | mean_popularity@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0.0686 | 0.0564 | 0.0598 | 0.0847 | 0.0542 | 0.1049 | 0.5198 | 956.1292 |
| als_tuned | 0.0906 | 0.0740 | 0.0715 | 0.1072 | 0.0677 | 0.1257 | 0.4541 | 1118.0085 |
| ease (tuned) | 0.1184 | 0.0959 | 0.0861 | 0.1322 | 0.0839 | 0.1473 | 0.2100 | 1575.2565 |
| item_item_cosine | 0.1201 | 0.0980 | 0.0786 | 0.1284 | 0.0836 | 0.1277 | 0.1274 | 1794.5672 |
| item_item_cosine_tuned | 0.1192 | 0.0983 | 0.0808 | 0.1313 | 0.0861 | 0.1384 | 0.1721 | 1634.2176 |
| lambdarank (tuned) | 0.1273 | 0.1031 | 0.0928 | 0.1436 | 0.0917 | 0.1591 | 0.3895 | 1314.6790 |
| lambdarank_drop_retriever_features (tuned) | 0.1000 | 0.0860 | 0.0622 | 0.1087 | 0.0761 | 0.1101 | 0.3736 | 1578.6552 |
| lambdarank_item_item (tuned) | 0.1264 | 0.1032 | 0.0909 | 0.1421 | 0.0910 | 0.1572 | 0.3270 | 1304.3512 |
| lambdarank_two_tower (tuned) | 0.1288 | 0.1051 | 0.0943 | 0.1445 | 0.0922 | 0.1607 | 0.3763 | 1318.6541 |
| lambdarank_union_balanced (tuned) | 0.1245 | 0.1019 | 0.0921 | 0.1412 | 0.0911 | 0.1576 | 0.3744 | 1309.8450 |
| most_popular | 0.0895 | 0.0787 | 0.0466 | 0.0951 | 0.0698 | 0.0874 | 0.0325 | 2325.9361 |
| no_ranker | 0.1191 | 0.0972 | 0.0897 | 0.1360 | 0.0872 | 0.1535 | 0.4669 | 1175.1585 |
| rp3beta (tuned) | 0.1138 | 0.0934 | 0.0782 | 0.1255 | 0.0818 | 0.1334 | 0.1756 | 1750.1566 |
| two_tower (tuned) | 0.1192 | 0.0966 | 0.0901 | 0.1362 | 0.0865 | 0.1541 | 0.4723 | 1167.6311 |

95% bootstrap CIs (NDCG@10):

| model | NDCG@10 point summary | interval seed | run point and 95% user CI |
| --- | --- | --- | --- |
| als | 0.0686 | single run | 0.0686 [0.0658, 0.0714] |
| als_tuned | 0.0906 | single run | 0.0906 [0.0874, 0.0940] |
| ease (tuned) | 0.1184 | single run | 0.1184 [0.1142, 0.1226] |
| item_item_cosine | 0.1201 | single run | 0.1201 [0.1158, 0.1242] |
| item_item_cosine_tuned | 0.1192 | single run | 0.1192 [0.1147, 0.1233] |
| lambdarank (tuned) | mean±std over seeds 0.1273±0.0013 | 42 | 0.1288 [0.1247, 0.1329] |
| lambdarank_drop_retriever_features (tuned) | 0.1000 | single run | 0.1000 [0.0964, 0.1037] |
| lambdarank_item_item (tuned) | 0.1264 | single run | 0.1264 [0.1225, 0.1306] |
| lambdarank_two_tower (tuned) | 0.1288 | single run | 0.1288 [0.1247, 0.1329] |
| lambdarank_union_balanced (tuned) | 0.1245 | single run | 0.1245 [0.1201, 0.1286] |
| most_popular | 0.0895 | single run | 0.0895 [0.0857, 0.0935] |
| no_ranker | 0.1191 | single run | 0.1191 [0.1150, 0.1230] |
| rp3beta (tuned) | 0.1138 | single run | 0.1138 [0.1095, 0.1178] |
| two_tower (tuned) | mean±std over seeds 0.1192±0.0001 | 42 | 0.1191 [0.1150, 0.1230] |

#### Retrieval recall (candidate generation)

Recall@100 / Recall@200 for models that report them, evaluated at the same cutoffs. This is test-list recall, not the validation candidate-set recall in the ranker section.

| model | recall@100 | recall@200 |
| --- | --- | --- |
| als | 0.3609 | 0.5302 |
| als_tuned | 0.3638 | 0.5014 |
| ease (tuned) | 0.4314 | 0.6003 |
| item_item_cosine | 0.3471 | 0.4989 |
| item_item_cosine_tuned | 0.3828 | 0.5414 |
| lambdarank (tuned) | 0.4481 ±0.0011 | 0.6025 ±0.0000 |
| lambdarank_drop_retriever_features (tuned) | 0.4004 | 0.6025 |
| lambdarank_item_item (tuned) | 0.4286 | 0.5414 |
| lambdarank_two_tower (tuned) | 0.4492 | 0.6025 |
| lambdarank_union_balanced (tuned) | 0.4468 | 0.5971 |
| most_popular | 0.2599 | 0.3961 |
| no_ranker | 0.4353 | 0.6025 |
| rp3beta (tuned) | 0.3723 | 0.5258 |
| two_tower (tuned) | 0.4354 ±0.0003 | 0.6037 ±0.0014 |

#### Two-tower seeds and gate (ADR-0006)

Chosen hyperparams: `{'batch_size': 1024, 'embedding_dim': 64, 'learning_rate': 0.003, 'max_epochs': 20, 'max_history': 50, 'patience': 3, 'temperature': 0.1, 'weight_decay': 0.0001}` (val NDCG@10=0.0826; early-stopping best_epoch=6; Refit used fixed epoch count = best validation epoch (6).). Tuning log: [`results/tuning/two_tower_ml-1m.json`](../results/tuning/two_tower_ml-1m.json).

| seed | ndcg@10 | ndcg@10 CI | recall@100 | recall@200 |
| --- | --- | --- | --- | --- |
| 42 | 0.1191 | [0.1150, 0.1230] | 0.4353 | 0.6025 |
| 43 | 0.1193 | [0.1154, 0.1233] | 0.4350 | 0.6028 |
| 44 | 0.1194 | [0.1156, 0.1236] | 0.4358 | 0.6056 |

Across seeds: NDCG@10 mean=0.1192, std=0.0001, min=0.1191, max=0.1194.

**Gate: negative result.** Two-tower does **not** beat both item–item bars on ml-1m-style NDCG@10 with CIs taken into account (see ADR-0006). The retriever or candidate set will be chosen in S4 by validation recall.

#### Ranker candidates, seeds, and gate (ADR-0007)

Candidate budget K=200. Selection metric: validation recall@200 (split: validation). Tie-break: recall@200, then recall@100, then union_balanced, item_item, two_tower, union_unbalanced. Winner: `two_tower`.

| candidate set | recall@100 | recall@200 | mean size |
| --- | --- | --- | --- |
| item_item | 0.435321 | 0.593272 | 200.000000 |
| two_tower | 0.489363 | 0.655227 | 200.000000 |
| union_balanced | 0.485470 | 0.653053 | 200.000000 |
| union_unbalanced | 0.485470 | 0.651810 | 280.720057 |

Early-stop user fraction=0.2 (split seed=42). Ranker seeds share candidate sets and retriever models.

| seed | best_iteration | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 42 | 95 | 0.1288 | [0.1247, 0.1329] | 0.0943 | 0.3763 |
| 43 | 94 | 0.1274 | [0.1232, 0.1316] | 0.0917 | 0.3919 |
| 44 | 43 | 0.1256 | [0.1213, 0.1301] | 0.0925 | 0.4003 |

Across seeds: NDCG@10 mean=0.1273, std=0.0013, min=0.1256, max=0.1288.

**Fixed item-item baseline: point estimate exceeded.** LambdaRank mean NDCG@10=0.1273 vs item_item_cosine 0.1201 [0.1158, 0.1242] (read from `metrics['item_item_cosine']['ndcg@10']`).

Seed NDCG@10 intervals are not all above the bar CI high (`all_seed_ci_low_above_bar_ci_high=False`).

Descriptive test-point comparison: strongest recorded prior model `item_item_cosine`=0.1201; ranker minus prior=+0.0071. Exceeds prior point: `True`. This comparison does not select a model or establish significance.
Unranked candidate-list point=0.1191; ranker minus unranked=+0.0082. Exceeds both points: `True`. The historical fixed-baseline gate above is a separate comparison.
A paired interval against the unranked list is unavailable in this record.

Ablations are the primary seed, except `ndcg@10`, `recall@10`, and `coverage@10` on the `lambdarank` row, which are means over the three ranker seeds. The `ndcg@10` CI and tail NDCG@10 on that row stay the primary seed; that CI is the primary-seed user bootstrap, not a confidence interval for the 3-seed mean. `no_ranker` keeps the winning candidate order. `lambdarank_drop_retriever_features` drops retriever score and rank.

| ablation | ndcg@10 point | interval run point and 95% CI | recall@10 | coverage@10 | tail ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| lambdarank | 0.1273 | 0.1288 [0.1247, 0.1329] | 0.0928 | 0.3895 | 0.0915 [0.0868, 0.0969] |
| lambdarank_drop_retriever_features | 0.1000 | 0.1000 [0.0964, 0.1037] | 0.0622 | 0.3736 | 0.0732 [0.0691, 0.0777] |
| lambdarank_item_item | 0.1264 | 0.1264 [0.1225, 0.1306] | 0.0909 | 0.3270 | 0.0492 [0.0456, 0.0533] |
| lambdarank_two_tower | 0.1288 | 0.1288 [0.1247, 0.1329] | 0.0943 | 0.3763 | 0.0915 [0.0868, 0.0969] |
| lambdarank_union_balanced | 0.1245 | 0.1245 [0.1201, 0.1286] | 0.0921 | 0.3744 | 0.0907 [0.0860, 0.0960] |
| no_ranker | 0.1191 | 0.1191 [0.1150, 0.1230] | 0.0897 | 0.4669 | 0.0825 [0.0782, 0.0873] |

Top feature gains (refit ranker, primary seed):

| feature | gain |
| --- | --- |
| two_tower_rank | 12083.8040 |
| item_item_rank | 5382.0176 |
| item_item_score | 4988.1077 |
| item_popularity | 4535.2138 |
| user_n_ratings | 4488.3243 |
| two_tower_score | 3534.1110 |
| item_recency | 2220.1169 |
| affinity_romance | 2140.1855 |
| user_mean_rating | 1925.5218 |
| affinity_comedy | 1850.7273 |

Refit ranker artifact (gitignored): `models/ml-1m/ranker.txt` schema_version=1. Recreate with `movielens-recommender run --config configs/ml-1m.yaml`.

Ranker stage runtime: 335.6810s.

#### Segment NDCG@10

User activity = train rating-count terciles (low/mid/high). Item head = top 20% of train items by popularity; tail = rest. Item-segment metrics restrict **relevant and recommended** items to the segment (users with no relevant items in-segment are excluded).

| model | activity low | activity mid | activity high | item head | item tail |
| --- | --- | --- | --- | --- | --- |
| als | 0.0747 [0.0689, 0.0803] | 0.0591 [0.0549, 0.0635] | 0.0721 [0.0675, 0.0770] | 0.0832 [0.0797, 0.0865] | 0.0600 [0.0561, 0.0638] |
| als_tuned | 0.0893 [0.0823, 0.0960] | 0.0800 [0.0748, 0.0849] | 0.1026 [0.0971, 0.1080] | 0.1034 [0.0993, 0.1074] | 0.0660 [0.0623, 0.0701] |
| ease (tuned) | 0.1017 [0.0943, 0.1092] | 0.0904 [0.0847, 0.0959] | 0.1630 [0.1552, 0.1711] | 0.1314 [0.1269, 0.1361] | 0.0743 [0.0699, 0.0788] |
| item_item_cosine | 0.0928 [0.0856, 0.1001] | 0.0943 [0.0881, 0.1002] | 0.1734 [0.1654, 0.1817] | 0.1311 [0.1265, 0.1358] | 0.0321 [0.0292, 0.0354] |
| item_item_cosine_tuned | 0.0917 [0.0844, 0.0988] | 0.0920 [0.0860, 0.0975] | 0.1740 [0.1655, 0.1826] | 0.1307 [0.1260, 0.1352] | 0.0467 [0.0432, 0.0509] |
| lambdarank (tuned) | 0.1165 [0.1092, 0.1242] | 0.0954 [0.0898, 0.1007] | 0.1745 [0.1657, 0.1829] | 0.1425 [0.1376, 0.1471] | 0.0915 [0.0868, 0.0969] |
| lambdarank_drop_retriever_features (tuned) | 0.0608 [0.0558, 0.0658] | 0.0759 [0.0709, 0.0814] | 0.1633 [0.1555, 0.1719] | 0.1074 [0.1035, 0.1114] | 0.0732 [0.0691, 0.0777] |
| lambdarank_item_item (tuned) | 0.1102 [0.1033, 0.1174] | 0.0958 [0.0902, 0.1009] | 0.1731 [0.1651, 0.1809] | 0.1395 [0.1347, 0.1439] | 0.0492 [0.0456, 0.0533] |
| lambdarank_two_tower (tuned) | 0.1165 [0.1092, 0.1242] | 0.0954 [0.0898, 0.1007] | 0.1745 [0.1657, 0.1829] | 0.1425 [0.1376, 0.1471] | 0.0915 [0.0868, 0.0969] |
| lambdarank_union_balanced (tuned) | 0.1107 [0.1033, 0.1182] | 0.0928 [0.0875, 0.0983] | 0.1701 [0.1617, 0.1778] | 0.1374 [0.1331, 0.1419] | 0.0907 [0.0860, 0.0960] |
| most_popular | 0.0395 [0.0351, 0.0440] | 0.0677 [0.0622, 0.0731] | 0.1614 [0.1530, 0.1710] | 0.0946 [0.0908, 0.0988] | 0.0000 [0.0000, 0.0001] |
| no_ranker | 0.1089 [0.1014, 0.1166] | 0.0932 [0.0874, 0.0990] | 0.1551 [0.1475, 0.1629] | 0.1364 [0.1319, 0.1407] | 0.0825 [0.0782, 0.0873] |
| rp3beta (tuned) | 0.0914 [0.0846, 0.0986] | 0.0885 [0.0824, 0.0938] | 0.1616 [0.1538, 0.1695] | 0.1243 [0.1197, 0.1287] | 0.0391 [0.0357, 0.0428] |
| two_tower (tuned) | 0.1089 [0.1014, 0.1166] | 0.0932 [0.0874, 0.0990] | 0.1551 [0.1475, 0.1629] | 0.1364 [0.1319, 0.1407] | 0.0825 [0.0782, 0.0873] |

Tuning log: [`results/tuning/ml-1m.json`](../results/tuning/ml-1m.json). Chosen ALS={'alpha': 20.0, 'factors': 128, 'iterations': 15, 'regularization': 0.1} (val NDCG@10=0.0646); item–item={'k_neighbors': 200, 'min_common': 1, 'shrinkage': 100.0} (val NDCG@10=0.0810). EASE={'l2': 5000.0} (val NDCG@10=0.0808); RP3beta={'alpha': 0.4, 'beta': 0.6, 'top_k': 2000} (val NDCG@10=0.0810). Two-tower={'batch_size': 1024, 'embedding_dim': 64, 'learning_rate': 0.003, 'max_epochs': 20, 'max_history': 50, 'patience': 3, 'temperature': 0.1, 'weight_decay': 0.0001} (val NDCG@10=0.0826, best_epoch=6; log [`results/tuning/two_tower_ml-1m.json`](../results/tuning/two_tower_ml-1m.json)).

Pipeline runtime: 668.3570s.

EASE/RP3beta tune and test runtime (same harness, not included in the pipeline runtime above): 536.9470s.

### `ml-32m` (from `results/ml-32m.json`)

Pinned version: `ml-32m@sha256:e4a68655d7386b8f95f2f2424b2ff975dfdd15ffd59e0d864a14dca43e99d6ee`.

Split: min_ratings=5, test_fraction=0.2, val_fraction=0.1, relevance_threshold=4.0, seed=42, ks=[10, 20], retrieval_ks=[100, 200], bootstrap=1000 @ alpha=0.05. Primary metric: **ndcg@10**. Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003). Names marked **(tuned)** used validation-selected hyperparameters (ADR-0005 / ADR-0006 / ADR-0007); others are S2 YAML defaults.

Evaluation users: seeded sample of 8000 out of 196517 warm-relevant users (seed=42, requested=8000, user_ids_sha256=`6b0d5cf6ce14cf5815e71f620b1dd5f37f7e6f09ac13ae5f3d138563b035c980`). 196,517 of 200,948 train users (97.8%) are eligible, the same eligibility rule the harness uses for test metrics (at least one test rating >= 4 on an item in the train catalog). The sample is partly test-informed: those users are chosen using their test ratings, and validation rows are restricted to the sample, so tuning and ranker training see users chosen partly by their test ratings. The expected effect is small. Drawing the sample without test ratings is the cleaner alternative and is a known limitation. Training uses every training interaction. Sampled users keep full histories. Tuning selection and ranker labels use this same sample.

EASE is restricted to the top 12000 items by train-interaction count (n_items_before=71364, n_items_fit=12000; ties: smaller item id). Items outside that head are not scored.

| model | ndcg@10 | precision@10 | recall@10 | ndcg@20 | precision@20 | recall@20 | coverage@10 | mean_popularity@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0.0800 | 0.0619 | 0.0758 | 0.1026 | 0.0592 | 0.1394 | 0.0497 | 24699.2246 |
| als_tuned | 0.0983 | 0.0765 | 0.0884 | 0.1204 | 0.0697 | 0.1561 | 0.0427 | 27539.4218 |
| ease (tuned) | 0.1243 | 0.0958 | 0.1010 | 0.1408 | 0.0805 | 0.1696 | 0.0255 | 36463.1416 |
| item_item_cosine | 0.1113 | 0.0853 | 0.0873 | 0.1269 | 0.0735 | 0.1493 | 0.0308 | 36710.5165 |
| item_item_cosine_tuned | 0.1093 | 0.0850 | 0.0858 | 0.1253 | 0.0734 | 0.1492 | 0.0318 | 36712.2740 |
| lambdarank (tuned) | 0.1423 | 0.1098 | 0.1142 | 0.1613 | 0.0934 | 0.1895 | 0.0435 | 32026.6521 |
| lambdarank_drop_retriever_features (tuned) | 0.0979 | 0.0776 | 0.0757 | 0.1107 | 0.0687 | 0.1243 | 0.0209 | 55270.9751 |
| lambdarank_item_item (tuned) | 0.1431 | 0.1107 | 0.1135 | 0.1615 | 0.0934 | 0.1891 | 0.0436 | 31007.8203 |
| lambdarank_two_tower (tuned) | 0.1429 | 0.1101 | 0.1148 | 0.1617 | 0.0936 | 0.1899 | 0.0428 | 31965.3612 |
| lambdarank_union_balanced (tuned) | 0.1403 | 0.1094 | 0.1134 | 0.1603 | 0.0934 | 0.1917 | 0.0430 | 31660.7558 |
| most_popular | 0.0699 | 0.0539 | 0.0515 | 0.0779 | 0.0458 | 0.0879 | 0.0021 | 70474.0101 |
| no_ranker | 0.1436 | 0.1103 | 0.1138 | 0.1608 | 0.0925 | 0.1872 | 0.0411 | 31974.7307 |
| rp3beta (tuned) | 0.0991 | 0.0766 | 0.0775 | 0.1143 | 0.0667 | 0.1368 | 0.0089 | 52004.6634 |
| two_tower (tuned) | 0.1436 | 0.1103 | 0.1138 | 0.1608 | 0.0925 | 0.1872 | 0.0411 | 31974.7307 |

95% bootstrap CIs (NDCG@10):

| model | NDCG@10 point summary | interval seed | run point and 95% user CI |
| --- | --- | --- | --- |
| als | 0.0800 | single run | 0.0800 [0.0774, 0.0830] |
| als_tuned | 0.0983 | single run | 0.0983 [0.0955, 0.1017] |
| ease (tuned) | 0.1243 | single run | 0.1243 [0.1209, 0.1282] |
| item_item_cosine | 0.1113 | single run | 0.1113 [0.1080, 0.1155] |
| item_item_cosine_tuned | 0.1093 | single run | 0.1093 [0.1061, 0.1132] |
| lambdarank (tuned) | mean±std over seeds 0.1423±0.0007 | 42 | 0.1429 [0.1391, 0.1472] |
| lambdarank_drop_retriever_features (tuned) | 0.0979 | single run | 0.0979 [0.0947, 0.1015] |
| lambdarank_item_item (tuned) | 0.1431 | single run | 0.1431 [0.1393, 0.1471] |
| lambdarank_two_tower (tuned) | 0.1429 | single run | 0.1429 [0.1391, 0.1472] |
| lambdarank_union_balanced (tuned) | 0.1403 | single run | 0.1403 [0.1367, 0.1444] |
| most_popular | 0.0699 | single run | 0.0699 [0.0671, 0.0728] |
| no_ranker | 0.1436 | single run | 0.1436 [0.1398, 0.1479] |
| rp3beta (tuned) | 0.0991 | single run | 0.0991 [0.0957, 0.1028] |
| two_tower (tuned) | 1 seed 0.1436 | 42 | 0.1436 [0.1398, 0.1479] |

#### Does the ranking hold at scale?

Two-tower 0.1436 [0.1398, 0.1479] and LambdaRank 3-seed mean 0.1423 (primary-seed CI [0.1391, 0.1472]) have overlapping individual intervals; this does not establish equivalence. Both beat item–item cosine 0.1113 [0.1080, 0.1155] with separated intervals, so the neural approach overtakes item–item at this scale, reversing the ml-1m result. LambdaRank adds no lift over its two-tower candidate list (`no_ranker` is also 0.1436). The two-tower's lead is on head items (head NDCG@10 0.1460 versus item–item 0.1114); its tail NDCG@10 (0.0006) is below item–item's (0.0022).

#### Retrieval recall (candidate generation)

Recall@100 / Recall@200 for models that report them, evaluated at the same cutoffs. This is test-list recall, not the validation candidate-set recall in the ranker section.

| model | recall@100 | recall@200 |
| --- | --- | --- |
| als | 0.4218 | 0.5802 |
| als_tuned | 0.4397 | 0.5919 |
| ease (tuned) | 0.4473 | 0.5961 |
| item_item_cosine | 0.4076 | 0.5541 |
| item_item_cosine_tuned | 0.4109 | 0.5589 |
| lambdarank (tuned) | 0.4845 ±0.0001 | 0.6243 ±0.0000 |
| lambdarank_drop_retriever_features (tuned) | 0.4148 | 0.6243 |
| lambdarank_item_item (tuned) | 0.4702 | 0.5589 |
| lambdarank_two_tower (tuned) | 0.4844 | 0.6243 |
| lambdarank_union_balanced (tuned) | 0.4834 | 0.6061 |
| most_popular | 0.2388 | 0.3462 |
| no_ranker | 0.4742 | 0.6243 |
| rp3beta (tuned) | 0.3629 | 0.4922 |
| two_tower (tuned) | 0.4742 (1 seed) | 0.6243 (1 seed) |

#### Two-tower seeds and gate (ADR-0006)

Chosen hyperparams: `{'batch_size': 1024, 'embedding_dim': 64, 'learning_rate': 0.001, 'max_epochs': 6, 'max_history': 50, 'patience': 2, 'temperature': 0.1, 'weight_decay': 0.0001}` (val NDCG@10=0.1008; early-stopping best_epoch=5; Refit used fixed epoch count = best validation epoch (5).). Tuning log: [`results/tuning/two_tower_ml-32m.json`](../results/tuning/two_tower_ml-32m.json).

| seed | ndcg@10 | ndcg@10 CI | recall@100 | recall@200 |
| --- | --- | --- | --- | --- |
| 42 | 0.1436 | [0.1398, 0.1479] | 0.4742 | 0.6243 |

Across seeds: NDCG@10 mean=0.1436, std=0.0000, min=0.1436, max=0.1436.

**Gate: win.** Two-tower beats both item–item default and tuned bars on NDCG@10 with CIs taken into account.

#### Ranker candidates, seeds, and gate (ADR-0007)

Candidate budget K=200. Selection metric: validation recall@200 (split: validation). Tie-break: recall@200, then recall@100, then union_balanced, item_item, two_tower, union_unbalanced. Winner: `two_tower`.

| candidate set | recall@100 | recall@200 | mean size |
| --- | --- | --- | --- |
| item_item | 0.454584 | 0.611488 | 200.000000 |
| two_tower | 0.528568 | 0.677032 | 200.000000 |
| union_balanced | 0.510032 | 0.662878 | 200.000000 |
| union_unbalanced | 0.510032 | 0.663260 | 269.830238 |

Early-stop user fraction=0.2 (split seed=42). Ranker seeds share candidate sets and retriever models.

| seed | best_iteration | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 42 | 33 | 0.1429 | [0.1391, 0.1472] | 0.1148 | 0.0428 |
| 43 | 85 | 0.1425 | [0.1388, 0.1468] | 0.1142 | 0.0443 |
| 44 | 20 | 0.1414 | [0.1376, 0.1456] | 0.1137 | 0.0434 |

Across seeds: NDCG@10 mean=0.1423, std=0.0007, min=0.1414, max=0.1429.

**Fixed item-item baseline: point estimate exceeded.** LambdaRank mean NDCG@10=0.1423 vs item_item_cosine 0.1113 [0.1080, 0.1155] (read from `metrics['item_item_cosine']['ndcg@10']`).

Every ranker-seed NDCG@10 CI low sits above the bar CI high.

Descriptive test-point comparison: strongest recorded prior model `two_tower`=0.1436; ranker minus prior=-0.0013. Exceeds prior point: `False`. This comparison does not select a model or establish significance.
Unranked candidate-list point=0.1436; ranker minus unranked=-0.0013. Exceeds both points: `False`. The historical fixed-baseline gate above is a separate comparison.
A paired interval against the unranked list is unavailable in this record.

Ablations are the primary seed, except `ndcg@10`, `recall@10`, and `coverage@10` on the `lambdarank` row, which are means over the three ranker seeds. The `ndcg@10` CI and tail NDCG@10 on that row stay the primary seed; that CI is the primary-seed user bootstrap, not a confidence interval for the 3-seed mean. `no_ranker` keeps the winning candidate order. `lambdarank_drop_retriever_features` drops retriever score and rank.

| ablation | ndcg@10 point | interval run point and 95% CI | recall@10 | coverage@10 | tail ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| lambdarank | 0.1423 | 0.1429 [0.1391, 0.1472] | 0.1142 | 0.0435 | 0.0009 [0.0000, 0.0024] |
| lambdarank_drop_retriever_features | 0.0979 | 0.0979 [0.0947, 0.1015] | 0.0757 | 0.0209 | 0.0009 [0.0000, 0.0024] |
| lambdarank_item_item | 0.1431 | 0.1431 [0.1393, 0.1471] | 0.1135 | 0.0436 | 0.0039 [0.0008, 0.0078] |
| lambdarank_two_tower | 0.1429 | 0.1429 [0.1391, 0.1472] | 0.1148 | 0.0428 | 0.0009 [0.0000, 0.0024] |
| lambdarank_union_balanced | 0.1403 | 0.1403 [0.1367, 0.1444] | 0.1134 | 0.0430 | 0.0011 [0.0000, 0.0032] |
| no_ranker | 0.1436 | 0.1436 [0.1398, 0.1479] | 0.1138 | 0.0411 | 0.0006 [0.0000, 0.0017] |

Top feature gains (refit ranker, primary seed):

| feature | gain |
| --- | --- |
| two_tower_rank | 15914.2058 |
| two_tower_score | 5808.6722 |
| user_n_ratings | 2605.8737 |
| item_popularity | 2357.3480 |
| item_item_score | 2000.1511 |
| affinity_thriller | 1694.5757 |
| item_year | 1575.1290 |
| item_recency | 1385.8150 |
| item_item_rank | 1209.6355 |
| affinity_musical | 911.5404 |

Refit ranker artifact (gitignored): `models/ml-32m/ranker.txt` schema_version=1. Recreate with `movielens-recommender run --config configs/ml-32m.yaml`.

Ranker stage runtime: 2841.7980s.

#### Segment NDCG@10

User activity = train rating-count terciles (low/mid/high). Item head = top 20% of train items by popularity; tail = rest. Item-segment metrics restrict **relevant and recommended** items to the segment (users with no relevant items in-segment are excluded).

| model | activity low | activity mid | activity high | item head | item tail |
| --- | --- | --- | --- | --- | --- |
| als | 0.0878 [0.0815, 0.0940] | 0.0695 [0.0653, 0.0738] | 0.0826 [0.0784, 0.0870] | 0.0801 [0.0772, 0.0831] | 0.0000 [0.0000, 0.0000] |
| als_tuned | 0.1014 [0.0949, 0.1076] | 0.0836 [0.0786, 0.0884] | 0.1099 [0.1050, 0.1149] | 0.0985 [0.0953, 0.1016] | 0.0000 [0.0000, 0.0000] |
| ease (tuned) | 0.1110 [0.1040, 0.1178] | 0.0982 [0.0933, 0.1036] | 0.1637 [0.1568, 0.1714] | 0.1245 [0.1207, 0.1284] | 0.0000 [0.0000, 0.0000] |
| item_item_cosine | 0.0963 [0.0901, 0.1033] | 0.0890 [0.0838, 0.0942] | 0.1486 [0.1416, 0.1555] | 0.1114 [0.1077, 0.1151] | 0.0022 [0.0003, 0.0048] |
| item_item_cosine_tuned | 0.0930 [0.0870, 0.0996] | 0.0867 [0.0820, 0.0919] | 0.1483 [0.1412, 0.1556] | 0.1095 [0.1060, 0.1132] | 0.0033 [0.0006, 0.0066] |
| lambdarank (tuned) | 0.1228 [0.1160, 0.1300] | 0.1162 [0.1108, 0.1217] | 0.1898 [0.1829, 0.1974] | 0.1432 [0.1393, 0.1473] | 0.0009 [0.0000, 0.0024] |
| lambdarank_drop_retriever_features (tuned) | 0.0796 [0.0739, 0.0854] | 0.0764 [0.0721, 0.0809] | 0.1378 [0.1314, 0.1446] | 0.0981 [0.0948, 0.1015] | 0.0009 [0.0000, 0.0024] |
| lambdarank_item_item (tuned) | 0.1223 [0.1148, 0.1299] | 0.1186 [0.1128, 0.1245] | 0.1883 [0.1811, 0.1953] | 0.1433 [0.1395, 0.1476] | 0.0039 [0.0008, 0.0078] |
| lambdarank_two_tower (tuned) | 0.1228 [0.1160, 0.1300] | 0.1162 [0.1108, 0.1217] | 0.1898 [0.1829, 0.1974] | 0.1432 [0.1393, 0.1473] | 0.0009 [0.0000, 0.0024] |
| lambdarank_union_balanced (tuned) | 0.1206 [0.1132, 0.1272] | 0.1131 [0.1072, 0.1188] | 0.1873 [0.1802, 0.1945] | 0.1406 [0.1370, 0.1448] | 0.0011 [0.0000, 0.0032] |
| most_popular | 0.0567 [0.0515, 0.0621] | 0.0527 [0.0488, 0.0570] | 0.1005 [0.0946, 0.1064] | 0.0701 [0.0673, 0.0729] | 0.0000 [0.0000, 0.0000] |
| no_ranker | 0.1220 [0.1149, 0.1294] | 0.1166 [0.1110, 0.1227] | 0.1922 [0.1851, 0.1999] | 0.1460 [0.1420, 0.1504] | 0.0006 [0.0000, 0.0017] |
| rp3beta (tuned) | 0.0818 [0.0758, 0.0880] | 0.0806 [0.0754, 0.0859] | 0.1348 [0.1280, 0.1420] | 0.0992 [0.0957, 0.1029] | 0.0049 [0.0004, 0.0110] |
| two_tower (tuned) | 0.1220 [0.1149, 0.1294] | 0.1166 [0.1110, 0.1227] | 0.1922 [0.1851, 0.1999] | 0.1460 [0.1420, 0.1504] | 0.0006 [0.0000, 0.0017] |

Tuning log: [`results/tuning/ml-32m.json`](../results/tuning/ml-32m.json). Chosen ALS={'alpha': 20.0, 'factors': 64, 'iterations': 15, 'regularization': 0.01} (val NDCG@10=0.0721); item–item={'k_neighbors': 50, 'min_common': 1, 'shrinkage': 0.0} (val NDCG@10=0.0789). EASE={'l2': 5000.0, 'max_items': 12000} (val NDCG@10=0.0866); RP3beta={'alpha': 0.5, 'beta': 0.5, 'top_k': 300} (val NDCG@10=0.0693). Two-tower={'batch_size': 1024, 'embedding_dim': 64, 'learning_rate': 0.001, 'max_epochs': 6, 'max_history': 50, 'patience': 2, 'temperature': 0.1, 'weight_decay': 0.0001} (val NDCG@10=0.1008, best_epoch=5; log [`results/tuning/two_tower_ml-32m.json`](../results/tuning/two_tower_ml-32m.json)).

Pipeline runtime: 6560.5750s.

### `ml-latest-small` (from `results/ml-latest-small.json`)

Pinned version: `ml-latest-small@sha256:696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`.

Split: min_ratings=5, test_fraction=0.2, val_fraction=0.1, relevance_threshold=4.0, seed=42, ks=[10, 20], retrieval_ks=[100, 200], bootstrap=1000 @ alpha=0.05. Primary metric: **ndcg@10**. Coverage@k is a point estimate only (no user-bootstrap CI; see ADR-0003). Names marked **(tuned)** used validation-selected hyperparameters (ADR-0005 / ADR-0006 / ADR-0007); others are S2 YAML defaults.

| model | ndcg@10 | precision@10 | recall@10 | ndcg@20 | precision@20 | recall@20 | coverage@10 | mean_popularity@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| als | 0.0795 | 0.0622 | 0.0693 | 0.0948 | 0.0531 | 0.1268 | 0.0965 | 108.2937 |
| als_tuned | 0.0875 | 0.0653 | 0.0764 | 0.1013 | 0.0553 | 0.1284 | 0.1119 | 96.1963 |
| ease (tuned) | 0.1022 | 0.0803 | 0.0855 | 0.1172 | 0.0674 | 0.1458 | 0.0618 | 146.4739 |
| item_item_cosine | 0.0899 | 0.0707 | 0.0771 | 0.1017 | 0.0583 | 0.1266 | 0.0565 | 122.4519 |
| item_item_cosine_tuned | 0.0997 | 0.0763 | 0.0774 | 0.1150 | 0.0674 | 0.1354 | 0.0485 | 160.4564 |
| lambdarank (tuned) | 0.0990 | 0.0785 | 0.0838 | 0.1151 | 0.0682 | 0.1402 | 0.0972 | 114.9737 |
| lambdarank_drop_retriever_features (tuned) | 0.0838 | 0.0644 | 0.0647 | 0.1003 | 0.0581 | 0.1258 | 0.1157 | 121.0568 |
| lambdarank_item_item (tuned) | 0.1071 | 0.0814 | 0.0899 | 0.1262 | 0.0719 | 0.1563 | 0.0924 | 118.4963 |
| lambdarank_two_tower (tuned) | 0.0933 | 0.0703 | 0.0766 | 0.1064 | 0.0594 | 0.1290 | 0.1003 | 129.1536 |
| lambdarank_union_balanced (tuned) | 0.1218 | 0.0890 | 0.0975 | 0.1355 | 0.0737 | 0.1539 | 0.0878 | 120.0193 |
| lambdarank_union_unbalanced (tuned) | 0.0852 | 0.0715 | 0.0692 | 0.1018 | 0.0646 | 0.1243 | 0.0999 | 113.1759 |
| most_popular | 0.0743 | 0.0563 | 0.0514 | 0.0802 | 0.0466 | 0.0859 | 0.0119 | 216.7525 |
| no_ranker | 0.0983 | 0.0714 | 0.0803 | 0.1156 | 0.0628 | 0.1415 | 0.1054 | 132.9802 |
| rp3beta (tuned) | 0.0858 | 0.0690 | 0.0674 | 0.0986 | 0.0591 | 0.1214 | 0.0220 | 187.9454 |
| two_tower (tuned) | 0.0910 | 0.0675 | 0.0869 | 0.1105 | 0.0589 | 0.1487 | 0.1396 | 98.8950 |

95% bootstrap CIs (NDCG@10):

| model | NDCG@10 point summary | interval seed | run point and 95% user CI |
| --- | --- | --- | --- |
| als | 0.0795 | single run | 0.0795 [0.0692, 0.0902] |
| als_tuned | 0.0875 | single run | 0.0875 [0.0768, 0.0982] |
| ease (tuned) | 0.1022 | single run | 0.1022 [0.0898, 0.1151] |
| item_item_cosine | 0.0899 | single run | 0.0899 [0.0784, 0.1011] |
| item_item_cosine_tuned | 0.0997 | single run | 0.0997 [0.0867, 0.1130] |
| lambdarank (tuned) | mean±std over seeds 0.0990±0.0114 | 42 | 0.0852 [0.0734, 0.0968] |
| lambdarank_drop_retriever_features (tuned) | 0.0838 | single run | 0.0838 [0.0722, 0.0960] |
| lambdarank_item_item (tuned) | 0.1071 | single run | 0.1071 [0.0940, 0.1199] |
| lambdarank_two_tower (tuned) | 0.0933 | single run | 0.0933 [0.0813, 0.1057] |
| lambdarank_union_balanced (tuned) | 0.1218 | single run | 0.1218 [0.1078, 0.1354] |
| lambdarank_union_unbalanced (tuned) | 0.0852 | single run | 0.0852 [0.0734, 0.0968] |
| most_popular | 0.0743 | single run | 0.0743 [0.0636, 0.0860] |
| no_ranker | 0.0983 | single run | 0.0983 [0.0860, 0.1110] |
| rp3beta (tuned) | 0.0858 | single run | 0.0858 [0.0735, 0.0985] |
| two_tower (tuned) | mean±std over seeds 0.0910±0.0023 | 42 | 0.0890 [0.0777, 0.1013] |

#### Retrieval recall (candidate generation)

Recall@100 / Recall@200 for models that report them, evaluated at the same cutoffs. This is test-list recall, not the validation candidate-set recall in the ranker section.

| model | recall@100 | recall@200 |
| --- | --- | --- |
| als | 0.3265 | 0.4272 |
| als_tuned | 0.2949 | 0.3848 |
| ease (tuned) | 0.4033 | 0.5426 |
| item_item_cosine | 0.3306 | 0.4507 |
| item_item_cosine_tuned | 0.3554 | 0.4773 |
| lambdarank (tuned) | 0.3650 ±0.0216 | 0.4976 ±0.0198 |
| lambdarank_drop_retriever_features (tuned) | 0.3282 | 0.4711 |
| lambdarank_item_item (tuned) | 0.3863 | 0.4773 |
| lambdarank_two_tower (tuned) | 0.3480 | 0.4712 |
| lambdarank_union_balanced (tuned) | 0.3948 | 0.4906 |
| lambdarank_union_unbalanced (tuned) | 0.3351 | 0.4702 |
| most_popular | 0.2337 | 0.3544 |
| no_ranker | 0.3667 | 0.4923 |
| rp3beta (tuned) | 0.3133 | 0.4432 |
| two_tower (tuned) | 0.3559 ±0.0010 | 0.4737 ±0.0019 |

#### Two-tower seeds and gate (ADR-0006)

Chosen hyperparams: `{'batch_size': 1024, 'embedding_dim': 64, 'learning_rate': 0.003, 'max_epochs': 20, 'max_history': 50, 'patience': 3, 'temperature': 0.1, 'weight_decay': 0.0001}` (val NDCG@10=0.0819; early-stopping best_epoch=7; Refit used fixed epoch count = best validation epoch (7).). Tuning log: [`results/tuning/two_tower_ml-latest-small.json`](../results/tuning/two_tower_ml-latest-small.json).

| seed | ndcg@10 | ndcg@10 CI | recall@100 | recall@200 |
| --- | --- | --- | --- | --- |
| 42 | 0.0890 | [0.0777, 0.1013] | 0.3566 | 0.4712 |
| 43 | 0.0898 | [0.0785, 0.1012] | 0.3567 | 0.4740 |
| 44 | 0.0942 | [0.0833, 0.1057] | 0.3544 | 0.4759 |

Across seeds: NDCG@10 mean=0.0910, std=0.0023, min=0.0890, max=0.0942.

**Gate: negative result.** Two-tower does **not** beat both item–item bars on ml-1m-style NDCG@10 with CIs taken into account (see ADR-0006). The retriever or candidate set will be chosen in S4 by validation recall.

#### Ranker candidates, seeds, and gate (ADR-0007)

Candidate budget K=200. Selection metric: validation recall@200 (split: validation). Tie-break: recall@200, then recall@100, then union_balanced, item_item, two_tower, union_unbalanced. Winner: `union_unbalanced`.

| candidate set | recall@100 | recall@200 | mean size |
| --- | --- | --- | --- |
| item_item | 0.415189 | 0.532968 | 200.000000 |
| two_tower | 0.403221 | 0.526670 | 200.000000 |
| union_balanced | 0.418921 | 0.553063 | 200.000000 |
| union_unbalanced | 0.418921 | 0.553083 | 309.819030 |

Early-stop user fraction=0.2 (split seed=42). Ranker seeds share candidate sets and retriever models.

| seed | best_iteration | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 42 | 1 | 0.0852 | [0.0734, 0.0968] | 0.0692 | 0.0999 |
| 43 | 9 | 0.0987 | [0.0867, 0.1102] | 0.0861 | 0.0931 |
| 44 | 72 | 0.1131 | [0.0990, 0.1257] | 0.0961 | 0.0985 |

Across seeds: NDCG@10 mean=0.0990, std=0.0114, min=0.0852, max=0.1131.

**Fixed item-item baseline: point estimate exceeded.** LambdaRank mean NDCG@10=0.0990 vs item_item_cosine 0.0899 [0.0784, 0.1011] (read from `metrics['item_item_cosine']['ndcg@10']`).

Seed NDCG@10 intervals are not all above the bar CI high (`all_seed_ci_low_above_bar_ci_high=False`).

Descriptive test-point comparison: strongest recorded prior model `ease`=0.1022; ranker minus prior=-0.0032. Exceeds prior point: `False`. This comparison does not select a model or establish significance.
Unranked candidate-list point=0.0983; ranker minus unranked=+0.0007. Exceeds both points: `False`. The historical fixed-baseline gate above is a separate comparison.
A paired interval against the unranked list is unavailable in this record.

Ablations are the primary seed, except `ndcg@10`, `recall@10`, and `coverage@10` on the `lambdarank` row, which are means over the three ranker seeds. The `ndcg@10` CI and tail NDCG@10 on that row stay the primary seed; that CI is the primary-seed user bootstrap, not a confidence interval for the 3-seed mean. `no_ranker` keeps the winning candidate order. `lambdarank_drop_retriever_features` drops retriever score and rank.

| ablation | ndcg@10 point | interval run point and 95% CI | recall@10 | coverage@10 | tail ndcg@10 |
| --- | --- | --- | --- | --- | --- |
| lambdarank | 0.0990 | 0.0852 [0.0734, 0.0968] | 0.0838 | 0.0972 | 0.0119 [0.0062, 0.0185] |
| lambdarank_drop_retriever_features | 0.0838 | 0.0838 [0.0722, 0.0960] | 0.0647 | 0.1157 | 0.0088 [0.0051, 0.0131] |
| lambdarank_item_item | 0.1071 | 0.1071 [0.0940, 0.1199] | 0.0899 | 0.0924 | 0.0015 [0.0000, 0.0041] |
| lambdarank_two_tower | 0.0933 | 0.0933 [0.0813, 0.1057] | 0.0766 | 0.1003 | 0.0105 [0.0061, 0.0157] |
| lambdarank_union_balanced | 0.1218 | 0.1218 [0.1078, 0.1354] | 0.0975 | 0.0878 | 0.0227 [0.0134, 0.0340] |
| lambdarank_union_unbalanced | 0.0852 | 0.0852 [0.0734, 0.0968] | 0.0692 | 0.0999 | 0.0119 [0.0062, 0.0185] |
| no_ranker | 0.0983 | 0.0983 [0.0860, 0.1110] | 0.0803 | 0.1054 | 0.0114 [0.0062, 0.0175] |

Top feature gains (refit ranker, primary seed):

| feature | gain |
| --- | --- |
| in_both | 275.5560 |
| two_tower_rank | 85.9516 |
| item_recency | 47.2973 |
| user_n_ratings | 42.1854 |
| affinity_drama | 41.1328 |
| item_item_rank | 32.6754 |
| affinity_comedy | 28.1487 |
| affinity_western | 24.4671 |
| item_item_score | 23.9401 |
| affinity_musical | 21.2151 |

Refit ranker artifact (gitignored): `models/ml-latest-small/ranker.txt` schema_version=1. Recreate with `movielens-recommender run --config configs/default.yaml`.

Ranker stage runtime: 41.7760s.

#### Segment NDCG@10

User activity = train rating-count terciles (low/mid/high). Item head = top 20% of train items by popularity; tail = rest. Item-segment metrics restrict **relevant and recommended** items to the segment (users with no relevant items in-segment are excluded).

| model | activity low | activity mid | activity high | item head | item tail |
| --- | --- | --- | --- | --- | --- |
| als | 0.0690 [0.0498, 0.0882] | 0.0770 [0.0611, 0.0936] | 0.0924 [0.0748, 0.1131] | 0.0840 [0.0734, 0.0947] | 0.0313 [0.0207, 0.0441] |
| als_tuned | 0.0841 [0.0632, 0.1046] | 0.0744 [0.0587, 0.0919] | 0.1040 [0.0856, 0.1235] | 0.0936 [0.0825, 0.1049] | 0.0363 [0.0255, 0.0468] |
| ease (tuned) | 0.0950 [0.0713, 0.1173] | 0.0780 [0.0626, 0.0945] | 0.1336 [0.1106, 0.1567] | 0.1087 [0.0959, 0.1214] | 0.0303 [0.0177, 0.0441] |
| item_item_cosine | 0.0815 [0.0601, 0.1030] | 0.0756 [0.0600, 0.0921] | 0.1126 [0.0920, 0.1356] | 0.0961 [0.0839, 0.1077] | 0.0105 [0.0051, 0.0179] |
| item_item_cosine_tuned | 0.0890 [0.0660, 0.1133] | 0.0756 [0.0603, 0.0924] | 0.1343 [0.1093, 0.1598] | 0.1056 [0.0925, 0.1195] | 0.0010 [0.0000, 0.0026] |
| lambdarank (tuned) | 0.0638 [0.0448, 0.0838] | 0.0757 [0.0599, 0.0934] | 0.1160 [0.0939, 0.1403] | 0.0920 [0.0805, 0.1037] | 0.0119 [0.0062, 0.0185] |
| lambdarank_drop_retriever_features (tuned) | 0.0692 [0.0493, 0.0905] | 0.0672 [0.0520, 0.0849] | 0.1150 [0.0896, 0.1402] | 0.0908 [0.0792, 0.1043] | 0.0088 [0.0051, 0.0131] |
| lambdarank_item_item (tuned) | 0.1040 [0.0801, 0.1294] | 0.0833 [0.0668, 0.1008] | 0.1339 [0.1103, 0.1570] | 0.1136 [0.0998, 0.1284] | 0.0015 [0.0000, 0.0041] |
| lambdarank_two_tower (tuned) | 0.0841 [0.0642, 0.1043] | 0.0734 [0.0574, 0.0894] | 0.1225 [0.0991, 0.1479] | 0.1004 [0.0882, 0.1132] | 0.0105 [0.0061, 0.0157] |
| lambdarank_union_balanced (tuned) | 0.1156 [0.0907, 0.1423] | 0.1016 [0.0812, 0.1221] | 0.1481 [0.1241, 0.1748] | 0.1296 [0.1154, 0.1443] | 0.0227 [0.0134, 0.0340] |
| lambdarank_union_unbalanced (tuned) | 0.0638 [0.0448, 0.0838] | 0.0757 [0.0599, 0.0934] | 0.1160 [0.0939, 0.1403] | 0.0920 [0.0805, 0.1037] | 0.0119 [0.0062, 0.0185] |
| most_popular | 0.0588 [0.0401, 0.0795] | 0.0610 [0.0461, 0.0779] | 0.1030 [0.0793, 0.1246] | 0.0783 [0.0673, 0.0914] | 0.0000 [0.0000, 0.0000] |
| no_ranker | 0.0936 [0.0716, 0.1166] | 0.0843 [0.0676, 0.1016] | 0.1170 [0.0948, 0.1380] | 0.1124 [0.0991, 0.1269] | 0.0114 [0.0062, 0.0175] |
| rp3beta (tuned) | 0.0693 [0.0506, 0.0896] | 0.0658 [0.0507, 0.0835] | 0.1221 [0.0965, 0.1453] | 0.0903 [0.0790, 0.1035] | 0.0000 [0.0000, 0.0000] |
| two_tower (tuned) | 0.0995 [0.0761, 0.1231] | 0.0828 [0.0652, 0.1016] | 0.0847 [0.0664, 0.1037] | 0.1136 [0.1003, 0.1273] | 0.0114 [0.0062, 0.0175] |

Tuning log: [`results/tuning/ml-latest-small.json`](../results/tuning/ml-latest-small.json). Chosen ALS={'alpha': 20.0, 'factors': 128, 'iterations': 15, 'regularization': 0.1} (val NDCG@10=0.0750); item–item={'k_neighbors': 40, 'min_common': 1, 'shrinkage': 100.0} (val NDCG@10=0.0873). EASE={'l2': 500.0} (val NDCG@10=0.0881); RP3beta={'alpha': 0.2, 'beta': 0.3, 'top_k': 2000} (val NDCG@10=0.0799). Two-tower={'batch_size': 1024, 'embedding_dim': 64, 'learning_rate': 0.003, 'max_epochs': 20, 'max_history': 50, 'patience': 3, 'temperature': 0.1, 'weight_decay': 0.0001} (val NDCG@10=0.0819, best_epoch=7; log [`results/tuning/two_tower_ml-latest-small.json`](../results/tuning/two_tower_ml-latest-small.json)).

Pipeline runtime: 129.4080s.

EASE/RP3beta tune and test runtime (same harness, not included in the pipeline runtime above): 289.6160s.

### Global-time-cutoff sanity check (`ml-1m`, secondary)

From `results/global_cutoff/ml-1m.json`. Secondary sanity check: whether model ranking holds when cross-user future signal is removed (ADR-0002). Not the headline table. Hyperparams: Per-user-protocol tuned configs (validation NDCG@10); not re-tuned for the global cutoff. most_popular has no hyperparameters. two_tower uses the per-user-protocol chosen hyperparameters and early-stopping epoch; not re-tuned on the global cutoff. lambdarank reuses the per-user candidate set and primary-seed best_iteration as a fixed num_boost_round. Labels are the chronological tail of the pre-cutoff train; features for that fit come from the head only. Retrievers and features are rebuilt on the full pre-cutoff train before scoring. Post-cutoff labels are not used. Not re-tuned. ease and rp3beta reuse the per-user-protocol validation-chosen hyperparameters; not re-tuned on the global cutoff. **Not re-tuned** for this protocol (`retuned=False`).

Cutoff: timestamp quantile=0.8 (cutoff_timestamp=975768738.0). Surviving: train users=5392, train items=3662, test interactions (after user filter)=103937, eval users (warm relevant)=1114.

| model | ndcg@10 | ndcg@10 CI | precision@10 | recall@10 | recall@100 | recall@200 |
| --- | --- | --- | --- | --- | --- | --- |
| als | 0.1553 | [0.1449, 0.1669] | 0.1479 | 0.0427 | 0.2613 | 0.3875 |
| ease | 0.2259 | [0.2116, 0.2407] | 0.2017 | 0.0613 | 0.2987 | 0.4479 |
| item_item_cosine | 0.2319 | [0.2162, 0.2473] | 0.2104 | 0.0575 | 0.2764 | 0.4171 |
| lambdarank | 0.2323 | [0.2193, 0.2465] | 0.2085 | 0.0720 | 0.3370 | 0.4801 |
| most_popular | 0.2136 | [0.1990, 0.2286] | 0.1987 | 0.0490 | 0.2648 | 0.3866 |
| no_ranker | 0.2140 | [0.2007, 0.2277] | 0.1974 | 0.0655 | 0.3224 | 0.4801 |
| rp3beta | 0.2217 | [0.2064, 0.2364] | 0.1991 | 0.0558 | 0.2579 | 0.3928 |
| two_tower | 0.2140 | [0.2007, 0.2277] | 0.1974 | 0.0655 | 0.3224 | 0.4801 |

### User demographic features (`ml-1m`, S4b)

From `results/demographics/ml-1m.json`. Candidate set `two_tower` (K=200; validation recall@200, then recall@100, then union_balanced, item_item, two_tower, union_unbalanced). Group-affinity statistics use fit-train only. NDCG@10 mean and std are over the three ranker seeds. The NDCG@10 CI, tail NDCG@10, activity slices, fairness slices, and paired difference are the primary seed.

S4 LambdaRank reference (results/ml-1m.json): NDCG@10 mean=0.1273, std=0.0013, winner=`two_tower`. This run's baseline: mean=0.1267, std=0.0008, abs diff=0.0006, matches at 4 decimals=False, candidate set matches=True.

| variant | ndcg@10 mean | std | ndcg@10 CI | recall@10 | coverage@10 | tail ndcg@10 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.1267 | 0.0008 | [0.1234, 0.1318] | 0.0918 | 0.3842 | 0.0905 [0.0858, 0.0959] |
| raw | 0.1249 | 0.0027 | [0.1217, 0.1298] | 0.0906 | 0.3811 | 0.0908 [0.0861, 0.0959] |
| affinity | 0.1317 | 0.0008 | [0.1287, 0.1371] | 0.0969 | 0.4009 | 0.0917 [0.0873, 0.0970] |
| both | 0.1334 | 0.0009 | [0.1282, 0.1366] | 0.0983 | 0.4012 | 0.0922 [0.0876, 0.0973] |

Paired bootstrap of NDCG@10 (both_minus_baseline, seed 42, n=5958): mean=0.0047 [0.0015, 0.0080], excludes_zero=True.

Primary-seed NDCG@10 by the existing user-activity terciles (train rating count).

| variant | activity low | activity mid | activity high |
| --- | --- | --- | --- |
| baseline | 0.1162 [0.1083, 0.1233] | 0.0937 [0.0881, 0.0991] | 0.1732 [0.1647, 0.1813] |
| raw | 0.1096 [0.1019, 0.1173] | 0.0945 [0.0889, 0.0999] | 0.1730 [0.1652, 0.1813] |
| affinity | 0.1181 [0.1106, 0.1253] | 0.1032 [0.0975, 0.1091] | 0.1772 [0.1690, 0.1852] |
| both | 0.1173 [0.1096, 0.1247] | 0.1025 [0.0969, 0.1083] | 0.1773 [0.1687, 0.1850] |

Simulated cold start: earliest N full-train ratings as the query profile. Test targets are unchanged. `most_popular` and `group_most_popular` do not use that profile (matrices: full train; fit-train age-bucket x gender counts).

| N | model | ndcg@10 | ndcg@10 CI | recall@10 | coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 5 | most_popular | 0.0895 | [0.0857, 0.0935] | 0.0466 | 0.0325 |
| 5 | group_most_popular | 0.0960 | [0.0923, 0.1001] | 0.0510 | 0.0614 |
| 5 | item_item | 0.0815 | [0.0775, 0.0850] | 0.0421 | 0.2345 |
| 5 | ranker_baseline | 0.0982 | [0.0944, 0.1017] | 0.0696 | 0.4920 |
| 5 | ranker_both | 0.1024 | [0.0986, 0.1062] | 0.0715 | 0.4688 |
| 10 | most_popular | 0.0895 | [0.0857, 0.0935] | 0.0466 | 0.0325 |
| 10 | group_most_popular | 0.0960 | [0.0923, 0.1001] | 0.0510 | 0.0614 |
| 10 | item_item | 0.0961 | [0.0921, 0.1002] | 0.0560 | 0.1800 |
| 10 | ranker_baseline | 0.1039 | [0.1002, 0.1077] | 0.0748 | 0.4328 |
| 10 | ranker_both | 0.1077 | [0.1039, 0.1117] | 0.0770 | 0.4265 |

Primary-seed NDCG@10 by gender and by age bucket, baseline ranker versus +both. Delta is both minus baseline.

| group | n | baseline ndcg@10 | both ndcg@10 | delta | delta CI |
| --- | --- | --- | --- | --- | --- |
| gender F | 1689 | 0.1167 | 0.1164 | -0.0003 | [-0.0061, 0.0055] |
| gender M | 4269 | 0.1321 | 0.1387 | 0.0066 | [0.0032, 0.0102] |
| age 1 (Under 18) | 220 | 0.1278 | 0.1302 | 0.0024 | [-0.0150, 0.0214] |
| age 18 (18-24) | 1085 | 0.1385 | 0.1426 | 0.0041 | [-0.0028, 0.0115] |
| age 25 (25-34) | 2070 | 0.1348 | 0.1376 | 0.0028 | [-0.0020, 0.0081] |
| age 35 (35-44) | 1180 | 0.1259 | 0.1351 | 0.0092 | [0.0017, 0.0171] |
| age 45 (45-49) | 543 | 0.1150 | 0.1142 | -0.0009 | [-0.0114, 0.0090] |
| age 50 (50-55) | 486 | 0.1091 | 0.1204 | 0.0113 | [0.0006, 0.0214] |
| age 56 (56+) | 374 | 0.1060 | 0.1088 | 0.0028 | [-0.0132, 0.0183] |

Demographic feature gains from the +both primary-seed refit booster. Rank is among every feature of that booster (1 = highest gain).

| feature | gain | rank |
| --- | --- | --- |
| demo_occupation | 3922.3723 | 2 |
| group_gender_pos_rate | 2584.2826 | 7 |
| group_age_pop_share | 2435.5404 | 8 |
| group_gender_pop_share | 1673.7744 | 10 |
| group_occupation_pop_share | 1341.6961 | 12 |
| group_occupation_pos_rate | 1333.0279 | 13 |
| group_age_pos_rate | 1324.1932 | 14 |
| demo_region | 760.1644 | 27 |
| demo_age | 275.3523 | 37 |
| demo_gender | 69.9087 | 41 |

**Decision: keep demographic features as the ranker default.** +both mean NDCG@10=0.1334 versus baseline 0.1267. The paired CI low is above 0.

Demographic experiment runtime: 400.0710s.

### New-user cold start (`ml-1m`, S5c)

This is not the S4b simulated cold start. S4b keeps each evaluated user in the training matrices, truncates the query to the earliest N full-train ratings (N of 5 and 10), and scores the original per-user test split. The user-id embedding is the one learned for that user, and item-item similarities include that user's later train ratings. Here the user is absent from every training row. The model sees only the first N chronological ratings, N in {1, 3, 5, 10}, and is scored on the later ratings with relevance at least 4. There is no user-id embedding at score time, and demographic features are not used.

The first pipeline is a negative result. It was trained on long histories and re-ranked one candidate source. Those numbers are copied below as `pipeline_v1` and were not recomputed. Round 2 was redesigned after those results on the same 604 held-out users: dropout p=0.0 and p=0.1, the short-profile ranker, and the per-N rule. The round-2 held-out numbers are not a fully fresh test. Every choice was frozen on validation before that second score.

Round 2 representation: `dropout_0.25` (validation NDCG@10 0.0719). Dropout grid edge: best p=0.25 (at edge: False); selected model at edge: False.

Cold-start ranker K=200 (134 trees, demographics off). Served method by profile size: N=1 `cold_start_ranker`, N=3 `cold_start_ranker`, N=5 `cold_start_ranker`, N=10 `cold_start_ranker`.

Cold-start most-popular NDCG is higher than the known-user most-popular number because the targets are every later rating, a long tail, and the short profile has not consumed the popular titles. The known-user number is a short per-user test tail after a long history.

Known users below are copied from `results/ml-1m.json`. They were in the training matrix. The new-user rows were not.

| known-user model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 |
| --- | --- | --- | --- | --- |
| most_popular | 0.0895 | [0.0857, 0.0935] | 0.0466 | 0.0325 |
| item_item_cosine | 0.1201 | [0.1158, 0.1242] | 0.0786 | 0.1274 |
| two_tower | 0.1192 | [0.1150, 0.1230] | 0.0901 | 0.4723 |
| lambdarank | 0.1273 | [0.1247, 0.1329] | 0.0928 | 0.3895 |

Primary protocol, fixed before this run: the model sees the first N chronological ratings, and the targets are all later ratings (relevance at least 4). Coverage has no interval.

| N | model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 | eval users |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | pipeline_v1 | 0.1346 | [0.1185, 0.1514] | 0.0159 | 0.3763 | 604 |
| 1 | cold_start_ranker | 0.4006 | [0.3797, 0.4194] | 0.0573 | 0.0129 | 604 |
| 1 | served | 0.4006 | [0.3797, 0.4194] | 0.0573 | 0.0129 | 604 |
| 1 | most_popular | 0.3944 | [0.3732, 0.4145] | 0.0570 | 0.0030 | 604 |
| 1 | item_item_fold_in | 0.2146 | [0.1914, 0.2372] | 0.0271 | 0.3290 | 604 |
| 1 | history_two_tower | 0.2448 | [0.2229, 0.2652] | 0.0331 | 0.2929 | 604 |
| 1 | ease_fold_in | 0.2037 | [0.1829, 0.2253] | 0.0257 | 0.2800 | 604 |
| 3 | pipeline_v1 | 0.2442 | [0.2252, 0.2631] | 0.0303 | 0.2893 | 604 |
| 3 | cold_start_ranker | 0.4089 | [0.3868, 0.4294] | 0.0588 | 0.0159 | 604 |
| 3 | served | 0.4089 | [0.3868, 0.4294] | 0.0588 | 0.0159 | 604 |
| 3 | most_popular | 0.3863 | [0.3646, 0.4068] | 0.0555 | 0.0033 | 604 |
| 3 | item_item_fold_in | 0.3088 | [0.2863, 0.3302] | 0.0391 | 0.1351 | 604 |
| 3 | history_two_tower | 0.3252 | [0.3027, 0.3475] | 0.0466 | 0.1381 | 604 |
| 3 | ease_fold_in | 0.2926 | [0.2706, 0.3143] | 0.0376 | 0.1526 | 604 |
| 5 | pipeline_v1 | 0.2829 | [0.2634, 0.3032] | 0.0383 | 0.2289 | 604 |
| 5 | cold_start_ranker | 0.4087 | [0.3871, 0.4290] | 0.0597 | 0.0202 | 604 |
| 5 | served | 0.4087 | [0.3871, 0.4290] | 0.0597 | 0.0202 | 604 |
| 5 | most_popular | 0.3767 | [0.3551, 0.3966] | 0.0534 | 0.0036 | 604 |
| 5 | item_item_fold_in | 0.3351 | [0.3123, 0.3581] | 0.0458 | 0.1058 | 604 |
| 5 | history_two_tower | 0.3583 | [0.3365, 0.3798] | 0.0503 | 0.0998 | 604 |
| 5 | ease_fold_in | 0.3237 | [0.3015, 0.3465] | 0.0442 | 0.1129 | 604 |
| 10 | pipeline_v1 | 0.3381 | [0.3179, 0.3582] | 0.0612 | 0.1627 | 604 |
| 10 | cold_start_ranker | 0.4076 | [0.3870, 0.4295] | 0.0701 | 0.0686 | 604 |
| 10 | served | 0.4076 | [0.3870, 0.4295] | 0.0701 | 0.0686 | 604 |
| 10 | most_popular | 0.3469 | [0.3253, 0.3670] | 0.0508 | 0.0044 | 604 |
| 10 | item_item_fold_in | 0.3775 | [0.3551, 0.3991] | 0.0621 | 0.0878 | 604 |
| 10 | history_two_tower | 0.3716 | [0.3490, 0.3938] | 0.0614 | 0.0561 | 604 |
| 10 | ease_fold_in | 0.3588 | [0.3390, 0.3800] | 0.0650 | 0.0859 | 604 |

Coverage trade-off: the served ranker's Coverage@10 is 0.0129, 0.0159, 0.0202, 0.0686 at N=1, 3, 5, 10. Most-popular is 0.0030, 0.0033, 0.0036, 0.0044. Item-item fold-in is 0.3290, 0.1351, 0.1058, 0.0878 and EASE fold-in is 0.2800, 0.1526, 0.1129, 0.0859. The served list stays close to popularity and far below those fold-in methods, so it leans on popular titles.

N=1: `cold_start_ranker` is above most-popular by 0.0062, and the interval [-0.0033, 0.0157] includes 0.
N=3: the served method `cold_start_ranker` beats most-popular. Difference 0.0226 [0.0124, 0.0333] (excludes 0).
N=5: the served method `cold_start_ranker` beats most-popular. Difference 0.0321 [0.0211, 0.0427] (excludes 0).
N=10: the served method `cold_start_ranker` beats most-popular. Difference 0.0607 [0.0465, 0.0769] (excludes 0).
N=10: served minus `item_item_fold_in` NDCG@10 0.0302 [0.0184, 0.0413] (excludes 0).

Round 1 paired gaps (pipeline minus the best simple baseline on that table) stay the recorded miss:

N=1: pipeline_v1 minus `most_popular` NDCG@10 -0.2598 [-0.2809, -0.2388] (excludes 0).
N=3: pipeline_v1 minus `most_popular` NDCG@10 -0.1421 [-0.1620, -0.1241] (excludes 0).
N=5: pipeline_v1 minus `most_popular` NDCG@10 -0.0937 [-0.1104, -0.0762] (excludes 0).
N=10: pipeline_v1 minus `item_item_fold_in` NDCG@10 -0.0394 [-0.0532, -0.0262] (excludes 0).

Sensitivity view, added after the first results. Targets are only each held-out user's last 20% of ratings (the harness tail), and only where that tail is after the first N. The serving rule was not re-chosen here. Pipeline v1 is not scored on this target.

| N | model | NDCG@10 | 95% CI | Recall@10 | Coverage@10 | eval users |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | cold_start_ranker | 0.0429 | [0.0364, 0.0497] | 0.0301 | 0.0129 | 599 |
| 1 | served | 0.0429 | [0.0364, 0.0497] | 0.0301 | 0.0129 | 599 |
| 1 | most_popular | 0.0397 | [0.0328, 0.0466] | 0.0289 | 0.0030 | 599 |
| 1 | item_item_fold_in | 0.0264 | [0.0213, 0.0326] | 0.0163 | 0.3284 | 599 |
| 1 | history_two_tower | 0.0281 | [0.0225, 0.0338] | 0.0187 | 0.2918 | 599 |
| 1 | ease_fold_in | 0.0287 | [0.0231, 0.0357] | 0.0181 | 0.2800 | 599 |
| 3 | cold_start_ranker | 0.0454 | [0.0383, 0.0523] | 0.0327 | 0.0159 | 599 |
| 3 | served | 0.0454 | [0.0383, 0.0523] | 0.0327 | 0.0159 | 599 |
| 3 | most_popular | 0.0405 | [0.0336, 0.0476] | 0.0293 | 0.0033 | 599 |
| 3 | item_item_fold_in | 0.0398 | [0.0320, 0.0484] | 0.0240 | 0.1351 | 599 |
| 3 | history_two_tower | 0.0420 | [0.0350, 0.0494] | 0.0296 | 0.1375 | 599 |
| 3 | ease_fold_in | 0.0382 | [0.0313, 0.0461] | 0.0219 | 0.1520 | 599 |
| 5 | cold_start_ranker | 0.0442 | [0.0378, 0.0509] | 0.0325 | 0.0202 | 599 |
| 5 | served | 0.0442 | [0.0378, 0.0509] | 0.0325 | 0.0202 | 599 |
| 5 | most_popular | 0.0414 | [0.0345, 0.0487] | 0.0296 | 0.0036 | 599 |
| 5 | item_item_fold_in | 0.0412 | [0.0338, 0.0498] | 0.0259 | 0.1050 | 599 |
| 5 | history_two_tower | 0.0418 | [0.0352, 0.0494] | 0.0295 | 0.0990 | 599 |
| 5 | ease_fold_in | 0.0421 | [0.0347, 0.0500] | 0.0258 | 0.1121 | 599 |
| 10 | cold_start_ranker | 0.0435 | [0.0366, 0.0504] | 0.0337 | 0.0686 | 599 |
| 10 | served | 0.0435 | [0.0366, 0.0504] | 0.0337 | 0.0686 | 599 |
| 10 | most_popular | 0.0442 | [0.0369, 0.0518] | 0.0320 | 0.0044 | 599 |
| 10 | item_item_fold_in | 0.0457 | [0.0386, 0.0541] | 0.0319 | 0.0861 | 599 |
| 10 | history_two_tower | 0.0450 | [0.0382, 0.0523] | 0.0330 | 0.0558 | 599 |
| 10 | ease_fold_in | 0.0379 | [0.0313, 0.0445] | 0.0262 | 0.0859 | 599 |

Sensitivity N=1: `cold_start_ranker` is above most-popular by 0.0032, and the interval [-0.0014, 0.0082] includes 0.
Sensitivity N=3: `cold_start_ranker` is above most-popular by 0.0049, and the interval [-0.0000, 0.0093] includes 0.
Sensitivity N=5: `cold_start_ranker` is above most-popular by 0.0028, and the interval [-0.0020, 0.0079] includes 0.
Sensitivity N=10: `cold_start_ranker` does not beat most-popular. Difference -0.0007 [-0.0072, 0.0061] (includes 0). The served ranker (0.0435) is below item-item fold-in (0.0457) and the history two-tower (0.0450).

Warmed new-user top-10 latency (five popular titles rated 5): 0.1523s. Method `cold_start_ranker`.

Cold-start experiment runtime: 960.1911s.

### Full-softmax two-tower (S3e)

From `results/two-tower-v2/`. The reference loss is the in-batch sampled softmax. Temperature, learning rate, and embedding dim were chosen on validation only. NDCG@10 in the table is the seed mean. The 95% CI column and the paired intervals are the primary seed, candidate minus reference.

#### `ml-1m`

Published reference two-tower (`results/ml-1m.json`): NDCG@10 0.1192 [0.1150, 0.1230], Recall@10 0.0901, Recall@100 0.4354, Recall@200 0.6037, Coverage@10 0.4723, head 0.1364, tail 0.0825. Validation NDCG@10 0.0826.

| loss | val NDCG@10 | best epoch | grid edges |
| --- | --- | --- | --- |
| reference in_batch | 0.0826 | 6 | published config |
| full_softmax | 0.0855 | 9 | embedding_dim=64.0 (high of [32.0, 64.0]); learning_rate=0.003 (high of [0.0003, 0.001, 0.003]); temperature=0.2 (high of [0.05, 0.1, 0.2]) |
| sampled_softmax | 0.0856 | 20 | embedding_dim=64.0 (high of [32.0, 64.0]); learning_rate=0.003 (high of [0.0003, 0.001, 0.003]); temperature=0.2 (high of [0.05, 0.1, 0.2]); best_epoch equals max_epochs 20 |

Validation winner: `sampled_softmax` (0.0856). Beats the reference on validation: True.

| model | NDCG@10 | 95% CI | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | head | tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in_batch (3-seed mean) | 0.1192 | [0.1150, 0.1230] | 0.0901 | 0.4354 | 0.6037 | 0.4723 | 0.1364 | 0.0825 |
| full_softmax (3-seed mean) | 0.1286 | [0.1247, 0.1340] | 0.0912 | 0.4383 | 0.6101 | 0.3310 | 0.1429 | 0.0867 |
| sampled_softmax (3-seed mean) | 0.1305 | [0.1254, 0.1347] | 0.0891 | 0.4280 | 0.6004 | 0.3002 | 0.1432 | 0.0816 |

Paired NDCG@10, full_softmax minus reference 0.0105 [0.0077, 0.0130] (excludes 0).
Paired head NDCG@10, full_softmax 0.0065 [0.0039, 0.0092] (excludes 0).
Paired tail NDCG@10, full_softmax 0.0042 [0.0008, 0.0074] (excludes 0).
Paired NDCG@10, sampled_softmax minus reference 0.0110 [0.0081, 0.0138] (excludes 0).
Paired head NDCG@10, sampled_softmax 0.0069 [0.0040, 0.0098] (excludes 0).
Paired tail NDCG@10, sampled_softmax -0.0009 [-0.0048, 0.0023] (includes 0).

Did full softmax help? Yes. Full softmax test NDCG@10 is 0.1286 against the reference rerun 0.1192. Paired difference 0.0105 [0.0077, 0.0130] (excludes 0). Validation was 0.0855 against reference 0.0826.

Does it fix the tail weakness? There was no tail gap versus item-item to fix. `full_softmax` is higher than the reference. `full_softmax` tail NDCG@10 is 0.0867 (head 0.1429). Published reference tail is 0.0825; item-item tail is 0.0321. Paired tail difference versus the reference rerun 0.0042 [0.0008, 0.0074] (excludes 0).

Does the ranker gain? No. The point estimate is above the candidate list and the paired interval includes 0. It does not beat the reference-tower ranker. New ranker NDCG@10 0.1301 versus `no_ranker` 0.1300. Paired versus no_ranker 0.0010 [-0.0025, 0.0044] (includes 0). Paired versus the reference-tower ranker -0.0013 [-0.0043, 0.0019] (includes 0). This is the first-round tower. For the final edge-extension tower the ranker is significantly worse. Published LambdaRank in `results/ml-1m.json` is 0.1273 (demographics off).

Compute: tune 7315.623s, test training 706.290s, ranker 325.702s, seeds [42, 43, 44].

Edge extension, run after the ml-1m test results existed. Sampled softmax only: dim {64, 128}, learning rate {0.003, 0.01}, temperature {0.2, 0.5}, max_epochs 40.

| dim | learning rate | temperature | val NDCG@10 | best epoch | edges |
| --- | --- | --- | --- | --- | --- |
| 64 | 0.003 | 0.2 | 0.0856 | 20 | embedding_dim=64.0 (low of [64.0, 128.0]); learning_rate=0.003 (low of [0.003, 0.01]); temperature=0.2 (low of [0.2, 0.5]) |
| 64 | 0.003 | 0.5 | 0.0704 | 15 | embedding_dim=64.0 (low of [64.0, 128.0]); learning_rate=0.003 (low of [0.003, 0.01]); temperature=0.5 (high of [0.2, 0.5]) |
| 64 | 0.01 | 0.2 | 0.0842 | 14 | embedding_dim=64.0 (low of [64.0, 128.0]); learning_rate=0.01 (high of [0.003, 0.01]); temperature=0.2 (low of [0.2, 0.5]) |
| 64 | 0.01 | 0.5 | 0.0694 | 15 | embedding_dim=64.0 (low of [64.0, 128.0]); learning_rate=0.01 (high of [0.003, 0.01]); temperature=0.5 (high of [0.2, 0.5]) |
| 128 | 0.003 | 0.2 | 0.0829 | 6 | embedding_dim=128.0 (high of [64.0, 128.0]); learning_rate=0.003 (low of [0.003, 0.01]); temperature=0.2 (low of [0.2, 0.5]) |
| 128 | 0.003 | 0.5 | 0.0669 | 6 | embedding_dim=128.0 (high of [64.0, 128.0]); learning_rate=0.003 (low of [0.003, 0.01]); temperature=0.5 (high of [0.2, 0.5]) |
| 128 | 0.01 | 0.2 | 0.0860 | 17 | embedding_dim=128.0 (high of [64.0, 128.0]); learning_rate=0.01 (high of [0.003, 0.01]); temperature=0.2 (low of [0.2, 0.5]) |
| 128 | 0.01 | 0.5 | 0.0709 | 17 | embedding_dim=128.0 (high of [64.0, 128.0]); learning_rate=0.01 (high of [0.003, 0.01]); temperature=0.5 (high of [0.2, 0.5]) |

The validation winner changed to dim 128, learning rate 0.01, temperature 0.2, val NDCG@10 0.0860.
The winner still sits on the high edge of dim 128 and learning rate 0.01, and the low edge of temperature 0.2.
Re-scored test NDCG@10 0.1309 [0.1255, 0.1346].
LambdaRank NDCG@10 0.1277 versus the reference-tower ranker 0.1334.
Paired versus no_ranker -0.0041 [-0.0073, -0.0008] (excludes 0).
Paired versus the reference-tower ranker -0.0063 [-0.0097, -0.0027] (excludes 0).
Paired NDCG@10, re-scored tower minus the reference refit 0.0111 [0.0082, 0.0141] (excludes 0).
The first ml-1m test still stands: both new losses beat the reference two-tower.
The ranker on the re-scored tower is worse, not flat: it is below its own candidate list and below the reference-tower ranker, and both paired intervals exclude 0.
The app stays. This change does not switch the Streamlit model.

#### `ml-32m`

Published reference two-tower (`results/ml-32m.json`): NDCG@10 0.1436 [0.1398, 0.1479], Recall@10 0.1138, Recall@100 0.4742, Recall@200 0.6243, Coverage@10 0.0411, head 0.1460, tail 0.0006. Validation NDCG@10 0.1008.

| loss | val NDCG@10 | best epoch | grid edges |
| --- | --- | --- | --- |
| reference in_batch | 0.1008 | 5 | published config |
| sampled_softmax | 0.0967 | 3 | embedding_dim=64.0 (only_value of [64.0]); temperature=0.1 (low of [0.1, 0.2]); best_epoch equals max_epochs 3 |

Validation winner: `sampled_softmax` (0.0967). Beats the reference on validation: False.

| model | NDCG@10 | 95% CI | Recall@10 | Recall@100 | Recall@200 | Coverage@10 | head | tail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sampled_softmax (1 seed) | 0.1379 | [0.1343, 0.1424] | 0.1106 | 0.4653 | 0.6144 | 0.0373 | 0.1383 | 0.0035 |

Paired interval versus the published two-tower was not computed. The published two-tower was not refit. The published results file has no per-user scores, so a paired interval versus that reference was not computed.
Unpaired point difference, `sampled_softmax` minus the published two-tower: NDCG@10 -0.0057, head -0.0076, tail 0.0029.
Paired interval versus the published LambdaRank was not computed. The published ranker was not refit and this run has no per-user scores for it.

Did full softmax help? Skipped for compute. One full-softmax epoch is 6860.3s, which does not fit beside the sampled-softmax work inside the 14400.0s cap.

Does it fix the tail weakness? The point estimate is above the published two-tower and above item-item. A paired interval was not computed. `sampled_softmax` tail NDCG@10 is 0.0035 (head 0.1383). Published reference tail is 0.0006; item-item tail is 0.0022.

Does the ranker gain? No. The new two-tower did not beat the reference on validation.

Compute: tune 7176.076s, test training 2400.219s, ranker not run, seeds [42].

Matched epoch budget, added after the first ml-32M test result. sampled softmax does not beat the in-batch reference on ml-32M at matched epoch budget.

Validation NDCG@10 0.0987 versus the published reference 0.1008. best_epoch 4, epochs trained 6, truncated False.
Test not re-scored: validation did not beat the published reference.
The matched run has no test score, so a paired interval was not computed.
Ranker: Validation did not beat the published reference.
Matched-round wall clock 4284.071s.
The app stays. This change does not switch the Streamlit model.

### Serving latency and batch (`ml-1m`, S5b)

From `results/serving/ml-1m.json`, written by `scripts/measure_serving.py`. Budgets were fixed in ADR-0014 before measuring. Requests go through the FastAPI test client in one process (each request is sent once untimed, then once timed). Host: 16 CPUs, Windows-11-10.0.26300-SP0.

| request | n | p50 ms | p95 ms | p99 ms | max ms | p95 budget | within |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Known user, top 10 with explanations | 500 | 33.1 | 62.8 | 85.5 | 123.2 | 300 | yes |
| Known user, top 10 without explanations | 500 | 19.5 | 22.1 | 26.3 | 51.4 | 100 | yes |
| New user, five ratings, top 10 | 200 | 136.7 | 151.1 | 174.3 | 287.2 | 300 | yes |
| Known user, explained, pre-S5b lookups | 500 | 80.0 | 187.7 | 292.3 | 404.3 |  |  |

The load-time lookups make the explained known-user call 2.42x faster at p50 than the pre-S5b table scans, on the same users.
Batch: top 10 for 6040 users (60400 rows) in 69.5s, 86.9 users/s, budget 600s, within budget: yes.

### Drift and refresh replay (`ml-1m`, S6)

From `results/ops/ml-1m.json`, written by `movielens-recommender drift-study`. ml-1m is replayed in time order: the first model is trained on every rating before 2000-08-01, then scored on 14 later periods (monthly to December 2000, then quarterly). Users need 5 earlier ratings; the target is what they rated 4 or more in the period. `frozen` is never retrained, `periodic` is retrained before every period, and `drift` retrains when the item divergence passes the threshold chosen on the first 3 periods (ADR-0016).

| model | frozen | periodic | drift | retrains (periodic / drift) |
| --- | --- | --- | --- | --- |
| most popular | 0.0896 | 0.0891 | 0.0896 | 13 / 0 |
| item-item | 0.1013 | 0.1009 | 0.1013 | 13 / 0 |
| EASE | 0.0997 | 0.1015 | 0.0997 | 13 / 0 |

Mean NDCG@10 over the test periods. Threshold chosen on item-item: tau=0.6 (qualifying: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).

| period | eval users | item divergence | new-item share | new-user share | item-item periodic minus frozen |
| --- | --- | --- | --- | --- | --- |
| 2000-08 | 187 | 0.3689 | 0.0059 | 0.9307 | 0.0000 [0.0000, 0.0000] |
| 2000-09 | 241 | 0.3981 | 0.0155 | 0.9524 | -0.0015 [-0.0091, 0.0063] |
| 2000-10 | 238 | 0.3912 | 0.0266 | 0.8240 | -0.0043 [-0.0107, 0.0016] |
| 2000-11 | 227 | 0.3935 | 0.0317 | 0.7995 | -0.0083 [-0.0158, -0.0013] |
| 2000-12 | 534 | 0.3853 | 0.0329 | 0.9799 | -0.0055 [-0.0119, 0.0011] |
| 2001-Q1 | 741 | 0.3930 | 0.0277 | 0.7497 | -0.0009 [-0.0071, 0.0047] |
| 2001-Q2 | 496 | 0.4244 | 0.0501 | 0.0032 | 0.0033 [-0.0036, 0.0107] |
| 2001-Q3 | 427 | 0.4506 | 0.0898 | 0.0025 | -0.0062 [-0.0130, 0.0003] |
| 2001-Q4 | 341 | 0.4341 | 0.0645 | 0.0000 | 0.0064 [-0.0013, 0.0142] |
| 2002-Q1 | 337 | 0.4443 | 0.0625 | 0.0147 | -0.0039 [-0.0126, 0.0041] |
| 2002-Q2 | 249 | 0.4279 | 0.0672 | 0.0136 | 0.0050 [-0.0028, 0.0133] |
| 2002-Q3 | 233 | 0.4192 | 0.0590 | 0.0000 | 0.0046 [-0.0042, 0.0141] |
| 2002-Q4 | 213 | 0.4383 | 0.0667 | 0.0000 | -0.0070 [-0.0182, 0.0040] |
| 2003-Q1 | 153 | 0.4219 | 0.0511 | 0.0325 | 0.0082 [-0.0020, 0.0202] |

Sampling-noise floor of the item divergence: 0.3693. Paired periodic-minus-frozen intervals that exclude 0: 3 of 39 (the first period is the same model).
Spearman correlation of each drift signal with the retraining gain: most popular item divergence -0.62, new item share -0.60, new user share +0.37; item-item item divergence +0.04, new item share +0.08, new user share -0.25; EASE item divergence +0.26, new item share +0.40, new user share -0.53.

### Beyond accuracy (`ml-1m`, ADR-0017)

From `results/beyond-accuracy/ml-1m.json`. Definitions: [`docs/adr/0017-beyond-accuracy.md`](../docs/adr/0017-beyond-accuracy.md). k=10, eval users=5958. Bootstrap: 1000 resamples, alpha=0.05, seed=42. Coverage@10 is a point estimate. Novelty, intra-list diversity, mean popularity, long-tail share, and NDCG@10 are user means with that bootstrap. Two-tower and LambdaRank lists are seed 42 only. Their committed NDCG@10 is that seed's test number in results/ml-1m.json. The headline NDCG@10 for those two is the mean over three seeds and is stored beside it.

| model | NDCG@10 | Coverage@10 | Novelty@10 | Intra-list diversity@10 | Mean popularity@10 | Long-tail share@10 | NDCG matches committed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| most_popular | 0.0895 [0.0857, 0.0935] | 0.0325 | 8.4476 [8.4436, 8.4519] | 0.6321 [0.6299, 0.6344] | 2325.9361 [2319.9340, 2331.5737] | 0.0000 [0.0000, 0.0000] | yes |
| item_item_cosine | 0.1201 [0.1158, 0.1242] | 0.1274 | 8.8996 [8.8910, 8.9077] | 0.6288 [0.6257, 0.6321] | 1794.5672 [1786.4853, 1803.2589] | 0.0031 [0.0022, 0.0040] | yes |
| als_tuned | 0.0906 [0.0874, 0.0940] | 0.4541 | 9.8391 [9.8212, 9.8573] | 0.6518 [0.6482, 0.6552] | 1118.0085 [1106.0058, 1129.8557] | 0.1117 [0.1076, 0.1156] | yes |
| ease | 0.1184 [0.1142, 0.1226] | 0.2100 | 9.1409 [9.1306, 9.1524] | 0.6301 [0.6264, 0.6336] | 1575.2565 [1565.4193, 1584.1931] | 0.0059 [0.0049, 0.0069] | yes |
| rp3beta | 0.1138 [0.1095, 0.1178] | 0.1756 | 8.9648 [8.9537, 8.9758] | 0.6119 [0.6084, 0.6157] | 1750.1566 [1740.9753, 1759.7112] | 0.0072 [0.0057, 0.0088] | yes |
| two_tower | 0.1191 [0.1150, 0.1230] | 0.4669 | 9.8603 [9.8386, 9.8841] | 0.6160 [0.6119, 0.6197] | 1175.1585 [1164.3297, 1186.3043] | 0.1040 [0.0999, 0.1084] | yes |
| lambdarank | 0.1293 [0.1248, 0.1333] | 0.3870 | 9.5047 [9.4908, 9.5193] | 0.6424 [0.6385, 0.6461] | 1315.4293 [1305.6593, 1325.1133] | 0.0459 [0.0430, 0.0489] | no |

`lambdarank` leads NDCG@10 at 0.1293 [0.1248, 0.1333], and `two_tower` has the highest novelty point estimate at 9.8603 [9.8386, 9.8841]; its interval overlaps `als_tuned`. `two_tower` trades accuracy for novelty: its NDCG@10 is 0.1191 [0.1150, 0.1230] against `lambdarank` at 0.1293 [0.1248, 0.1333], with coverage 0.4669 versus 0.3870. `most_popular` is the most popularity-biased list (mean popularity 2325.9361 [2319.9340, 2331.5737], long-tail share 0.0000 [0.0000, 0.0000]).

Recomputed NDCG@10 matches the committed value to 4 decimals for 6 of 7 models.
It does not match for lambdarank: 0.129253 versus committed seed-42 0.128786 (0.1293 versus 0.1288). Early stopping picked 126 trees versus 95 committed; hyperparameters and procedure unchanged.
Headline NDCG@10 in `results/ml-1m.json` differs at 4 decimals for `two_tower` list 0.1191 versus headline 0.1192; `lambdarank` list 0.1293 versus headline 0.1273.

Host: Linux 6.12.94+, x86_64, 4 CPUs, Python 3.12.3, SIMD AVX2, AVX512F, AVX512_ICL, AVX512_SKX, FMA3, SSE42.

ml-32M is future work. This panel is ml-1m only.

Beyond-accuracy runtime: 143.3350s.

Rebuild the lists with `python scripts/run_beyond_accuracy.py`.

### Implicit ALS retune (`ml-1m`)

From `results/als-retune/ml-1m.json`, with the grid in `results/tuning/als_retune_ml-1m.json`. Selection is validation NDCG@10 on the existing split. Test is scored once. The old ALS config is the committed winner, refit under the current top-n rule (ADR-0018).

Validation selected tfidf confidence, factors 64, regularization 1, iterations 30 with NDCG@10 0.0797; its sealed test NDCG@10 is 0.1138 [0.1099, 0.1181].
On this code the old ALS config scores 0.0906 [0.0874, 0.0940] and item–item scores 0.1201 [0.1158, 0.1242]; the paired retuned-minus-old difference is 0.0232 [0.0197, 0.0267] (excludes 0: yes) and the paired retuned-minus-item–item difference is -0.0063 [-0.0104, -0.0021] (excludes 0: yes).
Negative result: retuned ALS still trails item–item on test NDCG@10, and the selected config sits on no edge of the values scored, after one step past the low factors edge (64), the high regularization edge (1) and the high iterations edge (30).

Validation configs in the locked grid: 144. EASE had 8 and RP3beta had 125 in the committed ml-1m tuning file. Edge-extension trials scored after the base grid, under the ADR-0018 one-step rule: 3 (factors 32, regularization 10, iterations 60; all lower on validation).

| model | validation NDCG@10 | test NDCG@10 | 95% CI | committed test NDCG@10 |
| --- | --- | --- | --- | --- |
| old ALS | 0.0646 | 0.0906 | [0.0874, 0.0940] | 0.0906 |
| retuned ALS | 0.0797 | 0.1138 | [0.1099, 0.1181] | — |
| item–item cosine | — | 0.1201 | [0.1158, 0.1242] | 0.1201 |

### SASRec (`ml-1m`)

From `results/sasrec/ml-1m.json`, with the grid in `results/tuning/sasrec_ml-1m.json`. Selection is validation NDCG@10 on the existing split. Test is scored once. The shuffled model keeps the selected hyperparameters and permutes each window (ADR-0019). The SASRec test column is the mean across test seeds, and its interval is the primary seed.

Validation selected max_seq_len 50, n_layers 3, dropout 0.5, d_model 64, n_heads 1 at best epoch 30, equal to max_epochs (30), so the model may be under-trained at this budget, with NDCG@10 0.0505 [0.0475, 0.0535]; its sealed test NDCG@10 is 0.0924 across seeds (std 0.0014), and the primary seed is 0.0929 [0.0889, 0.0967].
The shuffled-order model scores 0.0892 [0.0852, 0.0930]; the paired real-minus-shuffled difference is 0.0037 [0.0024, 0.0052] (excludes 0: yes); on this run the two-tower scores 0.1191 [0.1150, 0.1230] and LambdaRank scores 0.1293 [0.1248, 0.1333] (best iteration 126); the paired SASRec-minus-two-tower difference is -0.0261 [-0.0306, -0.0214] (excludes 0: yes) and the paired SASRec-minus-LambdaRank difference is -0.0363 [-0.0408, -0.0319] (excludes 0: yes); published headline NDCG@10 is 0.1192 for the two-tower and 0.1273 for LambdaRank.
SASRec's coverage@10 of 0.0327 is close to most-popular's 0.0325, and its test NDCG@10 of 0.0924 is near most-popular's 0.0895.
Order helps on this split by a small margin: +0.0037 NDCG@10, about 4% relative.
The low max_seq_len edge (50) stays unresolved: ADR-0019 does not extend it, and max_seq_len 200 was scored and lost (best validation NDCG@10 0.0496). The selected config sits on the high n_layers edge (3), and no second step was allowed.
Test NDCG is higher than validation for every model on this split, because the test slice is twice the size (test fraction 0.2, validation fraction 0.1) and the test model is refit on full train.

Validation configs in the locked grid: 8. The two-tower grid had 8 and EASE had 8. Edge-extension trials scored after the base grid, under the ADR-0019 one-step rule: 2 (the n_layers step scored higher (validation NDCG@10 0.0505) and replaced the leader; the dropout step scored lower (validation NDCG@10 0.0481)).

Seeds: tuning 42, test 42, 43 and 44, shuffle 42, bootstrap 42.

| model | validation NDCG@10 | test NDCG@10 | 95% CI |
| --- | --- | --- | --- |
| SASRec | 0.0505 | 0.0924 | [0.0889, 0.0967] |
| SASRec shuffled | — | 0.0892 | [0.0852, 0.0930] |
| two-tower | — | 0.1191 | [0.1150, 0.1230] |
| LambdaRank | — | 0.1293 | [0.1248, 0.1333] |

### Data-Adaptive Normalized EASE (`EASE-DAN`, `ml-1m`)

From `results/ease-dan/ml-1m.json`, with the grid in `results/tuning/ease_dan_ml-1m.json` (ADR-0028). Selection is validation NDCG@10 on the existing split. Test is scored once. `results/ml-1m.json` is unchanged.

Validation selected `l2=5000`, `dropout_p=0.6`, `alpha=0`, `beta=0` (validation NDCG@10 0.0812, against committed EASE 0.0808); its sealed test NDCG@10 is 0.1195 [0.1151, 0.1238].
**Win against EASE^R (`beats_ease=True`).** Paired `ease_dan` minus `ease` test NDCG@10 is 0.0011 [0.0004, 0.0017] (excludes 0: yes); paired `ease_dan` minus `item_item_cosine` is -0.0007 [-0.0035, 0.0022] (excludes 0: no).

Validation configs in the locked grid: 108. Edge-extension trials scored under the ADR-0028 one-step rule: 2 (total scored: 110, selected sits on the high l2 edge (5000)).

| model | val NDCG@10 | test NDCG@10 | 95% CI | Recall@10 | Coverage@10 | fit (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `ease_dan` | 0.0812 | 0.1195 | [0.1151, 0.1238] | 0.0871 | 0.2053 | 2.44 |
| `ease` | 0.0808 | 0.1184 | [0.1142, 0.1226] | 0.0861 | 0.2100 | 2.39 |
| `item_item_cosine` | — | 0.1201 | [0.1158, 0.1242] | 0.0786 | 0.1274 | 1.34 |

### Sequential models: eSASRec and PCTM (`ml-1m`)

From `results/sequence-modern/ml-1m.json`, with the grid in `results/tuning/sequence_modern_ml-1m.json` (ADR-0030). Selection is validation NDCG@10 on the existing split. Test is scored once. `results/sasrec/ml-1m.json` and `results/ml-1m.json` are unchanged. The eSASRec test column is the mean across test seeds, and its interval is the primary seed. PCTM and the SASRec reference are single fits on the same users.

PCTM validation selected decay `harmonic`, max distance 20, smoothing tau 50, popularity beta -0.5 (validation NDCG@10 0.1863). eSASRec selected max sequence length 50, 2 LiGR layers, dropout 0.2, best epoch 13 (validation NDCG@10 0.0509). The SASRec reference keeps the committed architecture (validation NDCG@10 0.0505).

**Win (`pctm_beats_sasrec=True`).** Paired PCTM minus SASRec test NDCG@10 is 0.0857 [0.0796, 0.0917] (excludes 0: yes).
**Negative result (`esasrec_beats_sasrec=False`).** Paired eSASRec minus SASRec is -0.0134 [-0.0158, -0.0110] (excludes 0: yes). The three-seed eSASRec mean is 0.0836 (std 0.0029); the primary seed is 0.0795 [0.0761, 0.0827].
**Negative result (`higher_order_beats_pairwise_pctm=False`).** Paired eSASRec minus PCTM is -0.0991 [-0.1050, -0.0936] (excludes 0: yes). On this split the higher-order transformer stays below the pairwise probe.

PCTM catalog coverage@10 is 0.7112, against 0.0327 for the SASRec reference and 0.0341 for eSASRec. Published headline points from other runs, not paired in this file, are 0.1192 for the two-tower and 0.1273 for LambdaRank (0.1334 with demographics).

Validation configs in the locked grids: 8 eSASRec and 36 PCTM.

| model | val NDCG@10 | test NDCG@10 | 95% CI | Coverage@10 | Novelty@10 | Long-tail share@10 |
| --- | --- | --- | --- | --- | --- | --- |
| PCTM | 0.1863 | 0.1787 | [0.1737, 0.1837] | 0.7112 | 10.2004 | 0.1759 |
| SASRec reference | 0.0505 | 0.0929 | [0.0889, 0.0967] | 0.0327 | 8.5220 | 0.0000 |
| eSASRec | 0.0509 | 0.0836 | [0.0761, 0.0827] | 0.0341 | 8.7066 | 0.0000 |

### Discovery slider (`ml-1m` cold start)

From `results/slider-eval/ml-1m.json` (ADR-0032). The five Familiar-to-Adventurous positions re-rank the deployed cold-start list. Model weights stay fixed. The table is profile size N=5 on the 604 held-out cold-start users. N=3 and N=10 are in the same file.

Familiar NDCG@10 is 0.4095 [0.3879, 0.4296]. The committed cold-start point at N=5 is 0.4087 (matches at 4 decimals: False). The gap is the documented serving-snapshot refit, not a new model.
Closer changes NDCG@10 by -0.0062 [-0.0104, -0.0016] (excludes 0: yes) and coverage@10 from 0.0194 to 0.0271. In between reaches coverage@10 0.2114 and long-tail share@10 0.1995, with NDCG@10 0.2710.
On the last-20% target at N=5, Familiar NDCG@10 is 0.0444 and Closer is 0.0458.

| position | weight | NDCG@10 | 95% CI | Coverage@10 | Novelty@10 | Diversity@10 | Long-tail share@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Familiar | 0 | 0.4095 | [0.3879, 0.4296] | 0.0194 | 8.4396 | 0.6889 | 0.0000 |
| Closer | 0.25 | 0.4033 | [0.3814, 0.4236] | 0.0271 | 8.5080 | 0.7298 | 0.0012 |
| In between | 0.5 | 0.2710 | [0.2542, 0.2869] | 0.2114 | 9.8298 | 0.8220 | 0.1995 |
| Further | 0.75 | 0.0905 | [0.0812, 0.1002] | 0.3678 | 11.0415 | 0.8721 | 0.3983 |
| Adventurous | 1 | 0.0519 | [0.0450, 0.0588] | 0.3820 | 11.3095 | 0.8676 | 0.4611 |

### EASE-DAN as a third retriever (`ml-1m`)

From `results/ranker-ease-dan/ml-1m.json` (ADR-0031). Retriever hyperparameters stay frozen from `results/ml-1m.json` and `results/ease-dan/ml-1m.json`. The candidate budget is K=200. Validation Recall@200 chooses the K=200 set. Test NDCG@10 is the mean across ranker seeds 42, 43, and 44. The interval is the primary seed. `results/ml-1m.json` and `results/demographics/ml-1m.json` are unchanged.

Validation Recall@200: `two_tower` 0.6552, `union_balanced` 0.6531, `tri_union_balanced` 0.6607, `tri_union_unbalanced` 0.6598. Selected K=200 set: `tri_union_balanced`.

**Negative result (`tri_off_lifts_bi_off=False`).** Paired three-retriever minus two-retriever, demographics off, is 0.0003 [-0.0026, 0.0035] (excludes 0: no).
**Negative result (`tri_both_lifts_bi_both=False`).** Paired three-retriever minus two-retriever, demographics on, is -0.0032 [-0.0061, -0.0003] (excludes 0: yes).
**Negative result (`tri_unbalanced_both_lifts_bi_both=False`).** Paired unbalanced three-way union minus the two-retriever ranker, demographics on, is -0.0010 [-0.0038, 0.0020] (excludes 0: no).

| variant | candidate set | features | test NDCG@10 | 95% CI | Coverage@10 |
| --- | --- | --- | --- | --- | --- |
| 2 retrievers, demographics off | `two_tower` | 51 | 0.1288 | [0.1248, 0.1333] | 0.3897 |
| 3 retrievers, demographics off | `tri_union_balanced` | 54 | 0.1273 | [0.1251, 0.1336] | 0.3690 |
| 2 retrievers, demographics on | `two_tower` | 61 | 0.1327 | [0.1298, 0.1381] | 0.4013 |
| 3 retrievers, demographics on | `tri_union_balanced` | 64 | 0.1291 | [0.1265, 0.1348] | 0.3815 |
| 3 retrievers, unbalanced, demographics on | `tri_union_unbalanced` | 64 | 0.1325 | [0.1288, 0.1372] | 0.3916 |

### Pairwise transitions in the ranker, and the served model (`ml-1m`)

From `results/ranker-pctm/ml-1m.json` (ADR-0033). Item–item and two-tower hyperparameters stay frozen from `results/ml-1m.json`, and PCTM's from `results/sequence-modern/ml-1m.json`. The candidate budget is K=200. Validation Recall@200 chooses the sequence-aware set. Ranker rows are the mean across ranker seeds 42, 43, and 44, and their interval is the primary seed. PCTM alone is one fit. No other results file is changed.

Validation Recall@200: `two_tower` 0.6552, `pctm` 0.6942, `pctm_two_tower` 0.7251, `pctm_tri` 0.7189. Selected sequence-aware set: `pctm_two_tower`.

**Serving rule 2: serve the ranker over the sequence-aware set, demographics off (`serve=seq_off`).**

| paired test NDCG@10 difference | mean and 95% interval | win |
| --- | --- | --- |
| PCTM alone minus the earlier ranker | 0.0492 [0.0435, 0.0550] (excludes 0: yes) | yes |
| Sequence ranker, demographics off, minus PCTM alone | 0.0228 [0.0203, 0.0253] (excludes 0: yes) | yes |
| Sequence ranker, demographics on, minus PCTM alone | 0.0229 [0.0205, 0.0255] (excludes 0: yes) | yes |
| Demographics on minus demographics off | 0.0001 [-0.0013, 0.0018] (excludes 0: no) | no |
| Sequence ranker, demographics off, minus the earlier ranker | 0.0720 [0.0668, 0.0776] (excludes 0: yes) | yes |
| Sequence ranker, demographics on, minus the earlier ranker | 0.0721 [0.0670, 0.0781] (excludes 0: yes) | yes |

| variant | candidate set | features | test NDCG@10 | 95% CI | Coverage@10 |
| --- | --- | --- | --- | --- | --- |
| Earlier ranker: two-tower candidates, demographics on | `two_tower` | 61 | 0.1309 | [0.1251, 0.1336] | 0.3970 |
| PCTM list, no ranker | `pctm` | 0 | 0.1787 | [0.1737, 0.1837] | 0.7112 |
| Ranker over the sequence-aware set, demographics off | `pctm_two_tower` | 53 | 0.2021 | [0.1961, 0.2068] | 0.5100 |
| Ranker over the sequence-aware set, demographics on | `pctm_two_tower` | 63 | 0.2019 | [0.1966, 0.2067] | 0.4974 |

Top feature gains of the served ranker (primary seed): `pctm_rank` 68084, `two_tower_rank` 13229, `pctm_score` 10347, `item_item_score` 5509, `item_popularity` 3114, `two_tower_score` 1961.

### A larger Discover catalogue (`ml-32m`)

From `results/extended-catalog/ml-32m.json` (ADR-0034). The catalogue is the 8769 films listed in `ml-latest-small` with at least 50 ratings in ml-32m. 10000 users are held out of the fit, half for validation and half for test. The profile is a user's first N ratings and the target is every later rating of 4 or higher. Test NDCG@10 is scored once.

**Negative result (`ships=False`).** Validation kept `ml1m_choice`. Item–item fold-in against most-popular on the test users:

| N | test users | most popular | fold-in | paired fold-in − popular | fold-in Coverage@10 | popular Coverage@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4992 | 0.3455 | 0.2688 | -0.0767 [-0.0848, -0.0688] (excludes 0: yes) | 0.2773 | 0.0013 |
| 3 | 4992 | 0.3270 | 0.2921 | -0.0349 [-0.0437, -0.0274] (excludes 0: yes) | 0.2383 | 0.0015 |
| 5 | 4991 | 0.3149 | 0.2898 | -0.0251 [-0.0326, -0.0176] (excludes 0: yes) | 0.2340 | 0.0017 |
| 10 | 4990 | 0.2930 | 0.2751 | -0.0180 [-0.0260, -0.0103] (excludes 0: yes) | 0.2341 | 0.0022 |

From `results/extended-catalog/ml-32m-blend.json` (ADR-0035). The score is fold-in similarity scaled to 1, plus a weight λ on log rating count. The validation users are the same. The test users are a new sample, and all 15000 held-out users stay out of the fit.

**Negative result (`ships=False`).** Validation kept λ=4.0. Per-N serving choice on validation: N=1 `most_popular`, N=3 `item_item_popularity_blend`, N=5 `item_item_popularity_blend`, N=10 `item_item_popularity_blend`.

| N | test users | most popular | blend | paired blend − popular | blend Coverage@10 | popular Coverage@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4995 | 0.3406 | 0.3149 | -0.0256 [-0.0320, -0.0192] (excludes 0: yes) | 0.1024 | 0.0013 |
| 3 | 4993 | 0.3217 | 0.3250 | 0.0033 [-0.0028, 0.0096] (excludes 0: no) | 0.1019 | 0.0015 |
| 5 | 4992 | 0.3094 | 0.3228 | 0.0134 [0.0077, 0.0198] (excludes 0: yes) | 0.0981 | 0.0017 |
| 10 | 4988 | 0.2873 | 0.3104 | 0.0231 [0.0176, 0.0291] (excludes 0: yes) | 0.0998 | 0.0023 |

<!-- END RESULTS TABLE -->
