# ADR-0034: A larger Discover catalogue from MovieLens 32M

## Status

Accepted. The catalogue rule, the held-out users, the two configurations, the per-N serving rule, and the ship rule below are fixed before any ranking metric for them is computed. A budget probe counted films and ratings only. No existing results file is changed.

## Context

Discover serves ml-1m: 3,706 films, none released after 2000. The harness already runs on MovieLens 32M (ADR-0011), which has ratings through October 2023, but the demo does not serve it.

Two limits shape what can be served.

- **Data terms.** The demo does not redistribute ratings or titles. It downloads a GroupLens archive at startup and rebuilds its tables (ADR-0020). The ml-32m archive is 239 MB, too large for a Community Cloud cold start. `ml-latest-small` is 1 MB and uses the same movie ids. Its `movies.csv` holds 9,742 titles through 2018. The probe found 9,454 of those ids in ml-32m, with identical titles and genres.
- **Memory.** The server has 1 GB (ADR-0023). The ml-1m new-user pipeline (two-tower, dense EASE, LambdaRank) does not fit a second time at 2.4 times the catalogue, and its ranker was fit and measured on ml-1m only.

The probe: 8,769 of the `ml-latest-small` films have at least 50 ratings in ml-32m. They hold 29,956,318 ratings (93.6% of the file) from 200,945 users. 3,906 of them were released after 2000 and 1,376 after 2010.

## Decision

**Catalogue.** Films listed in `ml-latest-small/movies.csv` with at least 50 ratings in ml-32m. Titles and genres are read from the `ml-latest-small` download at startup. Ratings on other films are dropped before anything is fit.

**Model.** Item–item cosine with a sparse top-k neighbourhood (the ADR-0011 block path), scored by fold-in over the rated films, with most-popular as the other method. No tower, no EASE, and no ranker: this is the `item_item_fold_in` path the new-user server already has (ADR-0012). It ships as a neighbourhood matrix and a popularity order, which are item-level aggregates and hold no rating rows, user ids, or titles.

**Users.** Eligible users have at least 20 catalogue ratings. A seeded sample of 10,000 eligible users (seed 42) is held out and split in half: 5,000 validation users and 5,000 test users. No rating from either group enters the fit.

**Configurations.** Two, both on the same fit users: `k_neighbors=50, shrinkage=0` (the ml-32m validation choice in `results/tuning/ml-32m.json`) and `k_neighbors=200, shrinkage=100` (the ml-1m choice in `results/tuning/ml-1m.json`). The one with the higher validation NDCG@10 at N=5 is kept. A tie keeps the first.

**Protocol.** The ADR-0012 main protocol: the profile is a held-out user's first N ratings in time order, for N in 1, 3, 5, and 10. The target is every later rating of 4 or higher on a catalogue film. NDCG@10 with a 1,000-resample user bootstrap, alpha 0.05, seed 42. Coverage@10 and long-tail share@10 (ADR-0017) are reported.

**Per-N serving rule.** For each N, validation users choose between `item_item_fold_in` and `most_popular` by mean NDCG@10. A tie keeps `most_popular`. Profile sizes between grid points use the largest N that does not exceed them (ADR-0012).

**Ship rule.** Test users are scored once with the kept configuration. The extended catalogue ships when the paired test difference `item_item_fold_in − most_popular` has an interval low end above 0 at N=3, N=5, and N=10. Otherwise it does not ship and this record stays as a negative result.

**In the demo.** Discover gets a choice between the two catalogues. The ml-1m catalogue stays the default, because its list is the measured ranker pipeline. The extended one says that it uses a simpler similar-films model. Explore a MovieLens user is unchanged. If the extended parameters are absent or fail their checksum, Discover shows only the ml-1m catalogue.

**Known limits.** The split holds out users, not time: a fit user's later ratings can inform a list scored on an earlier held-out profile (the ADR-0002 caveat). Films in ml-32m but not in `ml-latest-small` cannot be shown, so nothing after September 2018 is in the catalogue.

Output: `results/extended-catalog/ml-32m.json`, written by `scripts/run_extended_catalog.py`.

## Outcome

**Negative result. The fold-in model does not ship.** Sealed run (`results/extended-catalog/ml-32m.json`, 276 s). The catalogue is 8,769 films and 29,956,318 ratings. 198,709 users are eligible. The fit used 28,450,773 ratings and took about 90 s per configuration.

Validation NDCG@10 at N=5 kept `k_neighbors=200, shrinkage=100` (`0.284568` against `0.265068`). At every N the validation users preferred `most_popular`, so the per-N rule chose it for all four profile sizes.

Test users, scored once with the kept configuration:

| N | users | most popular | item–item fold-in | paired fold-in − popular | fold-in Coverage@10 | popular Coverage@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4,992 | 0.345482 | 0.268777 | `-0.076705 [-0.084822, -0.068777]` | 0.277341 | 0.001254 |
| 3 | 4,992 | 0.327034 | 0.292142 | `-0.034893 [-0.043714, -0.027396]` | 0.238340 | 0.001482 |
| 5 | 4,991 | 0.314858 | 0.289770 | `-0.025088 [-0.032629, -0.017556]` | 0.234006 | 0.001711 |
| 10 | 4,990 | 0.293044 | 0.275086 | `-0.017958 [-0.025964, -0.010309]` | 0.234120 | 0.002167 |

The ship rule needed an interval low end above 0 at N=3, 5, and 10. All three intervals are entirely below 0 (`ships=false`). With every later rating of 4 or higher as the target, the most-rated films are hard to beat on NDCG@10: a list of them reaches `0.31` to `0.35` while covering under 0.3% of the catalogue. Fold-in covers 23% to 28% and scores lower. ADR-0012 met the same wall on ml-1m, and its answer was a model that starts from popular titles and re-orders them.

No parameter files were written, and the demo is unchanged by this record. ADR-0035 registers one follow-up on users this run did not score.
