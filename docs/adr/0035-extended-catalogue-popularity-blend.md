# ADR-0035: Similarity blended with popularity for the larger Discover catalogue

## Status

Accepted. The score, the grid, the users, and the ship rule below are fixed before any ranking metric for them is computed. ADR-0034's record (`results/extended-catalog/ml-32m.json`) stays read-only.

## Context

ADR-0034 fit item–item cosine on the 8,769-film catalogue and scored fold-in against most-popular. Fold-in lost at every profile size, with intervals entirely below 0, so nothing shipped. Most-popular scored `0.31` to `0.35` NDCG@10 while covering under 0.3% of the catalogue. Fold-in covered about 23% and scored `0.27` to `0.29`.

Those two lists use different information. One knows which films nearly everyone goes on to rate. The other knows which films sit close to the ones this viewer rated. The ml-1m new-user ranker beats popularity by combining both (ADR-0012). A ranker is not available here: it would need a second feature pipeline inside the 1 GB server (ADR-0023). A fixed blend of the two scores needs one number and the same parameter files.

ADR-0034's test users have been scored once, so they cannot decide this.

## Decision

**Score.** For a profile with fold-in scores `f` over the catalogue (the item–item neighbourhood matrix times the profile's ratings) and fit rating counts `c`:

`score(a) = f(a) / max(f) + λ · log(1 + c(a)) / log(1 + max(c))`

The first term is 0 for every film when `max(f)` is 0. Rated films are removed. Ties go to the lower item index (ADR-0015). `λ = 0` is fold-in and a very large `λ` is most-popular.

**Neighbourhood.** `k_neighbors=200, shrinkage=100`, the configuration ADR-0034's validation users kept. It is not chosen again.

**Users.** Validation users are ADR-0034's 5,000 validation users. Test users are a new seeded sample of 5,000 (seed 43) from eligible users who are in neither ADR-0034 group. The fit leaves out all 15,000: ADR-0034's validation and test users and the new test users.

**Grid.** `λ` in `0.25, 0.5, 1, 2, 4`. The value with the highest validation NDCG@10 at N=5 is kept. A tie keeps the smaller `λ`.

**Protocol.** As ADR-0034: profile is the first N ratings for N in 1, 3, 5, and 10, the target is every later rating of 4 or higher on a catalogue film, NDCG@10 with a 1,000-resample user bootstrap, alpha 0.05, seed 42, plus Coverage@10 and long-tail share@10.

**Per-N serving rule.** For each N, validation users choose between the blend and `most_popular` by mean NDCG@10. A tie keeps `most_popular`.

**Ship rule.** Test users are scored once. The larger catalogue ships when the paired test difference `blend − most_popular` has an interval low end above 0 at N=3, N=5, and N=10. Otherwise it does not ship, and no third attempt is made on this protocol without new test users.

**In the demo, if it ships.** As ADR-0034 described: Discover offers both catalogues, the ml-1m one stays the default, and the larger one says it uses a simpler model. The parameters are the neighbourhood matrix, the fit rating counts, and `λ`. If they are absent or fail their checksum, Discover shows only the ml-1m catalogue.

Output: `results/extended-catalog/ml-32m-blend.json`, written by `scripts/run_extended_blend.py`.

## Outcome

**Negative result by the registered rule. The larger catalogue does not ship.** Sealed run (`results/extended-catalog/ml-32m-blend.json`, 207 s). The fit used 27,693,021 ratings with 15,000 users left out. ADR-0034's record is unchanged.

Validation NDCG@10 at N=5 rose with every step of the grid (`0.289030` at `λ=0.25` to `0.318720` at `λ=4`), so `λ=4`, the largest registered value, was kept. Validation chose `most_popular` at N=1 and the blend at N=3, 5, and 10.

New test users, scored once with `λ=4`:

| N | users | most popular | blend | paired blend − popular | blend Coverage@10 | popular Coverage@10 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4,995 | 0.340565 | 0.314945 | `-0.025620 [-0.031997, -0.019197]` | 0.102406 | 0.001254 |
| 3 | 4,993 | 0.321694 | 0.324986 | `+0.003292 [-0.002751, +0.009599]` | 0.101950 | 0.001482 |
| 5 | 4,992 | 0.309413 | 0.322835 | `+0.013422 [+0.007713, +0.019762]` | 0.098073 | 0.001711 |
| 10 | 4,988 | 0.287292 | 0.310350 | `+0.023058 [+0.017573, +0.029084]` | 0.099783 | 0.002281 |

The rule needed a low end above 0 at N=3, 5, and 10. N=5 and N=10 clear it. N=3 does not: its interval includes 0 (`ships=false`). The blend is a real gain over popularity from five ratings on, and it covers about 10% of the catalogue against 0.2%. It is not yet a list worth serving to someone who has rated three films, which is where Discover starts.

Two readings of this record are worth carrying forward, and neither changes the verdict.

- The kept weight is at the edge of the grid. The validation curve was still rising, so the best fixed blend is probably above `λ=4`. That is a reason to register a wider grid with new test users. It is not a reason to rerun this one.
- A fixed blend has one weight for every profile size. The ml-1m new-user model beats popularity at N=3 because its ranker sees the profile size and many item features (ADR-0012). The same design on this catalogue, with item-level features shipped as aggregates, is the next thing to register.

No parameter files were written. The code this needs is in place and inactive: the similarity-only snapshot and the blend score (`serving/cold.py`), the startup rebuild from parameter files (`materialize_extended` in `space/runtime.py`), and a catalogue switch in Discover that appears only when such a snapshot exists. The live demo's Discover section is unchanged.
