# ADR-0025: A more useful demo, still on the served artifacts

## Status

Accepted. This decision records the first batch of additions to the public page. It does not rerun experiments, rewrite `results/`, retrain a model, or commit MovieLens files or an API key.

## Context

The public demo is the Streamlit Community Cloud app at `https://nikosmav-movielens-recommender.streamlit.app/`, entrypoint `space/app.py` (ADR-0020, ADR-0022, ADR-0023, ADR-0024). `main` at `090162c` is one themed page with Discover, Explore a MovieLens user, and How good is it?. Posters, both colour schemes, the phone layout, and the plain-language reasons are already there.

A visitor can rate films and read a top 10. They cannot ask which films sit next to a title, move the list away from the obvious picks, see the rated film that drove a reason, ask why a title is missing, or send the list to someone else. How good is it? shows NDCG@10 and does not show the beyond-accuracy panel that is already committed. The serving latencies are committed and are not drawn next to the funnel.

The weights the page already loads are enough for this batch. The two-tower checkpoint encodes an item vector per catalog film when the snapshot loads. Item–item similarity is the fallback when those vectors are absent. The ranker already scores a candidate list and returns ten of them. `results/beyond-accuracy/ml-1m.json` stores coverage, novelty, intra-list diversity, and long-tail share. `results/serving/ml-1m.json` stores latencies. `results/cold-start/ml-1m.json` stores whether the served list beats most-popular at 1 and at 3 ratings. None of those files is an input to training here.

## Decision

**More like this.** Every film on a pick, a shelf, a search hit, or an Explore card has a compact "More like this" control. It shows six nearest films as a poster grid. The neighbours come from cosine similarity of the two-tower item vectors already in memory. If those vectors are not loaded, the same control uses the item–item similarity row and does not densify the matrix. A caption names which of the two was used. The control does not rescore the top 10.

**Familiar to adventurous.** On Discover, a slider re-ranks the candidate list from the request that just ran. Position 0 is the default. It returns that list's measured top 10 in the measured order and does not consult embeddings or popularity. `tests/test_demo_recommendation_ids.py` keeps calling `recommend_for_user` and `recommend_new_user` and is not edited. Any position above 0 runs maximal marginal relevance: relevance is the ranker score (or the retriever score when the ranker did not run), diversity is cosine similarity of the item vectors, and a novelty term prefers lower training-popularity percentile. Ties break by lower item id. The page says, in that case, that this re-ranking has not been evaluated offline and that only the default position is the measured model. Under the films on screen, two readouts describe that list only: the count of distinct genres, and the average popularity percentile. The caption says they describe the list on screen and are not evaluation scores. They are not coverage, novelty, intra-list diversity, or NDCG.

**Evidence.** On a pick, the "Because you liked …" line is the one the page already builds from the explanation. A small poster of the first film named on that line sits beside the words. The film is taken from the explanation and from the same neighbour ranking the card already uses. A demographic attribute is not a reason and is not a poster.

**Why not this film.** Under the Discover picks, a title search answers for one named film, using the candidate list from that same request. Already rated: the list leaves it out. In the candidate list at rank r of N: the page says that rank, and whether the measured top 10 includes it. Absent from the candidate list: it was not retrieved. The answer does not invent a rank the request did not compute.

**Shareable picks.** "Copy link" writes the current ratings into the query string as movie id and stars only (`ratings=id-stars.id-stars`) and keeps `view=discover`. Opening that URL restores those ratings and, once there are at least three, the picks. The URL has no name, no MovieLens user id, and no demographic field.

**How it works.** A collapsible on Discover, and the same diagram on How good is it?, draws the funnel as catalog, then candidates, then LambdaRank, then the top 10. The candidate and scored counts are the counts from the request that just ran. The candidate stage is named for the source that request used (two-tower, the cold-start union, item–item, or popularity), not a hard-coded 200. When `results/serving/ml-1m.json` has latency blocks, the panel prints those stored medians and 95th percentiles and says they are the committed measurement, not the time for this click.

**Beyond accuracy.** How good is it? gains a panel of coverage@10, novelty@10, intra-list diversity@10, and long-tail share@10 for each model in `model_order` of `results/beyond-accuracy/ml-1m.json`. The numbers are the stored points. One sentence names, for each column, the model with the highest stored point and prints that point. The sentence is built from those fields. It is not a second caption typed beside them. Definitions shown on the panel are the strings in that file.

**Quick start.** Discover offers a few taste groups (animation, crime, science fiction) as buttons. Each button rates a handful of catalog ids at 5 stars, looking the titles up from the snapshot. Ids missing from the snapshot are skipped, and a group with fewer than three hits is not shown. The nudge says to rate at least three films. It cites the cold-start file only when that file says the served list does not beat most-popular at 1 rating and does beat it at 3.

**Poster matching.** A MovieLens title is still not a TMDB title. The search also tries the base title before a colon (`Star Wars` for the 1977 episode), a known rewrite when the catalog string is not the TMDB title (`Star Trek II: The Wrath of Khan`, `Babes in Toyland` for March of the Wooden Soldiers), and a hit whose title, original title, or subtitle matches. The year order stays the release year, then one year to either side, then no year. A poster image that fails in the browser drops its address once and does not request it again. The server still does not download image bytes.

**Search.** A word inside an alternate or foreign title counts only at a word start. A mid-word substring of the main title, if it is kept at all, ranks last and is capped, so `toy` does not surface a film whose French title merely contains those letters.

**Stale picks.** The notice that the list used earlier ratings stays. Reasons that name a film the person has since removed are not shown. The list is marked as the earlier one.

**How good is it?, the chart.** The ten ml-1m bars are grouped Baselines, Retrieval, and Full pipeline. The stored model id stays on the bar, in a larger chip. In the details, `negative_result` and `excludes_zero` are plain sentences (did not beat the baseline, interval excludes zero). The numbers are still the stored fields.

**Reasons on the card.** The model is unchanged. The line on the card follows the strongest stored reason. When that reason is popularity, the card says “A widely loved film that fits your mix” and does not borrow a weakly connected film from a later sentence. When the strongest reason names a rated film, the card still says “Because you liked …”.

**What must not change.** Top-10 item ids for the pinned users and the pinned new-user profile stay the ids in `tests/test_demo_recommendation_ids.py`. `results/` stays byte-identical. The ranker, the retrievers, and the scored features are not retrained. The HTTP new-user body does not grow: the extra candidate trace is removed before the response is sent. Gender, age, occupation, ZIP, and region are not shown as reasons. The server `VmHWM` on the ADR-0023 walk stays under 870,400 kB. The walk now also opens More like this, moves the slider, asks why a title is missing, copies the link, opens How it works, opens Beyond accuracy, and taps a taste group.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Retrain a diversified ranker | The slider is a control on one list. An offline number for it does not exist, and this batch does not invent one |
| Run MMR at the default slider position | The measured top 10 would move, and the pinned-id test would fail |
| Densify item–item similarity to find neighbours | ADR-0023 removed that copy. One sparse row is enough |
| Put genre, age, or occupation on the evidence poster | ADR-0009. The poster is the rated film the explanation already names |
| Store the shared list on the server | The query string is the list. Nothing new is written |
| Type a beyond-accuracy sentence by hand | It would drift from the JSON. The sentence is a max over the stored points |
| Show a latency the serving file does not contain | The funnel would be inventing a measurement |

## Consequences

- Community Cloud picks the page up from `space/app.py`. No new entrypoint.
- A deployment without `TMDB_API_KEY` still recommends. Neighbour posters and the evidence thumb use the same placeholder as the other cards.
- `results/` is still only read. Tests check the new panel against the JSON fields and check that reading it does not change the files.
- The candidate trace lives on the in-process result the page already computed. It is not a second retrieval.
