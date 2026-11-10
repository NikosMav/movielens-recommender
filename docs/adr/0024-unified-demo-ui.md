# ADR-0024: One MovieLens demo page

## Status

Accepted. This decision records the public page. It does not rerun experiments, rewrite `results/`, retrain a model, or commit MovieLens files or an API key.

## Context

The public demo is the Streamlit Community Cloud app at `https://nikosmav-movielens-recommender.streamlit.app/`, entrypoint `space/app.py` (ADR-0020, ADR-0022, ADR-0023). `main` at `d5422d7` serves two interfaces from that file.

The first is a compiled React page (Discover and Model Lab) mounted in a component iframe. It is the front door: a serif headline, a title search, a short ratings list, and a three-column profile comparison. It is light-only. In Streamlit's dark theme the iframe stays a light panel inside a dark frame. Posters are absent. Title search is a substring match, so "toy" ranks Babes in Toyland ahead of Toy Story. Rating a film is a dialog. Reasons repeat the ranker's sentences ("neural retriever", "nearest neighbours") and a generic popularity line.

The second is the earlier Streamlit page (Existing user, New user, How good is it?). It has posters, genre shelves, and the committed-metrics view, and its colours follow Streamlit's theme with `light-dark()` (ADR-0022). It is reached only by the button "Genre shelves, posters and evaluation". Its metrics view is a wall of raw identifiers and six-decimal tables, with no chart and no plain-language lead. Poster misses are cached, including rate-limit failures, so a card can stay on "No poster" after TMDB answers again.

A viewer should not have to learn which door holds which feature. The React page is the better first impression. The Streamlit page is the one that can follow Settings > Theme, because `light-dark()` reads the colour scheme on `.stApp`. `--st-*` variables and `st.context.theme.type` do not reliably restyle HTML inside a component iframe when the viewer changes theme (ADR-0022).

## Decision

**One page.** `space/app.py` is the only public interface. It has three sections in one navigation: Discover, Explore a MovieLens user, and How good is it?. Discover is the new-user cold start: search, genre shelves, ratings, and picks. Explore is the existing-user view plus the compare-models control that Model Lab used. How good is it? reads the committed results JSON. The classic radio and the "Genre shelves, posters and evaluation" button are not part of the page. The React bundle remains the local API client; the public process does not mount it.

**One visual system.** The page keeps the front door's type, purple accent, hero, search, and ratings ticket, and draws them in the Streamlit document with `light-dark()`. Light and dark both meet WCAG AA for text (4.5:1) and UI boundaries (3:1). The page does not set `color-scheme` and does not add `[theme] base`, so Settings > Theme and the system preference still choose the scheme. There is no second palette and no light-only island.

**Posters.** Every film on a pick, a shelf, a search hit, or a comparison uses the TMDB poster URL from ADR-0022, or a drawn placeholder (initials on a gradient). The server still does not download image bytes. Lookups retry transient HTTP failures with backoff, keep at most two searches in flight, and pause between requests to the real API. A hit, and a definitive miss, is stored with `st.cache_data` for one day. A rate-limit or network failure is not stored as a miss. The TMDB notice, logo, and link stay in the footer.

**Copy.** The main list shows at most two plain reasons, de-duplicated, without "neural retriever" or "nearest neighbours". The ranker's original sentences and contribution rows stay behind a disclosure. Demographic attributes stay filtered out (ADR-0009). How good is it? leads with which stored model is ahead, by how much, and whether the stored intervals overlap, plus one sentence on NDCG@10. A bar chart draws those stored points and intervals and highlights the leader. Figures on that page are formatted from the JSON at runtime. The full tables and the existing captions stay in a disclosure. Friendly names are display labels; the stored model id is shown beside them.

**What must not change.** Top-10 item ids for the pinned users and the pinned new-user profile stay the ids from `tests/test_demo_recommendation_ids.py`. `results/` stays byte-identical. The ranker, the retrievers, and the scored features are not retrained. The server `VmHWM` on the ADR-0023 walk stays under 870,400 kB. The walk's clicks follow the three sections above; the budget does not.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Keep the React iframe and restyle it | The iframe does not follow Settings > Theme through `light-dark()`. ADR-0022 already recorded that limit |
| Keep both pages and restyle the classic one to match | Two navigation models, two copies of ratings and reasons, and the stray button the audit called out |
| Hard-code the headline NDCG figures in the chart | They would drift from `results/`. The page reads the JSON |
| Cache a failed TMDB lookup as "no poster" | A 429 then sticks for the cache TTL. The audit saw posters flip to the placeholder across reloads |
| Show the ranker sentence unchanged on the card | The audit's main complaint about reasons was that jargon, repeated on almost every card |

## Consequences

- Community Cloud picks up the unified page from `space/app.py` without a new entrypoint.
- A deployment without `TMDB_API_KEY` still recommends. Cards use the placeholder and send no search.
- `results/` is still only read. A test checks the quality view against the JSON fields.
- The React source under `frontend/` still builds the local client and the unused bundle. CI keeps that bundle in sync. The public app does not load it.
