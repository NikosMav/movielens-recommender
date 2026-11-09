# ADR-0022: Posters, genre grid, and the public demo page

## Status

Accepted. ADR-0021 is already the results-and-reproducibility record, so this decision is ADR-0022. It does not rerun experiments, rewrite `results/`, or commit MovieLens files or an API key.

## Context

The public demo is the Streamlit Community Cloud app at `https://nikosmav-movielens-recommender.streamlit.app/`, entrypoint `space/app.py` (ADR-0020). Community Cloud gives this app 1 GB of RAM. ADR-0020 measured a fresh process that imported Streamlit, rebuilt both ml-1m snapshots, kept both models resident, and returned ten existing-user and ten new-user recommendations. Peak RSS was 781,632 KiB (`ru_maxrss`). That left a thin margin under 1 GB.

The page today is a form: a user dropdown, a title search that submits on Enter, and a second dropdown of matches. Recommendation rows are text. A smoke test of the live app found two further problems: the title search does not show matches until Enter, and switching between Existing user and New user spends a few seconds on screen with the previous mode still drawn.

A poster next to each explanation has to come from somewhere that a public non-commercial demo may use, with no secret committed to git. Two candidates were checked.

### Wikipedia and Wikidata

English Wikipedia lead images for films are often theatrical posters. Those files are hosted as non-free content. [Wikipedia:Non-free content](https://en.wikipedia.org/wiki/Wikipedia:Non-free_content) allows them on Wikipedia under a claim of fair use. It does not grant a licence to copy them onto another site. A public demo is that other site, so those page images are not an allowed poster source.

Wikidata can be queried without a key. A film's image claims (`P18`, and the logo claim `P154`) point at Wikimedia Commons files. Commons allows reuse outside Wikimedia, including hotlinking via `Special:FilePath`, and each file keeps its own licence ([Commons:Reusing content outside Wikimedia](https://commons.wikimedia.org/wiki/Commons:Reusing_content_outside_Wikimedia)). That licence often requires attribution of the author, which a single footer line does not satisfy. The files are also not reliable posters. A short probe against `wbsearchentities` / `wbgetentities` for well-known ml-1m-era titles, with a descriptive User-Agent, returned logos and unrelated photographs more often than posters: Toy Story resolved to a logo SVG, Four Weddings and a Funeral to a photograph of a church, Forrest Gump to a wax-museum photograph, The Godfather to a logo SVG. Some titles missed (Babe, The Usual Suspects, Lured) or hit the wrong work first (Jurassic Park hit a 1993 video game). The same burst then received HTTP 429. A genre grid would repeat that burst for every cold cache. Keyless Commons images are legal to reuse only file by file, and they are not a reliable poster shelf.

### TMDB

[TMDB's API terms of use](https://www.themoviedb.org/api-terms-of-use) allow a non-commercial application to call the API. Commercial use, and any use that earns revenue, needs a separate written agreement. The terms also forbid using TMDB content to train a machine-learning model. This demo does not train, does not sell access, and does not show ads. It displays posters for titles the already-trained ranker returned.

The same terms require three things this page will do:

- Use the TMDB logo, smaller than the application's own name, without implying endorsement.
- Show this notice prominently: "This application uses TMDB and the TMDB APIs but is not endorsed, certified, or otherwise approved by TMDB." The terms put the application kind in brackets; "application" is the matching word.
- Not cache information from the API for longer than 6 months.

TMDB does not claim ownership of the images. The developer FAQ states the same split: the API is free for non-commercial use when TMDB is attributed as the source. Search needs an API key. The poster file URL on `image.tmdb.org` does not include that key, so the browser can load the image after the server has stored only the path.

The key is not committed. The page reads `TMDB_API_KEY` from the environment, and if that variable is unset, from the Streamlit secret of the same name. A deployment without the secret still runs. Each card shows a drawn placeholder instead of a poster. No request is sent.

### Titles

Mapping a title and year to a TMDB id needs the title string. GroupLens's ml-1m README, quoted in ADR-0020, forbids redistributing the dataset without separate permission. A committed table of ml-1m titles, or of title-to-id pairs taken from `movies.dat`, would be that file. The Space already rebuilds the movie table in memory from the GroupLens download and does not commit it. Poster search uses that in-memory title and year. Nothing about the mapping is written to git. `st.cache_data` holds the returned poster path in the Streamlit process, with a one-day time-to-live, which is inside the six-month cap. The cache is not written to disk. Image bytes are not part of the cache: the card is an `<img src="https://image.tmdb.org/...">`, so the browser fetches the file and the server process does not.

A failed lookup, a missing key, a title that does not match exactly, or a network error produces the placeholder. The server does not retry inside the same cached call.

## Decision

**Poster source.** TMDB, optional key. Not Wikipedia page images, and not Wikidata/Commons. The footer carries the notice above, the TMDB logo, and a link to `https://www.themoviedb.org/`. The logo file is the short blue mark TMDB publishes with its API docs (`files.readme.io/29c6fee-blue_short.svg`), stored under `space/` so the footer does not depend on that host at render time. It is an attribution mark, not a MovieLens file.

**Lookup.** `GET /3/search/movie` with the display title, the year, and `include_adult=false`. A hit must match the normalised title (or original title) and the release year, and it must have a `poster_path`. The URL returned to the page is `https://image.tmdb.org/t/p/w185` plus that path. The API key is not put in the image URL and is not logged. `w185` is a small poster; the page does not request a larger size.

**New-user mode.** The ranker is unchanged: the same `recommend_new_user` call, the same minimum of three ratings, the same 1-to-5 scores. The form in front of it changes. The person picks one or more genres, then a shelf of about a dozen matching films ordered by the snapshot's existing popularity list (or by the feature-context popularity already in memory). They rate a film by clicking stars on its card, about five films, not by choosing from a dropdown. A separate title search remains for films outside the shelf. It is one `st.selectbox` over the in-memory catalog. Filtering happens in the widget, so matches are not gated on Enter. There is no `st.text_input` in front of a second dropdown.

**How good is it.** A third mode reads the committed JSON at runtime. It shows ml-1m NDCG@10 and the stored user-bootstrap interval for the headline models, the ranker gate against its item–item bar, the ml-32M two-tower and ranker points with their gates and the eval-sample note, and the SASRec real-minus-shuffled order check. Every number on that page is a field from `results/ml-1m.json`, `results/ml-32m.json`, or `results/sasrec/ml-1m.json`. Captions name those fields in plain language. The page does not compute a new metric, does not fill a missing interval, and does not rewrite a `negative_result` flag. A test compares the page's values to those JSON fields. `results/` stays byte-identical.

**Layout.** A hero with one line of pitch, then horizontal rows of cards. On a recommendation, the poster (or the placeholder) sits beside the plain-language reasons. The item–item versus two-tower comparison checkbox stays. Reasons still pass through `visible_reasons` and `visible_details`. Gender, age, occupation, ZIP, and region are not shown as reasons.

**Mode switch.** Both snapshots are loaded, and the catalog and the default existing-user list are cached, before the mode control is used again. `st.cache_resource` keeps the bundles. `st.cache_data` keeps the catalog, the popularity order, and each viewer's top 10. Switching modes opens inside `st.spinner`, so a slow path shows a spinner instead of leaving the previous mode's cards on screen with no indication. New-user scoring still runs only when the person asks for recommendations, not on every star.

**Memory budget.** Peak RSS of the demo process, measured with `ru_maxrss` after both models are resident and these features have run, stays under about 850 MiB. The features are not allowed to densify item–item similarity, decode poster bytes, or retain TMDB responses on disk. 850 MiB is about 870,400 KiB. ADR-0020's measured peak was 781,632 KiB on this class of VM, so the budget leaves roughly 90 MiB for the catalog, the URL cache, and the results JSON. The measurement is recorded in this ADR after the run. It is a VM measurement, not a Community Cloud measurement, same as ADR-0020.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| English Wikipedia `pageimages` | Lead art is typically a non-free poster. Wikipedia's non-free content policy does not allow that reuse on this demo |
| Wikidata `P18` / Commons only | Keyless and reusable file by file, but the probe returned logos and unrelated photos, missed titles, and was rate-limited. Per-file author attribution does not fit one footer line |
| Commit a title-to-TMDB id file | The strings would be the MovieLens title list. ADR-0020 does not redistribute that file |
| Require `TMDB_API_KEY` to boot | The live app must still recommend when the secret is absent. Placeholders cover that case |
| Download poster bytes in the server | Copies image weight into the 1 GB process. The browser can fetch the URL |
| Cache TMDB responses on disk or for the default `st.cache_data` lifetime | The terms forbid keeping API information for longer than 6 months. The TTL is one day, in memory |
| Replace the new-user ranker with a genre-average heuristic | The shelf is only how ratings are collected. Scoring stays `recommend_new_user` |
| Drop the title search | A genre shelf cannot reach every catalog title. One filtering selectbox keeps search without the Enter gate |
| Hard-code the README metrics into the page | They would drift from `results/`. The page reads the JSON |
| Show a demographic feature as a reason next to a poster | ADR-0009 and ADR-0020. The display filter stays |

## Consequences

- The README badge points at the live app instead of the pending anchor.
- Community Cloud shows posters only after `TMDB_API_KEY` is set in the app's secrets. Until then every card uses the placeholder. Setting the secret does not require a code change.
- The first cold lookup of a shelf is a handful of search calls, then the one-day cache. The image host is contacted by the browser, not by this process.
- `results/` is not an input to training here and must not change. The quality page is a reader.
- The RSS measurement below is the check against the 850 MiB budget. If a later change loads images in-process or densifies a similarity matrix, it is over budget even when the UI looks the same.
