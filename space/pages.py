"""Discover, Explore, and How good is it? One visual system, three sections."""

from __future__ import annotations

from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from cards import (
    funnel_html,
    hero_html,
    icon_html,
    metric_chart_html,
    neighbour_grid_html,
    poster_html,
    rated_line_html,
    recommendation_card_html,
    scroll_script,
)
from catalog import (
    clean_year,
    genre_parts,
    movie_records,
    popular_shelf,
    present_genres,
    rank_titles,
    select_grid,
    suggested_user_ids,
    upsert_rating,
)
from loading import (
    N_RECS,
    catalog,
    load_bundle,
    load_cold,
    popularity_lookup,
    poster_for,
    quality_view,
    recommendations,
    results_dir,
    similar_films,
)
from phrases import cap_popular, plain_method, plain_reasons, reason_choice
from quality import (
    NDCG_SENTENCE,
    beyond_definitions,
    beyond_rows,
    beyond_takeaway,
    chart_rows,
    detail_html,
    friendly_name,
    lead_paragraph,
    ml1_caption,
    ml32_caption,
    overview_models,
    rounded,
    sample_sentence,
    sasrec_caption,
    sasrec_lead,
    serving_latency_notes,
)

_SECTIONS = ("Discover", "Explore a MovieLens user", "How good is it?")
_SLUGS = {
    "Discover": "discover",
    "Explore a MovieLens user": "explore",
    "How good is it?": "quality",
}
_FROM_SLUG = {slug: name for name, slug in _SLUGS.items()}
_STARTERS = ("The Matrix", "Toy Story", "Fargo")
_MIN_RATINGS = 3

_BRAND = (
    '<a class="brand" href="?view=discover">'
    '<span class="brand-mark" aria-hidden="true">M</span> MovieLens</a>'
)


def _sync_nav() -> None:
    st.session_state["nav_scroll_pending"] = True
    slug = _SLUGS[str(st.session_state.get("nav") or "Discover")]
    if st.query_params.get("view") != slug:
        st.query_params["view"] = slug


def render_header() -> str:
    """Brand plus the three sections. The radio follows the query string on first load."""
    if "nav" not in st.session_state:
        requested = st.query_params.get("view", "discover")
        st.session_state["nav"] = _FROM_SLUG.get(str(requested), "Discover")
    with st.container(key="topbar"):
        brand_col, nav_col = st.columns([1.15, 2])
        with brand_col:
            st.markdown(_BRAND, unsafe_allow_html=True)
        with nav_col:
            choice = st.radio(
                "Section",
                _SECTIONS,
                key="nav",
                horizontal=True,
                label_visibility="collapsed",
                on_change=_sync_nav,
            )
    return str(choice)


def render_discover(cold_path: str, api_key: str) -> None:
    st.markdown(hero_html(), unsafe_allow_html=True)
    if not cold_path:
        st.markdown("The new-user snapshot is missing.")
        st.code("make space-snapshot", language="bash")
        return
    _discover_body(cold_path, api_key)


def render_explore(prod_path: str, api_key: str) -> None:
    if not prod_path:
        st.markdown("The production snapshot is missing.")
        st.code("make space-snapshot", language="bash")
        return
    _explore_body(prod_path, api_key)


def render_quality() -> None:
    _quality_body(str(results_dir().resolve()))


@st.fragment
def _discover_body(cold_path: str, api_key: str) -> None:
    records, ranks = catalog(cold_path)
    by_id = {int(record["item_id"]): record for record in records}
    _restore_shared_ratings(cold_path, by_id)
    ratings = _ratings()
    rated = {int(row["item_id"]): int(row["rating"]) for row in ratings}

    st.markdown(
        f'<h2 class="section-title icon-heading">{icon_html("film")}What have you watched?</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Search for a title, or browse a genre, then tap a star.</p>',
        unsafe_allow_html=True,
    )
    with st.container(key="workbench"):
        side, browse = st.columns([0.9, 1.7], gap="medium")
        with side:
            with st.container(key="ticket"):
                _ticket(ratings, by_id, cold_path, api_key)
        with browse:
            query = st.text_input(
                "Search for a movie",
                key="movie_query",
                placeholder="e.g. The Matrix, Toy Story, Fargo",
            )
            st.caption("Search the MovieLens 1M catalog: films released through 2000.")
            st.caption(
                "Or try a sample profile. This replaces your ratings with up to four 5-star films."
            )
            _taste_sets(by_id)
            st.caption("Try a title")
            with st.container(key="starters"):
                chips = st.columns(len(_STARTERS))
                for column, title in zip(chips, _STARTERS, strict=True):
                    column.button(
                        title, key=f"chip-{title}", on_click=_set_query, args=(title,)
                    )
            hits = rank_titles(query, records, ranks)
            if query.strip() and len(" ".join(query.split())) >= 2 and not hits:
                st.caption("No match in this catalog. Try another title or browse a genre below.")
            if hits:
                st.markdown("**Matches**")
                with st.container(key="matches-frame"):
                    _poster_grid(
                        hits, rated, api_key, grid_key="matches", films_path=cold_path, cold=True
                    )
            genres = present_genres(records)
            selected = st.pills(
                "Genres",
                genres,
                selection_mode="multi",
                default=[],
                key="shelf_genres",
            )
            chosen = list(selected or [])
            if chosen:
                shelf = select_grid(records, chosen, ranks)
                heading = "Genre shelf"
            else:
                shelf = popular_shelf(records, ranks)
                heading = "Popular films"
            shown = {int(record["item_id"]) for record in hits}
            shelf = [record for record in shelf if int(record["item_id"]) not in shown]
            if shelf:
                st.markdown(f"**{heading}**")
                st.markdown(
                    '<p class="sr-only">The star row names the film. '
                    "Filled stars show the current rating.</p>",
                    unsafe_allow_html=True,
                )
                with st.container(key="shelf-frame"):
                    _poster_grid(
                        shelf, rated, api_key, grid_key="shelf", films_path=cold_path, cold=True
                    )
            elif chosen:
                st.caption("No films in the snapshot match those genres.")

    genres_by_title = {
        str(record["title"]): str(record.get("genres") or "") for record in records
    }
    _picks(ratings, api_key, genres_by_title, cold_path, records, ranks, by_id)


@st.fragment
def _explore_body(prod_path: str, api_key: str) -> None:
    bundle, summaries = load_bundle(prod_path)
    genres_by_title = {
        str(record["title"]): str(record.get("genres") or "")
        for record in movie_records(bundle.movies)
    }
    user_ids = sorted(int(user_id) for user_id in summaries)
    st.markdown(
        f'<h1 class="display icon-heading">{icon_html("users")}Explore a MovieLens user</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Look up a viewer and see ten films the model would suggest '
        "from the films they rated. You can compare that list with two simpler ones.</p>",
        unsafe_allow_html=True,
    )
    if not user_ids:
        st.markdown("This snapshot has no viewers.")
        return
    lo, hi = user_ids[0], user_ids[-1]
    st.markdown(f"Viewer ids in this snapshot run from **{lo}** to **{hi}**.")
    suggestions = suggested_user_ids(user_ids)
    with st.container(key="viewer-bar"):
        field, action = st.columns([1, 1], vertical_alignment="bottom")
        with field:
            entered = st.number_input(
                "Viewer id",
                min_value=int(lo),
                max_value=int(hi),
                value=None,
                step=1,
                key="viewer_id",
                placeholder=str(suggestions[0]) if suggestions else str(lo),
                width=240,
            )
        with action:
            load_clicked = st.button("Load viewer", type="primary")
    if load_clicked:
        if entered is None:
            st.session_state["explore_error"] = f"Enter a viewer id from {lo} to {hi}."
        else:
            _load_viewer(prod_path, summaries, int(entered), lo, hi)
    if suggestions:
        st.caption("Try a viewer")
        with st.container(key="tries"):
            cols = st.columns(len(suggestions))
            for column, user_id in zip(cols, suggestions, strict=True):
                column.button(
                    f"Try {user_id}",
                    key=f"try-{user_id}",
                    on_click=_queue_viewer,
                    args=(int(user_id),),
                )
    requested = st.session_state.pop("explore_requested", None)
    if isinstance(requested, int):
        _load_viewer(prod_path, summaries, requested, lo, hi)
    error = str(st.session_state.get("explore_error") or "")
    if error:
        st.error(error)
    result = st.session_state.get("explore_result")
    if not result:
        st.markdown("Load a viewer to see ten films and why they were picked.")
        return
    if error:
        st.caption("The list below is still the previous viewer.")
    _viewer_result(result, api_key, genres_by_title, bundle, prod_path)


def _quality_body(path_str: str) -> None:
    st.markdown(
        f'<h1 class="display icon-heading">{icon_html("chart")}How good is it?</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(NDCG_SENTENCE)
    st.caption("Every number on this page is read from the committed results JSON.")
    try:
        view = quality_view(path_str)
    except FileNotFoundError:
        st.markdown("The committed results JSON is not next to this app.")
        return
    _dataset_block(
        "MovieLens 1M",
        view["ml1m"]["models"],
        "MovieLens 1M",
        "Compare the full model with similarity, retrieval, and popularity.",
    )
    _dataset_block(
        "MovieLens 32M",
        view["ml32m"]["models"],
        "MovieLens 32M",
        "The same models, scored on the larger set of viewers.",
    )
    st.markdown(sample_sentence(view["ml32m"]))
    st.markdown(
        f'<h2 class="section-title icon-heading">{icon_html("history")}Sequence model</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(sasrec_lead(view["sasrec"]))
    st.markdown(
        metric_chart_html(
            _sasrec_rows(view["sasrec"]),
            caption="The stored ranking score for the real watch order and a shuffled order.",
        ),
        unsafe_allow_html=True,
    )
    with st.expander("Detailed scores"):
        st.markdown("**MovieLens 1M**")
        st.markdown(detail_html(view["ml1m"]["models"]), unsafe_allow_html=True)
        st.markdown(ml1_caption(view))
        st.markdown("**MovieLens 32M**")
        st.markdown(detail_html(view["ml32m"]["models"]), unsafe_allow_html=True)
        st.markdown(ml32_caption(view))
        st.caption(str(view["ml32m"]["selection"]))
        st.markdown("**Sequence model**")
        st.markdown(sasrec_caption(view))
    _beyond_panel()
    _committed_funnel()


def _dataset_block(title: str, models: list[dict[str, Any]], dataset: str, note: str) -> None:
    st.markdown(f'<h2 class="section-title">{title}</h2>', unsafe_allow_html=True)
    summary = _escape(lead_paragraph(models, dataset=dataset))
    st.markdown(f'<p class="benchmark-summary">{summary}</p>', unsafe_allow_html=True)
    overview = overview_models(models)
    rows = [dict(row, group="") for row in chart_rows(overview)]
    st.markdown(metric_chart_html(rows, caption=note), unsafe_allow_html=True)
    if len(overview) < len(models):
        with st.expander(f"Compare all {len(models)} models · {title}"):
            st.markdown(
                metric_chart_html(chart_rows(models), caption="All recorded model variants."),
                unsafe_allow_html=True,
            )


def _sasrec_rows(block: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "model": "sasrec",
            "label": friendly_name("sasrec"),
            "ndcg": float(block["ndcg@10_mean"]),
            "low": float(block["primary_low"]),
            "high": float(block["primary_high"]),
        },
        {
            "model": "sasrec_shuffled",
            "label": friendly_name("sasrec_shuffled"),
            "ndcg": float(block["shuffled_ndcg@10"]),
            "low": float(block["shuffled_low"]),
            "high": float(block["shuffled_high"]),
        },
    ]
    winner = max(rows, key=lambda row: (row["ndcg"], row["model"]))["model"]
    for row in rows:
        row["best"] = row["model"] == winner
    return rows


def _read_json(name: str) -> dict[str, Any] | None:
    path = results_dir() / name
    if not path.is_file():
        return None
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _cold_start_nudge() -> str:
    from value import cold_start_nudge

    payload = _read_json("cold-start/ml-1m.json") or {}
    return cold_start_nudge(payload)


def _taste_sets(by_id: dict[int, dict[str, Any]]) -> None:
    from value import TASTE_SETS

    shown = []
    for group in TASTE_SETS:
        ids = [int(item) for item in group["item_ids"] if int(item) in by_id]
        if len(ids) < 3:
            continue
        shown.append((str(group["label"]), ids))
    if not shown:
        return
    columns = st.columns(len(shown))
    for column, (label, ids) in zip(columns, shown, strict=True):
        column.button(
            f"Try {label} sample",
            key=f"taste-{label}",
            on_click=_rate_taste,
            args=(ids, by_id),
        )


def _rate_taste(item_ids: list[int], by_id: dict[int, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for row in _ratings():
        st.session_state.pop(f"rate-{int(row['item_id'])}", None)
    for item_id in item_ids:
        record = by_id.get(int(item_id))
        if record is None:
            continue
        rows = upsert_rating(rows, int(item_id), str(record["title"]), 5)
        st.session_state[f"rate-{int(item_id)}"] = 5
    st.session_state["new_user_ratings"] = rows


def _restore_shared_ratings(cold_path: str, by_id: dict[int, dict[str, Any]]) -> None:
    """Apply a share link once, before the star widgets are created."""
    if st.session_state.get("share_restored"):
        return
    st.session_state["share_restored"] = True
    from value import decode_ratings

    mix = str(st.query_params.get("mix") or "0")
    if mix in {"0", "1", "2", "3", "4"}:
        st.session_state["familiarity"] = _FAMILIARITY[int(mix)]

    pairs = decode_ratings(str(st.query_params.get("ratings") or ""))
    rows: list[dict[str, Any]] = []
    for item_id, stars in pairs:
        record = by_id.get(int(item_id))
        if record is None:
            continue
        rows.append(
            {"item_id": int(item_id), "title": str(record["title"]), "rating": int(stars)}
        )
        st.session_state[f"rate-{int(item_id)}"] = int(stars)
    if not rows:
        return
    st.session_state["new_user_ratings"] = rows
    if len(rows) >= _MIN_RATINGS:
        _score(cold_path, rows, by_id)


def _beyond_table(rows: list[dict[str, Any]]) -> str:
    labels = ("Model", "Coverage", "Novelty", "Diversity", "Long-tail share")
    head = "".join(f"<th>{_escape(label)}</th>" for label in labels)
    body = []
    for row in rows:
        cells = (
            friendly_name(str(row["model"])),
            rounded(row["coverage@10"]),
            rounded(row["novelty@10"]),
            rounded(row["intra_list_diversity@10"]),
            rounded(row["long_tail_share@10"]),
        )
        body.append("<tr>" + "".join(f"<td>{_escape(cell)}</td>" for cell in cells) + "</tr>")
    return (
        '<div class="beyond-table"><table><thead><tr>'
        + head
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _beyond_panel() -> None:
    payload = _read_json("beyond-accuracy/ml-1m.json")
    if not payload or "models" not in payload:
        return
    rows = beyond_rows(payload)
    st.markdown(
        f'<h2 class="section-title icon-heading">{icon_html("compass")}Beyond accuracy</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(beyond_takeaway(rows))
    st.markdown(_beyond_table(rows), unsafe_allow_html=True)
    definitions = beyond_definitions(payload)
    if definitions:
        with st.expander("What these mean"):
            for label, text in definitions:
                st.markdown(f"**{label}.** {text}")


def _committed_funnel() -> None:
    from value import funnel_steps

    st.markdown(
        f'<h2 class="section-title icon-heading">{icon_html("route")}How it works</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(funnel_html(funnel_steps(None)), unsafe_allow_html=True)
    _latency_notes(_read_json("serving/ml-1m.json") or {})


def _live_funnel(pipeline: dict[str, Any] | None) -> None:
    from value import funnel_steps

    with st.expander("How it works"):
        st.markdown(funnel_html(funnel_steps(pipeline)), unsafe_allow_html=True)
        _latency_notes(_read_json("serving/ml-1m.json") or {})
        st.markdown("<div class='funnel-pad'></div>", unsafe_allow_html=True)


def _latency_notes(payload: dict[str, Any]) -> None:
    notes = serving_latency_notes(payload)
    if not notes:
        return
    st.caption("Committed serving measurement, not the time for this click.")
    rows = []
    for visible, tip in notes:
        rows.append(
            f'<p class="latency" title="{_escape(tip)}">{_escape(visible)}</p>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def _rated_label(count: int) -> str:
    if count <= 0:
        return "Nothing rated yet"
    if count == 1:
        return "1 film"
    return f"{count} films"


def _ratings() -> list[dict[str, Any]]:
    rows = st.session_state.get("new_user_ratings")
    if not isinstance(rows, list):
        rows = []
        st.session_state["new_user_ratings"] = rows
    return list(rows)


def _signature(ratings: list[dict[str, Any]]) -> tuple[tuple[int, int], ...]:
    return tuple((int(row["item_id"]), int(row["rating"])) for row in ratings)


def _set_rating(item_id: int, title: str, stars: int) -> None:
    st.session_state["new_user_ratings"] = upsert_rating(_ratings(), item_id, title, stars)


def _remove_rating(item_id: int) -> None:
    st.session_state["new_user_ratings"] = [
        row for row in _ratings() if int(row["item_id"]) != int(item_id)
    ]
    st.session_state.pop(f"rate-{int(item_id)}", None)


def _star_label(score: int) -> str:
    return f"{int(score)} out of 5"


def _apply_rating(item_id: int, title: str) -> None:
    raw = st.session_state.get(f"rate-{int(item_id)}")
    if raw is None:
        st.session_state["new_user_ratings"] = [
            row for row in _ratings() if int(row["item_id"]) != int(item_id)
        ]
        return
    _set_rating(int(item_id), str(title), int(raw))


def _set_query(title: str) -> None:
    st.session_state["movie_query"] = title


def _queue_viewer(user_id: int) -> None:
    st.session_state["viewer_id"] = int(user_id)
    st.session_state["explore_requested"] = int(user_id)


def _poster_grid(
    records: list[dict[str, Any]],
    rated: dict[int, int],
    api_key: str,
    *,
    grid_key: str,
    films_path: str,
    cold: bool,
) -> None:
    with st.container(key=grid_key):
        for record in records:
            item_id = int(record["item_id"])
            title = str(record["title"])
            with st.container(key=f"film-{grid_key}-{item_id}"):
                st.markdown(
                    recommendation_card_html(
                        title=title,
                        year=record.get("year"),
                        genres=str(record.get("genres") or ""),
                        reasons=[],
                        poster_url=poster_for(title, record.get("year"), api_key),
                        show_reasons=False,
                        kind="poster",
                    ),
                    unsafe_allow_html=True,
                )
                _stars(item_id, title, rated.get(item_id))
                _more_like_this(item_id, films_path, cold, api_key, scope=f"{grid_key}-{item_id}")


def _stars(item_id: int, title: str, current: int | None) -> None:
    st.segmented_control(
        f"Rate {title}",
        [1, 2, 3, 4, 5],
        default=None if current is None else int(current),
        key=f"rate-{int(item_id)}",
        on_change=_apply_rating,
        args=(int(item_id), title),
        label_visibility="collapsed",
        format_func=_star_label,
        width="content",
    )


def _ticket(
    ratings: list[dict[str, Any]],
    by_id: dict[int, dict[str, Any]],
    cold_path: str,
    api_key: str,
) -> None:
    filled = min(_MIN_RATINGS, len(ratings))
    bars = "".join(
        "<span class='on'></span>" if index < filled else "<span></span>"
        for index in range(_MIN_RATINGS)
    )
    st.markdown(
        f'<div class="ticket"><h2 class="icon-heading">{icon_html("star")}Your ratings</h2>'
        f'<p class="muted">{_rated_label(len(ratings))}</p>'
        f'<div class="progress" aria-hidden="true">{bars}</div></div>',
        unsafe_allow_html=True,
    )
    if not ratings:
        st.caption("No ratings yet. They stay in this browser session.")
    if ratings:
        with st.expander("Review or remove ratings"):
            for row in ratings:
                item_id = int(row["item_id"])
                title = str(row["title"])
                record = by_id.get(item_id, {})
                with st.container(key=f"rated-{item_id}"):
                    st.markdown(
                        rated_line_html(
                            title=title,
                            rating=int(row["rating"]),
                            poster_url=poster_for(title, record.get("year"), api_key),
                        ),
                        unsafe_allow_html=True,
                    )
                    st.button(
                        f"Remove {title}",
                        key=f"remove-{item_id}",
                        on_click=_remove_rating,
                        args=(item_id,),
                    )
    remaining = _MIN_RATINGS - len(ratings)
    if remaining == 1:
        st.caption("Rate 1 more film to get picks.")
    elif remaining > 1:
        st.caption(f"Rate {remaining} more films to get picks.")
    else:
        st.caption("You can get picks now, or rate a few more.")
    stored = st.session_state.get("new_user_result")
    stale = bool(stored) and stored.get("signature") != _signature(ratings)
    label = "Update picks" if stale else "Get my picks"
    if st.button(label, type="primary", disabled=len(ratings) < _MIN_RATINGS):
        _score(cold_path, ratings, by_id)


def _score(
    cold_path: str,
    ratings: list[dict[str, Any]],
    by_id: dict[int, dict[str, Any]],
) -> None:
    from movielens_recommender.serving.cold import recommend_new_user

    known = set(by_id)
    profile = [(int(row["item_id"]), float(row["rating"])) for row in ratings]
    if any(item_id not in known for item_id, _rating in profile):
        st.error("One of these films is not in the catalog. Remove it and try again.")
        return
    bundle = load_cold(cold_path)
    with st.spinner("Scoring your picks…"):
        result = recommend_new_user(bundle, profile, n=N_RECS)
    st.session_state["new_user_result"] = {
        "signature": _signature(ratings),
        "recommendations": result["recommendations"],
        "method": result.get("method"),
        "method_sentence": result.get("method_sentence"),
        "pipeline": result.get("pipeline"),
    }
    st.session_state["picks_nonce"] = int(st.session_state.get("picks_nonce") or 0) + 1


def _picks(
    ratings: list[dict[str, Any]],
    api_key: str,
    genres_by_title: dict[str, str],
    cold_path: str,
    records: list[dict[str, Any]],
    ranks: dict[int, int],
    by_id: dict[int, dict[str, Any]],
) -> None:
    stored = st.session_state.get("new_user_result")
    if not stored:
        return
    stale = stored.get("signature") != _signature(ratings)
    st.markdown('<div id="your-next-watch"></div>', unsafe_allow_html=True)
    st.markdown(
        '<h2 id="picks-heading" class="section-title icon-heading" tabindex="-1">'
        f'{icon_html("list")}Your next watch</h2>',
        unsafe_allow_html=True,
    )
    if stale:
        st.markdown('<div class="picks-stale" hidden></div>', unsafe_allow_html=True)
    if stale and len(ratings) < _MIN_RATINGS:
        st.warning("These picks need at least 3 ratings. Add a film to refresh them.")
    elif stale:
        st.warning("These picks used your earlier ratings. Update picks to match the list above.")
    if stale:
        if st.button("Refresh my picks", type="primary", disabled=len(ratings) < _MIN_RATINGS):
            _score(cold_path, ratings, by_id)
            st.rerun(scope="fragment")
        return
    method = str(stored.get("method") or "")
    st.caption(plain_method(method) if method else "Ranked from the films you rated.")
    if st.session_state.get("scroll_nonce") != st.session_state.get("picks_nonce"):
        st.session_state["scroll_nonce"] = st.session_state.get("picks_nonce")
        components.html(scroll_script(), height=0)
    cards = list(stored.get("recommendations") or [])
    if not cards:
        st.caption("No films were returned for these ratings.")
        return
    pipeline = stored.get("pipeline") if isinstance(stored.get("pipeline"), dict) else None
    can_adjust = bool(
        pipeline
        and pipeline.get("ranked_ids")
        and len(pipeline.get("ranked_scores") or []) == len(pipeline["ranked_ids"])
    )
    if not can_adjust:
        st.session_state["familiarity"] = "Familiar"
    shown = _reranked_cards(cards, pipeline, cold_path, by_id)
    _list_controls(shown, cold_path, can_adjust=can_adjust)
    history = _history_rows(ratings)
    liked_by_id = _liked_by_item(_item_item_model(cold_path, cold=True), shown, history)
    rated_titles = {str(row["title"]) for row in ratings}
    _card_grid(
        shown,
        api_key,
        grid_key="picks",
        kind="rich",
        explain=True,
        reason_limit=2,
        genres_by_title=genres_by_title,
        liked_by_id=liked_by_id,
        films_path=cold_path,
        cold=True,
        rated_titles=rated_titles,
        hide_unrated=stale,
        history=history,
    )
    _pick_tools(records, ranks, ratings, pipeline, shown)
    _live_funnel(pipeline)
    sentence = str(stored.get("method_sentence") or "")
    if sentence:
        with st.expander("How this list was built"):
            st.markdown(plain_method(str(stored.get("method") or "")))
            with st.expander("Technical wording"):
                st.markdown(sentence)


def _load_viewer(
    prod_path: str,
    summaries: dict[int, dict[str, Any]],
    user_id: int,
    lo: int,
    hi: int,
) -> None:
    if user_id not in summaries:
        st.session_state["explore_error"] = (
            f"No viewer {user_id}. Ids in this snapshot run from {lo} to {hi}."
        )
        return
    from movielens_recommender.serving.bundle import recent_history

    bundle, _summaries = load_bundle(prod_path)
    with st.spinner("Scoring this viewer…"):
        result = recommendations(prod_path, int(user_id), N_RECS)
    st.session_state["explore_error"] = ""
    st.session_state["explore_result"] = {
        "summary": summaries[int(user_id)],
        "recent": recent_history(bundle, int(user_id), 10),
        "production": result["production"],
        "item_item": result["item_item"],
        "two_tower": result["two_tower"],
        "pipeline": result.get("pipeline"),
    }


def _viewer_result(
    result: dict[str, Any],
    api_key: str,
    genres_by_title: dict[str, str],
    bundle: Any,
    prod_path: str,
) -> None:
    summary = result["summary"]
    genres = " · ".join(summary["top_genres"]) if summary["top_genres"] else "No genre stands out"
    st.markdown(
        f'<h2 class="section-title">A viewer with {int(summary["n_ratings"])} ratings</h2>',
        unsafe_allow_html=True,
    )
    st.caption(f"Viewer {int(summary['user_id'])} · {genres}")
    with st.expander("Recent ratings"):
        recent = list(result.get("recent") or [])
        if not recent:
            st.caption("No training ratings for this viewer.")
        else:
            st.caption("The per-user test holdout is excluded.")
            for row in recent:
                _history_row(row, api_key)
    compare = st.checkbox(
        "Compare three models",
        key="compare_models",
        help="Full model, similar films, and a taste match, from the same ratings.",
    )
    history = _bundle_history(bundle, int(summary["user_id"]))
    model = getattr(bundle, "item_item", None)
    if not compare:
        st.markdown('<h2 class="section-title">Top 10</h2>', unsafe_allow_html=True)
        st.caption("The list from the full model.")
        production = list(result["production"])
        _card_grid(
            production,
            api_key,
            grid_key="top10",
            kind="rich",
            explain=True,
            reason_limit=2,
            genres_by_title=genres_by_title,
            liked_by_id=_liked_by_item(model, production, history),
            films_path=prod_path,
            cold=False,
            history=history,
        )
        _live_funnel(result.get("pipeline") if isinstance(result.get("pipeline"), dict) else None)
        return
    options = ("Full model", "Similar films", "Taste match")
    with st.container(key="compare-switch"):
        choice = st.segmented_control(
            "Which list",
            options,
            default="Full model",
            key="compare_tab",
        )
    mode = {
        "Full model": "production",
        "Similar films": "item_item",
        "Taste match": "two_tower",
    }.get(choice or "Full model", "production")
    blocks = (
        ("Full model", "production", "col-production", result["production"]),
        ("Similar films", "item_item", "col-item-item", result["item_item"]),
        ("Taste match", "two_tower", "col-two-tower", result["two_tower"]),
    )
    blurbs = {
        "production": "The list this viewer would get.",
        "item_item": "Films close to ones they rated highly.",
        "two_tower": "A learned match between this viewer and each film.",
    }
    with st.container(key="compare"):
        st.markdown(
            f'<div class="compare-pick" data-mode="{mode}" hidden></div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(3)
        for column, (title, raw, col_key, cards) in zip(columns, blocks, strict=True):
            # Taste match is the two-tower list. Item–item closeness is not
            # that model's signal, so those cards stay a title and a poster.
            liked_by_id = (
                {}
                if raw == "two_tower"
                else _liked_by_item(model, list(cards), history)
            )
            prepared = (
                [[] for _card in cards]
                if raw == "two_tower"
                else _prepared_lines(
                    list(cards),
                    liked_by_id,
                    genres_by_title,
                    reason_limit=1,
                )
            )
            with column, st.container(key=col_key):
                st.markdown(f"### {title}")
                st.caption(blurbs[raw])
                for index, card in enumerate(cards, start=1):
                    _recommendation(
                        card,
                        api_key,
                        rank=index,
                        explain=False,
                        kind="compact",
                        scope=f"{raw}-{index}",
                        reason_limit=1,
                        genres_by_title=genres_by_title,
                        prepared=prepared[index - 1],
                        show_line=raw != "two_tower",
                        films_path=prod_path,
                        cold=False,
                        history=history,
                    )


def _history_row(row: dict[str, Any], api_key: str) -> None:
    title = str(row.get("title") or "")
    raw_year = row.get("year")
    year = clean_year(None if raw_year == "" else raw_year)
    genres = " · ".join(genre_parts(str(row.get("genres") or "")))
    bits = ["" if year is None else str(year), genres]
    meta = " · ".join(bit for bit in bits if bit)
    stars = str(row.get("stars") or "")
    st.markdown(
        '<div class="history-row">'
        f"{poster_html(title, poster_for(title, year, api_key))}"
        "<div>"
        f'<p class="pick-title">{_escape(title)}</p>'
        f'<p class="pick-meta">{_escape(meta)}</p>'
        "</div>"
        f'<p class="history-stars">{_escape(stars)}</p>'
        "</div>",
        unsafe_allow_html=True,
    )


def _card_grid(
    cards: list[dict[str, Any]],
    api_key: str,
    *,
    grid_key: str,
    kind: str,
    explain: bool,
    reason_limit: int,
    genres_by_title: dict[str, str] | None = None,
    liked_by_id: dict[int, list[tuple[str, float]]] | None = None,
    films_path: str = "",
    cold: bool = False,
    rated_titles: set[str] | None = None,
    hide_unrated: bool = False,
    history: list[tuple[int, str, float]] | None = None,
) -> None:
    prepared = _prepared_lines(
        cards,
        liked_by_id or {},
        genres_by_title,
        reason_limit=reason_limit,
    )
    with st.container(key=grid_key):
        for index, card in enumerate(cards, start=1):
            _recommendation(
                card,
                api_key,
                rank=index,
                explain=explain,
                kind=kind,
                scope=f"{grid_key}-{index}",
                reason_limit=reason_limit,
                genres_by_title=genres_by_title,
                prepared=prepared[index - 1],
                films_path=films_path,
                cold=cold,
                rated_titles=rated_titles,
                hide_unrated=hide_unrated,
                history=history,
            )


def _recommendation(
    card: dict[str, Any],
    api_key: str,
    *,
    rank: int,
    explain: bool,
    kind: str = "rich",
    scope: str = "card",
    reason_limit: int = 2,
    genres_by_title: dict[str, str] | None = None,
    prepared: list[str] | None = None,
    show_line: bool = True,
    films_path: str = "",
    cold: bool = False,
    rated_titles: set[str] | None = None,
    hide_unrated: bool = False,
    history: list[tuple[int, str, float]] | None = None,
) -> None:
    from runtime import visible_details, visible_reasons

    original = visible_reasons(list(card.get("reasons") or []))
    if prepared is not None:
        shown = list(prepared)
    elif show_line and (explain or kind == "compact"):
        shown = plain_reasons(
            visible_reasons(list(card.get("reasons") or [])),
            limit=reason_limit,
            genres_by_title=genres_by_title,
            candidate_genres=str(card.get("genres") or ""),
            candidate_title=str(card.get("title") or ""),
            candidate_year=card.get("year"),
        )
    else:
        shown = []
    if not explain and kind != "compact":
        shown = []
    if history is not None:
        from value import qualify_likes

        shown = qualify_likes(shown, {title for _item, title, stars in history if stars >= 4})
    if hide_unrated and shown:
        from value import cites_unrated

        kept = rated_titles or set()
        shown = [line for line in shown if not cites_unrated(line, kept)]
    item_id = int(card.get("item_id") or rank)
    with st.container(key=f"rec-{scope}-{item_id}"):
        st.markdown(
            recommendation_card_html(
                title=str(card["title"]),
                year=card.get("year"),
                genres=str(card.get("genres") or ""),
                reasons=shown,
                poster_url=poster_for(str(card["title"]), card.get("year"), api_key),
                rank=rank,
                show_reasons=bool(shown),
                kind=kind,
                evidence_html=_evidence_html(shown, history, api_key),
            ),
            unsafe_allow_html=True,
        )
        if films_path:
            _more_like_this(item_id, films_path, cold, api_key, scope=scope)
        if not explain:
            return
        details = visible_details(list(card.get("details") or []))
        with st.container(key=f"why-{scope}-{item_id}"):
            with st.expander("Why?"):
                if not original and not details:
                    st.markdown("No separate note was recorded for this title.")
                    return
                for text in dict.fromkeys(original):
                    st.markdown(text)
                if details or card.get("bias") is not None:
                    with st.expander("Score details"):
                        st.caption(
                            "Contributions to the score. They are not a reason to watch the film."
                        )
                        for row in details:
                            st.text(f"{row['feature']}: {row['contribution']:+.4f}")
                        bias = card.get("bias")
                        if bias is not None:
                            st.text(f"bias: {bias:+.4f}")


def _history_rows(ratings: list[dict[str, Any]]) -> list[tuple[int, str, float]]:
    return [
        (int(row["item_id"]), str(row["title"]), float(row["rating"])) for row in ratings
    ]


def _bundle_history(bundle: Any, user_id: int) -> list[tuple[int, str, float]]:
    try:
        from movielens_recommender.serving.bundle import user_history_items
        from movielens_recommender.serving.reasons import display_title
    except Exception:
        return []
    try:
        rows = user_history_items(bundle, int(user_id))
    except Exception:
        return []
    return [
        (int(item.item_id), display_title(str(item.title)), float(item.rating)) for item in rows
    ]


def _item_item_model(path_str: str, *, cold: bool) -> Any:
    try:
        bundle = load_cold(path_str) if cold else load_bundle(path_str)[0]
    except Exception:
        return None
    return getattr(bundle, "item_item", None)


def _prepared_lines(
    cards: list[dict[str, Any]],
    liked_by_id: dict[int, list[tuple[str, float]]],
    genres_by_title: dict[str, str] | None,
    *,
    reason_limit: int,
) -> list[list[str]]:
    from runtime import visible_reasons

    choices = []
    for card in cards:
        pairs = liked_by_id.get(int(card.get("item_id") or 0), [])
        choices.append(
            reason_choice(
                visible_reasons(list(card.get("reasons") or [])),
                limit=reason_limit,
                genres_by_title=genres_by_title,
                candidate_genres=str(card.get("genres") or ""),
                similar=pairs,
                candidate_year=card.get("year"),
            )
        )
    return cap_popular(choices)


def _liked_by_item(
    model: Any,
    cards: list[dict[str, Any]],
    history: list[tuple[int, str, float]],
) -> dict[int, list[tuple[str, float]]]:
    """Rated films with positive similarity to each pick, closest first.

    The order matches the explainer: similarity times the rating. The best
    cosine is kept even when it is not in that top pair, so a real neighbour
    is not dropped. This does not rescore the list.
    """
    from phrases import MEANINGFUL_SIM

    if model is None or not history or not hasattr(model, "neighbor_similarities"):
        return {}
    pool = [row for row in history if float(row[2]) >= 4.0]
    if not pool:
        return {}
    ids = [item_id for item_id, _title, _rating in pool]
    found: dict[int, list[tuple[str, float]]] = {}
    for card in cards:
        item_id = int(card.get("item_id") or 0)
        if not item_id:
            continue
        try:
            sims = model.neighbor_similarities(item_id, ids)
        except Exception:
            continue
        ranked: list[tuple[float, float, str]] = []
        for other, title, rating in pool:
            sim = float(sims.get(int(other), 0.0))
            if sim <= 0.0:
                continue
            ranked.append((sim * float(rating), sim, title))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
        pairs: list[tuple[str, float]] = []
        best: tuple[str, float] | None = None
        for _score, sim, title in ranked:
            if best is None or sim > best[1]:
                best = (title, sim)
            if any(name == title for name, _sim in pairs):
                continue
            pairs.append((title, sim))
            if len(pairs) == 2:
                break
        if (
            best is not None
            and best[1] >= MEANINGFUL_SIM
            and best[0] not in {name for name, _sim in pairs}
        ):
            pairs.append(best)
        if pairs:
            found[item_id] = pairs
    return found


def _reranked_cards(
    cards: list[dict[str, Any]],
    pipeline: dict[str, Any] | None,
    cold_path: str,
    by_id: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Position 0 keeps the measured cards. Above 0, reorder the scored candidates."""
    from value import embedding_rows, rerank_candidates

    adventurous = _adventurous_value()
    if adventurous <= 0.0 or not pipeline:
        return cards
    ids = [int(item) for item in pipeline.get("ranked_ids") or []]
    scores = [float(value) for value in pipeline.get("ranked_scores") or []]
    if not ids or len(scores) != len(ids):
        return cards
    lookup = popularity_lookup(cold_path, True)
    bundle = load_cold(cold_path)
    tower = getattr(bundle, "two_tower", None)
    features = getattr(tower, "_features", None)
    embeddings = embedding_rows(
        None if tower is None else getattr(tower, "_item_vectors", None),
        None if features is None else features.item_index,
        ids,
    )
    order = rerank_candidates(
        ids,
        scores,
        adventurous=adventurous,
        embeddings=embeddings,
        popularity=[float(lookup.get(item, 0.0)) for item in ids],
        n=len(cards) or 10,
    )
    by_card = {int(card["item_id"]): card for card in cards}
    shown: list[dict[str, Any]] = []
    for item_id in order:
        card = by_card.get(int(item_id))
        if card is not None:
            shown.append(card)
            continue
        record = by_id.get(int(item_id))
        if record is None:
            continue
        shown.append(
            {
                "item_id": int(item_id),
                "title": record["title"],
                "year": record.get("year"),
                "genres": record.get("genres") or "",
                "reasons": [],
            }
        )
    return shown or cards


_FAMILIARITY = ("Familiar", "Closer", "In between", "Further", "Adventurous")
_FAMILIARITY_VALUE = {
    "Familiar": 0.0,
    "Closer": 0.25,
    "In between": 0.5,
    "Further": 0.75,
    "Adventurous": 1.0,
}


def _adventurous_value() -> float:
    label = st.session_state.get("familiarity", "Familiar")
    if isinstance(label, int | float) and not isinstance(label, bool):
        return float(label)
    return float(_FAMILIARITY_VALUE.get(str(label), 0.0))


def _list_controls(
    cards: list[dict[str, Any]], cold_path: str, *, can_adjust: bool = True,
) -> None:
    from value import list_facts, readout_parts, unevaluated_note

    if "familiarity" not in st.session_state:
        st.session_state["familiarity"] = "Familiar"
    if can_adjust:
        st.select_slider(
            "Familiar or adventurous",
            options=list(_FAMILIARITY),
            key="familiarity",
            help="Familiar keeps the measured list. Adventurous prefers less popular films.",
        )
        note = unevaluated_note(_adventurous_value())
        if note:
            st.caption(note)
    else:
        st.caption("This list has no scored candidate details, so it cannot be adjusted.")
    found = popularity_lookup(cold_path, True) if cold_path else {}
    percentiles = [
        float(found[int(card["item_id"])])
        for card in cards
        if card.get("item_id") and int(card["item_id"]) in found
    ]
    facts = list_facts([str(card.get("genres") or "") for card in cards], percentiles)
    visible, tip = readout_parts(facts)
    st.markdown(
        f'<p class="list-readout" title="{_escape(tip)}">{_escape(visible)}</p>',
        unsafe_allow_html=True,
    )


def _pick_tools(
    records: list[dict[str, Any]],
    ranks: dict[int, int],
    ratings: list[dict[str, Any]],
    pipeline: dict[str, Any] | None,
    shown: list[dict[str, Any]],
) -> None:
    from value import share_url, why_not

    field, action = st.columns([3, 1], vertical_alignment="bottom")
    with field:
        query = st.text_input(
            "Why not this film?",
            key="why_not_query",
            placeholder="e.g. Titanic",
        )
    with action:
        share_open = st.checkbox("Share these picks", key="share-picks")
    if share_open:
        pairs = [(int(row["item_id"]), int(row["rating"])) for row in ratings]
        link = share_url(st.context.url or "", pairs, mix=round(_adventurous_value() * 4))
        st.caption(
            "Copy this link with the copy button on the code box. It includes your ratings "
            "and slider setting. Anyone with the link can see them."
        )
        st.code(link, language=None)
    if len(" ".join(str(query).split())) < 2:
        return
    if not pipeline or not pipeline.get("ranked_ids"):
        st.caption("Candidate details are unavailable for this list.")
        return
    hits = rank_titles(query, records, ranks, limit=10)
    if not hits:
        st.caption("No title matches that search.")
        return
    hit_id = st.selectbox(
        "Which film?",
        [int(hit["item_id"]) for hit in hits],
        format_func=lambda item: next(
            f"{hit['title']} ({hit.get('year') or 'year unknown'})"
            for hit in hits
            if int(hit["item_id"]) == item
        ),
        key="why-not-film",
    )
    hit = next(hit for hit in hits if int(hit["item_id"]) == hit_id)
    ranked = [] if not pipeline else [int(item) for item in pipeline.get("ranked_ids") or []]
    rated_ids = {int(row["item_id"]) for row in ratings}
    st.markdown(
        why_not(
            title=str(hit["title"]),
            item_id=int(hit["item_id"]),
            rated_ids=rated_ids,
            ranked_ids=ranked,
            shown_ids=[int(card["item_id"]) for card in shown],
            adjusted=_adventurous_value() > 0,
        )
    )


def _more_like_this(
    item_id: int,
    path: str,
    cold: bool,
    api_key: str,
    *,
    scope: str,
) -> None:
    from value import neighbour_caption

    if not path:
        return
    with st.popover("More like this"):
        payload = similar_films(path, int(item_id), cold)
        ids = [int(item) for item in payload.get("item_ids") or []]
        st.caption(neighbour_caption(str(payload.get("source") or "")))
        if not ids:
            return
        lookup = _title_lookup(path, cold)
        films = []
        for other in ids:
            record = lookup.get(other, {})
            title = str(record.get("title") or f"Item {other}")
            films.append(
                {
                    "title": title,
                    "year": record.get("year"),
                    "poster_url": poster_for(title, record.get("year"), api_key),
                }
            )
        st.markdown(neighbour_grid_html(films), unsafe_allow_html=True)


def _title_lookup(path: str, cold: bool) -> dict[int, dict[str, Any]]:
    if cold:
        records, _ranks = catalog(path)
    else:
        bundle, _summaries = load_bundle(path)
        records = movie_records(bundle.movies)
    return {int(record["item_id"]): record for record in records}


def _evidence_html(
    shown: list[str],
    history: list[tuple[int, str, float]] | None,
    api_key: str,
) -> str:
    if not shown or not shown[0].startswith("Because you liked ") or not history:
        return ""
    named = shown[0].removeprefix("Because you liked ")
    for _item_id, title, _rating in history:
        if title != named and not named.startswith(title + " and "):
            continue
        return poster_html(
            title,
            poster_for(title, None, api_key),
            extra_class="evidence-thumb",
        )
    return ""


def _escape(text: str) -> str:
    import html

    return html.escape(text)
