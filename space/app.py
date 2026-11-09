"""MovieLens demo for Streamlit Community Cloud. Loads parameters; it does not train.

The entrypoint is this file. ``MOVIELENS_ARTIFACTS`` and
``MOVIELENS_COLD_ARTIFACTS`` point at full snapshots (the test path). Otherwise
the page reads ``space-weights.tar.gz`` beside this file, or downloads the
pinned release archive, checks its SHA-256, extracts it into ``weights/``, and
rebuilds the snapshots after downloading ml-1m.

Poster bytes are not loaded here. A card either points the browser at a TMDB
image URL or draws a placeholder. See ADR-0022.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

_SPACE_DIR = Path(__file__).resolve().parent
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from cards import (  # noqa: E402
    THEME_CSS,
    footer_html,
    hero_html,
    load_logo_svg,
    recommendation_card_html,
)
from catalog import (  # noqa: E402
    clean_year,
    movie_records,
    popularity_order,
    present_genres,
    select_grid,
    upsert_rating,
)
from posters import POSTER_CACHE_TTL_SECONDS  # noqa: E402
from quality import (  # noqa: E402
    display_rows,
    load_quality,
    ml1_caption,
    ml32_caption,
    sasrec_caption,
)

_N_RECS = 10
_N_RECENT = 10
_MIN_NEW_RATINGS = 3
_CITATION = (
    "MovieLens ml-1m is downloaded from GroupLens and is not redistributed here. "
    "GroupLens and the University of Minnesota do not endorse this demo. "
    "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: "
    "History and Context. ACM Transactions on Interactive Intelligent Systems "
    "5, 4, Article 19. DOI 10.1145/2827872."
)
_MODES = ("Existing user", "New user", "How good is it?")


def _weights_root() -> Path:
    return _SPACE_DIR / "weights"


def _runtime_root() -> Path:
    override = os.environ.get("MOVIELENS_SPACE_RUNTIME")
    if override:
        return Path(override)
    return _SPACE_DIR / ".runtime"


def _results_dir() -> Path:
    override = os.environ.get("MOVIELENS_RESULTS")
    if override:
        return Path(override)
    return _SPACE_DIR.parent / "results"


def _read_manifest(path: Path) -> dict | None:
    file = path / "manifest.json"
    if not file.is_file():
        return None
    import json

    payload = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _usable_snapshot(path: Path) -> bool:
    manifest = _read_manifest(path)
    return manifest is not None and not manifest.get("weights_only")


def tmdb_api_key() -> str:
    """Environment first, then a Streamlit secret. Never a committed value."""
    env = os.environ.get("TMDB_API_KEY", "").strip()
    if env:
        return env
    secret_paths = (
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
    if not any(path.is_file() for path in secret_paths):
        return ""
    try:
        value = st.secrets["TMDB_API_KEY"]
    except Exception:
        return ""
    return str(value).strip()


@st.cache_resource(show_spinner=False)
def _load(path_str: str):
    from movielens_recommender.serving.bundle import load_bundle, user_summaries
    from runtime import use_sparse_item_item_loader

    use_sparse_item_item_loader()
    bundle = load_bundle(path_str)
    return bundle, user_summaries(bundle)


@st.cache_resource(show_spinner=False)
def _load_cold(path_str: str):
    from movielens_recommender.serving.cold import load_cold_start_bundle
    from runtime import use_sparse_item_item_loader

    use_sparse_item_item_loader()
    return load_cold_start_bundle(path_str)


@st.cache_data(show_spinner=False)
def _recommendations(path_str: str, user_id: int, n: int) -> dict[str, Any]:
    from movielens_recommender.serving.bundle import recommend_for_user

    bundle, _summaries = _load(path_str)
    return recommend_for_user(bundle, int(user_id), n=n)


@st.cache_data(show_spinner=False)
def _catalog(path_str: str) -> tuple[list[dict[str, Any]], dict[int, int]]:
    bundle = _load_cold(path_str)
    records = movie_records(bundle.movies)
    popular = (
        None
        if bundle.popular_item_ids is None
        else [int(item) for item in bundle.popular_item_ids]
    )
    item_ids = [int(item) for item in bundle.context.item_ids]
    popularity = [float(value) for value in bundle.context.popularity]
    return records, popularity_order(popular, item_ids, popularity)


@st.cache_data(show_spinner=False, ttl=POSTER_CACHE_TTL_SECONDS)
def _poster(title: str, year: int | None, api_key: str) -> str | None:
    from posters import poster_url

    return poster_url(title, year, api_key)


@st.cache_data(show_spinner=False)
def _quality(path_str: str) -> dict[str, Any]:
    return load_quality(path_str)


@st.cache_resource(show_spinner="Downloading MovieLens ml-1m and loading the model…")
def _materialize(weights_str: str, runtime_str: str) -> tuple[str, str]:
    from runtime import materialize_space

    production, cold = materialize_space(weights_str, runtime_str)
    return str(production), str(cold)


def _warm(prod: Path, cold: Path, *, has_prod: bool, has_cold: bool) -> None:
    """Load both modes once so a later switch reads the cache."""
    if has_prod:
        _bundle, summaries = _load(str(prod.resolve()))
        if summaries:
            user_id = sorted(summaries)[0]
            _recommendations(str(prod.resolve()), int(user_id), _N_RECS)
    if has_cold:
        _catalog(str(cold.resolve()))


def _label(summary: dict) -> str:
    genres = ", ".join(summary["top_genres"]) if summary["top_genres"] else "no genres"
    return f"{summary['user_id']} · {summary['n_ratings']} ratings · {genres}"


def _poster_for(title: str, year: object, api_key: str) -> str | None:
    if not api_key:
        return None
    return _poster(title, clean_year(year), api_key)


def _chunks(items: list[dict], size: int) -> list[list[dict]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _show_cards(cards: list[dict], *, explain: bool, api_key: str, per_row: int) -> None:
    from runtime import visible_details, visible_reasons

    if not cards:
        st.caption("No recommendations for this user.")
        return
    for start in range(0, len(cards), per_row):
        chunk = cards[start : start + per_row]
        columns = st.columns(len(chunk))
        for offset, (column, card) in enumerate(zip(columns, chunk, strict=True)):
            with column:
                reasons = visible_reasons(list(card.get("reasons") or [])) if explain else []
                st.markdown(
                    recommendation_card_html(
                        title=str(card["title"]),
                        year=card.get("year"),
                        genres=str(card.get("genres") or ""),
                        reasons=reasons,
                        poster_url=_poster_for(str(card["title"]), card.get("year"), api_key),
                        rank=start + offset + 1,
                        show_reasons=explain,
                    ),
                    unsafe_allow_html=True,
                )
                if not explain:
                    continue
                with st.expander("Why this result?"):
                    if reasons:
                        for text in reasons:
                            st.markdown(text)
                    else:
                        st.markdown("The ranker did not give a positive reason for this title.")
                    with st.expander("Details"):
                        st.caption(
                            "Contributions to this title's ranker score. "
                            "They explain the score, not why the person would like the film."
                        )
                        for row in visible_details(list(card.get("details") or [])):
                            st.text(f"{row['feature']}: {row['contribution']:+.4f}")
                        bias = card.get("bias")
                        if bias is not None:
                            st.text(f"bias: {bias:+.4f}")


def _missing(prod: Path, cold: Path) -> None:
    st.markdown(
        "No model parameters were found. `space-weights.tar.gz` is missing "
        "or its SHA-256 does not match. From a checkout of the GitHub repo, "
        "with the `rank`, `deep`, and `ui` extras installed:"
    )
    st.code("make space-snapshot", language="bash")
    st.markdown(
        "Pack that `weights/` directory as `space-weights.tar.gz` and place it "
        "next to `app.py`, or publish that archive at the pinned release URL. "
        "The app extracts it into `weights/` and downloads ml-1m from GroupLens. "
        "It does not train."
    )
    st.caption(f"Looked in `{prod}` and `{cold}`.")


def _existing_user(path: Path, api_key: str) -> None:
    path_str = str(path.resolve())
    bundle, summaries = _load(path_str)
    manifest = bundle.manifest
    candidate_set = str(manifest.get("candidate_set", "two_tower"))
    demographics = str(manifest.get("demographics", "off"))
    dataset = str(manifest.get("dataset", "ml-1m"))
    demo_sentence = ""
    if demographics != "off":
        demo_sentence = (
            " Demographic group-affinity features are in the ranker. "
            "They are not named as personal attributes; a contribution from them "
            "reads “Popular with viewers similar to you.”"
        )
    st.markdown(
        f"Production model on `{dataset}`: candidates from the validation-chosen "
        f"`{candidate_set}` set, re-ranked by LightGBM LambdaRank.{demo_sentence} "
        "Recorded metrics are on the How good is it page."
    )
    st.caption(
        "This mode uses an existing MovieLens user. Switch to New user to rate "
        "a few films yourself. A new profile is kept in this session only."
    )

    user_ids = sorted(summaries)
    if not user_ids:
        st.markdown("The snapshot has no users.")
        return
    selected = st.selectbox(
        "User",
        options=user_ids,
        format_func=lambda uid: _label(summaries[int(uid)]),
    )
    summary = summaries[int(selected)]
    genre_text = ", ".join(summary["top_genres"]) if summary["top_genres"] else "—"
    st.markdown(f"**{summary['n_ratings']}** training ratings. Most common genres: {genre_text}.")
    from movielens_recommender.serving.bundle import recent_history

    history = recent_history(bundle, int(selected), _N_RECENT)
    with st.expander("Recent ratings the model was allowed to use"):
        st.caption("The per-user test holdout is excluded.")
        if history:
            st.dataframe(history, hide_index=True, width="stretch")
        else:
            st.caption("No training ratings for this user.")

    compare = st.checkbox("Compare with item–item cosine and two-tower alone", value=False)
    with st.spinner("Scoring this viewer…"):
        result = _recommendations(path_str, int(selected), _N_RECS)
    if not compare:
        st.subheader("Top 10")
        _show_cards(result["production"], explain=True, api_key=api_key, per_row=2)
        return

    production, item_item, tower = st.columns(3)
    with production:
        st.subheader("Production ranker")
        _show_cards(result["production"], explain=True, api_key=api_key, per_row=1)
    with item_item:
        st.subheader("Item–item cosine")
        _show_cards(result["item_item"], explain=False, api_key=api_key, per_row=1)
    with tower:
        st.subheader("Two-tower")
        _show_cards(result["two_tower"], explain=False, api_key=api_key, per_row=1)


def _choice_label(record: dict[str, Any]) -> str:
    year = record.get("year")
    title = str(record["title"])
    if year in (None, ""):
        return title
    return f"{title} ({year})"


def _apply_stars(item_id: int, title: str, stars: int | None) -> None:
    if stars is None:
        return
    rows = list(st.session_state["new_user_ratings"])
    current = next((row for row in rows if int(row["item_id"]) == int(item_id)), None)
    if current is not None and int(current["rating"]) == int(stars):
        return
    st.session_state["new_user_ratings"] = upsert_rating(rows, item_id, title, stars)
    st.session_state["new_user_result"] = None


def _star_control(record: dict[str, Any], api_key: str) -> None:
    item_id = int(record["item_id"])
    generation = int(st.session_state.get("rate_generation", 0))
    st.markdown(
        recommendation_card_html(
            title=str(record["title"]),
            year=record.get("year"),
            genres=str(record.get("genres") or ""),
            reasons=[],
            poster_url=_poster_for(str(record["title"]), record.get("year"), api_key),
            show_reasons=False,
        ),
        unsafe_allow_html=True,
    )
    value = st.feedback("stars", key=f"rate-{generation}-{item_id}")
    stars = None if value is None else int(value) + 1
    _apply_stars(item_id, str(record["title"]), stars)


def _new_user(path: Path, api_key: str) -> None:
    path_str = str(path.resolve())
    bundle = _load_cold(path_str)
    records, ranks = _catalog(path_str)
    manifest = bundle.manifest
    dataset = str(manifest.get("dataset", "ml-1m"))
    st.markdown(
        f"New-user model on `{dataset}`: pick one or more genres, then rate films "
        "on the cards. The page does not ask for gender, age, occupation, or ZIP. "
        "Recorded cold-start metrics are on the How good is it page."
    )
    st.caption(
        "Rate at least 3 films. About 5 is a good start. "
        "Ratings stay in this browser session and are not saved."
    )
    if "new_user_ratings" not in st.session_state:
        st.session_state["new_user_ratings"] = []

    genres = present_genres(records)
    selected_genres = st.multiselect(
        "Genres",
        options=genres,
        placeholder="Pick one or more genres",
    )
    by_id = {int(record["item_id"]): record for record in records}
    ordered = sorted(records, key=lambda record: str(record["title"]).lower())
    found = st.selectbox(
        "Find a title",
        options=[int(record["item_id"]) for record in ordered],
        format_func=lambda item_id: _choice_label(by_id[int(item_id)]),
        index=None,
        placeholder="Type to filter titles",
        filter_mode="fuzzy",
    )
    featured_id = None if found is None else int(found)
    if featured_id is not None and featured_id in by_id:
        _star_control(by_id[featured_id], api_key)

    shelf = select_grid(records, list(selected_genres), ranks)
    if featured_id is not None:
        shelf = [record for record in shelf if int(record["item_id"]) != featured_id]
    if not selected_genres:
        st.caption("Pick one or more genres to open a shelf of films.")
    elif not shelf and featured_id is None:
        st.caption("No films in the snapshot match those genres.")
    else:
        st.caption("Click the stars on a card. The list underneath uses those ratings.")
        for chunk in _chunks(shelf, 4):
            columns = st.columns(len(chunk))
            for column, record in zip(columns, chunk, strict=True):
                with column:
                    _star_control(record, api_key)

    ratings = list(st.session_state["new_user_ratings"])
    if ratings:
        st.markdown("**Your ratings**")
        st.dataframe(
            [{"title": row["title"], "stars": row["rating"]} for row in ratings],
            hide_index=True,
            width="stretch",
        )
        if st.button("Clear ratings"):
            st.session_state["new_user_ratings"] = []
            st.session_state["new_user_result"] = None
            generation = int(st.session_state.get("rate_generation", 0))
            st.session_state["rate_generation"] = generation + 1
            st.rerun()
    else:
        st.caption("No ratings yet.")

    ready = len(ratings) >= _MIN_NEW_RATINGS
    if not ready:
        remaining = _MIN_NEW_RATINGS - len(ratings)
        st.caption(f"Add {remaining} more rating(s) to continue.")
    if st.button("Get recommendations", disabled=not ready):
        from movielens_recommender.serving.cold import recommend_new_user

        profile = [(int(row["item_id"]), float(row["rating"])) for row in ratings]
        with st.spinner("Scoring this profile…"):
            result = recommend_new_user(bundle, profile, n=_N_RECS)
        st.session_state["new_user_result"] = result

    result = st.session_state.get("new_user_result")
    if not result:
        return
    latency = float(result["latency_sec"])
    sentence = result.get("method_sentence")
    if sentence:
        st.markdown(str(sentence))
    st.caption(f"Scored in {latency:.3f}s. Rated movies are left out of the list.")
    st.subheader("Top 10")
    _show_cards(result["recommendations"], explain=True, api_key=api_key, per_row=2)


def _quality_page() -> None:
    st.subheader("How good is it?")
    st.markdown(
        "Every number on this page is read from the committed results JSON. "
        "The page does not rescore the models."
    )
    try:
        view = _quality(str(_results_dir().resolve()))
    except FileNotFoundError:
        st.markdown("The committed results JSON is not next to this app.")
        return
    st.markdown(f"ml-1m model against model, from `{view['sources']['ml-1m']}`.")
    st.dataframe(display_rows(view["ml1m"]["models"]), hide_index=True, width="stretch")
    st.markdown(ml1_caption(view))
    st.markdown(f"ml-32M, from `{view['sources']['ml-32m']}`.")
    st.dataframe(display_rows(view["ml32m"]["models"]), hide_index=True, width="stretch")
    st.markdown(ml32_caption(view))
    st.markdown(f"SASRec order check, from `{view['sources']['sasrec']}`.")
    st.markdown(sasrec_caption(view))


def _downloaded_weights() -> Path | None:
    """Extract ``space-weights.tar.gz`` from the Space root into ``weights/``."""
    from runtime import install_space_weights

    return install_space_weights(_SPACE_DIR)


def _resolve() -> tuple[Path, Path] | None:
    prod_env = os.environ.get("MOVIELENS_ARTIFACTS")
    cold_env = os.environ.get("MOVIELENS_COLD_ARTIFACTS")
    if prod_env and _usable_snapshot(Path(prod_env)):
        prod = Path(prod_env)
        cold = Path(cold_env) if cold_env else prod / "cold_start"
        return prod, cold
    try:
        weights = _downloaded_weights()
    except Exception:
        return None
    if weights is None:
        return None
    try:
        prod_str, cold_str = _materialize(str(weights), str(_runtime_root()))
    except Exception:
        return None
    return Path(prod_str), Path(cold_str)


def _footer() -> None:
    st.caption(_CITATION)
    logo = load_logo_svg(_SPACE_DIR / "tmdb_blue_short.svg")
    st.markdown(
        footer_html(logo_svg=logo, api_key_set=bool(tmdb_api_key())),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="MovieLens recommender", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(hero_html(), unsafe_allow_html=True)
    api_key = tmdb_api_key()
    resolved = _resolve()
    if resolved is None:
        prod_env = os.environ.get("MOVIELENS_ARTIFACTS")
        cold_env = os.environ.get("MOVIELENS_COLD_ARTIFACTS")
        prod = Path(prod_env) if prod_env else _weights_root() / "production"
        cold = Path(cold_env) if cold_env else _weights_root() / "cold_start"
        _missing(prod, cold)
        _footer()
        return

    prod, cold = resolved
    has_prod = (prod / "manifest.json").is_file() and _usable_snapshot(prod)
    has_cold = (cold / "manifest.json").is_file() and _usable_snapshot(cold)
    if not has_prod and not has_cold:
        _missing(prod, cold)
        _footer()
        return

    with st.spinner("Loading both recommenders…"):
        _warm(prod, cold, has_prod=has_prod, has_cold=has_cold)

    default = 0 if has_prod else 1
    mode = st.radio(
        "Who is this for?",
        options=list(_MODES),
        index=default,
        horizontal=True,
    )
    with st.spinner("Opening this view…"):
        if mode == "Existing user":
            if not has_prod:
                st.markdown("The production snapshot is missing.")
                st.code("make space-snapshot", language="bash")
            else:
                _existing_user(prod, api_key)
        elif mode == "New user":
            if not has_cold:
                st.markdown("The new-user snapshot is missing.")
                st.code("make space-snapshot", language="bash")
            else:
                _new_user(cold, api_key)
        else:
            _quality_page()
    _footer()


main()
