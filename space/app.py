"""MovieLens demo for a Hugging Face Space. Loads parameters; it does not train.

Push this directory as the Space repository root. ``MOVIELENS_ARTIFACTS`` and
``MOVIELENS_COLD_ARTIFACTS`` point at full snapshots (the test path). Otherwise
the page reads ``space-weights.tar.gz`` beside this file, checks its SHA-256,
extracts it into ``weights/``, and rebuilds the snapshots after downloading ml-1m.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

_SPACE_DIR = Path(__file__).resolve().parent
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

RESULTS_URL = "https://github.com/NikosMav/movielens-recommender/blob/main/README.md#results"
_N_RECS = 10
_N_RECENT = 10
_MIN_NEW_RATINGS = 3
_SEARCH_LIMIT = 30
_CITATION = (
    "MovieLens ml-1m is downloaded from GroupLens and is not redistributed here. "
    "GroupLens and the University of Minnesota do not endorse this demo. "
    "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: "
    "History and Context. ACM Transactions on Interactive Intelligent Systems "
    "5, 4, Article 19. DOI 10.1145/2827872."
)


def _weights_root() -> Path:
    return _SPACE_DIR / "weights"


def _runtime_root() -> Path:
    override = os.environ.get("MOVIELENS_SPACE_RUNTIME")
    if override:
        return Path(override)
    return _SPACE_DIR / ".runtime"


def _read_manifest(path: Path) -> dict | None:
    file = path / "manifest.json"
    if not file.is_file():
        return None
    payload = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _usable_snapshot(path: Path) -> bool:
    manifest = _read_manifest(path)
    return manifest is not None and not manifest.get("weights_only")


@st.cache_resource(show_spinner="Loading the fitted model…")
def _load(path_str: str):
    from movielens_recommender.serving.bundle import load_bundle, user_summaries

    bundle = load_bundle(path_str)
    return bundle, user_summaries(bundle)


@st.cache_resource(show_spinner="Loading the new-user model…")
def _load_cold(path_str: str):
    from movielens_recommender.serving.cold import load_cold_start_bundle

    return load_cold_start_bundle(path_str)


@st.cache_resource(show_spinner="Downloading MovieLens ml-1m and loading the model…")
def _materialize(weights_str: str, runtime_str: str) -> tuple[str, str]:
    from runtime import materialize_space

    production, cold = materialize_space(weights_str, runtime_str)
    return str(production), str(cold)


def _label(summary: dict) -> str:
    genres = ", ".join(summary["top_genres"]) if summary["top_genres"] else "no genres"
    return f"{summary['user_id']} · {summary['n_ratings']} ratings · {genres}"


def _show_list(cards: list[dict], *, explain: bool) -> None:
    from runtime import visible_details, visible_reasons

    if not cards:
        st.caption("No recommendations for this user.")
        return
    for rank, card in enumerate(cards, start=1):
        year = card["year"]
        year_text = str(year) if year not in (None, "") else ""
        genres = card["genres"] or "—"
        bits = " · ".join(bit for bit in (year_text, genres) if bit)
        st.markdown(f"**{rank}. {card['title']}**")
        st.caption(bits)
        if explain:
            with st.expander("Why this result?"):
                reasons = visible_reasons(list(card.get("reasons") or []))
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
        "Pack that `weights/` directory as `space-weights.tar.gz`, copy it "
        "next to `app.py`, then `git lfs install`, add, and push `space/` "
        "as the Hugging Face Space root. The Space extracts the archive "
        "into `weights/` and downloads ml-1m from GroupLens. It does not train."
    )
    st.caption(f"Looked in `{prod}` and `{cold}`.")


def _existing_user(path: Path) -> None:
    bundle, summaries = _load(str(path.resolve()))
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
        f"Recorded metrics are in the [README results]({RESULTS_URL})."
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
    st.caption(
        "Recent ratings the model was allowed to use. The per-user test holdout is excluded."
    )
    from movielens_recommender.serving.bundle import recent_history, recommend_for_user

    history = recent_history(bundle, int(selected), _N_RECENT)
    if history:
        st.dataframe(history, hide_index=True, width="stretch")
    else:
        st.caption("No training ratings for this user.")

    compare = st.checkbox("Compare with item–item cosine and two-tower alone", value=False)
    result = recommend_for_user(bundle, int(selected), n=_N_RECS)
    if not compare:
        st.subheader("Top 10")
        _show_list(result["production"], explain=True)
        return

    production, item_item, tower = st.columns(3)
    with production:
        st.subheader("Production ranker")
        _show_list(result["production"], explain=True)
    with item_item:
        st.subheader("Item–item cosine")
        _show_list(result["item_item"], explain=False)
    with tower:
        st.subheader("Two-tower")
        _show_list(result["two_tower"], explain=False)


def _movie_choices(bundle) -> list[tuple[int, str]]:
    from movielens_recommender.serving.reasons import display_title

    rows = []
    for record in bundle.movies.itertuples(index=False):
        title = display_title(str(record.title))
        year = record.year
        year_text = "" if year != year else str(int(year))
        label = f"{title} ({year_text})" if year_text else title
        rows.append((int(record.item_id), label))
    rows.sort(key=lambda pair: pair[1].lower())
    return rows


def _new_user(path: Path) -> None:
    bundle = _load_cold(str(path.resolve()))
    manifest = bundle.manifest
    dataset = str(manifest.get("dataset", "ml-1m"))
    st.markdown(
        f"New-user model on `{dataset}`: rate a few films and get a top 10. "
        "The page does not ask for gender, age, occupation, or ZIP. "
        f"Recorded cold-start metrics are in the [README results]({RESULTS_URL})."
    )
    st.caption(
        "Rate at least 3 films. About 5 is a good start. "
        "Ratings stay in this browser session and are not saved."
    )
    if "new_user_ratings" not in st.session_state:
        st.session_state["new_user_ratings"] = []

    query = st.text_input("Search titles", value="")
    catalog = _movie_choices(bundle)
    needle = query.strip().lower()
    matches = [pair for pair in catalog if needle and needle in pair[1].lower()]
    matches = matches[:_SEARCH_LIMIT]
    if not needle:
        st.caption("Type part of a title, then add a 1 to 5 star rating.")
    elif not matches:
        st.caption("No titles match that search.")
    else:
        labels = {item_id: label for item_id, label in matches}
        chosen = st.selectbox("Movie", options=list(labels), format_func=lambda item: labels[item])
        stars = st.slider("Stars", min_value=1, max_value=5, value=4, step=1)
        if st.button("Add rating"):
            rows = [
                row
                for row in st.session_state["new_user_ratings"]
                if int(row["item_id"]) != int(chosen)
            ]
            rows.append(
                {"item_id": int(chosen), "title": labels[int(chosen)], "rating": int(stars)}
            )
            st.session_state["new_user_ratings"] = rows
            st.session_state["new_user_result"] = None

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
    _show_list(result["recommendations"], explain=True)


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


def main() -> None:
    st.set_page_config(page_title="MovieLens recommender", layout="wide")
    st.title("MovieLens recommender")
    st.caption(_CITATION)
    resolved = _resolve()
    if resolved is None:
        prod_env = os.environ.get("MOVIELENS_ARTIFACTS")
        cold_env = os.environ.get("MOVIELENS_COLD_ARTIFACTS")
        prod = Path(prod_env) if prod_env else _weights_root() / "production"
        cold = Path(cold_env) if cold_env else _weights_root() / "cold_start"
        _missing(prod, cold)
        return

    prod, cold = resolved
    has_prod = (prod / "manifest.json").is_file() and _usable_snapshot(prod)
    has_cold = (cold / "manifest.json").is_file() and _usable_snapshot(cold)
    if not has_prod and not has_cold:
        _missing(prod, cold)
        return

    default = 0 if has_prod else 1
    mode = st.radio(
        "Who is this for?",
        options=["Existing user", "New user"],
        index=default,
        horizontal=True,
    )
    if mode == "Existing user":
        if not has_prod:
            st.markdown("The production snapshot is missing.")
            st.code("make space-snapshot", language="bash")
            return
        _existing_user(prod)
        return
    if not has_cold:
        st.markdown("The new-user snapshot is missing.")
        st.code("make space-snapshot", language="bash")
        return
    _new_user(cold)


main()
