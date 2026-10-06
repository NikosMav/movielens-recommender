"""One-page MovieLens demo. Loads a prebuilt snapshot; it does not train.

Run from the repo root after ``movielens-recommender build-artifacts``:

    streamlit run app/streamlit_app.py

``MOVIELENS_ARTIFACTS`` overrides the snapshot directory (default ``artifacts/ml-1m``).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

DEFAULT_ARTIFACTS = Path("artifacts/ml-1m")
RESULTS_URL = "https://github.com/NikosMav/movielens-recommender/blob/main/README.md#results"
_N_RECS = 10
_N_RECENT = 10


def _artifact_dir() -> Path:
    return Path(os.environ.get("MOVIELENS_ARTIFACTS", str(DEFAULT_ARTIFACTS)))


@st.cache_resource(show_spinner="Loading the fitted model…")
def _load(path_str: str):
    from movielens_recommender.serving.bundle import load_bundle, user_summaries

    bundle = load_bundle(path_str)
    return bundle, user_summaries(bundle)


def _label(summary: dict) -> str:
    genres = ", ".join(summary["top_genres"]) if summary["top_genres"] else "no genres"
    return f"{summary['user_id']} · {summary['n_ratings']} ratings · {genres}"


def _show_list(cards: list[dict], *, explain: bool) -> None:
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
                reasons = card.get("reasons") or []
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
                    for row in card.get("details") or []:
                        st.text(f"{row['feature']}: {row['contribution']:+.4f}")
                    bias = card.get("bias")
                    if bias is not None:
                        st.text(f"bias: {bias:+.4f}")


def main() -> None:
    st.set_page_config(page_title="MovieLens recommender", layout="wide")
    st.title("MovieLens recommender")
    path = _artifact_dir()
    if not (path / "manifest.json").is_file():
        st.markdown(
            "No fitted snapshot was found. From the repo root, with the "
            "`rank`, `deep`, and `ui` extras installed:"
        )
        st.code(
            "movielens-recommender build-artifacts --config configs/ml-1m.yaml\n"
            "streamlit run app/streamlit_app.py",
            language="bash",
        )
        st.caption(f"Looked in `{path}`.")
        return

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
        "Existing MovieLens users only. A new profile is not supported: the neural "
        "retriever has no embedding for a user it was not trained on."
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


main()
