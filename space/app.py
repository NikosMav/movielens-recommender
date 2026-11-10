"""MovieLens demo for Streamlit Community Cloud. Loads parameters; it does not train.

The entrypoint is this file. Discover, Explore a MovieLens user, and
How good is it? are the three sections (ADR-0024).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_SPACE_DIR = Path(__file__).resolve().parent
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from cards import THEME_CSS, footer_html, load_logo_svg  # noqa: E402
from loading import (  # noqa: E402
    resolve_snapshots,
    tmdb_api_key,
    usable_snapshot,
    warm,
    weights_root,
)
from pages import render_discover, render_explore, render_header, render_quality  # noqa: E402

_CITATION = (
    "MovieLens ml-1m is downloaded from GroupLens and is not redistributed here. "
    "GroupLens and the University of Minnesota do not endorse this demo. "
    "F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: "
    "History and Context. ACM Transactions on Interactive Intelligent Systems "
    "5, 4, Article 19. DOI 10.1145/2827872."
)


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


def _footer(api_key: str) -> None:
    st.caption(_CITATION)
    logo = load_logo_svg(_SPACE_DIR / "tmdb_blue_short.svg")
    st.markdown(
        footer_html(logo_svg=logo, api_key_set=bool(api_key)),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="MovieLens", layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    api_key = tmdb_api_key()

    resolved = resolve_snapshots()
    if resolved is None:
        import os

        prod_env = os.environ.get("MOVIELENS_ARTIFACTS")
        cold_env = os.environ.get("MOVIELENS_COLD_ARTIFACTS")
        prod = Path(prod_env) if prod_env else weights_root() / "production"
        cold = Path(cold_env) if cold_env else weights_root() / "cold_start"
        _missing(prod, cold)
        _footer(api_key)
        return

    prod, cold = resolved
    has_prod = (prod / "manifest.json").is_file() and usable_snapshot(prod)
    has_cold = (cold / "manifest.json").is_file() and usable_snapshot(cold)
    if not has_prod and not has_cold:
        _missing(prod, cold)
        _footer(api_key)
        return

    section = render_header()
    with st.spinner("Loading both recommenders…"):
        warm(prod, cold, has_prod=has_prod, has_cold=has_cold)
    with st.spinner("Opening this view…"):
        if section == "Discover":
            render_discover(str(cold.resolve()) if has_cold else "", api_key)
        elif section == "Explore a MovieLens user":
            render_explore(str(prod.resolve()) if has_prod else "", api_key)
        else:
            render_quality()
    _footer(api_key)


main()
