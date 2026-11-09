"""Serve the Astryx build as a bidirectional Streamlit component."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import streamlit as st
from streamlit.components.v1 import declare_component

BUILD = Path(__file__).resolve().parent / "astryx_ui"
_KEY = "astryx_movie_request"
_REPLY = "astryx_movie_reply"


def render(
    *,
    search: Callable[[str], list[dict[str, Any]]],
    recommend: Callable[[list[tuple[int, int]]], dict[str, Any]],
    compare: Callable[[int], dict[str, Any]],
) -> None:
    component = declare_component("astryx_movies", path=BUILD)

    def respond() -> None:
        request = st.session_state.get(_KEY)
        if not isinstance(request, dict) or not isinstance(request.get("id"), str):
            return
        status, data = dispatch(request, search=search, recommend=recommend, compare=compare)
        st.session_state[_REPLY] = {"id": request["id"], "status": status, "data": data}

    component(key=_KEY, reply=st.session_state.get(_REPLY), on_change=respond)


def dispatch(
    request: dict[str, Any],
    *,
    search: Callable[[str], list[dict[str, Any]]],
    recommend: Callable[[list[tuple[int, int]]], dict[str, Any]],
    compare: Callable[[int], dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """Handle the same three operations the standalone frontend sends to FastAPI."""
    path = request.get("path")
    if not isinstance(path, str) or len(path) > 1_000:
        return 422, {}
    try:
        route = urlsplit(path)
        if route.path == "/movies/search":
            query = parse_qs(route.query).get("q", [""])[0].strip()
            if not 1 <= len(query) <= 200:
                return 422, {}
            return 200, {"movies": search(query)}
        if route.path == "/recommendations/new-user":
            profile = _profile(request.get("body"))
            return 200, recommend(profile)
        parts = route.path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "users" and parts[2] == "comparison":
            user_id = int(parts[1])
            if user_id <= 0:
                return 422, {}
            return 200, compare(user_id)
    except KeyError:
        return 404, {}
    except ValueError:
        return 422, {}
    except FileNotFoundError:
        return 503, {}
    return 404, {}


def _profile(body: Any) -> list[tuple[int, int]]:
    rows = body.get("ratings") if isinstance(body, dict) else None
    if not isinstance(rows, list) or not 3 <= len(rows) <= 200:
        raise ValueError("Rate between 3 and 200 films")
    profile = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each rating must be an object")
        item_id, rating = row.get("item_id"), row.get("rating")
        if type(item_id) is not int or item_id < 1 or item_id in seen:
            raise ValueError("Each movie must have a distinct ID")
        if type(rating) is not int or not 1 <= rating <= 5:
            raise ValueError("Use a whole number from 1 to 5")
        seen.add(item_id)
        profile.append((item_id, rating))
    return profile
