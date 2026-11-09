"""TMDB poster addresses. The server stores a URL, never image bytes.

The API key is read by the caller from the environment or from a Streamlit
secret. This module does not open a secrets file and does not log the key.
A missing key, a network error, or a title that does not match exactly
returns ``None`` so the page can draw a placeholder.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_ROOT = "https://image.tmdb.org/t/p/w185"
# One day. TMDB's terms forbid keeping API information for longer than 6 months.
POSTER_CACHE_TTL_SECONDS = 24 * 60 * 60
_RESPONSE_LIMIT = 1_000_000
_USER_AGENT = "movielens-recommender-demo"

ATTRIBUTION_NOTICE = (
    "This application uses TMDB and the TMDB APIs but is not endorsed, "
    "certified, or otherwise approved by TMDB."
)
ATTRIBUTION_LINK = "https://www.themoviedb.org/"

Opener = Callable[..., Any]


def poster_url(
    title: str,
    year: int | None,
    api_key: str,
    *,
    opener: Opener | None = None,
) -> str | None:
    """Return a browser-loadable poster URL, or ``None`` for the placeholder.

    The opener is called only for the search endpoint. The returned string
    points at ``image.tmdb.org`` and does not contain the API key.
    """
    key = (api_key or "").strip()
    if not key or not str(title).strip():
        return None
    open_ = opener or _urlopen
    try:
        payload = _search(str(title).strip(), year, key, open_)
    except Exception:
        return None
    path = _matching_poster_path(str(title).strip(), year, payload)
    if path is None:
        return None
    return TMDB_IMAGE_ROOT + path


def _urlopen(request: urllib.request.Request, timeout: float) -> Any:
    return urllib.request.urlopen(request, timeout=timeout)


def _search(title: str, year: int | None, api_key: str, opener: Opener) -> dict[str, Any]:
    query = {
        "query": title,
        "include_adult": "false",
        "language": "en-US",
        "api_key": api_key,
    }
    if year is not None:
        query["year"] = f"{int(year):04d}"
    url = TMDB_SEARCH_URL + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
    )
    response = opener(request, timeout=8)
    try:
        if hasattr(response, "__enter__"):
            with response as handle:
                return _decode_payload(_read_limited(handle))
        return _decode_payload(_read_limited(response))
    finally:
        if not hasattr(response, "__enter__"):
            close = getattr(response, "close", None)
            if close is not None:
                close()


def _decode_payload(body: bytes) -> dict[str, Any]:
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("TMDB search did not return an object")
    return payload


def _read_limited(response: Any) -> bytes:
    read = getattr(response, "read", None)
    if read is None:
        raise ValueError("TMDB response has no body")
    body = read(_RESPONSE_LIMIT + 1)
    if not isinstance(body, bytes | bytearray):
        raise ValueError("TMDB response body is not bytes")
    if len(body) > _RESPONSE_LIMIT:
        raise ValueError("TMDB response is too large")
    return bytes(body)


def _matching_poster_path(title: str, year: int | None, payload: dict[str, Any]) -> str | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    target = _normalise_title(title)
    if not target:
        return None
    matches: list[tuple[float, str]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        path = row.get("poster_path")
        if not isinstance(path, str) or not path.startswith("/") or ".." in path:
            continue
        names = {
            _normalise_title(str(row.get("title") or "")),
            _normalise_title(str(row.get("original_title") or "")),
        }
        if target not in names:
            continue
        if year is not None and not _release_year_matches(row.get("release_date"), int(year)):
            continue
        popularity = row.get("popularity")
        score = float(popularity) if isinstance(popularity, int | float) else 0.0
        matches.append((score, path))
    if not matches:
        return None
    matches.sort(key=lambda item: (-item[0], item[1]))
    return matches[0][1]


def _release_year_matches(release_date: Any, year: int) -> bool:
    if not isinstance(release_date, str) or len(release_date) < 4:
        return False
    return release_date[:4] == f"{year:04d}"


def _normalise_title(title: str) -> str:
    cleaned = []
    previous_space = False
    for char in title.casefold():
        if char.isalnum():
            cleaned.append(char)
            previous_space = False
        elif not previous_space:
            cleaned.append(" ")
            previous_space = True
    return "".join(cleaned).strip()
