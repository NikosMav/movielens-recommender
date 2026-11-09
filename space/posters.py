"""TMDB poster addresses. The server stores a URL, never image bytes.

The API key is read by the caller from the environment or from a Streamlit
secret. This module does not open a secrets file and does not log the key.
A missing key returns ``None`` before any request. A network error, a bad
response, or a title that still does not match returns ``None`` so the page
can draw a placeholder.

MovieLens titles are not TMDB titles. A trailing article (``Matrix, The``)
is moved to the front, a parenthetical alternate title is searched on its
own, and the part after ``': '`` or ``' - '`` is tried when the full string
misses. The release year is sent as ``primary_release_year``, then one year
to either side, then with no year filter. The first hit stops the search.
"""

from __future__ import annotations

import json
import re
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

# Same trailing articles as ``display_title``. Longer tokens come first so
# "An" is not consumed by "A", and "Les" is not consumed by "Le".
_TRAILING_ARTICLES = (
    "The",
    "Les",
    "Das",
    "Der",
    "Die",
    "La",
    "Le",
    "Il",
    "El",
    "An",
    "A",
    "L'",
)
_ARTICLE_CANON = {article.casefold(): article for article in _TRAILING_ARTICLES}
_TRAILING_ARTICLE = re.compile(
    r"^(?P<body>.+), (?P<article>"
    + "|".join(re.escape(article) for article in _TRAILING_ARTICLES)
    + r")(?P<rest> \(.*\))?$",
    re.IGNORECASE,
)
_YEAR_SUFFIX = re.compile(r"\s*\((\d{4})\)\s*$")
_PAREN = re.compile(r"\(([^)]*)\)")
_YEAR_ONLY = re.compile(r"^\d{4}$")
_LEADING_ARTICLES = tuple(
    sorted(
        (article.casefold() for article in _TRAILING_ARTICLES if article != "L'"),
        key=len,
        reverse=True,
    )
)

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
    points at ``image.tmdb.org`` and does not contain the API key. A missing
    or blank key returns before the opener is called.
    """
    key = (api_key or "").strip()
    if not key or not str(title).strip():
        return None
    queries = candidate_queries(title)
    keys = _match_keys(queries)
    if not queries or not keys:
        return None
    open_ = opener or _urlopen
    prefer = int(year) if year is not None else None
    for release_year in _year_attempts(prefer):
        for query in queries:
            try:
                payload = _search(query, release_year, key, open_)
            except Exception:
                return None
            path = _matching_poster_path(
                keys,
                payload,
                accepted_year=release_year,
                prefer_year=prefer,
            )
            if path is not None:
                return TMDB_IMAGE_ROOT + path
    return None


def candidate_queries(title: str) -> list[str]:
    """Search strings for one MovieLens title, most specific first.

    The year suffix is not part of a query. No request is made here.
    """
    text = _move_trailing_article(_strip_year_suffix(str(title)).strip())
    if not text:
        return []
    alternates: list[str] = []
    for inner in _PAREN.findall(text):
        inner = _collapse(inner)
        if not inner or _YEAR_ONLY.match(inner):
            continue
        alternates.append(_move_trailing_article(inner))
    primary = _collapse(_PAREN.sub(" ", text))
    seeds = [primary] if primary else []
    seeds.extend(alternates)
    extras: list[str] = []
    for seed in seeds:
        extras.extend(_subtitle_parts(seed))
    ordered: list[str] = []
    seen: set[str] = set()
    for query in seeds + extras:
        cleaned = _collapse(query)
        if len(cleaned) < 2:
            continue
        marker = cleaned.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(cleaned)
    return ordered


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
        query["primary_release_year"] = str(int(year))
    url = TMDB_SEARCH_URL + "?" + urllib.parse.urlencode(query)
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
    )
    # Positional, so a stand-in can name the second parameter anything.
    response = opener(request, 8)
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


def _year_attempts(year: int | None) -> list[int | None]:
    if year is None:
        return [None]
    return [year, year - 1, year + 1, None]


def _matching_poster_path(
    keys: set[str],
    payload: dict[str, Any],
    *,
    accepted_year: int | None,
    prefer_year: int | None,
) -> str | None:
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    scored: list[tuple[int, float, str]] = []
    for row in results:
        if not isinstance(row, dict):
            continue
        path = row.get("poster_path")
        if not _safe_poster_path(path):
            continue
        names = {
            key
            for field in ("title", "original_title")
            if (key := _match_key(str(row.get(field) or "")))
        }
        if not names & keys:
            continue
        row_year = _release_year(row.get("release_date"))
        if accepted_year is not None and row_year != accepted_year:
            continue
        if prefer_year is None or row_year is None:
            distance = 0 if prefer_year is None else 100
        else:
            distance = abs(row_year - prefer_year)
        popularity = row.get("popularity")
        if isinstance(popularity, bool) or not isinstance(popularity, int | float):
            score = 0.0
        else:
            score = float(popularity)
        scored.append((distance, -score, path))
    if not scored:
        return None
    scored.sort()
    return scored[0][2]


def _safe_poster_path(path: Any) -> bool:
    return isinstance(path, str) and path.startswith("/") and ".." not in path


def _release_year(release_date: Any) -> int | None:
    if not isinstance(release_date, str) or len(release_date) < 4:
        return None
    head = release_date[:4]
    if not head.isdigit():
        return None
    return int(head)


def _match_keys(queries: list[str]) -> set[str]:
    return {key for query in queries if (key := _match_key(query))}


def _match_key(title: str) -> str:
    text = _move_trailing_article(_strip_year_suffix(str(title)).strip())
    text = _collapse(_PAREN.sub(" ", text))
    return _strip_leading_article(_normalise_title(text))


def _strip_year_suffix(title: str) -> str:
    match = _YEAR_SUFFIX.search(title)
    if match is None:
        return title
    year = int(match.group(1))
    if 1870 <= year <= 2100:
        return title[: match.start()].rstrip()
    return title


def _move_trailing_article(title: str) -> str:
    match = _TRAILING_ARTICLE.match(title)
    if match is None:
        return title
    article = _ARTICLE_CANON.get(match.group("article").casefold(), match.group("article"))
    body = match.group("body").strip()
    rest = match.group("rest") or ""
    if article.endswith("'"):
        return f"{article}{body}{rest}"
    return f"{article} {body}{rest}"


def _subtitle_parts(title: str) -> list[str]:
    parts: list[str] = []
    if ": " in title:
        after_colon = title.split(": ", 1)[1].strip()
        if after_colon:
            parts.append(after_colon)
    sources = [title, *parts]
    for source in sources:
        if " - " not in source:
            continue
        after_dash = source.rsplit(" - ", 1)[1].strip()
        if after_dash:
            parts.append(after_dash)
    return parts


def _strip_leading_article(normalised: str) -> str:
    for article in _LEADING_ARTICLES:
        prefix = article + " "
        if normalised.startswith(prefix) and len(normalised) > len(prefix):
            return normalised[len(prefix) :]
    return normalised


def _collapse(text: str) -> str:
    return " ".join(text.split())


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
