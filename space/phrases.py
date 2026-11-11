"""Display copy for recommendation cards.

The ranker sentences stay in the payload and in the technical disclosure.
The main card shows at most two short lines. A rated film with a real
similarity is "Because you liked …". "A widely loved film that fits your mix"
is used only when popularity is the stored signal and no rated film is close.
The second line names genres shared with a rated film, or the decade.
This module does not score anything and does not rewrite the stored sentences.
"""

from __future__ import annotations

import re

_POPULAR = frozenset(
    {
        "Popular with many viewers",
        "Widely rated by other viewers",
    }
)
POPULAR_LINE = "A widely loved film that fits your mix"
# Item–item cosine. At or above this, the neighbour is close enough to name.
MEANINGFUL_SIM = 0.20
# Below this there is no film worth citing, even as a fallback.
TRACE_SIM = 0.05
_NEIGHBOR = re.compile(
    r"^You rated (?P<title>.+) (?P<stars>\d+(?:\.\d+)?★)"
    r" and this is one of its nearest neighbou?rs$"
)
_BECAUSE = re.compile(
    r"^Because you rated (?P<title>.+?)(?: (?P<stars>\d+(?:\.\d+)?★)| highly)$"
)
_GENRE_RATED = re.compile(
    r"^Matches your liking for (?P<genre>.+?) — you rated "
    r"(?P<title>.+) (?P<stars>\d+(?:\.\d+)?★)$"
)
_GENRE_LIKE = re.compile(r"^Matches your liking for (?P<genre>.+)$")
_ITS = re.compile(r"^It's an? (?P<genre>.+) film$")

_METHOD = {
    "most_popular": "This list is the films people have rated most often.",
    "item_item_fold_in": "This list is films similar to ones you rated.",
    "history_two_tower": "This list matches the films you rated, without using a saved profile.",
    "cold_start_ranker": (
        "Ranked from the films you rated: popular titles, and ones close to them."
    ),
}


def plain_method(method: str) -> str:
    """One sentence for how a new-user list was chosen. No profile-size code."""
    return _METHOD.get(method, "This list is based on the films you rated.")


def plain_reasons(
    reasons: list[str],
    *,
    limit: int = 2,
    genres_by_title: dict[str, str] | None = None,
    candidate_genres: str = "",
    similar_titles: list[str] | None = None,
    similar: list[tuple[str, float]] | None = None,
    candidate_title: str = "",
    candidate_year: object = None,
) -> list[str]:
    """At most ``limit`` lines for one card. See ``reason_choice``."""
    del candidate_title
    return reason_choice(
        reasons,
        limit=limit,
        genres_by_title=genres_by_title,
        candidate_genres=candidate_genres,
        similar_titles=similar_titles,
        similar=similar,
        candidate_year=candidate_year,
    )["lines"]


def reason_choice(
    reasons: list[str],
    *,
    limit: int = 2,
    genres_by_title: dict[str, str] | None = None,
    candidate_genres: str = "",
    similar_titles: list[str] | None = None,
    similar: list[tuple[str, float]] | None = None,
    candidate_year: object = None,
) -> dict[str, object]:
    """The card lines, and the film lines to use if a popularity cap swaps this one.

    ``similar`` is ``(title, cosine)`` for rated films. A cosine at or above
    ``MEANINGFUL_SIM`` is named. A weaker one can still be the fallback when a
    list would otherwise be all popularity. Titles passed without a cosine are
    treated as meaningful, because the caller already chose them.
    """
    liked, popular_signal = _scan(reasons)
    scored = [(str(title), float(sim)) for title, sim in (similar or []) if str(title).strip()]
    if not scored and similar_titles:
        scored = [(str(title), 1.0) for title in similar_titles if str(title).strip()]
    scored.sort(key=lambda row: (-row[1], row[0]))
    best = scored[0][1] if scored else None
    strong = [title for title, sim in scored if sim >= MEANINGFUL_SIM][:2]
    traced = [title for title, sim in scored if sim >= TRACE_SIM][:2]
    if not scored and liked:
        strong = liked[:2]
        traced = liked[:2]
    lookup = genres_by_title or {}
    if strong:
        lines = _compose(strong, candidate_genres, lookup, candidate_year, limit)
        film_lines = lines
        popular = False
        best_out = 1.0 if best is None else best
    elif popular_signal or not liked:
        lines = _compose_popular(traced, candidate_genres, lookup, candidate_year, limit)
        film_lines = _compose(traced, candidate_genres, lookup, candidate_year, limit)
        popular = True
        best_out = 0.0 if best is None else best
    else:
        lines = _compose(liked[:2], candidate_genres, lookup, candidate_year, limit)
        film_lines = lines
        popular = False
        best_out = 1.0 if best is None else best
    return {
        "lines": lines,
        "popular": popular,
        "best_sim": best_out,
        "film_lines": film_lines,
    }


def cap_popular(choices: list[dict[str, object]]) -> list[list[str]]:
    """At most one widely-loved card per three, unless nothing else can be said.

    A card with no rated film above ``TRACE_SIM`` stays widely loved even past
    that cap. That is the case the similarities themselves require.
    """
    count = len(choices)
    cap = count // 3
    forced: list[int] = []
    optional: list[int] = []
    for index, choice in enumerate(choices):
        if not choice.get("popular"):
            continue
        film_lines = choice.get("film_lines") or []
        sim = float(choice.get("best_sim") or 0.0)
        if film_lines and sim >= TRACE_SIM:
            optional.append(index)
        else:
            forced.append(index)
    if len(forced) >= cap:
        keep = set(forced)
    else:
        optional.sort(key=lambda index: (float(choices[index].get("best_sim") or 0.0), index))
        keep = set(forced) | set(optional[: cap - len(forced)])
    shown: list[list[str]] = []
    for index, choice in enumerate(choices):
        lines = list(choice.get("lines") or [])
        film_lines = list(choice.get("film_lines") or [])
        if choice.get("popular") and index not in keep and film_lines:
            shown.append(film_lines)
        else:
            shown.append(lines)
    return shown


def _scan(reasons: list[str]) -> tuple[list[str], bool]:
    liked: list[str] = []
    popular = False
    for raw in reasons:
        text = " ".join(str(raw).split())
        if not text:
            continue
        if text in _POPULAR:
            popular = True
            continue
        neighbor = _NEIGHBOR.match(text)
        if neighbor is not None:
            _remember(liked, neighbor.group("title"))
            continue
        because = _BECAUSE.match(text)
        if because is not None:
            _remember(liked, because.group("title"))
            continue
        genre_rated = _GENRE_RATED.match(text)
        if genre_rated is not None:
            _remember(liked, genre_rated.group("title"))
    return liked, popular


def _compose(
    titles: list[str],
    candidate_genres: str,
    genres_by_title: dict[str, str],
    year: object,
    limit: int,
) -> list[str]:
    if not titles:
        return []
    if len(titles) >= 2:
        head = f"Because you liked {titles[0]} and {titles[1]}"
    else:
        head = f"Because you liked {titles[0]}"
    lines = [head]
    follow = _second_line(titles, candidate_genres, genres_by_title, year)
    if follow and follow not in lines:
        lines.append(follow)
    return lines[:limit]


def _compose_popular(
    titles: list[str],
    candidate_genres: str,
    genres_by_title: dict[str, str],
    year: object,
    limit: int,
) -> list[str]:
    lines = [POPULAR_LINE]
    follow = _second_line(titles, candidate_genres, genres_by_title, year)
    if follow:
        lines.append(follow)
    return lines[:limit]


def _second_line(
    titles: list[str],
    candidate_genres: str,
    genres_by_title: dict[str, str],
    year: object,
) -> str | None:
    catalog = _genre_names(candidate_genres)
    best_title = ""
    best: list[str] = []
    for title in titles:
        owned = set(_genre_names(genres_by_title.get(title, "")))
        shared = [genre for genre in catalog if genre in owned]
        if len(shared) > len(best):
            best = shared
            best_title = title
    if best and best_title:
        if len(best) == 1:
            return f"Shares {best[0]} with {best_title}"
        return f"Shares {best[0]} and {best[1]} with {best_title}"
    return _era(year)


def _era(year: object) -> str | None:
    try:
        value = int(float(year))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if value < 1888 or value > 2100:
        return None
    decade = (value // 10) * 10
    return f"A {decade}s film"


def _remember(titles: list[str], title: str) -> None:
    cleaned = " ".join(title.split())
    if cleaned and cleaned not in titles:
        titles.append(cleaned)


def _genre_names(genres: str) -> list[str]:
    names: list[str] = []
    for part in str(genres).split("|"):
        name = part.strip()
        if not name or name == "(no genres listed)" or name in names:
            continue
        names.append(name)
    return names

