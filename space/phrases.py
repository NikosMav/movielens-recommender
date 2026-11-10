"""Display copy for recommendation cards.

The ranker sentences stay in the payload and in the technical disclosure.
The main card shows at most two short lines. The first follows the strongest
stored reason: a film that reason names, or a plain popularity line when
popularity leads. A later film name does not override that. The second line
adds a genre and does not repeat those titles. This module does not score
anything and does not rewrite the stored sentences.
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
    candidate_title: str = "",
) -> list[str]:
    """At most ``limit`` lines, specific to this pick.

    The first line follows the first stored reason that says something
    specific. Popularity leads with a plain sentence and does not borrow a
    film from a later line or from ``similar_titles``. A film the leading
    reason names is "Because you liked …". A second contributing film, when
    ``similar_titles`` includes one and popularity did not lead, is joined
    on that same line. The next line adds a genre and does not repeat titles.
    """
    if limit < 1:
        return []
    liked: list[str] = []
    affinity: list[str] = []
    plain_genres: list[str] = []
    popular_at: int | None = None
    lead: str | None = None
    for index, raw in enumerate(reasons):
        text = " ".join(str(raw).split())
        if not text:
            continue
        if text in _POPULAR:
            if popular_at is None:
                popular_at = index
            if lead is None:
                lead = "popular"
            continue
        neighbor = _NEIGHBOR.match(text)
        if neighbor is not None:
            _remember(liked, neighbor.group("title"))
            if lead is None:
                lead = "film"
            continue
        because = _BECAUSE.match(text)
        if because is not None:
            _remember(liked, because.group("title"))
            if lead is None:
                lead = "film"
            continue
        genre_rated = _GENRE_RATED.match(text)
        if genre_rated is not None:
            _remember(affinity, genre_rated.group("genre"))
            _remember(liked, genre_rated.group("title"))
            if lead is None:
                lead = "film"
            continue
        genre_like = _GENRE_LIKE.match(text)
        if genre_like is not None:
            _remember(affinity, genre_like.group("genre"))
            if lead is None:
                lead = "genre"
            continue
        its = _ITS.match(text)
        if its is not None:
            _remember(plain_genres, its.group("genre"))
            if lead is None:
                lead = "genre"
    # A neighbour list is a signal only when the stored reasons did not
    # already say the film is here because it is popular.
    if lead != "popular" and similar_titles:
        for title in similar_titles:
            _remember(liked, title)
        if liked and lead is None:
            lead = "film"
    if lead == "popular":
        liked = []
    named = liked[:2]
    lines: list[str] = []
    if lead == "popular":
        lines.append(POPULAR_LINE)
    elif len(named) == 2:
        lines.append(f"Because you liked {named[0]} and {named[1]}")
    elif named:
        lines.append(f"Because you liked {named[0]}")
    follow = _followup(
        affinity=affinity,
        plain_genres=plain_genres,
        candidate_genres=candidate_genres,
        named=named,
        genres_by_title=genres_by_title,
        popular=lead != "popular" and popular_at == 0,
        salt=candidate_title or (named[0] if named else ""),
    )
    if follow:
        lines.append(follow)
    chosen: list[str] = []
    for line in lines:
        if line in chosen:
            continue
        chosen.append(line)
        if len(chosen) >= limit:
            break
    return chosen


def _remember(titles: list[str], title: str) -> None:
    cleaned = " ".join(title.split())
    if cleaned and cleaned not in titles:
        titles.append(cleaned)


def _followup(
    *,
    affinity: list[str],
    plain_genres: list[str],
    candidate_genres: str,
    named: list[str],
    genres_by_title: dict[str, str] | None,
    popular: bool,
    salt: str,
) -> str | None:
    """A second line that does not repeat the films named on the first."""
    if affinity:
        return f"Shares {_join(affinity[:2])}"
    if plain_genres and not candidate_genres:
        return _its_line(plain_genres[0])
    shared: list[str] = []
    catalog = _genre_names(candidate_genres)
    if genres_by_title and named:
        pool: set[str] = set()
        for title in named:
            pool.update(_genre_names(genres_by_title.get(title, "")))
        shared = [genre for genre in catalog if genre in pool]
    # One shared genre on every card reads as a repeated sentence. Rotate
    # across the shared genres, a genre only this film has, and popularity.
    options: list[str] = [f"Shares {genre}" for genre in shared]
    options.extend(_its_line(genre) for genre in catalog if genre not in shared)
    if not options and plain_genres:
        options.append(_its_line(plain_genres[0]))
    # Popularity is the fallback, not another copy of the same line. A genre
    # this film actually has already distinguishes two picks that name the
    # same rated films.
    if popular and not options:
        options.append("Popular with viewers")
    if not options:
        return None
    return options[_mix(salt or options[0]) % len(options)]


def _join(genres: list[str]) -> str:
    if len(genres) == 1:
        return genres[0]
    return f"{genres[0]} and {genres[1]}"


def _mix(text: str) -> int:
    value = 17
    for char in text:
        value = (value * 33 + ord(char)) & 0xFFFFFFFF
    return value


def _its_line(genre: str) -> str:
    article = "an" if genre[:1].lower() in "aeiou" else "a"
    return f"It's {article} {genre} film"


def _genre_names(genres: str) -> list[str]:
    names: list[str] = []
    for part in str(genres).split("|"):
        name = part.strip()
        if not name or name == "(no genres listed)" or name in names:
            continue
        names.append(name)
    return names

