"""Display copy for recommendation cards.

The ranker sentences stay in the payload and in the technical disclosure.
The main card shows at most two short lines drawn from those sentences:
the rated film they name, a shared genre, and popularity only when that
sentence is the first (strongest) stored reason. This module does not
score anything and does not rewrite the stored sentences.
"""

from __future__ import annotations

import re

_POPULAR = frozenset(
    {
        "Popular with many viewers",
        "Widely rated by other viewers",
    }
)
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
        "This list reorders popular films and films similar to ones you rated."
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
) -> list[str]:
    """At most ``limit`` lines, specific to this pick.

    A named film becomes "Because you liked {title}". A genre sentence
    becomes "Shares {genre} with {title}" when the stored sentence names
    both, otherwise it names the genre. "Popular with viewers" is included
    only when a popularity sentence is the first stored reason.
    """
    if limit < 1:
        return []
    liked: list[str] = []
    affinity: list[tuple[str, str | None]] = []
    plain_genres: list[str] = []
    popular_at: int | None = None
    for index, raw in enumerate(reasons):
        text = " ".join(str(raw).split())
        if not text:
            continue
        if text in _POPULAR:
            if popular_at is None:
                popular_at = index
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
            genre = genre_rated.group("genre").strip()
            title = genre_rated.group("title").strip()
            affinity.append((genre, title))
            _remember(liked, title)
            continue
        genre_like = _GENRE_LIKE.match(text)
        if genre_like is not None:
            affinity.append((genre_like.group("genre").strip(), None))
            continue
        its = _ITS.match(text)
        if its is not None:
            plain_genres.append(its.group("genre").strip())
    lines: list[str] = []
    title = liked[0] if liked else None
    if title:
        lines.append(f"Because you liked {title}")
    if affinity:
        lines.append(_share_line(affinity, title))
    elif plain_genres:
        lines.append(_its_line(plain_genres[0]))
    elif title and genres_by_title:
        shared = _shared_genres(candidate_genres, genres_by_title.get(title, ""))
        if shared:
            shown = shared[:2]
            genre_text = shown[0] if len(shown) == 1 else f"{shown[0]} and {shown[1]}"
            lines.append(f"Shares {genre_text} with {title}")
    # Popularity is the strongest stored family only when it is the first reason.
    # A named film and a shared genre already fill the two lines.
    if popular_at == 0 and len(lines) < limit:
        lines.insert(0, "Popular with viewers")
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


def _share_line(affinity: list[tuple[str, str | None]], fallback: str | None) -> str:
    genre, partner = affinity[0]
    extra = next((name for name, _who in affinity[1:] if name != genre), None)
    genre_text = genre if extra is None else f"{genre} and {extra}"
    who = partner or fallback or "films you rated"
    return f"Shares {genre_text} with {who}"


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


def _shared_genres(candidate: str, liked: str) -> list[str]:
    liked_names = set(_genre_names(liked))
    return [name for name in _genre_names(candidate) if name in liked_names]
