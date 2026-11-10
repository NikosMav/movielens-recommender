"""Display copy for recommendation cards.

The ranker sentences stay in the payload and in the technical disclosure.
The main card shows a shorter wording, at most two lines, without the
retriever jargon. This module does not score anything.
"""

from __future__ import annotations

import re

_EXACT = {
    "Ranked highly by the neural retriever for your taste profile": (
        "Matches films you tend to like"
    ),
    "Lines up with movies you have already rated": "Similar to films you have already rated",
    "Close to movies you have already rated": "Similar to films you have already rated",
    "Both the neighbourhood model and the neural retriever suggested it": (
        "More than one model ranked it highly"
    ),
    "Popular with many viewers": "Popular with many viewers",
    "Widely rated by other viewers": "Lots of people have rated this",
    "Popular with viewers similar to you": "Popular with viewers like you",
    "Recently active in other viewers' ratings": "People have been rating this recently",
    "Its release year fits the films you rate": "The release year fits what you rate",
    "Your rating history is a pattern this ranker scores up": "Fits the way you rate films",
}

_GENERIC = frozenset(
    {
        "Popular with many viewers",
        "Lots of people have rated this",
        "Popular with viewers like you",
        "People have been rating this recently",
        "The release year fits what you rate",
        "Fits the way you rate films",
    }
)

_NEIGHBOR = re.compile(
    r"^You rated (?P<title>.+) (?P<stars>\d+(?:\.\d+)?★)"
    r" and this is one of its nearest neighbou?rs$"
)
_RATED = re.compile(
    r"^Because you rated (?P<title>.+?)(?: (?P<stars>\d+(?:\.\d+)?★)| highly)$"
)
_BANNED = (
    (re.compile(r"neural retriever", re.IGNORECASE), "taste model"),
    (re.compile(r"nearest neighbou?rs", re.IGNORECASE), "similar films"),
)

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


def plain_reasons(reasons: list[str], *, limit: int = 2) -> list[str]:
    """At most ``limit`` distinct lines. Specific lines come before generic ones."""
    if limit < 1:
        return []
    specific: list[str] = []
    generic: list[str] = []
    stories: dict[tuple[str, str], int] = {}
    for raw in reasons:
        text = _to_plain(str(raw))
        if not text:
            continue
        bucket = generic if text in _GENERIC else specific
        _place(bucket, text, _story(text), stories if bucket is specific else None)
    chosen: list[str] = []
    for text in specific + generic:
        if text in chosen:
            continue
        chosen.append(text)
        if len(chosen) >= limit:
            break
    return chosen


def _place(
    bucket: list[str],
    text: str,
    story: tuple[str, str] | None,
    stories: dict[tuple[str, str], int] | None,
) -> None:
    if stories is not None and story is not None and story in stories:
        index = stories[story]
        if "★" in text and "★" not in bucket[index]:
            bucket[index] = text
        return
    if text in bucket:
        return
    bucket.append(text)
    if stories is not None and story is not None:
        stories[story] = len(bucket) - 1


def _to_plain(text: str) -> str:
    match = _NEIGHBOR.match(text)
    if match is not None:
        text = f"Because you rated {match.group('title')} {match.group('stars')}"
    else:
        text = _EXACT.get(text, text)
    for pattern, replacement in _BANNED:
        text = pattern.sub(replacement, text)
    return " ".join(text.split())


def _story(text: str) -> tuple[str, str] | None:
    match = _RATED.match(text)
    if match is None:
        return None
    return ("rated", match.group("title"))
