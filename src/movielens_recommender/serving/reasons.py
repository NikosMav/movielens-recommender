"""Map LightGBM pred_contrib values to a few plain-language reasons.

The mapping is a template, not a causal account. Contributions explain the
ranker's score for this candidate. Demographic columns are collapsed to one
generic sentence so gender, age, occupation, and region are never named.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from movielens_recommender.movies import GENRES
from movielens_recommender.ranker.features import slug_genre

_YEAR_SUFFIX = re.compile(r"\s*\(\d{4}\)\s*$")

# MovieLens files the leading article at the end ("Muse, The"). Longer tokens
# come first so "An" is not consumed by "A", and "Les" is not consumed by "Le".
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
_TRAILING_ARTICLE = re.compile(
    r"^(?P<body>.+), (?P<article>"
    + "|".join(re.escape(article) for article in _TRAILING_ARTICLES)
    + r")(?P<rest> \(.*\))?$"
)

# One sentence per genre family. Built from the canonical genre list so a
# feature slug cannot leak into the main text as a raw column name.
_SLUG_TO_GENRE: dict[str, str] = {slug_genre(genre): genre for genre in GENRES}

_NEIGHBOR_FEATURES = frozenset({"item_item_score", "item_item_rank"})
_RETRIEVER_FEATURES = frozenset({"two_tower_score", "two_tower_rank"})
_ACTIVITY_FEATURES = frozenset({"user_n_ratings", "user_mean_rating", "user_std_rating"})

_GENERIC_SIMILAR_VIEWERS = "Popular with viewers similar to you"
_RETRIEVER_TEXT = "Ranked highly by the neural retriever for your taste profile"
_NEIGHBOR_FALLBACK = "Close to movies you have already rated"
_IN_BOTH_TEXT = "Both the neighbourhood model and the neural retriever suggested it"
_POPULARITY_TEXT = "Widely rated by other viewers"
_RECENCY_TEXT = "Recently active in other viewers' ratings"
_YEAR_TEXT = "Its release year fits the films you rate"
_ACTIVITY_TEXT = "Your rating history is a pattern this ranker scores up"


@dataclass(frozen=True)
class HistoryItem:
    """One rating the ranker was allowed to see."""

    item_id: int
    title: str
    genres: str
    rating: float
    timestamp: int = 0


@dataclass(frozen=True)
class Reason:
    """One user-facing sentence and the positive contribution mass behind it."""

    text: str
    contribution: float


@dataclass(frozen=True)
class RecommendationExplanation:
    """Top reasons for the main view, plus raw contributions for a details pane."""

    reasons: list[Reason]
    details: list[tuple[str, float]]
    bias: float
    raw_score: float | None = None

    def main_text(self) -> str:
        return "\n".join(reason.text for reason in self.reasons)


def display_title(title: str) -> str:
    """Show a MovieLens title in reading order, without repeating the year.

    The catalog stores a leading article at the end (``Muse, The``). This
    moves that article back to the front for display only. A trailing
    ``(YYYY)`` is dropped because the year is already a separate column.
    Alternate titles in parentheses stay where they are.
    """
    text = _YEAR_SUFFIX.sub("", str(title)).strip()
    if not text:
        return str(title)
    match = _TRAILING_ARTICLE.match(text)
    if match is None:
        return text
    article = match.group("article")
    body = match.group("body").strip()
    rest = match.group("rest") or ""
    # French elision: "Enfer, L'" is "L'Enfer", not "L' Enfer".
    if article.endswith("'"):
        return f"{article}{body}{rest}"
    return f"{article} {body}{rest}"


def format_stars(rating: float) -> str:
    value = float(rating)
    if value.is_integer():
        return f"{int(value)}★"
    return f"{value:.1f}★"


def _family(feature: str) -> str | None:
    if feature in _NEIGHBOR_FEATURES:
        return "neighbor"
    if feature in _RETRIEVER_FEATURES:
        return "retriever"
    if feature == "in_both":
        return "in_both"
    if feature == "item_popularity":
        return "popularity"
    if feature == "item_recency":
        return "recency"
    if feature == "item_year":
        return "year"
    if feature in _ACTIVITY_FEATURES:
        return "activity"
    if feature.startswith("affinity_"):
        return f"genre:{feature[len('affinity_') :]}"
    if feature.startswith("genre_"):
        return f"genre:{feature[len('genre_') :]}"
    if feature.startswith("demo_") or feature.startswith("group_"):
        return "similar_viewers"
    return None


def _genre_names(genres: str) -> set[str]:
    names = {part.strip() for part in str(genres).split("|") if part.strip()}
    if "Children's" in names:
        names.add("Children")
    return names


def _best_neighbor(
    history: Sequence[HistoryItem],
    similarity: Mapping[int, float],
) -> HistoryItem | None:
    best: HistoryItem | None = None
    best_key: tuple[float, float, int] | None = None
    for row in history:
        sim = float(similarity.get(int(row.item_id), 0.0))
        if sim <= 0.0:
            continue
        # Item–item scores are ``S @ r``, so the driver is similarity times
        # the rating, not similarity alone.
        key = (sim * float(row.rating), sim, int(row.timestamp))
        if best_key is None or key > best_key:
            best_key = key
            best = row
    return best


def _best_in_genre(history: Sequence[HistoryItem], genre: str) -> HistoryItem | None:
    hits = [row for row in history if genre in _genre_names(row.genres)]
    if not hits:
        return None
    return max(hits, key=lambda row: (float(row.rating), int(row.timestamp)))


def _neighbor_text(history: Sequence[HistoryItem], similarity: Mapping[int, float]) -> str:
    neighbor = _best_neighbor(history, similarity)
    if neighbor is None:
        return _NEIGHBOR_FALLBACK
    title = display_title(neighbor.title)
    stars = format_stars(neighbor.rating)
    return f"You rated {title} {stars} and this is one of its nearest neighbours"


def _genre_text(
    slug: str,
    *,
    affinity_positive: bool,
    history: Sequence[HistoryItem],
) -> str | None:
    genre = _SLUG_TO_GENRE.get(slug)
    if genre is None or genre == "(no genres listed)":
        return None
    if not affinity_positive:
        return f"It's a {genre} film"
    driver = _best_in_genre(history, genre)
    if driver is not None and float(driver.rating) >= 4.0:
        title = display_title(driver.title)
        stars = format_stars(driver.rating)
        return f"Matches your liking for {genre} — you rated {title} {stars}"
    return f"Matches your liking for {genre}"


def explain_recommendation(
    contributions: Mapping[str, float],
    *,
    history: Sequence[HistoryItem],
    neighbor_similarity: Mapping[int, float] | None = None,
    bias: float = 0.0,
    raw_score: float | None = None,
    top_k: int = 3,
    detail_k: int = 8,
    affinity_positive: Mapping[str, bool] | None = None,
) -> RecommendationExplanation:
    """Turn positive contributions into at most ``top_k`` sentences.

    Features that share a sentence are summed. Only strictly positive
    contributions count toward a reason. ``neighbor_similarity`` maps a
    history item id to its item–item similarity with the candidate.
    ``affinity_positive`` overrides the per-genre affinity sign when the
    caller has already aggregated columns; by default a genre family uses
    the affinity wording when ``affinity_<slug>`` itself is positive.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    sims = {} if neighbor_similarity is None else neighbor_similarity
    grouped: dict[str, float] = {}
    affinity_pos: dict[str, bool] = {}
    for name, raw in contributions.items():
        if name == "bias":
            continue
        value = float(raw)
        family = _family(str(name))
        if family is None or value <= 0.0:
            continue
        grouped[family] = grouped.get(family, 0.0) + value
        if str(name).startswith("affinity_"):
            affinity_pos[str(name)[len("affinity_") :]] = True

    if affinity_positive is not None:
        affinity_pos.update({str(k): bool(v) for k, v in affinity_positive.items()})

    ranked = sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    reasons: list[Reason] = []
    for family, mass in ranked:
        if len(reasons) >= top_k:
            break
        text = _render(family, history, sims, affinity_pos)
        if text is None:
            continue
        reasons.append(Reason(text=text, contribution=float(mass)))

    detail_rows = [
        (str(name), float(value))
        for name, value in contributions.items()
        if str(name) != "bias" and float(value) != 0.0
    ]
    detail_rows.sort(key=lambda item: (-abs(item[1]), item[0]))
    if detail_k >= 0:
        detail_rows = detail_rows[:detail_k]

    return RecommendationExplanation(
        reasons=reasons,
        details=detail_rows,
        bias=float(bias),
        raw_score=None if raw_score is None else float(raw_score),
    )


def _render(
    family: str,
    history: Sequence[HistoryItem],
    similarity: Mapping[int, float],
    affinity_pos: Mapping[str, bool],
) -> str | None:
    if family == "neighbor":
        return _neighbor_text(history, similarity)
    if family == "retriever":
        return _RETRIEVER_TEXT
    if family == "in_both":
        return _IN_BOTH_TEXT
    if family == "popularity":
        return _POPULARITY_TEXT
    if family == "recency":
        return _RECENCY_TEXT
    if family == "year":
        return _YEAR_TEXT
    if family == "activity":
        return _ACTIVITY_TEXT
    if family == "similar_viewers":
        return _GENERIC_SIMILAR_VIEWERS
    if family.startswith("genre:"):
        slug = family[len("genre:") :]
        return _genre_text(
            slug,
            affinity_positive=bool(affinity_pos.get(slug, False)),
            history=history,
        )
    return None
