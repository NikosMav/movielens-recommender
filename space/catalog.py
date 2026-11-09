"""Genre shelf for the new-user page. Ratings stay 1 to 5 for the existing ranker."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

GRID_LIMIT = 12

# Same vocabulary as ``movielens_recommender.movies.GENRES``, minus the empty bucket.
# Kept here so a shelf test does not import the training stack.
GENRE_ORDER: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def genre_parts(genres: str) -> list[str]:
    return [part.strip() for part in str(genres).split("|") if part.strip()]


def present_genres(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Genres that appear on at least one film, in catalog order."""
    found: set[str] = set()
    for record in records:
        found.update(genre_parts(str(record.get("genres", ""))))
    return [genre for genre in GENRE_ORDER if genre in found]


def popularity_order(
    popular_item_ids: Sequence[int] | None,
    item_ids: Sequence[int],
    popularity: Sequence[float],
) -> dict[int, int]:
    """Lower rank means more popular. ``popular_item_ids`` wins when it is present."""
    if popular_item_ids is not None:
        return {int(item): index for index, item in enumerate(popular_item_ids)}
    paired = list(zip(item_ids, popularity, strict=True))
    paired.sort(key=lambda item: (-float(item[1]), int(item[0])))
    return {int(item_id): index for index, (item_id, _score) in enumerate(paired)}


def select_grid(
    records: Sequence[Mapping[str, Any]],
    genres: Sequence[str],
    popularity_rank: Mapping[int, int],
    *,
    limit: int = GRID_LIMIT,
) -> list[dict[str, Any]]:
    """Films that match any selected genre, most popular first."""
    wanted = {genre.strip() for genre in genres if genre and genre.strip()}
    if not wanted or limit <= 0:
        return []
    chosen: list[dict[str, Any]] = []
    for record in records:
        parts = set(genre_parts(str(record.get("genres", ""))))
        if not parts & wanted:
            continue
        chosen.append(dict(record))
    missing = len(popularity_rank) + 1
    chosen.sort(
        key=lambda row: (
            popularity_rank.get(int(row["item_id"]), missing),
            int(row["item_id"]),
        )
    )
    return chosen[:limit]


def upsert_rating(
    rows: Sequence[Mapping[str, Any]],
    item_id: int,
    title: str,
    stars: int,
) -> list[dict[str, Any]]:
    """Replace one film's stars, or append it. Order of the other films stays put."""
    score = int(stars)
    if score < 1 or score > 5:
        raise ValueError("a rating is a whole number from 1 to 5")
    updated: list[dict[str, Any]] = []
    found = False
    for row in rows:
        if int(row["item_id"]) == int(item_id):
            updated.append({"item_id": int(item_id), "title": title, "rating": score})
            found = True
        else:
            updated.append(
                {
                    "item_id": int(row["item_id"]),
                    "title": str(row["title"]),
                    "rating": int(row["rating"]),
                }
            )
    if not found:
        updated.append({"item_id": int(item_id), "title": title, "rating": score})
    return updated


def movie_records(movies: Any) -> list[dict[str, Any]]:
    """Plain records from a snapshot movie table. Titles are display titles."""
    from movielens_recommender.serving.reasons import display_title

    records = []
    for record in movies.itertuples(index=False):
        year = getattr(record, "year", None)
        year_out = clean_year(year)
        records.append(
            {
                "item_id": int(record.item_id),
                "title": display_title(str(record.title)),
                "year": year_out,
                "genres": str(record.genres),
            }
        )
    return records


def clean_year(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
