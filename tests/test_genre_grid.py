"""Genre shelf: which films are offered, and how a click becomes a 1–5 rating."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[1] / "space"


def _catalog():
    path = SPACE / "catalog.py"
    spec = importlib.util.spec_from_file_location("space_catalog_grid_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _films():
    return [
        {"item_id": 1, "title": "Alpha", "year": 1994, "genres": "Action|Comedy"},
        {"item_id": 2, "title": "Bravo", "year": 1995, "genres": "Comedy"},
        {"item_id": 3, "title": "Charlie", "year": 1999, "genres": "Drama"},
        {"item_id": 4, "title": "Delta", "year": 1990, "genres": "Comedy|Drama"},
    ]


def test_genre_order_matches_the_package():
    from movielens_recommender.movies import GENRES

    catalog = _catalog()
    assert catalog.GENRE_ORDER == tuple(genre for genre in GENRES if genre != "(no genres listed)")


def test_empty_selection_is_an_empty_shelf():
    catalog = _catalog()
    assert catalog.select_grid(_films(), [], {1: 0, 2: 1, 3: 2, 4: 3}) == []
    assert catalog.select_grid(_films(), ["  "], {1: 0}) == []


def test_shelf_is_the_union_most_popular_first_and_capped():
    catalog = _catalog()
    ranks = {1: 3, 2: 0, 3: 2, 4: 1}
    comedy = catalog.select_grid(_films(), ["Comedy"], ranks)
    assert [row["item_id"] for row in comedy] == [2, 4, 1]
    both = catalog.select_grid(_films(), ["Drama", "Action"], ranks)
    assert [row["item_id"] for row in both] == [4, 3, 1]
    capped = catalog.select_grid(_films(), ["Comedy"], ranks, limit=1)
    assert [row["item_id"] for row in capped] == [2]


def test_present_genres_follow_catalog_order():
    catalog = _catalog()
    assert catalog.present_genres(_films()) == ["Action", "Comedy", "Drama"]


def test_popularity_order_prefers_the_snapshot_list():
    catalog = _catalog()
    assert catalog.popularity_order([4, 2, 1], [1, 2, 4], [9.0, 8.0, 7.0]) == {4: 0, 2: 1, 1: 2}
    fallback = catalog.popularity_order(None, [1, 2, 4], [1.0, 5.0, 5.0])
    assert fallback == {2: 0, 4: 1, 1: 2}


def test_upsert_rating_replaces_one_film_and_keeps_order():
    catalog = _catalog()
    rows = catalog.upsert_rating([], 2, "Bravo", 5)
    rows = catalog.upsert_rating(rows, 4, "Delta", 4)
    rows = catalog.upsert_rating(rows, 2, "Bravo", 3)
    assert rows == [
        {"item_id": 2, "title": "Bravo", "rating": 3},
        {"item_id": 4, "title": "Delta", "rating": 4},
    ]
    with pytest.raises(ValueError):
        catalog.upsert_rating(rows, 1, "Alpha", 0)
