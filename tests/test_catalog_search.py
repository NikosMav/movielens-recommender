"""Title search prefers an exact title, then a prefix, then a word, then a substring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPACE = Path(__file__).resolve().parents[1] / "space"


def _catalog():
    path = SPACE / "catalog.py"
    spec = importlib.util.spec_from_file_location("space_catalog_search_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(item_id: int, title: str) -> dict:
    return {"item_id": item_id, "title": title, "year": 1999, "genres": "Drama"}


def test_search_ranks_exact_then_prefix_then_word_then_substring():
    catalog = _catalog()
    records = [
        _row(1, "Babes in Toyland"),
        _row(2, "Toy Story"),
        _row(3, "Toy"),
        _row(4, "The Story of Toys"),
    ]
    ranks = {1: 1, 2: 50, 3: 9, 4: 2}
    titles = [row["title"] for row in catalog.rank_titles("toy", records, ranks)]
    assert titles == ["Toy", "Toy Story", "Babes in Toyland", "The Story of Toys"]


def test_star_wars_beats_a_star_is_born_and_unrelated_titles_drop_out():
    catalog = _catalog()
    records = [
        _row(1, "A Star Is Born"),
        _row(2, "Bastard Out of Carolina"),
        _row(3, "Brenda Starr"),
        _row(4, "Star Wars: Episode IV - A New Hope"),
    ]
    ranks = {1: 1, 2: 1, 3: 1, 4: 80}
    titles = [row["title"] for row in catalog.rank_titles("star", records, ranks)]
    assert titles[0].startswith("Star Wars")
    assert titles.index("Star Wars: Episode IV - A New Hope") < titles.index("A Star Is Born")
    assert titles.index("A Star Is Born") < titles.index("Bastard Out of Carolina")


def test_matrix_matches_the_word_and_popularity_breaks_ties():
    catalog = _catalog()
    records = [
        _row(1, "Mass Transit"),
        _row(2, "Madonna"),
        _row(3, "The Matrix"),
        _row(4, "Matrix Reloaded"),
    ]
    ranks = {3: 40, 4: 3}
    titles = [row["title"] for row in catalog.rank_titles("matr", records, ranks)]
    assert titles == ["Matrix Reloaded", "The Matrix"]


def test_short_query_returns_nothing():
    catalog = _catalog()
    assert catalog.rank_titles("t", [_row(1, "Toy Story")], {1: 1}) == []


def test_suggested_ids_prefer_known_viewers_and_stay_inside_the_snapshot():
    catalog = _catalog()
    assert catalog.suggested_user_ids([1, 2, 3, 4, 5, 6, 7, 8]) == [1, 5, 8]
    assert catalog.suggested_user_ids(range(1, 6041))[:3] == [1, 155, 4000]
