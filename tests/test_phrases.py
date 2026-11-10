"""Card copy stays short, names the film, and drops retriever jargon."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPACE = Path(__file__).resolve().parents[1] / "space"


def _phrases():
    path = SPACE / "phrases.py"
    spec = importlib.util.spec_from_file_location("space_phrases_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plain_reasons_name_the_rated_film_and_drop_the_generic_pair():
    phrases = _phrases()
    reasons = [
        "Ranked highly by the neural retriever for your taste profile",
        "Popular with many viewers",
        "Because you rated Toy Story highly",
        "You rated Toy Story 5★ and this is one of its nearest neighbours",
        "Lines up with movies you have already rated",
    ]
    shown = phrases.plain_reasons(reasons)
    assert shown == ["Because you liked Toy Story"]
    assert all("neural" not in line and "neighbour" not in line for line in shown)
    assert "Popular with viewers" not in shown


def test_genre_line_names_the_shared_genre_and_the_film():
    phrases = _phrases()
    shown = phrases.plain_reasons(
        [
            "Matches your liking for Comedy — you rated Toy Story 4★",
            "Ranked highly by the neural retriever for your taste profile",
        ]
    )
    assert shown == [
        "Because you liked Toy Story",
        "Shares Comedy",
    ]
    assert "Toy Story" not in shown[1]


def test_two_picks_do_not_share_the_same_sentence():
    phrases = _phrases()
    alien = phrases.plain_reasons(
        [
            "You rated Alien 5★ and this is one of its nearest neighbours",
            "Matches your liking for Horror — you rated Alien 5★",
        ]
    )
    toy = phrases.plain_reasons(
        [
            "Because you rated Toy Story highly",
            "It's a Comedy film",
        ]
    )
    assert alien == ["Because you liked Alien", "Shares Horror"]
    assert toy == ["Because you liked Toy Story", "It's a Comedy film"]
    assert "Alien" not in alien[1]
    assert alien != toy


def test_popularity_is_shown_only_when_it_is_the_first_reason():
    phrases = _phrases()
    dominant = phrases.plain_reasons(
        ["Popular with many viewers", "Widely rated by other viewers"]
    )
    assert dominant == ["Popular with viewers"]
    later = phrases.plain_reasons(
        [
            "Popular with many viewers",
            "Because you rated Toy Story highly",
        ]
    )
    assert later == ["Because you liked Toy Story", "Popular with viewers"]
    buried = phrases.plain_reasons(
        [
            "Ranked highly by the neural retriever for your taste profile",
            "Widely rated by other viewers",
            "Because you rated Fargo highly",
        ]
    )
    assert buried == ["Because you liked Fargo"]


def test_shared_genres_come_from_the_named_film_when_the_sentence_has_none():
    phrases = _phrases()
    lookup = {
        "Men in Black": "Action|Adventure|Comedy|Sci-Fi",
        "American Beauty": "Comedy|Drama",
    }
    wars = phrases.plain_reasons(
        [
            "Popular with many viewers",
            "Lines up with movies you have already rated",
            "Because you rated Men in Black highly",
        ],
        genres_by_title=lookup,
        candidate_genres="Action|Adventure|Sci-Fi",
    )
    ryan = phrases.plain_reasons(
        [
            "Popular with many viewers",
            "Because you rated American Beauty highly",
        ],
        genres_by_title=lookup,
        candidate_genres="Action|Drama|War",
    )
    assert wars[0] == "Because you liked Men in Black"
    assert ryan[0] == "Because you liked American Beauty"
    assert "Men in Black" not in wars[1]
    assert "American Beauty" not in ryan[1]
    assert wars[1] != ryan[1]
    assert wars[1] in {"Shares Action", "Shares Adventure", "Shares Sci-Fi"}
    assert ryan[1] in {"Shares Drama", "It's an Action film", "It's a War film"}


def test_two_contributors_share_the_first_line_and_the_second_line_varies():
    phrases = _phrases()
    reasons = [
        "Popular with many viewers",
        "Because you rated Men in Black highly",
    ]
    lookup = {
        "Men in Black": "Action|Adventure|Comedy|Sci-Fi",
        "Toy Story": "Adventure|Animation|Children's|Comedy",
    }
    shown = phrases.plain_reasons(
        reasons,
        similar_titles=["Men in Black", "Toy Story"],
        genres_by_title=lookup,
        candidate_genres="Action|Adventure|Comedy",
        candidate_title="Rush Hour",
    )
    assert shown[0] == "Because you liked Men in Black and Toy Story"
    assert len(shown) == 2
    assert "Men in Black" not in shown[1]
    assert "Toy Story" not in shown[1]
    repeats = []
    for title in ("Alpha One", "Bravo Two", "Charlie Three", "Delta Four"):
        line = phrases.plain_reasons(
            reasons,
            genres_by_title=lookup,
            candidate_genres="Action|Adventure|Sci-Fi",
            candidate_title=title,
        )[1]
        repeats.append(line)
    assert len(set(repeats)) > 1


def test_method_sentence_does_not_name_the_retriever():
    phrases = _phrases()
    text = phrases.plain_method("cold_start_ranker")
    assert "neural" not in text
    assert "N=" not in text
