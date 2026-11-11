"""Card copy stays short, names a close film, and drops retriever jargon."""

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
    assert shown[0] == "Because you liked Toy Story"
    assert all("neural" not in line and "neighbour" not in line for line in shown)
    named = phrases.plain_reasons(
        [
            "Because you rated Toy Story highly",
            "Popular with many viewers",
        ]
    )
    assert named == ["Because you liked Toy Story"]


def test_genre_line_names_the_shared_genre_and_the_film():
    phrases = _phrases()
    shown = phrases.plain_reasons(
        [
            "Matches your liking for Comedy — you rated Toy Story 4★",
            "Ranked highly by the neural retriever for your taste profile",
        ],
        candidate_genres="Comedy|Romance",
        genres_by_title={"Toy Story": "Comedy|Romance|Animation"},
    )
    assert shown == [
        "Because you liked Toy Story",
        "Shares Comedy and Romance with Toy Story",
    ]


def test_two_picks_do_not_share_the_same_sentence():
    phrases = _phrases()
    alien = phrases.plain_reasons(
        [
            "You rated Alien 5★ and this is one of its nearest neighbours",
            "Matches your liking for Horror — you rated Alien 5★",
        ],
        candidate_genres="Horror|Thriller",
        genres_by_title={"Alien": "Horror|Sci-Fi"},
    )
    toy = phrases.plain_reasons(
        [
            "Because you rated Toy Story highly",
            "It's a Comedy film",
        ],
        candidate_year=1999,
    )
    assert alien == ["Because you liked Alien", "Shares Horror with Alien"]
    assert toy == ["Because you liked Toy Story", "A 1990s film"]
    assert "It's a" not in " ".join(toy + alien)
    assert alien != toy


def test_popularity_needs_no_close_film():
    phrases = _phrases()
    dominant = phrases.plain_reasons(
        ["Popular with many viewers", "Widely rated by other viewers"]
    )
    assert dominant == [phrases.POPULAR_LINE]
    close = phrases.plain_reasons(
        [
            "Popular with many viewers",
            "Because you rated Toy Story highly",
        ],
        similar=[("Toy Story", 0.45)],
    )
    assert close[0] == "Because you liked Toy Story"
    weak = phrases.plain_reasons(
        ["Popular with many viewers", "Because you rated Toy Story highly"],
        similar=[("Toy Story", 0.02)],
        candidate_year=1994,
    )
    assert weak[0] == phrases.POPULAR_LINE
    assert "Toy Story" not in " ".join(weak)
    assert weak[1] == "A 1990s film"
    buried = phrases.plain_reasons(
        [
            "Ranked highly by the neural retriever for your taste profile",
            "Because you rated Fargo highly",
            "Widely rated by other viewers",
        ]
    )
    assert buried == ["Because you liked Fargo"]


def test_shared_genres_name_the_rated_film():
    phrases = _phrases()
    lookup = {
        "Men in Black": "Action|Adventure|Comedy|Sci-Fi",
        "American Beauty": "Comedy|Drama",
    }
    wars = phrases.plain_reasons(
        [
            "Because you rated Men in Black highly",
            "Popular with many viewers",
            "Lines up with movies you have already rated",
        ],
        genres_by_title=lookup,
        candidate_genres="Action|Adventure|Sci-Fi",
    )
    ryan = phrases.plain_reasons(
        [
            "Because you rated American Beauty highly",
            "Popular with many viewers",
        ],
        genres_by_title=lookup,
        candidate_genres="Action|Drama|War",
    )
    assert wars == [
        "Because you liked Men in Black",
        "Shares Action and Adventure with Men in Black",
    ]
    assert ryan == [
        "Because you liked American Beauty",
        "Shares Drama with American Beauty",
    ]


def test_two_contributors_share_the_first_line():
    phrases = _phrases()
    reasons = [
        "Because you rated Men in Black highly",
        "Popular with many viewers",
    ]
    weak = phrases.plain_reasons(
        ["Popular with many viewers", "Because you rated American Beauty highly"],
        similar=[("American Beauty", 0.02), ("Toy Story", 0.01)],
        candidate_year=1999,
    )
    assert weak[0] == phrases.POPULAR_LINE
    assert "American Beauty" not in " ".join(weak)
    lookup = {
        "Men in Black": "Action|Adventure|Comedy|Sci-Fi",
        "Toy Story": "Adventure|Animation|Children's|Comedy",
    }
    shown = phrases.plain_reasons(
        reasons,
        similar=[("Men in Black", 0.61), ("Toy Story", 0.44)],
        genres_by_title=lookup,
        candidate_genres="Action|Adventure|Comedy",
    )
    assert shown[0] == "Because you liked Men in Black and Toy Story"
    assert shown[1] == "Shares Action and Adventure with Men in Black"


def test_widely_loved_is_at_most_one_card_in_three_unless_nothing_is_close():
    phrases = _phrases()
    sims = [0.55, 0.08, 0.07, 0.06, 0.01, 0.40]
    choices = [
        phrases.reason_choice(
            ["Popular with many viewers"],
            similar=[("Toy Story", sim)],
            genres_by_title={"Toy Story": "Comedy"},
            candidate_genres="Comedy",
            candidate_year=1995,
        )
        for sim in sims
    ]
    shown = phrases.cap_popular(choices)
    popular = [lines for lines in shown if lines and lines[0] == phrases.POPULAR_LINE]
    # Six cards, so at most two. The 0.01 neighbour has no film line, so it stays.
    assert len(popular) == 2
    assert shown[0][0] == "Because you liked Toy Story"
    assert shown[4][0] == phrases.POPULAR_LINE
    assert shown[5][0] == "Because you liked Toy Story"


def test_method_sentence_does_not_name_the_retriever():
    phrases = _phrases()
    text = phrases.plain_method("cold_start_ranker")
    assert "neural" not in text
    assert "N=" not in text
