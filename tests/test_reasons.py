"""Plain-language reason templates. No MovieLens download."""

from __future__ import annotations

import re

from movielens_recommender.ranker.features import feature_names
from movielens_recommender.serving.reasons import HistoryItem, explain_recommendation

_FORBIDDEN = ("shap", "gender", "occupation", "region", "zip", "male", "female")
_AGE = re.compile(r"\bage\b", re.IGNORECASE)


def _history() -> list[HistoryItem]:
    return [
        HistoryItem(1, "Alien (1979)", "Sci-Fi|Horror", 5.0, timestamp=10),
        HistoryItem(2, "Toy Story (1995)", "Animation|Comedy|Children's", 4.0, timestamp=20),
        HistoryItem(3, "The Room (2003)", "Drama", 1.0, timestamp=30),
    ]


def _assert_main_is_plain(text: str) -> None:
    lowered = text.lower()
    for word in _FORBIDDEN:
        assert word not in lowered
    assert _AGE.search(text) is None
    for name in feature_names("both"):
        assert name not in text


def test_top_three_and_neighbor_template():
    explanation = explain_recommendation(
        {
            "item_item_score": 0.9,
            "item_item_rank": 0.4,
            "affinity_sci_fi": 0.3,
            "two_tower_score": 0.2,
            "item_popularity": 0.1,
            "item_recency": 0.05,
        },
        history=_history(),
        neighbor_similarity={1: 0.8, 2: 0.1, 3: 0.0},
    )
    assert len(explanation.reasons) == 3
    assert explanation.reasons[0].text == (
        "You rated Alien 5★ and this is one of its nearest neighbours"
    )
    assert explanation.reasons[0].contribution == 1.3
    assert "Sci-Fi" in explanation.reasons[1].text
    assert "Alien" in explanation.reasons[1].text
    assert explanation.reasons[2].text == (
        "Ranked highly by the neural retriever for your taste profile"
    )
    _assert_main_is_plain(explanation.main_text())


def test_demographic_contributions_collapse_to_one_generic_reason():
    explanation = explain_recommendation(
        {
            "demo_gender": 1.2,
            "demo_age": 0.4,
            "demo_occupation": 0.3,
            "demo_region": 0.2,
            "group_gender_pos_rate": 0.8,
            "group_age_pop_share": 0.3,
            "group_occupation_pos_rate": 0.1,
            "two_tower_rank": 0.15,
            "item_popularity": 0.05,
        },
        history=_history(),
        neighbor_similarity={},
    )
    texts = [reason.text for reason in explanation.reasons]
    assert texts[0] == "Popular with viewers similar to you"
    assert texts.count("Popular with viewers similar to you") == 1
    assert len(explanation.reasons) == 3
    assert explanation.reasons[0].contribution == 1.2 + 0.4 + 0.3 + 0.2 + 0.8 + 0.3 + 0.1
    _assert_main_is_plain(explanation.main_text())
    detail_names = [name for name, _value in explanation.details]
    assert "demo_gender" in detail_names
    assert "group_age_pop_share" in detail_names


def test_negative_contributions_are_not_reasons_and_raw_names_stay_in_details():
    explanation = explain_recommendation(
        {
            "item_popularity": -4.0,
            "affinity_comedy": 0.2,
            "genre_comedy": 0.1,
            "demo_gender": -0.5,
        },
        history=_history(),
        neighbor_similarity={},
    )
    assert len(explanation.reasons) == 1
    assert explanation.reasons[0].text == (
        "Matches your liking for Comedy — you rated Toy Story 4★"
    )
    _assert_main_is_plain(explanation.main_text())
    assert any(name == "item_popularity" and value == -4.0 for name, value in explanation.details)
    assert any(name == "demo_gender" for name, _value in explanation.details)


def test_neighbor_driver_is_similarity_times_rating():
    history = [
        HistoryItem(1, "Wayne's World (1992)", "Comedy", 1.0, timestamp=1),
        HistoryItem(2, "You've Got Mail (1998)", "Comedy|Romance", 4.0, timestamp=2),
    ]
    explanation = explain_recommendation(
        {"item_item_score": 0.5, "item_item_rank": 0.2},
        history=history,
        # High similarity on the 1★ title, but 0.4 * 4 beats 0.9 * 1.
        neighbor_similarity={1: 0.9, 2: 0.4},
    )
    assert explanation.reasons[0].text == (
        "You rated You've Got Mail 4★ and this is one of its nearest neighbours"
    )


def test_missing_neighbor_does_not_invent_a_title():
    explanation = explain_recommendation(
        {"item_item_score": 0.4},
        history=_history(),
        neighbor_similarity={1: 0.0, 2: 0.0},
    )
    assert explanation.reasons[0].text == "Close to movies you have already rated"
    assert "Alien" not in explanation.main_text()


def test_low_rated_genre_example_is_not_cited_as_liking():
    explanation = explain_recommendation(
        {"affinity_drama": 0.4},
        history=_history(),
        neighbor_similarity={},
    )
    assert explanation.reasons[0].text == "Matches your liking for Drama"
    assert "The Room" not in explanation.main_text()
