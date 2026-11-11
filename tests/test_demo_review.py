"""Regressions for share links, adjusted lists, and changing a taste profile."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

SPACE = Path(__file__).resolve().parents[1] / "space"


@pytest.fixture
def value(monkeypatch):
    monkeypatch.syspath_prepend(str(SPACE))
    import value

    return value


def test_share_url_replaces_old_profile_and_preserves_the_slider(value):
    url = value.share_url(
        "https://demo.streamlit.app/~/+/?view=quality&ratings=9-1&mix=0&embed=true#old",
        [(260, 5), (1, 4)],
        mix=3,
    )
    parsed = urlsplit(url)
    assert parsed.netloc == "demo.streamlit.app"
    assert parsed.path == "/"
    assert not parsed.fragment
    assert parse_qs(parsed.query) == {
        "view": ["discover"],
        "ratings": ["260-5.1-4"],
        "mix": ["3"],
    }
    assert value.decode_ratings(parse_qs(parsed.query)["ratings"][0]) == [(260, 5), (1, 4)]
    with pytest.raises(ValueError):
        value.share_url("http://localhost:8501", [(1, 4)], mix=5)


def test_why_not_distinguishes_displayed_picks_from_the_measured_order(value):
    kwargs = dict(
        rated_ids={1}, ranked_ids=[10, 20, 30, 40], top_n=2, shown_ids=[40, 30], adjusted=True
    )
    assert "number 1 in the list you are viewing" in value.why_not(
        title="Moved up",
        item_id=40,
        **kwargs,
    )
    demoted = value.why_not(title="Moved down", item_id=10, **kwargs)
    assert "not in the list you are viewing" in demoted
    assert "before slider adjustment" in demoted
    assert "not among the candidates" in value.why_not(title="Missing", item_id=99, **kwargs)
    assert "already in your ratings" in value.why_not(title="Seen", item_id=1, **kwargs)


def test_low_ratings_do_not_become_claims_of_liking_a_film(value):
    lines = ["Because you liked Toy Story", "Shares Comedy with Toy Story"]
    assert value.qualify_likes(lines, {"Fargo"}) == [
        "Similar to films you rated: Toy Story",
        "Shares Comedy with Toy Story",
    ]
    assert value.qualify_likes(lines, {"Toy Story"}) == lines
    assert value.qualify_likes(
        ["Because you liked Beauty and the Beast"], {"Beauty and the Beast"}
    ) == [
        "Because you liked Beauty and the Beast",
    ]
    assert not value.cites_unrated(
        "Because you liked Beauty and the Beast and Fargo", {"Beauty and the Beast", "Fargo"}
    )


def _app(source: str):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_string(
        f"import sys\nsys.path.insert(0, {str(SPACE)!r})\n" + source,
        default_timeout=20,
    ).run()


def test_stale_profile_hides_cards_and_sharing_until_refreshed():
    app = _app("""
import streamlit as st
from unittest.mock import patch
import pages
ratings = [{"item_id": 1, "title": "Current film", "rating": 1}]
st.session_state["new_user_result"] = {
    "signature": ((1, 5),),
    "recommendations": [{"item_id": 2, "title": "An old pick"}],
}
with patch.object(pages, "_reranked_cards", side_effect=AssertionError("stale list used")):
    pages._picks(ratings, "", {}, "unused", [], {}, {})
""")
    assert not app.exception
    assert "at least 3 ratings" in app.warning[0].value
    assert app.button[0].label == "Refresh my picks"
    assert app.button[0].disabled
    assert not app.checkbox
    assert not any("An old pick" in mark.value for mark in app.markdown)


def test_sample_replaces_profile_and_resets_existing_star_widgets():
    app = _app("""
import streamlit as st
import pages
st.session_state["new_user_ratings"] = [{"item_id": 1, "title": "Old", "rating": 2}]
st.session_state["rate-1"] = 2
by_id = {item: {"title": f"Sample {item}"} for item in [2, 3, 4]}
pages._rate_taste([2, 3, 4, 999], by_id)
""")
    assert not app.exception
    assert [(r["item_id"], r["rating"]) for r in app.session_state["new_user_ratings"]] == [
        (2, 5),
        (3, 5),
        (4, 5),
    ]
    assert "rate-1" not in app.session_state


def test_shared_profile_restores_slider_and_ratings_before_scoring():
    app = _app("""
import streamlit as st
from unittest.mock import patch
import pages
st.query_params.update({"ratings": "1-5.2-4.3-3", "mix": "3"})
by_id = {item: {"title": f"Film {item}"} for item in [1, 2, 3]}
with patch.object(pages, "_score") as score:
    pages._restore_shared_ratings("snapshot", by_id)
    st.session_state["scored"] = score.call_count
    pages._restore_shared_ratings("snapshot", by_id)
    st.session_state["scored_twice"] = score.call_count
""")
    assert not app.exception
    assert app.session_state["familiarity"] == "Further"
    assert app.session_state["scored"] == app.session_state["scored_twice"] == 1
    assert [r["rating"] for r in app.session_state["new_user_ratings"]] == [5, 4, 3]


def test_sharing_does_not_claim_clipboard_success_or_mutate_the_page_url():
    app = _app("""
import streamlit as st
import pages
st.query_params["view"] = "discover"
pages._pick_tools([], {}, [{"item_id": 1, "rating": 5}], None, [])
""")
    app.checkbox[0].check().run()
    assert not app.exception
    assert "ratings=1-5" in app.code[0].value
    assert "mix=0" in app.code[0].value
    assert "ratings" not in app.query_params
    assert not any("Link copied" in caption.value for caption in app.caption)
