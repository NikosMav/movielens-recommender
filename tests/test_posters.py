"""Poster lookup never keeps image bytes, and a dead network falls back."""

from __future__ import annotations

import importlib.util
import json
import urllib.parse
from pathlib import Path

SPACE = Path(__file__).resolve().parents[1] / "space"


def _load(name: str):
    path = SPACE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"space_{name}_poster_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> bool:
        self.closed = True
        return False

    def read(self, _n: int = -1) -> bytes:
        return self._payload


def test_missing_key_does_not_touch_the_network():
    posters = _load("posters")

    def _open(*_args, **_kwargs):
        raise AssertionError("a missing key must not call the network")

    assert posters.poster_url("Toy Story", 1995, "", opener=_open) is None
    assert posters.poster_url("Toy Story", 1995, "   ", opener=_open) is None


def test_network_error_falls_back_to_none():
    posters = _load("posters")

    def _open(*_args, **_kwargs):
        raise OSError("offline")

    assert posters.poster_url("Toy Story", 1995, "test-key", opener=_open) is None


def _params(request) -> dict[str, list[str]]:
    parsed = urllib.parse.urlparse(request.full_url)
    return urllib.parse.parse_qs(parsed.query)


def test_exact_match_returns_an_image_url_without_the_key():
    posters = _load("posters")
    seen: dict[str, str] = {}
    calls = {"n": 0}

    def _open(request, timeout):
        calls["n"] += 1
        seen["url"] = request.full_url
        seen["timeout"] = str(timeout)
        assert "image.tmdb.org" not in request.full_url
        query = _params(request)
        assert query["primary_release_year"] == ["1995"]
        assert "year" not in query
        assert query["include_adult"] == ["false"]
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Toy Story 2",
                        "original_title": "Toy Story 2",
                        "release_date": "1999-11-24",
                        "poster_path": "/other.jpg",
                        "popularity": 99,
                    },
                    {
                        "title": "Toy Story",
                        "original_title": "Toy Story",
                        "release_date": "1995-11-22",
                        "poster_path": "/abc.jpg",
                        "popularity": 10,
                    },
                ]
            }
        ).encode()
        return _Body(body)

    url = posters.poster_url("Toy Story", 1995, "test-key", opener=_open)
    assert url == "https://image.tmdb.org/t/p/w185/abc.jpg"
    assert "test-key" not in (url or "")
    assert "api_key=test-key" in seen["url"]
    assert seen["timeout"] == "8"
    assert calls["n"] == 1


def test_wrong_title_and_unsafe_path_are_placeholders():
    posters = _load("posters")
    calls = {"n": 0}

    def _open(_request, _timeout):
        calls["n"] += 1
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Toy Story 2",
                        "release_date": "1995-11-22",
                        "poster_path": "/wrong.jpg",
                        "popularity": 50,
                    },
                    {
                        "title": "Toy Story",
                        "release_date": "1995-11-22",
                        "poster_path": "/../secrets.jpg",
                        "popularity": 100,
                    },
                    {
                        "title": "Toy Story",
                        "release_date": "1995-11-22",
                        "poster_path": "secrets.jpg",
                        "popularity": 80,
                    },
                ]
            }
        ).encode()
        return _Body(body)

    assert posters.poster_url("Toy Story", 1995, "test-key", opener=_open) is None
    # One title, four year attempts, and still no usable path.
    assert calls["n"] == 4


def test_card_placeholder_has_no_image_request():
    cards = _load("cards")
    html = cards.recommendation_card_html(
        title="Toy Story",
        year=1995,
        genres="Animation",
        reasons=["You rated a neighbour 5 stars."],
        poster_url=None,
    )
    assert "No poster" in html
    assert "<img" not in html
    linked = cards.recommendation_card_html(
        title="Toy Story",
        year=1995,
        genres="Animation",
        reasons=["You rated a neighbour 5 stars."],
        poster_url="https://image.tmdb.org/t/p/w185/abc.jpg",
    )
    assert 'src="https://image.tmdb.org/t/p/w185/abc.jpg"' in linked
    assert "test-key" not in linked
    assert "No poster" not in linked


def test_missing_key_stays_at_zero_requests_for_a_hard_title():
    posters = _load("posters")
    calls = {"n": 0}

    def _open(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("a missing key must not call the network")

    title = "Star Wars: Episode V - The Empire Strikes Back (1980)"
    assert posters.poster_url(title, 1980, "", opener=_open) is None
    assert calls["n"] == 0


def test_network_error_stops_the_ladder():
    posters = _load("posters")
    calls = {"n": 0}

    def _open(_request, _timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Body(b'{"results":[]}')
        raise OSError("offline")

    assert posters.poster_url("Toy Story", 1995, "test-key", opener=_open) is None
    # Empty answer, then the next year fails four times and the ladder stops.
    assert calls["n"] == 1 + posters.MAX_ATTEMPTS


def test_trailing_articles_become_reading_order_queries():
    posters = _load("posters")
    expected = {
        "Matrix, The (1999)": "The Matrix",
        "Walk in the Clouds, A (1995)": "A Walk in the Clouds",
        "Awfully Big Adventure, An (1995)": "An Awfully Big Adventure",
        "Misérables, Les (1995)": "Les Misérables",
        "Ciudad, La": "La Ciudad",
        "Postino, Il (1994)": "Il Postino",
        "Superweib, Das (1996)": "Das Superweib",
        "Bewegte Mann, Der": "Der Bewegte Mann",
        "Mariachi, El (1992)": "El Mariachi",
        "Colonel Chabert, Le (1994)": "Le Colonel Chabert",
        "Harder They Come, Die": "Die Harder They Come",
        "Enfer, L' (1994)": "L'Enfer",
    }
    for raw, query in expected.items():
        assert posters.candidate_queries(raw)[0] == query
    assert posters.candidate_queries("Paris, Texas (1984)") == ["Paris, Texas"]
    assert posters.candidate_queries("The Matrix") == ["The Matrix"]


def test_parenthetical_alternate_is_a_later_query():
    posters = _load("posters")
    assert posters.candidate_queries("Shichinin no samurai (Seven Samurai) (1954)") == [
        "Shichinin no samurai",
        "Seven Samurai",
    ]
    assert posters.candidate_queries("Postino, Il (The Postman) (1994)") == [
        "Il Postino",
        "The Postman",
    ]
    # The display form already moved the article. The alternate still comes second.
    assert posters.candidate_queries("Il Postino (The Postman)") == [
        "Il Postino",
        "The Postman",
    ]


def test_star_wars_falls_back_to_the_part_after_colon_or_dash():
    posters = _load("posters")
    calls: list[tuple[str, str | None]] = []

    def _open(request, _timeout):
        query = _params(request)
        year = query.get("primary_release_year", [None])[0]
        text = query["query"][0]
        calls.append((text, year))
        assert "year" not in query
        results = []
        if text == "The Empire Strikes Back" and year == "1980":
            results = [
                {
                    "title": "The Empire Strikes Back",
                    "original_title": "The Empire Strikes Back",
                    "release_date": "1980-05-21",
                    "poster_path": "/empire.jpg",
                    "popularity": 20,
                }
            ]
        return _Body(json.dumps({"results": results}).encode())

    url = posters.poster_url(
        "Star Wars: Episode V - The Empire Strikes Back (1980)",
        1980,
        "test-key",
        opener=_open,
    )
    assert url == "https://image.tmdb.org/t/p/w185/empire.jpg"
    assert "test-key" not in (url or "")
    assert calls[0] == ("Star Wars: Episode V - The Empire Strikes Back", "1980")
    assert ("Episode V - The Empire Strikes Back", "1980") in calls
    hit = calls.index(("The Empire Strikes Back", "1980"))
    assert all(year == "1980" for _text, year in calls[: hit + 1])
    assert posters.candidate_queries("Star Wars: Episode V - The Empire Strikes Back") == [
        "Star Wars: Episode V - The Empire Strikes Back",
        "Episode V - The Empire Strikes Back",
        "The Empire Strikes Back",
    ]


def test_year_ladder_uses_primary_release_year_then_neighbours_then_none():
    posters = _load("posters")
    calls: list[str | None] = []

    def _open(request, _timeout):
        query = _params(request)
        assert query["query"] == ["Toy Story"]
        assert "year" not in query
        year = query.get("primary_release_year", [None])[0]
        calls.append(year)
        if year is not None:
            return _Body(b'{"results":[]}')
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Toy Story",
                        "original_title": "Toy Story",
                        "release_date": "2010-06-18",
                        "poster_path": "/far.jpg",
                        "popularity": 90,
                    },
                    {
                        "title": "Toy Story",
                        "original_title": "Toy Story",
                        "release_date": "1995-11-22",
                        "poster_path": "/near.jpg",
                        "popularity": 1,
                    },
                ]
            }
        ).encode()
        return _Body(body)

    url = posters.poster_url("Toy Story", 1995, "test-key", opener=_open)
    assert calls == ["1995", "1994", "1996", None]
    assert url == "https://image.tmdb.org/t/p/w185/near.jpg"


def test_one_year_off_matches_before_dropping_the_year():
    posters = _load("posters")
    calls: list[str | None] = []

    def _open(request, _timeout):
        query = _params(request)
        year = query.get("primary_release_year", [None])[0]
        calls.append(year)
        if year != "1996":
            return _Body(b'{"results":[]}')
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Toy Story",
                        "release_date": "1996-01-01",
                        "poster_path": "/year.jpg",
                        "popularity": 4,
                    }
                ]
            }
        ).encode()
        return _Body(body)

    url = posters.poster_url("Toy Story", 1995, "test-key", opener=_open)
    assert url == "https://image.tmdb.org/t/p/w185/year.jpg"
    assert calls == ["1995", "1994", "1996"]


def test_alternate_title_is_searched_at_the_same_year_before_relaxing_it():
    posters = _load("posters")
    calls: list[tuple[str, str | None]] = []

    def _open(request, _timeout):
        query = _params(request)
        text = query["query"][0]
        year = query.get("primary_release_year", [None])[0]
        calls.append((text, year))
        if text == "Seven Samurai" and year == "1954":
            body = {
                "results": [
                    {
                        "title": "Seven Samurai",
                        "original_title": "七人の侍",
                        "release_date": "1954-04-26",
                        "poster_path": "/seven.jpg",
                        "popularity": 30,
                    }
                ]
            }
            return _Body(json.dumps(body).encode())
        return _Body(b'{"results":[]}')

    url = posters.poster_url(
        "Shichinin no samurai (Seven Samurai) (1954)",
        1954,
        "test-key",
        opener=_open,
    )
    assert url == "https://image.tmdb.org/t/p/w185/seven.jpg"
    assert calls == [
        ("Shichinin no samurai", "1954"),
        ("Seven Samurai", "1954"),
    ]


def test_poster_query_script_lists_a_sample_without_network():
    import subprocess
    import sys

    script = SPACE.parent / "scripts" / "poster_queries.py"
    sample = subprocess.run(
        [sys.executable, str(script)],
        input="",
        capture_output=True,
        text=True,
        check=True,
    )
    assert sample.stderr == ""
    assert "The Matrix" in sample.stdout
    assert "The Empire Strikes Back" in sample.stdout
    assert "Shichinin no samurai" in sample.stdout
    assert "Seven Samurai" in sample.stdout
    chosen = subprocess.run(
        [sys.executable, str(script)],
        input="Toy Story (1995)\n",
        capture_output=True,
        text=True,
        check=True,
    )
    assert chosen.stdout.strip() == "Toy Story (1995)\n  Toy Story"
    assert "The Matrix" not in chosen.stdout


def test_rate_limit_is_retried_and_then_returns_the_poster():
    from email.message import Message
    from urllib.error import HTTPError

    posters = _load("posters")
    calls = {"n": 0}
    waits: list[float] = []

    def _open(request, _timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            headers = Message()
            headers["Retry-After"] = "2"
            raise HTTPError(request.full_url, 429, "Too Many Requests", headers, None)
        body = json.dumps(
            {
                "results": [
                    {
                        "title": "Toy Story",
                        "original_title": "Toy Story",
                        "release_date": "1995-11-22",
                        "poster_path": "/toy.jpg",
                        "popularity": 80,
                    }
                ]
            }
        ).encode()
        return _Body(body)

    url, definitive = posters.lookup_poster(
        "Toy Story",
        1995,
        "test-key",
        opener=_open,
        sleeper=waits.append,
    )
    assert url == "https://image.tmdb.org/t/p/w185/toy.jpg"
    assert definitive is True
    assert calls["n"] == 3
    assert waits == [2.0, 2.0]


def test_parallel_searches_stay_within_the_flight_cap():
    import threading
    import time

    posters = _load("posters")
    current = {"n": 0}
    peak = {"n": 0}
    lock = threading.Lock()

    def _open(_request, _timeout):
        with lock:
            current["n"] += 1
            peak["n"] = max(peak["n"], current["n"])
        time.sleep(0.05)
        with lock:
            current["n"] -= 1
        return _Body(b'{"results":[]}')

    threads = [
        threading.Thread(
            target=posters.poster_url,
            args=(f"Film {index}", 1995, "test-key"),
            kwargs={"opener": _open},
        )
        for index in range(6)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak["n"] <= posters.MAX_IN_FLIGHT


def test_poster_cache_ttl_stays_one_day_in_memory():
    posters = _load("posters")
    loading = (SPACE / "loading.py").read_text(encoding="utf-8")
    assert posters.POSTER_CACHE_TTL_SECONDS == 24 * 60 * 60
    assert "ttl=POSTER_CACHE_TTL_SECONDS" in loading
    assert "st.cache_data" in loading


def test_fixture_posters_stay_off_the_tmdb_path(monkeypatch):
    posters = _load("posters")
    monkeypatch.setenv("MOVIELENS_FAKE_POSTERS", "1")
    calls = {"n": 0}

    def _open(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("the fixture flag must not change poster_url")

    assert posters.poster_url("Toy Story", 1995, "", opener=_open) is None
    assert calls["n"] == 0
    first = posters.fixture_poster("Toy Story")
    second = posters.fixture_poster("Toy Story")
    assert first == second
    assert first.startswith("data:image/svg+xml,")
    assert "image.tmdb.org" not in first
    assert "api_key" not in first
    source = (SPACE / "posters.py").read_text(encoding="utf-8")
    assert "MOVIELENS_FAKE_POSTERS" not in source
    loading = (SPACE / "loading.py").read_text(encoding="utf-8")
    assert loading.index("MOVIELENS_FAKE_POSTERS") < loading.index("if not api_key:")
