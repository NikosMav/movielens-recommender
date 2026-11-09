"""Poster lookup never keeps image bytes, and a dead network falls back."""

from __future__ import annotations

import importlib.util
import json
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


def test_exact_match_returns_an_image_url_without_the_key():
    posters = _load("posters")
    seen: dict[str, str] = {}

    def _open(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = str(timeout)
        assert "image.tmdb.org" not in request.full_url
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


def test_title_or_year_mismatch_and_unsafe_path_are_placeholders():
    posters = _load("posters")

    def _open(_request, _timeout):
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
                        "release_date": "1996-01-01",
                        "poster_path": "/year.jpg",
                        "popularity": 40,
                    },
                    {
                        "title": "Toy Story",
                        "release_date": "1995-11-22",
                        "poster_path": "/../secrets.jpg",
                        "popularity": 100,
                    },
                ]
            }
        ).encode()
        return _Body(body)

    assert posters.poster_url("Toy Story", 1995, "test-key", opener=_open) is None


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
