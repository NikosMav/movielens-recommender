"""S5b HTTP API. Synthetic snapshots only; no MovieLens download."""

from __future__ import annotations

import re

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from movielens_recommender.serving.api import create_app  # noqa: E402

_DEMOGRAPHIC = re.compile(r"gender|occupation|zip|\bage\b|demo_|group_", re.IGNORECASE)


@pytest.fixture(scope="module")
def client(artifact_dir, cold_artifact_dir):
    return TestClient(create_app(artifact_dir, cold_artifact_dir))


def test_health_and_snapshot(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "production": True, "new_user": True}
    snapshot = client.get("/v1/snapshot").json()
    assert snapshot["dataset"] == "synthetic"
    assert snapshot["dataset_sha256"] == "synthetic-sha256"
    assert snapshot["candidate_set"]
    assert snapshot["new_user_snapshot"] is True


def test_known_user_recommendations(client, artifact_dir):
    from movielens_recommender.serving.bundle import load_bundle, recommend_for_user

    response = client.get("/v1/users/1/recommendations", params={"n": 5})
    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 1
    expected = recommend_for_user(load_bundle(artifact_dir), 1, n=5)["production"]
    assert [card["item_id"] for card in body["recommendations"]] == [
        card["item_id"] for card in expected
    ]
    for card in body["recommendations"]:
        assert card["reasons"]
        assert _DEMOGRAPHIC.search(" ".join(card["reasons"])) is None
        assert "score" in card

    plain = client.get("/v1/users/1/recommendations", params={"n": 5, "explain": False}).json()
    assert [card["item_id"] for card in plain["recommendations"]] == [
        card["item_id"] for card in body["recommendations"]
    ]
    assert all("reasons" not in card for card in plain["recommendations"])


def test_known_user_errors(client):
    missing = client.get("/v1/users/999/recommendations")
    assert missing.status_code == 404
    assert "999" in missing.json()["detail"]
    assert client.get("/v1/users/1/recommendations", params={"n": 0}).status_code == 422
    assert client.get("/v1/users/1/recommendations", params={"n": 101}).status_code == 422
    assert client.get("/v1/users/abc/recommendations").status_code == 422


def test_movie_search(client):
    hits = client.get("/v1/movies/search", params={"q": "movie 1"}).json()["movies"]
    assert {hit["item_id"] for hit in hits} == {1, 10, 11, 12}
    assert hits == sorted(hits, key=lambda hit: hit["title"].lower())
    capped = client.get("/v1/movies/search", params={"q": "movie", "limit": 3}).json()
    assert len(capped["movies"]) == 3
    assert client.get("/v1/movies/search", params={"q": ""}).status_code == 422
    assert client.get("/v1/movies/search", params={"q": "zzz"}).json()["movies"] == []


def test_new_user(client):
    payload = {
        "ratings": [
            {"item_id": 1, "rating": 5},
            {"item_id": 2, "rating": 4.5},
            {"item_id": 3, "rating": 4},
        ],
        "n": 4,
    }
    response = client.post("/v1/recommendations/new-user", json=payload)
    assert response.status_code == 200
    body = response.json()
    items = [card["item_id"] for card in body["recommendations"]]
    assert items
    assert len(items) <= 4
    assert set(items).isdisjoint({1, 2, 3})
    for card in body["recommendations"]:
        assert _DEMOGRAPHIC.search(" ".join(card["reasons"])) is None
    assert body["latency_sec"] >= 0.0


def test_new_user_validation(client):
    url = "/v1/recommendations/new-user"
    assert client.post(url, json={"ratings": []}).status_code == 422
    too_high = {"ratings": [{"item_id": 1, "rating": 6}]}
    assert client.post(url, json=too_high).status_code == 422
    unknown = client.post(url, json={"ratings": [{"item_id": 4242, "rating": 4}]})
    assert unknown.status_code == 422
    assert "4242" in unknown.json()["detail"]
    with_profile = {"ratings": [{"item_id": 1, "rating": 4}], "gender": "F"}
    assert client.post(url, json=with_profile).status_code == 422


def test_missing_snapshots_answer_503(tmp_path, artifact_dir):
    nothing = TestClient(create_app(tmp_path / "missing", tmp_path / "missing_cold"))
    health = nothing.get("/health")
    assert health.status_code == 503
    assert "build-artifacts" in health.json()["detail"]
    assert nothing.get("/v1/users/1/recommendations").status_code == 503
    assert nothing.get("/v1/movies/search", params={"q": "a"}).status_code == 503

    no_cold = TestClient(create_app(artifact_dir, tmp_path / "missing_cold"))
    assert no_cold.get("/health").json() == {
        "status": "ok",
        "production": True,
        "new_user": False,
    }
    response = no_cold.post(
        "/v1/recommendations/new-user", json={"ratings": [{"item_id": 1, "rating": 4}]}
    )
    assert response.status_code == 503
    assert "cold-start" in response.json()["detail"]


def _metric(text: str, name: str, **labels: str) -> float:
    """Value of one Prometheus sample, matched on name and exact label set."""
    want = ",".join(f'{key}="{value}"' for key, value in labels.items())
    target = f"{name}{{{want}}}" if labels else name
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        sample, _, value = line.rpartition(" ")
        if sample == target:
            return float(value)
    raise AssertionError(f"{target} not in /metrics")


def test_metrics_count_requests_by_route_template(artifact_dir, cold_artifact_dir):
    fresh = TestClient(create_app(artifact_dir, cold_artifact_dir))
    fresh.get("/v1/users/1/recommendations", params={"n": 3})
    fresh.get("/v1/users/2/recommendations", params={"n": 3})
    fresh.get("/v1/users/999/recommendations")
    fresh.get("/no/such/path")
    response = fresh.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    text = response.text
    route = "/v1/users/{user_id}/recommendations"
    assert _metric(text, "movielens_requests_total", route=route, method="GET", status="200") == 2
    assert _metric(text, "movielens_requests_total", route=route, method="GET", status="404") == 1
    assert (
        _metric(text, "movielens_requests_total", route="unmatched", method="GET", status="404")
        == 1
    )
    assert _metric(text, "movielens_request_seconds_count", route=route) == 3
    assert _metric(text, "movielens_request_seconds_bucket", route=route, le="+Inf") == 3
    assert _metric(text, "movielens_request_seconds_sum", route=route) > 0
    assert "/v1/users/1/" not in text and "/v1/users/999" not in text
    assert _metric(text, "movielens_snapshot_loaded", snapshot="production") == 1
    assert _metric(text, "movielens_snapshot_loaded", snapshot="new_user") == 1
    assert _metric(text, "movielens_snapshot_age_seconds") > 0


def test_metrics_track_new_user_inputs_against_the_training_reference(
    artifact_dir, cold_artifact_dir
):
    fresh = TestClient(create_app(artifact_dir, cold_artifact_dir))
    before = fresh.get("/metrics").text
    assert _metric(before, "movielens_newuser_window_requests") == 0
    reference = _metric(before, "movielens_newuser_reference_popularity_percentile")
    assert 0.0 < reference <= 1.0
    assert 0.0 <= _metric(before, "movielens_newuser_reference_tail_share") <= 1.0
    payload = {"ratings": [{"item_id": 1, "rating": 5}, {"item_id": 2, "rating": 4}]}
    assert fresh.post("/v1/recommendations/new-user", json=payload).status_code == 200
    after = fresh.get("/metrics").text
    assert _metric(after, "movielens_newuser_window_requests") == 1
    assert 0.0 <= _metric(after, "movielens_newuser_popularity_percentile_mean") <= 1.0
    assert 0.0 <= _metric(after, "movielens_newuser_tail_share") <= 1.0


def test_request_log_is_one_json_line_without_ids(artifact_dir, cold_artifact_dir, caplog):
    import json
    import logging

    fresh = TestClient(create_app(artifact_dir, cold_artifact_dir))
    with caplog.at_level(logging.INFO, logger="movielens_recommender.api"):
        fresh.get("/v1/users/1/recommendations", params={"n": 4})
        fresh.post(
            "/v1/recommendations/new-user",
            json={"ratings": [{"item_id": 3, "rating": 5}], "n": 2},
        )
    lines = [r.getMessage() for r in caplog.records if r.name == "movielens_recommender.api"]
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first == {
        "route": "/v1/users/{user_id}/recommendations",
        "method": "GET",
        "status": 200,
        "latency_ms": first["latency_ms"],
        "n": 4,
    }
    assert first["latency_ms"] > 0
    assert second["route"] == "/v1/recommendations/new-user"
    assert second["n"] == 2
    assert "item" not in " ".join(lines) and '"rating' not in " ".join(lines)


def test_metrics_work_without_snapshots(tmp_path):
    empty = TestClient(create_app(tmp_path / "missing", tmp_path / "missing_cold"))
    empty.get("/health")
    text = empty.get("/metrics").text
    assert _metric(text, "movielens_snapshot_loaded", snapshot="production") == 0
    health = {"route": "/health", "method": "GET", "status": "503"}
    assert _metric(text, "movielens_requests_total", **health) == 1
