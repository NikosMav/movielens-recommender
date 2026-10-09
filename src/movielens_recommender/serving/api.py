"""HTTP API over the serving snapshots (ADR-0014). Optional extra: ``[api]``.

Run locally::

    uvicorn --factory movielens_recommender.serving.api:create_app_from_env

``MOVIELENS_ARTIFACTS`` (default ``artifacts/ml-1m``) and
``MOVIELENS_COLD_ARTIFACTS`` (default ``<production>/cold_start``) pick the
snapshots, as in the Streamlit page. Snapshots load once, when the app is made.
A missing snapshot does not stop the app: the routes that need it answer 503
with the command that builds it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from movielens_recommender import __version__
from movielens_recommender.serving.bundle import (
    ServingBundle,
    load_bundle,
    movie_row,
    rank_for_user,
    recommend_for_user,
)
from movielens_recommender.serving.cold import (
    ColdStartBundle,
    load_cold_start_bundle,
    recommend_new_user,
)
from movielens_recommender.serving.monitoring import ItemPopularity, Monitor
from movielens_recommender.serving.reasons import display_title

DEFAULT_ARTIFACTS = Path("artifacts/ml-1m")
MAX_N = 100
MAX_PROFILE = 200
MAX_SEARCH = 50

log = logging.getLogger("movielens_recommender.api")


class Rating(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: int
    rating: float = Field(ge=0.5, le=5.0)


class NewUserRequest(BaseModel):
    """Only ratings. Demographic fields are rejected, not ignored (ADR-0012)."""

    model_config = ConfigDict(extra="forbid")

    ratings: list[Rating] = Field(min_length=1, max_length=MAX_PROFILE)
    n: int = Field(default=10, ge=1, le=MAX_N)


class _State:
    def __init__(self, artifacts: Path, cold_artifacts: Path) -> None:
        self.lock = threading.Lock()
        self.production: ServingBundle | None = None
        self.production_error: str | None = None
        self.cold: ColdStartBundle | None = None
        self.cold_error: str | None = None
        self.catalog: frozenset[int] = frozenset()
        self.titles: list[tuple[str, str, dict[str, Any]]] = []
        try:
            self.production = load_bundle(artifacts)
        except (FileNotFoundError, ValueError) as exc:
            self.production_error = str(exc)
        try:
            self.cold = load_cold_start_bundle(cold_artifacts)
            self.catalog = frozenset(int(item) for item in self.cold.movies["item_id"])
        except (FileNotFoundError, ValueError) as exc:
            self.cold_error = str(exc)
        if self.production is not None:
            for item in self.production.movies["item_id"].tolist():
                meta = movie_row(self.production, int(item))
                shown = display_title(str(meta["title"]))
                haystack = f"{shown.lower()}\n{str(meta['title']).lower()}"
                card = {"item_id": int(item), "title": shown, "year": meta["year"]}
                self.titles.append((shown.lower(), haystack, card))
            self.titles.sort(key=lambda row: (row[0], row[2]["item_id"]))

    def need_production(self) -> ServingBundle:
        if self.production is None:
            raise HTTPException(status_code=503, detail=self.production_error)
        return self.production

    def need_cold(self) -> ColdStartBundle:
        if self.cold is None:
            raise HTTPException(status_code=503, detail=self.cold_error)
        return self.cold


def _plain(value: Any) -> Any:
    """JSON-safe copy: numpy scalars and arrays become Python numbers and lists."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return value


def create_app(
    artifacts: Path | str = DEFAULT_ARTIFACTS,
    cold_artifacts: Path | str | None = None,
) -> FastAPI:
    """Load the snapshots once and return the app."""
    artifacts = Path(artifacts)
    cold = Path(cold_artifacts) if cold_artifacts is not None else artifacts / "cold_start"
    state = _State(artifacts, cold)
    app = FastAPI(
        title="movielens-recommender",
        version=__version__,
        description="Top-n movie recommendations from the ml-1m serving snapshots (ADR-0014).",
    )
    app.state.serving = state
    production = state.production
    monitor = Monitor(
        snapshot_created_at=production.manifest.get("created_at") if production else None,
        loaded={"production": production is not None, "new_user": state.cold is not None},
        popularity=(
            ItemPopularity(production.histories["item_id"].to_numpy()) if production else None
        ),
    )
    app.state.monitor = monitor

    @app.middleware("http")
    async def observe(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)
        started = monitor.start()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", None) or "unmatched"
            elapsed = monitor.finish(started, route=route, method=request.method, status=status)
            record: dict[str, Any] = {
                "route": route,
                "method": request.method,
                "status": status,
                "latency_ms": round(elapsed * 1000.0, 3),
            }
            n = getattr(request.state, "log_n", None) or request.query_params.get("n")
            if n is not None and str(n).isdigit():
                record["n"] = int(n)
            log.info(json.dumps(record))

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(monitor.render(), media_type="text/plain; version=0.0.4")

    @app.get("/health")
    def health() -> JSONResponse:
        if state.production is None:
            return JSONResponse(status_code=503, content={"detail": state.production_error})
        return JSONResponse(
            content={"status": "ok", "production": True, "new_user": state.cold is not None}
        )

    @app.get("/v1/snapshot")
    def snapshot() -> dict[str, Any]:
        bundle = state.need_production()
        manifest = bundle.manifest
        return _plain(
            {
                "dataset": manifest.get("dataset"),
                "dataset_sha256": manifest.get("dataset_sha256"),
                "git_sha": manifest.get("git_sha"),
                "created_at": manifest.get("created_at"),
                "schema_version": manifest.get("schema_version"),
                "candidate_set": bundle.candidate_set,
                "candidate_k": bundle.candidate_k,
                "new_user_snapshot": state.cold is not None,
            }
        )

    @app.get("/v1/users/{user_id}/recommendations")
    def user_recommendations(
        user_id: int,
        n: Annotated[int, Query(ge=1, le=MAX_N)] = 10,
        explain: bool = True,
    ) -> dict[str, Any]:
        bundle = state.need_production()
        try:
            with state.lock:
                if explain:
                    cards = recommend_for_user(bundle, user_id, n=n)["production"]
                else:
                    cards = []
                    for item_id, score in rank_for_user(bundle, user_id, n=n):
                        meta = movie_row(bundle, item_id)
                        cards.append(
                            {
                                "item_id": item_id,
                                "title": display_title(str(meta["title"])),
                                "year": meta["year"],
                                "genres": meta["genres"],
                                "score": score,
                            }
                        )
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"user {user_id} is not in the serving snapshot"
            ) from None
        return _plain(
            {
                "user_id": user_id,
                "candidate_set": bundle.candidate_set,
                "n": n,
                "recommendations": cards,
            }
        )

    @app.post("/v1/recommendations/new-user")
    def new_user(body: NewUserRequest, request: Request) -> dict[str, Any]:
        request.state.log_n = body.n
        bundle = state.need_cold()
        unknown = sorted({r.item_id for r in body.ratings} - state.catalog)
        if unknown:
            raise HTTPException(
                status_code=422, detail=f"item ids not in the catalog: {unknown[:10]}"
            )
        profile = [(r.item_id, r.rating) for r in body.ratings]
        monitor.observe_new_user([item for item, _rating in profile])
        with state.lock:
            result = recommend_new_user(bundle, profile, n=body.n)
        return _plain(result)

    @app.get("/v1/movies/search")
    def search_movies(
        q: Annotated[str, Query(min_length=1, max_length=200)],
        limit: Annotated[int, Query(ge=1, le=MAX_SEARCH)] = 20,
    ) -> dict[str, Any]:
        state.need_production()
        needle = q.strip().lower()
        if not needle:
            raise HTTPException(status_code=422, detail="q must not be blank")
        hits = [card for _key, haystack, card in state.titles if needle in haystack]
        return _plain({"query": q, "movies": hits[:limit]})

    return app


def create_app_from_env() -> FastAPI:
    """Factory for ``uvicorn --factory``; reads the same variables as the Streamlit page."""
    artifacts = Path(os.environ.get("MOVIELENS_ARTIFACTS", str(DEFAULT_ARTIFACTS)))
    cold = os.environ.get("MOVIELENS_COLD_ARTIFACTS")
    return create_app(artifacts, Path(cold) if cold else None)
