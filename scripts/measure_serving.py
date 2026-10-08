"""Measure S5b serving latency and batch wall time on ml-1m (ADR-0014).

Needs both snapshots on disk (``build-artifacts`` and ``cold-start``) and the
``[api]`` extra. Writes ``results/serving/<dataset>.json``. The samples and the
budgets are the ones fixed in ADR-0014 before any measurement.

    python scripts/measure_serving.py --artifacts artifacts/ml-1m
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from movielens_recommender.serving import bundle as bundle_mod
from movielens_recommender.serving.api import create_app
from movielens_recommender.serving.batch import batch_recommend
from movielens_recommender.serving.build import git_sha
from movielens_recommender.serving.reasons import HistoryItem

SEED = 42
N = 10
KNOWN_USERS = 500
NEW_USER_PROFILES = 200
NEW_USER_POOL = 500
NEW_USER_RATINGS = 5
BUDGET_MS = {
    "known_user_explained": 300.0,
    "known_user_ranked": 100.0,
    "new_user_five_ratings": 300.0,
}
BATCH_BUDGET_SEC = 600.0


def _percentiles(samples_sec: list[float]) -> dict[str, float]:
    ms = np.asarray(samples_sec, dtype=np.float64) * 1000.0
    return {
        "n": int(ms.size),
        "mean_ms": round(float(ms.mean()), 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 3),
        "p95_ms": round(float(np.percentile(ms, 95)), 3),
        "p99_ms": round(float(np.percentile(ms, 99)), 3),
        "max_ms": round(float(ms.max()), 3),
    }


def _timed(client: TestClient, method: str, url: str, **kwargs: Any) -> float:
    """Warm the exact call once, then time it once (ADR-0014)."""
    for attempt in range(2):
        started = time.perf_counter()
        response = client.request(method, url, **kwargs)
        elapsed = time.perf_counter() - started
        if response.status_code != 200:
            raise RuntimeError(f"{method} {url} -> {response.status_code}: {response.text[:200]}")
        if attempt == 1:
            return elapsed
    raise AssertionError("unreachable")


def _with_budget(name: str, stats: dict[str, float]) -> dict[str, Any]:
    budget = BUDGET_MS.get(name)
    out: dict[str, Any] = dict(stats)
    if budget is not None:
        out["budget_p95_ms"] = budget
        out["within_budget"] = bool(stats["p95_ms"] <= budget)
    return out


class _ScanHelpers:
    """The pre-S5b helper bodies, copied verbatim, for the indexes-off timing."""

    def __init__(self) -> None:
        self._saved: dict[str, Any] = {}

    @staticmethod
    def movie_row(bundle, item_id):
        frame = bundle.movies
        hit = frame.loc[frame["item_id"] == int(item_id)]
        if hit.empty:
            return {"item_id": int(item_id), "title": f"Item {int(item_id)}", "year": None,
                    "genres": ""}
        row = hit.iloc[0]
        year = row["year"]
        year_out = None if year != year else int(year)
        return {"item_id": int(item_id), "title": str(row["title"]), "year": year_out,
                "genres": str(row["genres"])}

    @staticmethod
    def user_history_items(bundle, user_id):
        frame = bundle.histories
        rows = frame.loc[frame["user_id"] == int(user_id)]
        if rows.empty:
            return []
        ordered = rows.sort_values(["timestamp", "item_id"], kind="mergesort")
        items = []
        for record in ordered.itertuples(index=False):
            meta = bundle_mod.movie_row(bundle, int(record.item_id))
            items.append(
                HistoryItem(
                    item_id=int(record.item_id),
                    title=str(meta["title"]),
                    genres=str(meta["genres"]),
                    rating=float(record.rating),
                    timestamp=int(record.timestamp),
                )
            )
        return items

    @staticmethod
    def lookups(bundle):
        # Old path: set(known_user_ids(bundle)) on every request.
        known = sorted(int(uid) for uid in bundle.histories["user_id"].unique())
        return bundle_mod._Lookups(
            known_users=frozenset(known), sorted_users=known, history_rows={}, movie_rows={}
        )

    def __enter__(self) -> _ScanHelpers:
        for name, replacement in (
            ("movie_row", self.movie_row),
            ("user_history_items", self.user_history_items),
            ("_lookups", self.lookups),
        ):
            self._saved[name] = getattr(bundle_mod, name)
            setattr(bundle_mod, name, replacement)
        return self

    def __exit__(self, *exc: object) -> None:
        for name, original in self._saved.items():
            setattr(bundle_mod, name, original)


def _memory_mib() -> float | None:
    if sys.platform == "win32":

        class _Status(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Status()
        status.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.ullTotalPhys / 2**20, 1)
        return None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024, 1)
    return None


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "torch", "lightgbm", "fastapi", "starlette", "pydantic"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "missing"
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifacts", default="artifacts/ml-1m")
    parser.add_argument("--cold-artifacts", default=None)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    artifacts = Path(args.artifacts)
    cold = Path(args.cold_artifacts) if args.cold_artifacts else artifacts / "cold_start"
    app = create_app(artifacts, cold)
    state = app.state.serving
    if state.production is None or state.cold is None:
        raise SystemExit(state.production_error or state.cold_error)
    client = TestClient(app)
    bundle = state.production

    rng = np.random.default_rng(SEED)
    users = bundle_mod.known_user_ids(bundle)
    sample = sorted(int(u) for u in rng.choice(users, size=min(KNOWN_USERS, len(users)),
                                               replace=False))

    print(f"Known users, explained ({len(sample)})...")
    explained = [
        _timed(client, "GET", f"/v1/users/{uid}/recommendations", params={"n": N})
        for uid in sample
    ]
    print(f"Known users, ranked only ({len(sample)})...")
    ranked = [
        _timed(client, "GET", f"/v1/users/{uid}/recommendations",
               params={"n": N, "explain": False})
        for uid in sample
    ]
    print(f"Known users, explained, indexes off ({len(sample)})...")
    with _ScanHelpers():
        scanned = [
            _timed(client, "GET", f"/v1/users/{uid}/recommendations", params={"n": N})
            for uid in sample
        ]

    counts = bundle.histories.groupby("item_id", sort=False).size()
    counts = counts.reset_index(name="n").sort_values(
        ["n", "item_id"], ascending=[False, True], kind="mergesort"
    )
    pool = counts["item_id"].head(NEW_USER_POOL).astype(int).to_numpy()
    profiles = []
    for _ in range(NEW_USER_PROFILES):
        items = rng.choice(pool, size=NEW_USER_RATINGS, replace=False)
        ratings = rng.choice([3.0, 4.0, 5.0], size=NEW_USER_RATINGS)
        profiles.append(
            [{"item_id": int(i), "rating": float(r)} for i, r in zip(items, ratings, strict=True)]
        )
    print(f"New users, {NEW_USER_RATINGS} ratings ({len(profiles)})...")
    new_user = [
        _timed(client, "POST", "/v1/recommendations/new-user", json={"ratings": p, "n": N})
        for p in profiles
    ]

    print(f"Batch, every known user ({len(users)})...")
    batch_dir = Path(tempfile.mkdtemp(prefix="movielens-batch-"))
    try:
        batch = batch_recommend(artifacts, batch_dir, n=N)
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)

    explained_stats = _percentiles(explained)
    scanned_stats = _percentiles(scanned)
    payload = {
        "dataset": bundle.manifest.get("dataset"),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(),
        "adr": "docs/adr/0014-batch-api-docker.md",
        "snapshot": {
            "dataset_sha256": bundle.manifest.get("dataset_sha256"),
            "git_sha": bundle.manifest.get("git_sha"),
            "created_at": bundle.manifest.get("created_at"),
            "candidate_set": bundle.candidate_set,
            "candidate_k": bundle.candidate_k,
            "n_known_users": len(users),
        },
        "protocol": {
            "client": "fastapi.testclient.TestClient, one process, no network",
            "warmup": "each request is sent once untimed, then once timed",
            "n": N,
            "seed": SEED,
            "known_user_sample": len(sample),
            "new_user_profiles": len(profiles),
            "new_user_pool": f"{NEW_USER_POOL} most-rated training items",
            "new_user_ratings": f"{NEW_USER_RATINGS} distinct items, ratings from {{3, 4, 5}}",
        },
        "latency": {
            "known_user_explained": _with_budget("known_user_explained", explained_stats),
            "known_user_ranked": _with_budget("known_user_ranked", _percentiles(ranked)),
            "new_user_five_ratings": _with_budget("new_user_five_ratings",
                                                  _percentiles(new_user)),
            "known_user_explained_indexes_off": scanned_stats,
        },
        "index_speedup_p50": round(scanned_stats["p50_ms"] / explained_stats["p50_ms"], 2),
        "batch": {
            "n": batch["n"],
            "n_users": batch["n_users"],
            "n_rows": batch["n_rows"],
            "wall_sec": batch["wall_sec"],
            "users_per_sec": batch["users_per_sec"],
            "budget_sec": BATCH_BUDGET_SEC,
            "within_budget": bool(batch["wall_sec"] <= BATCH_BUDGET_SEC),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "memory_total_mib": _memory_mib(),
        },
        "versions": _versions(),
    }
    out = Path(args.results_dir) / "serving" / f"{payload['dataset']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["latency"], indent=2))
    print(json.dumps(payload["batch"], indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
