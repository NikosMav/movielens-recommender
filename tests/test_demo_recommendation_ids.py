"""Top-10 item ids pinned from main at 268c75a, before the server RSS change.

The demo path (sparse item–item similarity, seen ids left as arrays) must
return these ids for the same snapshots. The ids were read from that main
checkout. Titles are not stored here. Snapshots are not in git. Point
``MOVIELENS_ARTIFACTS`` and ``MOVIELENS_COLD_ARTIFACTS`` at them, or leave
the local rebuild at ``space/.runtime``. CI has neither and does not
download MovieLens, so the test skips there.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"

# Existing ml-1m users, production item ids, top 10.
EXISTING_TOP10: dict[int, list[int]] = {
    1: [1196, 34, 1198, 2078, 318, 588, 2087, 2355, 364, 595],
    42: [2054, 849, 3697, 2532, 1676, 1587, 1690, 2353, 1391, 2363],
    100: [457, 589, 1374, 1220, 1610, 1376, 1240, 1242, 1097, 1036],
    500: [2096, 3114, 1265, 34, 1136, 1080, 235, 2918, 1259, 1],
    1000: [1580, 1197, 1270, 588, 2028, 2987, 364, 2628, 2762, 733],
    6040: [778, 1265, 1270, 1186, 1179, 2064, 1258, 800, 32, 3006],
}

# First five Comedy-shelf films in popularity order, rated 5, 4, 5, 3, 4.
NEW_USER_RATINGS: list[tuple[int, float]] = [
    (2858, 5.0),
    (1270, 4.0),
    (1580, 5.0),
    (1197, 3.0),
    (2396, 4.0),
]
NEW_USER_TOP10: list[int] = [260, 1196, 1210, 2571, 2997, 480, 2762, 1265, 589, 110]


def _usable(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return False
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return isinstance(payload, dict) and not payload.get("weights_only")


def _snapshot_dirs() -> tuple[Path, Path] | None:
    prod_env = os.environ.get("MOVIELENS_ARTIFACTS", "").strip()
    cold_env = os.environ.get("MOVIELENS_COLD_ARTIFACTS", "").strip()
    if prod_env:
        prod = Path(prod_env)
        cold = Path(cold_env) if cold_env else prod / "cold_start"
    else:
        prod = SPACE / ".runtime" / "production"
        cold = SPACE / ".runtime" / "cold_start"
    if _usable(prod) and _usable(cold):
        return prod, cold
    return None


def test_demo_top10_ids_match_main():
    found = _snapshot_dirs()
    if found is None:
        pytest.skip("serving snapshots are not on disk; CI does not download MovieLens")
    prod, cold = found
    pytest.importorskip("torch")
    pytest.importorskip("lightgbm")
    import sys

    import movielens_recommender.serving.bundle as bundle_mod

    sys.path.insert(0, str(SPACE))
    from movielens_recommender.serving.bundle import load_bundle, recommend_for_user
    from movielens_recommender.serving.cold import load_cold_start_bundle, recommend_new_user
    from runtime import use_compact_seen_ids, use_sparse_item_item_loader

    original_item_item = bundle_mod._load_item_item
    original_tower = bundle_mod._load_two_tower
    try:
        use_sparse_item_item_loader()
        use_compact_seen_ids()
        bundle = load_bundle(prod)
        for user_id, expected in EXISTING_TOP10.items():
            result = recommend_for_user(bundle, user_id, n=10)
            got = [int(card["item_id"]) for card in result["production"]]
            assert got == expected
        cold_bundle = load_cold_start_bundle(cold)
        result = recommend_new_user(cold_bundle, NEW_USER_RATINGS, n=10)
        got = [int(card["item_id"]) for card in result["recommendations"]]
        assert got == NEW_USER_TOP10
    finally:
        bundle_mod._load_item_item = original_item_item
        bundle_mod._load_two_tower = original_tower
        bundle_mod._space_sparse_item_item = False
        bundle_mod._space_compact_seen = False
        if sys.path and sys.path[0] == str(SPACE):
            sys.path.pop(0)
