"""Score every user in a serving snapshot and write one file (ADR-0014)."""

from __future__ import annotations

import csv
import gzip
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from movielens_recommender.serving.bundle import (
    known_user_ids,
    load_bundle,
    movie_row,
    rank_for_user,
)
from movielens_recommender.serving.reasons import display_title

RECOMMENDATIONS_FILENAME = "recommendations.csv.gz"
BATCH_MANIFEST_FILENAME = "batch_manifest.json"
COLUMNS = ("user_id", "rank", "item_id", "score", "title", "year")


def batch_recommend(
    artifacts: Path | str,
    out_dir: Path | str,
    *,
    n: int = 10,
    user_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Write the production top ``n`` for ``user_ids`` (default: every known user).

    Unknown ids raise ``KeyError`` before anything is written. The output holds
    rating-derived data, so ``out_dir`` belongs under the gitignored ``artifacts/``.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    bundle = load_bundle(artifacts)
    known = known_user_ids(bundle)
    if user_ids is None:
        users = known
    else:
        users = [int(uid) for uid in user_ids]
        missing = sorted(set(users) - set(known))
        if missing:
            raise KeyError(f"users not in the serving snapshot: {missing[:10]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n_rows = 0
    with gzip.open(out / RECOMMENDATIONS_FILENAME, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for uid in users:
            for rank, (item_id, score) in enumerate(rank_for_user(bundle, uid, n=n), start=1):
                meta = movie_row(bundle, item_id)
                year = "" if meta["year"] is None else int(meta["year"])
                writer.writerow(
                    [uid, rank, item_id, f"{score:.6f}", display_title(meta["title"]), year]
                )
                n_rows += 1
    wall = time.perf_counter() - started
    manifest = {
        "n": int(n),
        "n_users": len(users),
        "n_rows": n_rows,
        "wall_sec": round(wall, 3),
        "users_per_sec": round(len(users) / wall, 3) if wall > 0 else None,
        "dataset": bundle.manifest.get("dataset"),
        "dataset_sha256": bundle.manifest.get("dataset_sha256"),
        "git_sha": bundle.manifest.get("git_sha"),
        "snapshot_created_at": bundle.manifest.get("created_at"),
        "candidate_set": bundle.candidate_set,
        "file": RECOMMENDATIONS_FILENAME,
        "columns": list(COLUMNS),
    }
    (out / BATCH_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
