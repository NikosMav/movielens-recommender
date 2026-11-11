"""Check discovery controls against a saved snapshot, without training.

This is a behavioral probe, not an offline relevance or user-satisfaction
evaluation. Output describes the displayed lists for four fixed profiles.
Run from the repository root with the UI, deep, and rank extras installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "space"))


def check(snapshot: Path) -> list[dict]:
    from movielens_recommender.serving.cold import load_cold_start_bundle, recommend_new_user
    from runtime import use_sparse_item_item_loader
    from value import (
        TASTE_SETS,
        embedding_rows,
        list_facts,
        popularity_percentiles,
        rerank_candidates,
    )

    use_sparse_item_item_loader()
    bundle = load_cold_start_bundle(snapshot)
    profiles = {
        str(group["label"]): [(int(item), 5.0) for item in group["item_ids"]]
        for group in TASTE_SETS
    }
    profiles["Mixed ratings"] = [(2858, 5.0), (1270, 4.0), (1580, 5.0), (1197, 3.0), (2396, 4.0)]
    from catalog import movie_records

    movies = {int(row["item_id"]): row for row in movie_records(bundle.movies)}
    pop = popularity_percentiles(bundle.context.item_ids, bundle.context.popularity)
    tower = bundle.two_tower
    findings = []
    for name, profile in profiles.items():
        result = recommend_new_user(bundle, profile, n=10)
        cards = result["recommendations"]
        original = [int(card["item_id"]) for card in cards]
        pipeline = result["pipeline"]
        candidates, scores = pipeline["ranked_ids"], pipeline["ranked_scores"]
        embeddings = embedding_rows(tower._item_vectors, tower._features.item_index, candidates)
        positions = []
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            ids = rerank_candidates(
                candidates,
                scores,
                adventurous=weight,
                embeddings=embeddings,
                popularity=[pop[int(item)] for item in candidates],
                n=10,
            )
            assert len(ids) == len(set(ids)) == 10
            assert not set(ids) & {item for item, _ in profile}
            assert set(ids) <= set(candidates)
            if weight == 0:
                assert ids == original, "Default position changed the served model"
            facts = list_facts(
                [movies[item]["genres"] for item in ids],
                [pop[item] for item in ids],
            )
            positions.append(
                {
                    "adventurous": weight,
                    "overlap_with_default": len(set(ids) & set(original)),
                    "genre_count": facts["genre_count"],
                    "mean_popularity_percentile": round(facts["mean_popularity_percentile"], 2),
                }
            )
        findings.append({"profile": name, "method": result["method"], "positions": positions})
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "space/.runtime/cold_start")
    args = parser.parse_args()
    print(json.dumps(check(args.snapshot), indent=2))


if __name__ == "__main__":
    main()
