"""Controls on top of a list the ranker already scored.

Nothing here trains a model, writes a metric, or reads a MovieLens file.
Neighbours, the familiarity slider, the why-not answer, and the share
link are pure functions of ids, scores, and vectors the caller already has.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

NEIGHBOUR_COUNT = 6
_TOKEN = re.compile(r"^(\d+)-([1-5])$")
_POPULAR_LINE = "A widely loved film that fits your mix"
_UNEVALUATED = (
    "This re-ranking has not been evaluated offline. "
    "Only the default position is the measured model."
)

TASTE_SETS: tuple[dict[str, Any], ...] = (
    {"label": "Animation", "item_ids": (1, 3114, 2355, 588)},
    {"label": "Crime", "item_ids": (296, 608, 50, 858)},
    {"label": "Science fiction", "item_ids": (260, 1196, 1210, 2571)},
)


def neighbour_caption(source: str) -> str:
    """One line under a neighbour grid. It names the vectors that were used."""
    if source == "two-tower item embeddings":
        return "Nearest films in the two-tower item embeddings."
    if source == "item–item similarity":
        return (
            "Nearest films by item–item similarity. "
            "Item embeddings were not in the loaded weights."
        )
    return "No similar films were available."


def nearest_ids(
    vectors: np.ndarray,
    item_index: Mapping[int, int],
    catalog_ids: Sequence[int],
    item_id: int,
    k: int,
) -> list[int]:
    """``k`` nearest catalog ids by cosine similarity. The film itself is left out."""
    row = item_index.get(int(item_id))
    if row is None or k <= 0:
        return []
    mat = np.asarray(vectors, dtype=np.float64)
    query = mat[int(row)]
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        return []
    norms = np.linalg.norm(mat, axis=1)
    scores = mat @ query
    denom = norms * query_norm
    sims = np.full(len(scores), -np.inf, dtype=np.float64)
    ok = denom > 0.0
    sims[ok] = scores[ok] / denom[ok]
    sims[int(row)] = -np.inf
    order = np.argsort(-sims, kind="mergesort")
    catalog = [int(item) for item in catalog_ids]
    found: list[int] = []
    for index in order:
        if not np.isfinite(sims[int(index)]):
            continue
        found.append(catalog[int(index)])
        if len(found) >= k:
            break
    return found


def nearest_from_similarity(
    similarity: Any,
    item_index: Mapping[int, int],
    item_ids: Sequence[int],
    item_id: int,
    k: int,
) -> list[int]:
    """``k`` nearest ids from one similarity row. A sparse row stays one row."""
    row = item_index.get(int(item_id))
    if row is None or k <= 0:
        return []
    getrow = getattr(similarity, "getrow", None)
    if getrow is not None:
        sparse_row = getrow(int(row))
        indices = np.asarray(sparse_row.indices)
        data = np.asarray(sparse_row.data, dtype=np.float64)
    else:
        dense = np.asarray(similarity[int(row)]).ravel()
        indices = np.arange(dense.shape[0])
        data = dense.astype(np.float64, copy=False)
    if len(data) == 0:
        return []
    order = np.argsort(-data, kind="mergesort")
    catalog = [int(item) for item in item_ids]
    found: list[int] = []
    for pos in order:
        column = int(indices[int(pos)])
        if column == int(row) or column < 0 or column >= len(catalog):
            continue
        value = float(data[int(pos)])
        if not np.isfinite(value):
            continue
        found.append(catalog[column])
        if len(found) >= k:
            break
    return found


def embedding_rows(
    vectors: np.ndarray | None,
    item_index: Mapping[int, int] | None,
    item_ids: Sequence[int],
) -> np.ndarray | None:
    """Rows of ``vectors`` aligned with ``item_ids``, or None when any id is missing."""
    if vectors is None or item_index is None or not item_ids:
        return None
    columns: list[int] = []
    for item_id in item_ids:
        column = item_index.get(int(item_id))
        if column is None:
            return None
        columns.append(int(column))
    return np.asarray(vectors, dtype=np.float64)[columns]


def rerank_candidates(
    item_ids: Sequence[int],
    scores: Sequence[float],
    *,
    adventurous: float,
    embeddings: np.ndarray | None,
    popularity: Sequence[float] | None,
    n: int,
) -> list[int]:
    """Re-rank a candidate list. Position 0 returns the measured order unchanged.

    ``popularity`` is a percentile from 0 to 100, higher meaning more popular.
    Above 0, relevance is blended with a preference for lower popularity, and
    maximal marginal relevance penalises cosine similarity to films already
    chosen. Ties break toward the lower item id.
    """
    ids = [int(item) for item in item_ids]
    if n <= 0 or not ids:
        return []
    if float(adventurous) <= 0.0:
        return ids[:n]
    if len(scores) != len(ids):
        raise ValueError("scores must match item ids")
    weight = min(float(adventurous), 1.0)
    raw = np.asarray(scores, dtype=np.float64)
    low = float(np.min(raw))
    high = float(np.max(raw))
    if high > low:
        relevance = (raw - low) / (high - low)
    else:
        relevance = np.ones(len(ids), dtype=np.float64)
    if popularity is None:
        novelty = np.zeros(len(ids), dtype=np.float64)
    else:
        if len(popularity) != len(ids):
            raise ValueError("popularity must match item ids")
        pop = np.clip(np.asarray(popularity, dtype=np.float64), 0.0, 100.0)
        novelty = 1.0 - pop / 100.0
    combined = (1.0 - weight) * relevance + weight * novelty
    similarity = _unit_cosine(embeddings)
    chosen: list[int] = []
    remaining = set(range(len(ids)))
    while remaining and len(chosen) < n:
        best_index: int | None = None
        best_key: tuple[float, int, int] | None = None
        for index in remaining:
            penalty = 0.0
            if similarity is not None and chosen:
                penalty = float(max(similarity[index, earlier] for earlier in chosen))
            value = float(combined[index]) - weight * penalty
            key = (value, -ids[index], -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            break
        chosen.append(best_index)
        remaining.remove(best_index)
    return [ids[index] for index in chosen]


def unevaluated_note(adventurous: float) -> str | None:
    """Shown only when the slider has left the measured list."""
    if float(adventurous) <= 0.0:
        return None
    return _UNEVALUATED


def popularity_percentiles(
    item_ids: Sequence[int],
    popularity: Sequence[float],
) -> dict[int, float]:
    """Percent of the catalog with a strictly lower popularity. Ties share it."""
    ids = [int(item) for item in item_ids]
    values = np.asarray(list(popularity), dtype=np.float64)
    if len(ids) != len(values):
        raise ValueError("popularity must match item ids")
    count = len(ids)
    if count == 0:
        return {}
    order = np.argsort(values, kind="mergesort")
    less = np.zeros(count, dtype=np.int64)
    for rank, index in enumerate(order.tolist()):
        if rank == 0 or values[index] != values[int(order[rank - 1])]:
            start = rank
        less[index] = start
    denom = max(count - 1, 1)
    return {ids[index]: 100.0 * float(less[index]) / float(denom) for index in range(count)}


def list_facts(
    genres: Sequence[str],
    percentiles: Sequence[float],
) -> dict[str, Any]:
    """Distinct genres and mean popularity percentile for the films on screen."""
    seen: set[str] = set()
    for text in genres:
        for part in str(text).split("|"):
            name = part.strip()
            if not name or name == "(no genres listed)":
                continue
            seen.add(name)
    mean = None
    if percentiles:
        mean = float(sum(float(value) for value in percentiles)) / len(percentiles)
    return {"genre_count": len(seen), "mean_popularity_percentile": mean}


def readout_sentence(facts: Mapping[str, Any]) -> str:
    """Describes the list on screen. It is not an evaluation score."""
    count = int(facts["genre_count"])
    genre_word = "genre" if count == 1 else "genres"
    parts = [f"This list: {count} distinct {genre_word}"]
    mean = facts.get("mean_popularity_percentile")
    if mean is not None:
        parts.append(f"average popularity percentile {mean:.0f}")
    sentence = ", ".join(parts) + "."
    return sentence + " These describe the films on screen. They are not evaluation scores."


def why_not(
    *,
    title: str,
    item_id: int,
    rated_ids: set[int],
    ranked_ids: Sequence[int],
    top_n: int = 10,
) -> str:
    """Where one film sat in the candidate list this request actually scored."""
    name = " ".join(str(title).split()) or "This film"
    if int(item_id) in rated_ids:
        return f"{name} is already in your ratings, so the list leaves it out."
    try:
        rank = list(ranked_ids).index(int(item_id)) + 1
    except ValueError:
        return f"{name} was not among the candidates retrieved for this list."
    total = len(list(ranked_ids))
    if rank <= int(top_n):
        return (
            f"{name} is rank {rank} of {total} candidates. "
            f"The measured model places it in the top {int(top_n)}."
        )
    return (
        f"{name} is rank {rank} of {total} candidates, "
        f"below the top {int(top_n)}."
    )


def encode_ratings(pairs: Sequence[tuple[int, int]]) -> str:
    """``id-stars`` joined by dots. Stars are 1 to 5. No other fields."""
    parts: list[str] = []
    for item_id, stars in pairs:
        item = int(item_id)
        score = int(stars)
        if item < 1 or score < 1 or score > 5:
            raise ValueError("ratings are positive movie ids and stars from 1 to 5")
        parts.append(f"{item}-{score}")
    return ".".join(parts)


def decode_ratings(text: str) -> list[tuple[int, int]]:
    """Parse a share link. Bad tokens are skipped. A repeated id keeps the last stars."""
    latest: dict[int, int] = {}
    order: list[int] = []
    for token in str(text).split("."):
        if not token:
            continue
        match = _TOKEN.fullmatch(token)
        if match is None:
            continue
        item = int(match.group(1))
        score = int(match.group(2))
        if item < 1:
            continue
        if item not in latest:
            order.append(item)
        latest[item] = score
    return [(item, latest[item]) for item in order]


def cites_unrated(line: str, rated_titles: set[str]) -> bool:
    """True when a because-line names a film that is no longer rated."""
    prefix = "Because you liked "
    if not line.startswith(prefix):
        return False
    named = line[len(prefix) :]
    parts = named.rsplit(" and ", 1) if " and " in named else [named]
    return any(part not in rated_titles for part in parts)


def funnel_steps(pipeline: Mapping[str, Any] | None) -> list[dict[str, str]]:
    """Four stages. Counts come from the request; the names follow its source."""
    if not pipeline:
        return [
            {"name": "Catalog", "detail": "Every film the snapshot can recommend"},
            {"name": "Two-tower candidates", "detail": "A short list retrieved for this person"},
            {"name": "LambdaRank", "detail": "Scores those candidates"},
            {"name": "Top 10", "detail": "The list on screen"},
        ]
    source = str(pipeline.get("candidate_source") or "")
    source_name = {
        "two_tower": "Two-tower candidates",
        "cold-start union": "Candidates",
        "popularity": "Popularity list",
        "item–item": "Item–item candidates",
        "history_two_tower": "Two-tower candidates",
    }.get(source, "Candidates")
    scored = int(pipeline.get("n_scored") or 0)
    if pipeline.get("scored_by_ranker"):
        scored_detail = f"{scored} films scored"
    else:
        scored_detail = "The ranker did not score this request"
    return [
        {"name": "Catalog", "detail": f"{int(pipeline.get('n_catalog') or 0)} films"},
        {
            "name": source_name,
            "detail": f"{int(pipeline.get('n_candidates') or 0)} candidates",
        },
        {"name": "LambdaRank", "detail": scored_detail},
        {"name": "Top 10", "detail": "The measured list, before any slider"},
    ]


def cold_start_nudge(payload: Mapping[str, Any]) -> str:
    """Cite the cold-start file only when its stored flags say 3 beats 1."""
    blocks: dict[int, Mapping[str, Any]] = {}
    for block in payload.get("by_n") or []:
        if isinstance(block, dict) and "n" in block:
            blocks[int(block["n"])] = block
    base = "Rate at least 3 films."
    one = blocks.get(1)
    three = blocks.get(3)
    if one is None or three is None:
        return base
    one_pair = one.get("paired_vs_most_popular") or {}
    three_pair = three.get("paired_vs_most_popular") or {}
    if (
        one.get("beats_most_popular") is False
        and isinstance(one_pair, dict)
        and one_pair.get("excludes_zero") is False
        and three.get("beats_most_popular") is True
        and isinstance(three_pair, dict)
        and three_pair.get("excludes_zero") is True
    ):
        return (
            "Rate at least 3 films. In results/cold-start/ml-1m.json the served list "
            "does not beat most-popular at 1 rating, and it does at 3 ratings."
        )
    return base


def _unit_cosine(embeddings: np.ndarray | None) -> np.ndarray | None:
    if embeddings is None:
        return None
    mat = np.asarray(embeddings, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0:
        return None
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = mat / norms
    return unit @ unit.T
