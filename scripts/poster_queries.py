#!/usr/bin/env python3
"""Print TMDB search queries for movie titles.

No network, no API key, and no MovieLens file. Titles are read from stdin,
one per line. With no titles, a built-in sample of MovieLens-style strings
is printed instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# A handful of title shapes. This is not the MovieLens file.
SAMPLE = (
    "Matrix, The (1999)",
    "Walk in the Clouds, A (1995)",
    "Awfully Big Adventure, An (1995)",
    "Misérables, Les (1995)",
    "Ciudad, La",
    "Postino, Il (The Postman) (1994)",
    "Superweib, Das (1996)",
    "Bewegte Mann, Der",
    "Mariachi, El (1992)",
    "Shichinin no samurai (Seven Samurai) (1954)",
    "Star Wars: Episode V - The Empire Strikes Back (1980)",
    "Toy Story (1995)",
    "Paris, Texas (1984)",
)


def _candidate_queries():
    path = ROOT / "space" / "posters.py"
    spec = importlib.util.spec_from_file_location("space_poster_queries", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.candidate_queries


def titles_from_stdin() -> list[str]:
    if sys.stdin.isatty():
        return list(SAMPLE)
    lines = [line.strip() for line in sys.stdin.read().splitlines()]
    titles = [line for line in lines if line]
    return titles or list(SAMPLE)


def render(titles: list[str]) -> str:
    queries_for = _candidate_queries()
    blocks = []
    for title in titles:
        lines = [title]
        lines.extend(f"  {query}" for query in queries_for(title))
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + ("\n" if blocks else "")


def main() -> None:
    sys.stdout.write(render(titles_from_stdin()))


if __name__ == "__main__":
    main()
