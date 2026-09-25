"""Most-popular baseline: rank items by training interaction count."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MostPopularRecommender:
    """Recommend the globally most-interacted items (ties broken by item_id)."""

    def __init__(self) -> None:
        self._ranked_items: np.ndarray = np.array([], dtype=np.int64)

    def fit(self, train: pd.DataFrame) -> MostPopularRecommender:
        counts = train.groupby("item_id").size().reset_index(name="count")
        # Descending count, ascending item_id for determinism on ties.
        ranked = counts.sort_values(
            ["count", "item_id"], ascending=[False, True], kind="mergesort"
        )
        self._ranked_items = ranked["item_id"].to_numpy(dtype=np.int64)
        return self

    def recommend(self, user_id: int, n: int) -> list[int]:
        """Return top-n globally popular items (caller filters seen items)."""
        del user_id  # global ranking
        if n <= 0:
            return []
        return [int(x) for x in self._ranked_items[:n]]
