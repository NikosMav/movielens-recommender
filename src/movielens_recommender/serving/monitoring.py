"""Request metrics and new-user input drift for the API (ADR-0016).

Prometheus text format is written by hand: a few counters, one histogram, and
some gauges do not need a client library. Labels are route templates, never
raw paths, so user ids do not become label values.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
WINDOW = 500
TAIL_FRACTION = 0.8  # items outside the top 20% by training count are "tail"


class ItemPopularity:
    """Training-popularity percentile per item, and the reference values for drift."""

    def __init__(self, item_ids: Sequence[int]) -> None:
        items, counts = np.unique(np.asarray(item_ids, dtype=np.int64), return_counts=True)
        # Percentile of each item's count among catalog items: 1.0 is the most rated.
        order = np.argsort(counts, kind="stable")
        ranks = np.empty(len(items), dtype=np.float64)
        ranks[order] = np.arange(1, len(items) + 1)
        pct = ranks / max(len(items), 1)
        self.percentile: dict[int, float] = dict(zip(items.tolist(), pct.tolist(), strict=True))
        weights = counts / counts.sum() if counts.sum() else counts
        self.reference_percentile = float(np.sum(weights * pct)) if len(items) else 0.0
        self.reference_tail_share = float(np.sum(weights * (pct <= TAIL_FRACTION)))

    def summarize(self, item_ids: Sequence[int]) -> tuple[float, int, int]:
        """(sum of percentiles, number of known items, number of tail items)."""
        total, known, tail = 0.0, 0, 0
        for item in item_ids:
            pct = self.percentile.get(int(item))
            if pct is None:
                continue
            total += pct
            known += 1
            tail += int(pct <= TAIL_FRACTION)
        return total, known, tail


class Monitor:
    def __init__(
        self,
        *,
        snapshot_created_at: str | None,
        loaded: Mapping[str, bool],
        popularity: ItemPopularity | None,
    ) -> None:
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = {}
        self._hist: dict[str, list[int]] = {}
        self._sum: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._in_flight = 0
        self._window: deque[tuple[float, int, int]] = deque(maxlen=WINDOW)
        self._created = _parse_utc(snapshot_created_at)
        self._loaded = dict(loaded)
        self._popularity = popularity

    def start(self) -> float:
        with self._lock:
            self._in_flight += 1
        return time.perf_counter()

    def finish(self, started: float, *, route: str, method: str, status: int) -> float:
        elapsed = time.perf_counter() - started
        with self._lock:
            self._in_flight -= 1
            key = (route, method, str(status))
            self._requests[key] = self._requests.get(key, 0) + 1
            buckets = self._hist.setdefault(route, [0] * len(BUCKETS))
            for i, bound in enumerate(BUCKETS):
                if elapsed <= bound:
                    buckets[i] += 1
            self._sum[route] = self._sum.get(route, 0.0) + elapsed
            self._count[route] = self._count.get(route, 0) + 1
        return elapsed

    def observe_new_user(self, item_ids: Sequence[int]) -> None:
        if self._popularity is None:
            return
        summary = self._popularity.summarize(item_ids)
        with self._lock:
            self._window.append(summary)

    def render(self) -> str:
        with self._lock:
            requests = dict(self._requests)
            hist = {k: list(v) for k, v in self._hist.items()}
            sums, counts = dict(self._sum), dict(self._count)
            in_flight = self._in_flight
            window = list(self._window)
        out: list[str] = []

        def head(name: str, kind: str, text: str) -> None:
            out.append(f"# HELP {name} {text}")
            out.append(f"# TYPE {name} {kind}")

        head("movielens_requests_total", "counter", "Requests by route template and status.")
        for (route, method, status), value in sorted(requests.items()):
            out.append(
                f'movielens_requests_total{{route="{route}",method="{method}",'
                f'status="{status}"}} {value}'
            )
        head("movielens_request_seconds", "histogram", "Request latency by route template.")
        for route in sorted(hist):
            for bound, value in zip(BUCKETS, hist[route], strict=True):
                out.append(
                    f'movielens_request_seconds_bucket{{route="{route}",le="{bound}"}} {value}'
                )
            out.append(
                f'movielens_request_seconds_bucket{{route="{route}",le="+Inf"}} {counts[route]}'
            )
            out.append(f'movielens_request_seconds_sum{{route="{route}"}} {sums[route]:.6f}')
            out.append(f'movielens_request_seconds_count{{route="{route}"}} {counts[route]}')
        head("movielens_requests_in_flight", "gauge", "Requests being handled now.")
        out.append(f"movielens_requests_in_flight {in_flight}")
        head("movielens_snapshot_loaded", "gauge", "1 when the snapshot loaded at startup.")
        for name, ok in sorted(self._loaded.items()):
            out.append(f'movielens_snapshot_loaded{{snapshot="{name}"}} {int(ok)}')
        if self._created is not None:
            head("movielens_snapshot_age_seconds", "gauge", "Seconds since the snapshot was built.")
            out.append(f"movielens_snapshot_age_seconds {time.time() - self._created:.0f}")
        if self._popularity is not None:
            total = sum(s for s, _k, _t in window)
            known = sum(k for _s, k, _t in window)
            tail = sum(t for _s, _k, t in window)
            head(
                "movielens_newuser_window_requests",
                "gauge",
                f"New-user requests in the drift window (last {WINDOW}).",
            )
            out.append(f"movielens_newuser_window_requests {len(window)}")
            head(
                "movielens_newuser_popularity_percentile_mean",
                "gauge",
                "Mean training-popularity percentile of items rated in the window.",
            )
            out.append(
                f"movielens_newuser_popularity_percentile_mean {total / known if known else 0:.6f}"
            )
            head(
                "movielens_newuser_tail_share",
                "gauge",
                "Share of rated items outside the top 20% by training count, in the window.",
            )
            out.append(f"movielens_newuser_tail_share {tail / known if known else 0:.6f}")
            head(
                "movielens_newuser_reference_popularity_percentile",
                "gauge",
                "The same mean over the training ratings.",
            )
            out.append(
                "movielens_newuser_reference_popularity_percentile "
                f"{self._popularity.reference_percentile:.6f}"
            )
            head(
                "movielens_newuser_reference_tail_share",
                "gauge",
                "The same tail share over the training ratings.",
            )
            out.append(
                "movielens_newuser_reference_tail_share "
                f"{self._popularity.reference_tail_share:.6f}"
            )
        return "\n".join(out) + "\n"


def _parse_utc(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    except ValueError:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
