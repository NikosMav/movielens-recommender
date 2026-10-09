"""Replay periods, data-only drift signals, and the refresh threshold rule (ADR-0016)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Period:
    """Half-open window ``[start, end)`` in Unix seconds, UTC."""

    name: str
    start: int
    end: int


def _utc(text: str) -> int:
    return int(pd.Timestamp(text, tz="UTC").timestamp())


def ml1m_periods() -> list[Period]:
    """Monthly August–December 2000, then quarterly through 2003 Q1."""
    periods = [
        Period(f"2000-{month:02d}", _utc(f"2000-{month:02d}-01"), _utc(f"2000-{month + 1:02d}-01"))
        for month in range(8, 12)
    ]
    periods.append(Period("2000-12", _utc("2000-12-01"), _utc("2001-01-01")))
    for year in (2001, 2002):
        for quarter, (first, after) in enumerate(
            (("01", "04"), ("04", "07"), ("07", "10"), ("10", None)), start=1
        ):
            end = f"{year}-{after}-01" if after else f"{year + 1}-01-01"
            periods.append(Period(f"{year}-Q{quarter}", _utc(f"{year}-{first}-01"), _utc(end)))
    periods.append(Period("2003-Q1", _utc("2003-01-01"), _utc("2003-04-01")))
    return periods


def window_before(periods: Sequence[Period], index: int) -> Period:
    """The "last period" for ``periods[index]``: the one before it, or an equal-length window."""
    if index > 0:
        return periods[index - 1]
    first = periods[0]
    return Period("before", first.start - (first.end - first.start), first.start)


def js_divergence(left: Mapping[int, float], right: Mapping[int, float]) -> float:
    """Jensen–Shannon divergence, base 2, between two count maps. 0 to 1."""
    keys = sorted(set(left) | set(right))
    if not keys:
        return 0.0
    p = np.array([float(left.get(k, 0.0)) for k in keys])
    q = np.array([float(right.get(k, 0.0)) for k in keys])
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / m[mask])))

    return float(min(1.0, max(0.0, 0.5 * _kl(p) + 0.5 * _kl(q))))


def item_divergence(
    last_items: Iterable[int],
    train_items: Iterable[int],
    *,
    sample_size: int = 2000,
    seed: int = 42,
) -> float:
    """JS divergence between equal-size samples (without replacement) of two item streams.

    Equal sizes keep a small quarter comparable with a large month. One
    generator draws both samples, so passing the same stream twice gives two
    independent samples, which is the sampling-noise floor.
    """
    a = np.asarray(list(last_items), dtype=np.int64)
    b = np.asarray(list(train_items), dtype=np.int64)
    if a.size == 0 or b.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    size = min(sample_size, a.size, b.size)
    sa = rng.choice(a, size=size, replace=False)
    sb = rng.choice(b, size=size, replace=False)
    return js_divergence(Counter(sa.tolist()), Counter(sb.tolist()))


def new_item_share(
    last_items: Sequence[int], train_counts: Mapping[int, int], *, min_count: int = 5
) -> float:
    """Share of ratings on items with fewer than ``min_count`` training ratings."""
    if len(last_items) == 0:
        return 0.0
    rare = sum(1 for item in last_items if train_counts.get(int(item), 0) < min_count)
    return rare / len(last_items)


def new_user_share(frame: pd.DataFrame, first_seen: Mapping[int, int], period: Period) -> float:
    """Share of ``frame``'s ratings from users whose first rating falls inside ``period``."""
    if frame.empty:
        return 0.0
    firsts = frame["user_id"].map(lambda uid: first_seen[int(uid)])
    new = (firsts >= period.start) & (firsts < period.end)
    return float(new.mean())


def select_tau(
    tuning: Mapping[float, Mapping[str, float]],
    *,
    periodic_mean: float,
    tolerance: float = 0.005,
) -> dict:
    """ADR-0016 rule: fewest tuning retrains among τ within ``tolerance`` of periodic.

    Ties go to the larger τ. If no τ qualifies, the smallest τ is used.
    """
    taus = sorted(tuning)
    floor = periodic_mean - tolerance - 1e-12
    qualifying = [tau for tau in taus if tuning[tau]["mean_ndcg"] >= floor]
    if qualifying:
        fewest = min(tuning[tau]["retrains"] for tau in qualifying)
        chosen = max(tau for tau in qualifying if tuning[tau]["retrains"] == fewest)
    else:
        chosen = taus[0]
    return {"tau": chosen, "qualifying": qualifying, "tolerance": tolerance}
