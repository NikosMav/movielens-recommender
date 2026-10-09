"""Top-n selection must not depend on the CPU's SIMD sort path.

``np.argpartition`` picks among tied scores differently on AVX2, AVX-512, and
baseline builds. Every top-n in the package breaks ties by lowest index, the
same as a full stable sort, whatever the hardware.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from movielens_recommender.baselines.common import recommend_from_scores, top_n_indices

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _tie_heavy(rng: np.random.Generator, size: int) -> np.ndarray:
    scores = np.zeros(size)
    hot = rng.choice(size, size=min(size, int(rng.integers(0, 15))), replace=False)
    scores[hot] = rng.integers(1, 4, size=hot.size).astype(np.float64)
    return scores


def test_top_n_matches_a_full_stable_sort():
    rng = np.random.default_rng(0)
    for _ in range(300):
        size = int(rng.integers(1, 400))
        scores = _tie_heavy(rng, size)
        n = int(rng.integers(1, size + 3))
        expected = np.argsort(-scores, kind="stable")[:n]
        np.testing.assert_array_equal(top_n_indices(scores, n), expected)


def test_top_n_keeps_the_lowest_tied_indices_at_the_cut():
    scores = np.zeros(3706)
    scores[[3000, 17, 2500]] = [0.9, 0.5, 0.5]
    assert top_n_indices(scores, 10).tolist() == [3000, 17, 2500, 0, 1, 2, 3, 4, 5, 6]
    ids = np.arange(100, 100 + 3706)
    assert recommend_from_scores(scores, ids, 5) == [3100, 117, 2600, 100, 101]


def test_top_n_drops_non_finite_and_handles_edges():
    scores = np.array([-np.inf, 2.0, np.nan, 2.0, 1.0])
    assert top_n_indices(scores, 3).tolist() == [1, 3, 4]
    assert top_n_indices(scores, 10).tolist() == [1, 3, 4]
    assert top_n_indices(scores, 0).tolist() == []
    assert top_n_indices(np.array([]), 3).tolist() == []


def test_host_info_names_the_platform_and_cpu_path():
    import json

    from movielens_recommender.cli import host_info

    info = host_info()
    assert json.loads(json.dumps(info)) == info
    assert info["os"] and info["machine"] and info["python"]
    assert isinstance(info["cpu_count"], int)
    assert isinstance(info["numpy_simd"], list)


_PROBE = """
import hashlib, numpy as np
from movielens_recommender.baselines.common import top_n_indices
rng = np.random.default_rng(1)
parts = []
for _ in range(200):
    scores = np.zeros(3706)
    hot = rng.choice(3706, size=int(rng.integers(0, 15)), replace=False)
    scores[hot] = rng.random(hot.size)
    parts.append(top_n_indices(scores, 10).tobytes())
print(hashlib.sha256(b"".join(parts)).hexdigest())
"""

_NO_SIMD = "AVX2 FMA3 AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"


@pytest.mark.parametrize("disabled", ["", _NO_SIMD])
def test_top_n_is_the_same_with_simd_sorts_disabled(disabled):
    def run(features: str) -> str:
        env = dict(os.environ, PYTHONPATH=_SRC)
        env.pop("NPY_DISABLE_CPU_FEATURES", None)
        if features:
            env["NPY_DISABLE_CPU_FEATURES"] = features
        done = subprocess.run(
            [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    assert run(disabled) == run("")
