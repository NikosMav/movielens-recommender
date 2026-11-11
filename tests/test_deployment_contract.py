"""Exercise the installed deployment pin, isolated from pytest's src/ path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_deployment_package_returns_scored_candidate_trace(cold_artifact_dir):
    target = os.environ.get("MOVIELENS_DEPLOYMENT_PACKAGE")
    if not target:
        pytest.skip("the deployment check installs the pinned wheel in a separate directory")
    package = Path(target).resolve()
    assert (package / "movielens_recommender").is_dir()
    program = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import movielens_recommender
assert Path(movielens_recommender.__file__).resolve().is_relative_to(Path(sys.argv[1]))
from movielens_recommender.serving.cold import load_cold_start_bundle, recommend_new_user
bundle = load_cold_start_bundle(sys.argv[2])
result = recommend_new_user(bundle, [(1, 5.0), (2, 4.0), (3, 5.0)], n=3)
trace = result["pipeline"]
assert trace["scored_by_ranker"]
assert len(trace["ranked_ids"]) == len(trace["ranked_scores"]) > 0
assert trace["ranked_ids"][:3] == [row["item_id"] for row in result["recommendations"]]
assert not set(trace["ranked_ids"]) & {1, 2, 3}
'''
    checked = subprocess.run(
        [sys.executable, "-I", "-c", program, str(package), str(cold_artifact_dir)],
        capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
