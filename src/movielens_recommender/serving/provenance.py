"""Snapshot identities bind metadata to the files actually loaded by serving."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(manifest: dict[str, Any]) -> str:
    metadata_fields = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    encoded = json.dumps(metadata_fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Record the saving environment and hash sibling artifacts, excluding sub-bundles."""
    payload = dict(manifest)
    payload["created_at"] = payload.get("created_at") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    payload.setdefault("config", None)
    payload["config_sha256"] = (
        hashlib.sha256(json.dumps(payload["config"], sort_keys=True).encode()).hexdigest()
        if payload["config"] is not None else None
    )
    payload.setdefault("git_sha", "unknown")
    payload.setdefault("benchmark_record", None)
    versions = {}
    for name in ("movielens-recommender", "numpy", "scipy", "pandas", "torch", "lightgbm"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    payload["library_versions"] = versions
    payload["host"] = {
        "os": platform.system(), "release": platform.release(), "machine": platform.machine(),
        "python": platform.python_version(), "cpu_count": os.cpu_count(),
    }
    payload["provenance_version"] = 1
    payload["artifact_sha256"] = {
        path.name: file_sha256(path) for path in sorted(directory.iterdir())
        if path.is_file() and path.name != "manifest.json"
    }
    payload["snapshot_id"] = _identity(payload)
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def verify_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    """Verify new manifests before deserializing artifacts. Legacy files stay readable."""
    if not any(key in manifest for key in ("provenance_version", "snapshot_id", "artifact_sha256")):
        return
    if (manifest.get("provenance_version") != 1
            or manifest.get("snapshot_id") != _identity(manifest)):
        raise ValueError(f"Snapshot metadata checksum mismatch at {directory}")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f"Snapshot artifact checksums missing at {directory}")
    files = {p.name for p in directory.iterdir() if p.is_file() and p.name != "manifest.json"}
    if files != set(hashes):
        raise ValueError(f"Snapshot file set does not match checksums at {directory}")
    for name, expected in hashes.items():
        if Path(name).name != name or file_sha256(directory / name) != expected:
            raise ValueError(f"Snapshot artifact checksum mismatch: {name}")


def snapshot_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    """Public provenance for a bundle that has passed its loader's verification."""
    keys = (
        "snapshot_id", "dataset", "dataset_sha256", "git_sha", "created_at", "schema_version",
        "candidate_set", "candidate_source", "candidate_k", "best_iteration", "demographics",
        "ranker_mode", "library_versions", "host", "benchmark_record", "config_sha256",
    )
    summary = {key: manifest.get(key) for key in keys}
    summary["artifact_integrity"] = (
        "verified" if manifest.get("snapshot_id") else "legacy_unverified"
    )
    summary["metadata_missing"] = [
        key for key in ("config", "created_at", "git_sha", "library_versions", "host")
        if not manifest.get(key) or manifest.get(key) == "unknown"
    ]
    return summary


def git_sha(cwd: Path | str | None = None) -> str:
    """HEAD sha, with a ``-dirty`` suffix when the worktree is not clean."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.environ.get("MOVIELENS_BUILD_SHA", "unknown")
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return sha
    if dirty.strip():
        return f"{sha}-dirty"
    return sha
