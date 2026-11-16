#!/usr/bin/env python3
"""Pack the demo's parameter archive from trees that already exist.

``make space-snapshot`` refits everything. This command does not fit. It takes
a full production snapshot (``build-artifacts --out-dir``), a weights-only
new-user tree (so a published one can be carried over byte for byte), and an
optional larger-catalogue tree (ADR-0034), checks that no rating rows, user
ids, or titles are inside, and writes one ``.tar.gz`` with a fixed member
order and zeroed timestamps. It prints the SHA-256 to pin in
``space/runtime.py``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import io
import shutil
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runtime():
    path = REPO_ROOT / "space" / "runtime.py"
    spec = importlib.util.spec_from_file_location("space_runtime_pack", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage(
    dest: Path,
    *,
    production: Path,
    cold_weights: Path,
    extended: Path | None,
) -> Path:
    """Lay out ``production/``, ``cold_start/``, and optionally ``extended/`` under ``dest``."""
    runtime = _load_runtime()
    runtime._export_one(production, dest / "production", extra={})
    shutil.copytree(cold_weights, dest / "cold_start")
    if extended is not None:
        shutil.copytree(extended, dest / runtime.EXTENDED_DIR)
    for name in ("production", "cold_start"):
        runtime._require_weights_manifest(runtime._read_json(dest / name / "manifest.json"))
    runtime.assert_weights_only(dest)
    return dest


def pack(tree: Path, out: Path) -> str:
    """Write a reproducible ``.tar.gz`` of ``tree`` and return its SHA-256."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path in sorted(tree.rglob("*")):
            info = archive.gettarinfo(str(path), arcname=path.relative_to(tree).as_posix())
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if path.is_dir() else 0o644
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    out.parent.mkdir(parents=True, exist_ok=True)
    # An empty filename and a zero mtime keep the gzip header the same for any ``out``.
    with out.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as packed:
            packed.write(buffer.getvalue())
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--production", required=True, help="Full production snapshot directory")
    parser.add_argument("--cold-weights", required=True, help="Weights-only new-user tree")
    parser.add_argument("--extended", default=None, help="Weights-only larger-catalogue tree")
    parser.add_argument("--out", required=True, help="Archive path to write")
    args = parser.parse_args()
    out = Path(args.out)
    with tempfile.TemporaryDirectory() as tmp:
        tree = stage(
            Path(tmp) / "weights",
            production=Path(args.production),
            cold_weights=Path(args.cold_weights),
            extended=None if args.extended is None else Path(args.extended),
        )
        runtime = _load_runtime()
        for name, size in runtime.weight_sizes(tree):
            print(f"{size:>12} {name}")
        digest = pack(tree, out)
    print(f"{out.stat().st_size:>12} {out}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
