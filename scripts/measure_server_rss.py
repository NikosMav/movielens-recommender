#!/usr/bin/env python3
"""Measure the long-lived Streamlit server's peak resident memory.

This is the procedure in ADR-0023. It starts ``streamlit run space/app.py``
headless, walks the three views with two browser sessions open at once, and
prints the server's ``VmHWM`` and the steady ``VmRSS`` after the browsers
close. A fresh process then rebuilds both snapshots and prints ``ru_maxrss``.

The gate is ``peak_vmhwm_kb`` for the server process. Browser processes are
not included. Playwright and a Chromium binary are required. CI does not run
this script.

Prints four lines:

    peak_vmhwm_kb=<int>
    steady_vmrss_kb=<int>
    fresh_ru_maxrss_kib=<int>
    fresh_vmhwm_kb=<int>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
ARCHIVE_NAME = "space-weights.tar.gz"
WEIGHTS_SHA256 = "f6cbc0d77d435834493b85835215211874307d827ac2910121894bb56be7dbfd"
WEIGHTS_URL = (
    "https://huggingface.co/NikosMav/movielens-recommender-weights/"
    "resolve/main/space-weights.tar.gz"
)
ML1M_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
CHROME = Path("/usr/local/bin/google-chrome")
STAR_CLICKS = (4, 3, 4, 2, 3)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    with urlopen(url, timeout=120) as response, partial.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    partial.replace(dest)


def _ensure_archive() -> None:
    archive = SPACE / ARCHIVE_NAME
    if archive.is_file() and _sha256(archive) == WEIGHTS_SHA256:
        print(f"weights archive ok: {archive}", file=sys.stderr)
        return
    cache = Path(tempfile.mkdtemp(prefix="ml-weights-")) / ARCHIVE_NAME
    print(f"downloading weights to {cache}", file=sys.stderr)
    _download(WEIGHTS_URL, cache)
    if _sha256(cache) != WEIGHTS_SHA256:
        raise SystemExit(f"weight archive checksum mismatch: {cache}")
    if archive.is_symlink() or archive.exists():
        archive.unlink()
    archive.symlink_to(cache)


def _zip_candidates() -> list[Path]:
    found: list[Path] = []
    override = os.environ.get("MOVIELENS_ML1M_ZIP", "").strip()
    if override:
        found.append(Path(override))
    found.append(Path("/tmp/ml-data/ml-1m.zip"))
    return found


def _ensure_ml1m(data_dir: Path) -> None:
    ratings = data_dir / "ml-1m" / "ml-1m" / "ratings.dat"
    if ratings.is_file():
        print(f"ml-1m already at {ratings}", file=sys.stderr)
        return
    archive: Path | None = None
    for candidate in _zip_candidates():
        if candidate.is_file() and _sha256(candidate) == ML1M_SHA256:
            archive = candidate
            break
    if archive is None:
        archive = Path(tempfile.mkdtemp(prefix="ml-1m-")) / "ml-1m.zip"
        print(f"downloading ml-1m to {archive}", file=sys.stderr)
        _download(ML1M_URL, archive)
        if _sha256(archive) != ML1M_SHA256:
            raise SystemExit(f"ml-1m checksum mismatch: {archive}")
    dest = data_dir / "ml-1m"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(dest)
    if not ratings.is_file():
        raise SystemExit(f"ml-1m extract is missing {ratings}")
    print(f"ml-1m extracted under {dest}", file=sys.stderr)


def _proc_kb(pid: int) -> tuple[int, int]:
    hwm = rss = None
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            hwm = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            rss = int(line.split()[1])
    if hwm is None or rss is None:
        raise SystemExit(f"/proc/{pid}/status has no VmHWM/VmRSS")
    return hwm, rss


def _log_rss(pid: int, step: str) -> None:
    hwm, rss = _proc_kb(pid)
    print(f"step {step}: VmRSS={rss} kB VmHWM={hwm} kB", file=sys.stderr, flush=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_healthy(base: str, proc: subprocess.Popen[bytes], log_path: Path) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(encoding="utf-8")[-4000:]
            raise SystemExit(f"server exited early:\n{tail}")
        try:
            with urlopen(base + "/_stcore/health", timeout=2) as response:  # noqa: S310
                if b"ok" in response.read():
                    return
        except OSError:
            time.sleep(0.3)
    tail = log_path.read_text(encoding="utf-8")[-4000:]
    raise SystemExit(f"server did not become healthy:\n{tail}")


def _launch_browser(playwright: object):
    kwargs: dict[str, object] = {
        "headless": True,
        "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
    }
    if CHROME.is_file():
        kwargs["executable_path"] = str(CHROME)
    return playwright.chromium.launch(**kwargs)  # type: ignore[attr-defined]


def _click_mode(page: object, name: str) -> None:
    page.locator("[data-testid='stRadio']").get_by_text(name, exact=True).click(force=True)  # type: ignore[attr-defined]


def _wait_text(page: object, text: str, timeout: int = 180_000) -> None:
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout)  # type: ignore[attr-defined]


def _open_existing(page: object, base: str) -> None:
    page.goto(f"{base}?view=classic", wait_until="domcontentloaded", timeout=120_000)  # type: ignore[attr-defined]
    page.locator("[data-testid='stRadio']").get_by_text("Existing user", exact=True).wait_for(  # type: ignore[attr-defined]
        timeout=300_000
    )
    _wait_text(page, "Production model", timeout=300_000)
    _wait_text(page, "Top 10", timeout=300_000)


def _rate_comedy_shelf(page: object) -> None:
    _click_mode(page, "New user")
    _wait_text(page, "Pick one or more genres")
    page.locator("[data-testid='stMultiSelect']").click()  # type: ignore[attr-defined]
    option = page.get_by_role("option", name="Comedy", exact=True)  # type: ignore[attr-defined]
    option.wait_for(timeout=60_000)
    option.click()
    page.keyboard.press("Escape")  # type: ignore[attr-defined]
    page.wait_for_function(  # type: ignore[attr-defined]
        """() => document.querySelectorAll("[data-testid='stFeedback']").length >= 5""",
        timeout=180_000,
    )
    for index, star in enumerate(STAR_CLICKS):
        widget = page.locator("[data-testid='stFeedback']").nth(index)  # type: ignore[attr-defined]
        widget.locator("button").nth(star).click()
        page.wait_for_function(  # type: ignore[attr-defined]
            """([index, star]) => {
                const widgets = document.querySelectorAll("[data-testid='stFeedback']");
                const widget = widgets[index];
                if (!widget) return false;
                const buttons = [...widget.querySelectorAll("button")];
                let last = -1;
                buttons.forEach((button, i) => {
                    if (button.getAttribute("aria-checked") === "true") last = i;
                });
                return last === star;
            }""",
            arg=[index, star],
            timeout=60_000,
        )
        if index + 1 < 3:
            _wait_text(page, f"Add {3 - (index + 1)} more", timeout=60_000)
        else:
            _wait_text(page, "Your ratings", timeout=60_000)
    page.get_by_role("button", name="Get recommendations").click()  # type: ignore[attr-defined]
    _wait_text(page, "Scored in", timeout=180_000)
    _wait_text(page, "Top 10")


def _walk(base: str, pid: int) -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "playwright is required to drive the walk "
            "(pip install playwright). CI does not run this script."
        ) from exc

    with sync_playwright() as playwright:
        browser = _launch_browser(playwright)
        session_a = browser.new_context(viewport={"width": 1400, "height": 900})
        session_b = browser.new_context(viewport={"width": 1280, "height": 800})
        page_a = session_a.new_page()
        page_b = session_b.new_page()
        try:
            _open_existing(page_a, base)
            _log_rss(pid, "A existing user")
            _open_existing(page_b, base)
            _log_rss(pid, "B existing user")
            _rate_comedy_shelf(page_a)
            _log_rss(pid, "A new-user recommendations")
            _click_mode(page_b, "How good is it?")
            _wait_text(page_b, "ml-1m model against model")
            _click_mode(page_b, "New user")
            _wait_text(page_b, "Pick one or more genres")
            _log_rss(pid, "B quality and new user")
            _click_mode(page_a, "How good is it?")
            _wait_text(page_a, "ml-1m model against model")
            _click_mode(page_a, "Existing user")
            _wait_text(page_a, "Production model")
            _click_mode(page_a, "New user")
            _wait_text(page_a, "Your ratings")
            _click_mode(page_a, "How good is it?")
            _wait_text(page_a, "SASRec")
            _log_rss(pid, "A mode switches")
        except Exception:
            shot = Path(tempfile.gettempdir()) / "measure_server_rss_failure.png"
            try:
                page_a.screenshot(path=str(shot), full_page=True)
                print(f"failure screenshot: {shot}", file=sys.stderr)
            except Exception:
                pass
            raise
        finally:
            browser.close()


def _server_env(runtime: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("TMDB_API_KEY", "MOVIELENS_ARTIFACTS", "MOVIELENS_COLD_ARTIFACTS", "WEIGHTS_URL"):
        env.pop(key, None)
    env["MOVIELENS_SPACE_RUNTIME"] = str(runtime)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    return env


def _start_server(runtime: Path, port: int) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = runtime / "server.log"
    handle = log_path.open("wb")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "space/app.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--browser.gatherUsageStats",
        "false",
        "--server.fileWatcherType",
        "none",
        "--server.runOnSave",
        "false",
    ]
    proc = subprocess.Popen(  # noqa: S603
        command,
        cwd=ROOT,
        env=_server_env(runtime),
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    handle.close()
    return proc, log_path


def _stop_server(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def _run_fresh(runtime: Path) -> tuple[int, int]:
    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(Path(__file__).resolve()), "--fresh", str(runtime)],
        cwd=ROOT,
        env=_server_env(runtime),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(
            "fresh process failed\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    values: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        if line.startswith("fresh_ru_maxrss_kib=") or line.startswith("fresh_vmhwm_kb="):
            key, raw = line.split("=", 1)
            values[key] = int(raw)
    if "fresh_ru_maxrss_kib" not in values or "fresh_vmhwm_kb" not in values:
        raise SystemExit(f"fresh process did not print both numbers:\n{completed.stdout}")
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        if not completed.stderr.endswith("\n"):
            sys.stderr.write("\n")
    return values["fresh_ru_maxrss_kib"], values["fresh_vmhwm_kb"]


def _fresh_main(runtime: Path) -> None:
    """ADR-0022 shape: one new interpreter, both models resident, no browser."""
    import resource

    import streamlit

    sys.path.insert(0, str(SPACE))
    from cards import recommendation_card_html
    from catalog import movie_records, popularity_order, select_grid
    from movielens_recommender.serving.bundle import load_bundle
    from movielens_recommender.serving.cold import load_cold_start_bundle
    from posters import poster_url
    from quality import load_quality
    from runtime import materialize_space, use_sparse_item_item_loader

    use_sparse_item_item_loader()
    production, cold = materialize_space(SPACE / "weights", runtime)
    prod = load_bundle(production)
    cold_bundle = load_cold_start_bundle(cold)
    records = movie_records(cold_bundle.movies)
    popular = (
        None
        if cold_bundle.popular_item_ids is None
        else [int(item) for item in cold_bundle.popular_item_ids]
    )
    order = popularity_order(
        popular,
        [int(item) for item in cold_bundle.context.item_ids],
        [float(value) for value in cold_bundle.context.popularity],
    )
    shelf = select_grid(records, ["Comedy"], order)
    first = shelf[0]
    card = recommendation_card_html(
        title=str(first["title"]),
        year=first.get("year"),
        genres=str(first.get("genres") or ""),
        reasons=[],
        poster_url=None,
        show_reasons=False,
    )
    view = load_quality(str(ROOT / "results"))
    missing = poster_url(str(first["title"]), None, "")
    held = (streamlit.__version__, prod, cold_bundle, card, view, missing)
    print(f"fresh held {len(held)} objects, shelf {len(shelf)}", file=sys.stderr)
    hwm, _rss = _proc_kb(os.getpid())
    ru = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print(f"fresh_ru_maxrss_kib={ru}")
    print(f"fresh_vmhwm_kb={hwm}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.fresh is not None:
        _fresh_main(args.fresh)
        return

    _ensure_archive()
    runtime = Path(tempfile.mkdtemp(prefix="ml-server-rss-"))
    _ensure_ml1m(runtime / "data")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc, log_path = _start_server(runtime, port)
    try:
        _wait_healthy(base, proc, log_path)
        assert proc.pid is not None
        _log_rss(proc.pid, "server healthy")
        _walk(base, proc.pid)
        time.sleep(2)
        if proc.poll() is not None:
            raise SystemExit(f"server exited during the walk:\n{log_path.read_text()[-4000:]}")
        hwm, rss = _proc_kb(proc.pid)
        _log_rss(proc.pid, "browsers closed")
    finally:
        _stop_server(proc)

    fresh_ru, fresh_hwm = _run_fresh(runtime)
    print(f"peak_vmhwm_kb={hwm}")
    print(f"steady_vmrss_kb={rss}")
    print(f"fresh_ru_maxrss_kib={fresh_ru}")
    print(f"fresh_vmhwm_kb={fresh_hwm}")


if __name__ == "__main__":
    main()
