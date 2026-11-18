#!/usr/bin/env python3
"""Record a short walkthrough of the public demo, and the README screenshots.

``video``: drives the live page in a headless browser with Playwright's
recorder, then trims the loading time and writes an H.264 ``.mp4`` (for a
LinkedIn post) and a small ``.gif`` (for the README) with ffmpeg. The output
goes to ``artifacts/demo/``, which is gitignored.

``screenshots``: writes the three README images under ``docs/demo/``.

The page must be awake: run ``scripts/keep_awake.py`` first. Posters appear
because the live app has a TMDB key; a local run without one shows
placeholders. ``--channel chrome`` uses the installed Chrome.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = "https://nikosmav-movielens-recommender.streamlit.app/~/+/"
OUT = ROOT / "artifacts" / "demo"
SHOTS = ROOT / "docs" / "demo"
SAMPLE = "Try a sample profile"
# Ids rated 5 in the README's picks screenshot: The Matrix, Star Wars V, IV, VI.
SHARED = "?view=discover&ratings=2571-5.1196-5.260-5.1210-5&mix=0"


def _wait_text(page, text: str, timeout_sec: int = 120) -> None:
    page.get_by_text(text, exact=False).first.wait_for(timeout=timeout_sec * 1000)


def _glide(page, total: int, *, step: int = 120, pause: float = 0.05) -> None:
    """Scroll by ``total`` pixels in small steps, so the video reads as a scroll."""
    page.mouse.move(640, 450)
    moved = 0
    while abs(moved) < abs(total):
        delta = step if total > 0 else -step
        page.mouse.wheel(0, delta)
        moved += delta
        time.sleep(pause)


def _settle(page, seconds: float) -> None:
    page.wait_for_timeout(int(seconds * 1000))


def record(channel: str | None) -> Path:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT / "raw"
    shutil.rmtree(raw_dir, ignore_errors=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel=channel)
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            color_scheme="light",
            record_video_dir=str(raw_dir),
            record_video_size={"width": 1280, "height": 720},
        )
        started = time.monotonic()
        page = context.new_page()
        page.goto(APP + "?view=discover", wait_until="domcontentloaded")
        _wait_text(page, "Find your next film")
        page.locator("img.poster").first.wait_for(timeout=60_000)
        lead_in = time.monotonic() - started
        _settle(page, 2.0)

        _glide(page, 360)
        _settle(page, 1.0)
        page.get_by_role("button", name=SAMPLE).click()
        _settle(page, 2.0)
        _wait_text(page, "Your next watch")
        _settle(page, 1.5)
        page.get_by_text("Your next watch").first.scroll_into_view_if_needed()
        _settle(page, 2.5)
        _glide(page, 900, step=60, pause=0.06)
        _settle(page, 1.5)
        page.get_by_text("Why this pick?", exact=True).first.click()
        _settle(page, 3.0)

        page.get_by_text("How good is it?", exact=True).first.click()
        _wait_text(page, "The model in this demo")
        _settle(page, 1.0)
        page.get_by_text("The model in this demo").first.scroll_into_view_if_needed()
        _settle(page, 1.5)
        # End on the served-model chart: 0.202 against 0.131.
        _glide(page, 240, step=40, pause=0.06)
        _settle(page, 4.5)

        video = page.video
        context.close()
        browser.close()
        raw = Path(video.path())

    trimmed_mp4 = OUT / "movielens-demo.mp4"
    gif = OUT / "movielens-demo.gif"
    start = max(0.0, lead_in - 0.5)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        target = OUT / "movielens-demo.webm"
        shutil.copy(raw, target)
        print(f"ffmpeg not found. Untrimmed recording: {target}")
        return target
    subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(raw),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", "-crf", "20",
            "-movflags", "+faststart", "-an", str(trimmed_mp4),
        ],
        check=True,
    )
    palette = OUT / "palette.png"
    scale = "fps=8,scale=720:-1:flags=lanczos"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(trimmed_mp4),
         "-vf", f"{scale},palettegen=stats_mode=diff", str(palette)],
        check=True,
    )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(trimmed_mp4), "-i", str(palette),
         "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5", str(gif)],
        check=True,
    )
    palette.unlink()
    shutil.rmtree(raw_dir, ignore_errors=True)
    for path in (trimmed_mp4, gif):
        print(f"{path.stat().st_size:>10} {path}")
    return trimmed_mp4


def _crop(page, top_text: str, height: int, out: Path, *, above: int = 24) -> None:
    from PIL import Image

    box = page.get_by_text(top_text, exact=False).first.bounding_box()
    if box is None:
        raise RuntimeError(f"{top_text!r} is not on the page")
    full = OUT / "full.png"
    page.screenshot(path=str(full), full_page=False)
    image = Image.open(full).convert("RGB")
    top = max(0, int(box["y"]) - above)
    image.crop((0, top, 1280, min(image.height, top + height))).save(out, optimize=True)
    full.unlink()
    print(f"{out.stat().st_size:>10} {out}")


def screenshots(channel: str | None) -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel=channel)
        page = browser.new_page(viewport={"width": 1280, "height": 4800}, color_scheme="light")

        page.goto(APP + "?view=discover", wait_until="domcontentloaded")
        _wait_text(page, "Find your next film")
        page.locator("img.poster").nth(5).wait_for(timeout=60_000)
        _settle(page, 3.0)
        _crop(page, "Search for a movie", 1080, SHOTS / "discover-rate.png", above=40)

        page.goto(APP + SHARED, wait_until="domcontentloaded")
        _wait_text(page, "Your next watch")
        _settle(page, 6.0)
        _crop(page, "Your next watch", 940, SHOTS / "discover-picks.png", above=40)

        page.goto(APP + "?view=quality", wait_until="domcontentloaded")
        _wait_text(page, "The model in this demo")
        _settle(page, 3.0)
        _crop(page, "How good is it?", 790, SHOTS / "quality.png", above=60)
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("video", "screenshots", "both"))
    parser.add_argument("--channel", default=None, help="e.g. chrome for the installed Chrome")
    args = parser.parse_args()
    if args.what in {"screenshots", "both"}:
        screenshots(args.channel)
    if args.what in {"video", "both"}:
        record(args.channel)


if __name__ == "__main__":
    main()
