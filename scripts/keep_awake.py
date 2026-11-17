#!/usr/bin/env python3
"""Open the public demo in a headless browser so Community Cloud does not put it to sleep.

A plain HTTP request does not count as a visit: the app only wakes when a
browser loads it, and a sleeping app waits for someone to click its wake
button. This script loads the page, clicks that button when it is there, and
waits until the app has drawn its first section. It exits non-zero when the
app does not come up, so a scheduled run that fails is visible.

Locally, ``KEEP_AWAKE_CHANNEL=chrome`` uses the installed Chrome instead of a
Playwright browser download.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_URL = "https://nikosmav-movielens-recommender.streamlit.app/"
WAKE_BUTTON = "Yes, get this app back up"
# Text the app draws once its models are loaded (the Discover hero).
READY_TEXT = "Find your next film"
MISSING_TEXT = ("No model parameters were found", "could not finish starting")


def _app_text(page) -> str:
    """Visible text of the page and of every frame. The app runs inside an iframe."""
    chunks = []
    for frame in page.frames:
        try:
            chunks.append(frame.locator("body").inner_text(timeout=2_000))
        except Exception:
            continue
    return "\n".join(chunks)


def visit(url: str, *, timeout_sec: int, channel: str | None) -> tuple[bool, str]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel=channel)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        started = time.monotonic()
        woke = False
        while time.monotonic() - started < timeout_sec:
            button = page.get_by_role("button", name=WAKE_BUTTON)
            if not woke and button.count():
                button.first.click()
                woke = True
                print("The app was asleep. Clicked the wake button.", flush=True)
            text = _app_text(page)
            if READY_TEXT in text:
                browser.close()
                waited = round(time.monotonic() - started)
                return True, f"ready after {waited}s ({'woken' if woke else 'awake'})"
            for marker in MISSING_TEXT:
                if marker in text:
                    browser.close()
                    return False, f"the app started but reports: {marker}"
            time.sleep(5)
        browser.close()
    return False, f"not ready after {timeout_sec}s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.environ.get("DEMO_URL", DEFAULT_URL))
    parser.add_argument("--timeout", type=int, default=420, help="Seconds to wait for the app")
    args = parser.parse_args()
    ok, detail = visit(
        args.url, timeout_sec=args.timeout, channel=os.environ.get("KEEP_AWAKE_CHANNEL") or None
    )
    print(f"{args.url}: {detail}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
