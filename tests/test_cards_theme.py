"""Card colours have to work on Streamlit's light theme and its dark theme."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"

# Body text 4.5:1. Borders and other UI edges 3:1. WCAG 2.1 AA is the guide.
_TEXT = (
    ("ink", "app"),
    ("ink", "card"),
    ("pitch", "app"),
    ("kicker", "app"),
    ("kicker", "card"),
    ("meta", "card"),
    ("meta", "app"),
    ("why", "card"),
    ("footer", "app"),
    ("footer", "card"),
    ("placeholder_fg", "placeholder_bg"),
)


def _load(name: str):
    path = SPACE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"space_{name}_theme_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _channels(color: str) -> tuple[float, float, float]:
    text = color.removeprefix("#")
    return tuple(int(text[index : index + 2], 16) / 255 for index in (0, 2, 4))


def _luminance(color: str) -> float:
    def channel(value: float) -> float:
        if value <= 0.04045:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = _channels(color)
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def _contrast(foreground: str, background: str) -> float:
    lighter = max(_luminance(foreground), _luminance(background))
    darker = min(_luminance(foreground), _luminance(background))
    return (lighter + 0.05) / (darker + 0.05)


def test_both_palettes_meet_wcag_aa_for_text_and_borders():
    cards = _load("cards")
    for name in ("LIGHT", "DARK"):
        palette = getattr(cards, name)
        for foreground, background in _TEXT:
            ratio = _contrast(palette[foreground], palette[background])
            assert ratio >= 4.5, (name, foreground, background, ratio)
        edge = _contrast(palette["border"], palette["card"])
        page = _contrast(palette["border"], palette["app"])
        assert edge >= 3, (name, "border", "card", edge)
        assert page >= 3, (name, "border", "app", page)
        plate = _contrast(palette["border"], palette["logo_plate"])
        assert plate >= 3, (name, "border", "logo_plate", plate)
    # The official TMDB mark stays on white in both themes.
    assert cards.LIGHT["logo_plate"] == cards.DARK["logo_plate"] == "#ffffff"
    dark_placeholder = _contrast(cards.DARK["meta"], cards.DARK["placeholder_bg"])
    assert dark_placeholder >= 3


def test_css_follows_streamlit_color_scheme_and_does_not_force_a_theme():
    cards = _load("cards")
    css = cards.THEME_CSS
    assert "light-dark(" in css
    assert cards.LIGHT["app"] in css
    assert cards.DARK["app"] in css
    assert cards.LIGHT["ink"] in css
    assert cards.DARK["ink"] in css
    assert 'data-testid="stApp"' in css
    # Setting color-scheme here would ignore Settings > Theme.
    assert "color-scheme" not in css
    config = ROOT / ".streamlit" / "config.toml"
    text = config.read_text(encoding="utf-8")
    assert 'toolbarMode = "minimal"' in text
    assert "[theme]" not in text
    assert "color-scheme" not in text
    assert "base =" not in text
    sys.path.insert(0, str(SPACE))
    try:
        footer = cards.footer_html(logo_svg="<svg viewBox='0 0 1 1'></svg>", api_key_set=False)
    finally:
        sys.path.remove(str(SPACE))
    assert 'class="tmdb-mark"' in footer
    assert 'class="tmdb-logo"' in footer
    assert "No poster request is sent without it." in footer
