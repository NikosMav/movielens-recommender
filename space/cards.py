"""Card HTML. Poster images are referenced by URL so the server does not decode them.

Colours follow the colour scheme Streamlit sets on ``.stApp``. That scheme is
light or dark from the active theme (Settings, or the system preference when
the theme follows the system). ``light-dark()`` reads it. A light fallback is
declared first for a browser that does not understand ``light-dark()``.
"""

from __future__ import annotations

import html
from pathlib import Path

# Warm paper in light mode. The dark values are the same hues, lightened or
# darkened until body text is at least 4.5:1 and borders at least 3:1.
LIGHT = {
    "app": "#f4efe6",
    "card": "#fffdf9",
    "ink": "#1f1a16",
    "pitch": "#3f3832",
    "kicker": "#8a3a32",
    "meta": "#6d645b",
    "why": "#2c2723",
    "footer": "#5c554e",
    "border": "#8a7b6c",
    "placeholder_bg": "#2c2622",
    "placeholder_fg": "#f3eadf",
    "logo_plate": "#ffffff",
}
DARK = {
    "app": "#14110e",
    "card": "#241f1c",
    "ink": "#f7f1e8",
    "pitch": "#e7dfd4",
    "kicker": "#f0b2a4",
    "meta": "#c4b8a8",
    "why": "#f3ece3",
    "footer": "#ddd4c8",
    "border": "#8d8278",
    "placeholder_bg": "#3c342e",
    "placeholder_fg": "#f6f0e6",
    "logo_plate": "#ffffff",
}


def _paint(prop: str, light: str, dark: str) -> str:
    """A plain colour, then the same property via ``light-dark()``.

    ``light-dark()`` takes colours only, so borders set the width here and
    the colour on the longhand.
    """
    return f"{prop}: {light};\n  {prop}: light-dark({light}, {dark});"


def _line(prop: str, light: str, dark: str) -> str:
    return (
        f"{prop}: 1px solid {light};\n"
        f"  {prop}-color: {light};\n"
        f"  {prop}-color: light-dark({light}, {dark});"
    )


def _theme_css() -> str:
    light, dark = LIGHT, DARK
    return f"""
<style>
.stApp[data-testid="stApp"] {{
  {_paint("background", light["app"], dark["app"])}
}}
.block-container {{
  padding-top: 1.2rem;
}}
.hero {{
  padding: 0.2rem 0 0.8rem;
  {_line("border-bottom", light["border"], dark["border"])}
  margin-bottom: 0.8rem;
}}
.hero .kicker {{
  margin: 0;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 0.75rem;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.hero h1 {{
  margin: 0.15rem 0 0;
  font-size: 2.15rem;
  line-height: 1.1;
  letter-spacing: -0.03em;
  {_paint("color", light["ink"], dark["ink"])}
}}
.hero .pitch {{
  margin: 0.45rem 0 0;
  font-size: 1.12rem;
  {_paint("color", light["pitch"], dark["pitch"])}
  max-width: 40rem;
}}
.rec-card {{
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  {_paint("background", light["card"], dark["card"])}
  {_line("border", light["border"], dark["border"])}
  {_paint("color", light["ink"], dark["ink"])}
  border-radius: 14px;
  padding: 0.65rem;
  margin-bottom: 0.45rem;
  min-height: 132px;
}}
.poster {{
  width: 84px;
  height: 126px;
  object-fit: cover;
  border-radius: 8px;
  flex: 0 0 auto;
  {_paint("background", light["placeholder_bg"], dark["placeholder_bg"])}
}}
.poster.placeholder {{
  display: flex;
  align-items: flex-end;
  box-sizing: border-box;
  padding: 0.45rem;
  {_paint("color", light["placeholder_fg"], dark["placeholder_fg"])}
  {_line("border", light["border"], dark["meta"])}
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}
.rec-body {{ min-width: 0; }}
.rec-rank {{
  {_paint("color", light["kicker"], dark["kicker"])}
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0;
}}
.rec-title {{
  margin: 0.05rem 0 0;
  font-size: 1rem;
  font-weight: 650;
  {_paint("color", light["ink"], dark["ink"])}
  word-break: normal;
}}
.rec-meta {{
  margin: 0.12rem 0 0.3rem;
  {_paint("color", light["meta"], dark["meta"])}
  font-size: 0.82rem;
  word-break: normal;
}}
.rec-why {{
  margin: 0;
  padding-left: 1rem;
  {_paint("color", light["why"], dark["why"])}
  font-size: 0.88rem;
}}
.rec-why li {{ margin: 0.12rem 0; }}
.shelf-card .rec-card {{ min-height: 0; }}
.tmdb-footer {{
  margin-top: 1.4rem;
  padding-top: 0.8rem;
  {_line("border-top", light["border"], dark["border"])}
  {_paint("color", light["footer"], dark["footer"])}
  font-size: 0.82rem;
}}
.tmdb-footer a {{
  {_paint("color", light["ink"], dark["ink"])}
}}
.tmdb-mark {{
  display: inline-block;
  background: {light["logo_plate"]};
  {_line("border", light["border"], dark["border"])}
  border-radius: 8px;
  padding: 0.35rem 0.55rem 0.3rem;
  line-height: 0;
}}
.tmdb-logo {{ width: 92px; height: auto; display: block; margin: 0; }}
.stApp [data-testid="stCaptionContainer"] {{
  opacity: 1;
  {_paint("color", light["footer"], dark["meta"])}
}}
</style>
"""


THEME_CSS = _theme_css()

PITCH = "Pick a viewer or rate a few films, then read why each title made the list."


def hero_html() -> str:
    return (
        '<div class="hero">'
        '<p class="kicker">MovieLens ml-1m</p>'
        "<h1>A recommender that shows its work</h1>"
        f'<p class="pitch">{html.escape(PITCH)}</p>'
        "</div>"
    )


def poster_html(title: str, poster_url: str | None) -> str:
    """An ``<img>`` for a URL, or a drawn placeholder. Neither path reads image bytes."""
    if poster_url:
        src = html.escape(poster_url, quote=True)
        return f'<img class="poster" alt="" width="84" height="126" src="{src}">'
    label = html.escape(title)
    return (
        f'<div class="poster placeholder" role="img" aria-label="No poster for {label}">'
        "<span>No poster</span></div>"
    )


def recommendation_card_html(
    *,
    title: str,
    year: object,
    genres: str,
    reasons: list[str],
    poster_url: str | None,
    rank: int | None = None,
    show_reasons: bool = True,
) -> str:
    bits = []
    if year not in (None, ""):
        bits.append(str(year))
    if genres:
        bits.append(" · ".join(part for part in genres.split("|") if part))
    meta = " · ".join(bits) if bits else "—"
    why = ""
    if show_reasons:
        if reasons:
            items = "".join(f"<li>{html.escape(text)}</li>" for text in reasons)
        else:
            items = "<li>The ranker did not give a positive reason for this title.</li>"
        why = f'<ul class="rec-why">{items}</ul>'
    rank_html = f'<p class="rec-rank">{int(rank)}</p>' if rank is not None else ""
    return (
        '<div class="rec-card">'
        f"{poster_html(title, poster_url)}"
        '<div class="rec-body">'
        f"{rank_html}"
        f'<p class="rec-title">{html.escape(title)}</p>'
        f'<p class="rec-meta">{html.escape(meta)}</p>'
        f"{why}"
        "</div></div>"
    )


def footer_html(*, logo_svg: str, api_key_set: bool) -> str:
    from posters import ATTRIBUTION_LINK, ATTRIBUTION_NOTICE

    if api_key_set:
        extra = (
            "Poster images load in your browser from image.tmdb.org. "
            "This server stores the image address only."
        )
    else:
        extra = (
            "Cards use a placeholder until a TMDB_API_KEY secret is set. "
            "No poster request is sent without it."
        )
    logo = logo_svg.replace("<svg ", '<svg class="tmdb-logo" ', 1)
    return (
        '<div class="tmdb-footer">'
        '<span class="tmdb-mark">'
        f'<a href="{html.escape(ATTRIBUTION_LINK, quote=True)}">{logo}</a>'
        "</span>"
        f"<p>{html.escape(ATTRIBUTION_NOTICE)}</p>"
        f"<p>{html.escape(extra)}</p>"
        "</div>"
    )


def load_logo_svg(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")
