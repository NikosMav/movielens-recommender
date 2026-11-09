"""Card HTML. Poster images are referenced by URL so the server does not decode them."""

from __future__ import annotations

import html
from pathlib import Path

THEME_CSS = """
<style>
.stApp {
  background: #f4efe6;
}
.block-container {
  padding-top: 1.2rem;
}
.hero {
  padding: 0.2rem 0 0.8rem;
  border-bottom: 1px solid #e2d8c8;
  margin-bottom: 0.8rem;
}
.hero .kicker {
  margin: 0;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  font-size: 0.75rem;
  color: #8a3a32;
}
.hero h1 {
  margin: 0.15rem 0 0;
  font-size: 2.15rem;
  line-height: 1.1;
  letter-spacing: -0.03em;
  color: #1f1a16;
}
.hero .pitch {
  margin: 0.45rem 0 0;
  font-size: 1.12rem;
  color: #3f3832;
  max-width: 40rem;
}
.rec-card {
  display: flex;
  gap: 0.75rem;
  align-items: flex-start;
  background: #fffdf9;
  border: 1px solid #e4d9c8;
  border-radius: 14px;
  padding: 0.65rem;
  margin-bottom: 0.45rem;
  min-height: 132px;
}
.poster {
  width: 84px;
  height: 126px;
  object-fit: cover;
  border-radius: 8px;
  flex: 0 0 auto;
  background: #2c2622;
}
.poster.placeholder {
  display: flex;
  align-items: flex-end;
  box-sizing: border-box;
  padding: 0.45rem;
  color: #f3eadf;
  font-size: 0.72rem;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.rec-body { min-width: 0; }
.rec-rank {
  color: #8a3a32;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0;
}
.rec-title {
  margin: 0.05rem 0 0;
  font-size: 1rem;
  font-weight: 650;
  color: #1f1a16;
  word-break: normal;
}
.rec-meta {
  margin: 0.12rem 0 0.3rem;
  color: #6d645b;
  font-size: 0.82rem;
  word-break: normal;
}
.rec-why {
  margin: 0;
  padding-left: 1rem;
  color: #2c2723;
  font-size: 0.88rem;
}
.rec-why li { margin: 0.12rem 0; }
.shelf-card .rec-card { min-height: 0; }
.tmdb-footer {
  margin-top: 1.4rem;
  padding-top: 0.8rem;
  border-top: 1px solid #e2d8c8;
  color: #5c554e;
  font-size: 0.82rem;
}
.tmdb-footer a { color: #1f1a16; }
.tmdb-logo { width: 92px; height: auto; display: block; margin: 0.35rem 0; }
</style>
"""

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
        f'<a href="{html.escape(ATTRIBUTION_LINK, quote=True)}">{logo}</a>'
        f"<p>{html.escape(ATTRIBUTION_NOTICE)}</p>"
        f"<p>{html.escape(extra)}</p>"
        "</div>"
    )


def load_logo_svg(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")
