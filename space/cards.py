"""Card HTML and the shared palette.

Colours follow the colour scheme Streamlit sets on ``.stApp``. That scheme is
light or dark from the active theme (Settings, or the system preference when
the theme follows the system). ``light-dark()`` reads it. A light fallback is
declared first for a browser that does not understand ``light-dark()``.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

# Purple paper in light mode, the same hue darkened for dark mode.
# Body text is at least 4.5:1 and borders at least 3:1 on both.
LIGHT = {
    "app": "#f6f5fb",
    "card": "#ffffff",
    "ink": "#221c42",
    "pitch": "#3a345c",
    "kicker": "#3d2d86",
    "meta": "#4a4468",
    "why": "#2a2448",
    "footer": "#3e385c",
    "border": "#6d658c",
    "placeholder_bg": "#3a3164",
    "placeholder_fg": "#f6f3ff",
    "logo_plate": "#ffffff",
}
DARK = {
    "app": "#13111a",
    "card": "#211e2e",
    "ink": "#f5f2ff",
    "pitch": "#e3def6",
    "kicker": "#d4c6ff",
    "meta": "#cfc6e6",
    "why": "#f3eefe",
    "footer": "#ddd6f2",
    "border": "#8a82a8",
    "placeholder_bg": "#2a2540",
    "placeholder_fg": "#f6f3ff",
    "logo_plate": "#ffffff",
}

_ARTICLES = frozenset(
    {"the", "a", "an", "le", "la", "les", "el", "il", "das", "der", "die", "l"}
)
_WORD = re.compile(r"[A-Za-z0-9]+")


def _paint(prop: str, light: str, dark: str) -> str:
    """A plain colour, then the same property via ``light-dark()``."""
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
@import url("https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700&family=Fraunces:opsz,wght@9..144,560;9..144,650&display=swap");
.stApp[data-testid="stApp"] {{
  {_paint("background", light["app"], dark["app"])}
  {_paint("color", light["ink"], dark["ink"])}
  font-family: Figtree, Arial, sans-serif;
  font-size: 16px;
}}
[data-testid="stHeader"] {{
  {_paint("background", light["app"], dark["app"])}
}}
.block-container {{
  max-width: 1120px;
  padding-top: 6rem;
  padding-left: 1.25rem;
  padding-right: 1.25rem;
}}
.stApp h1, .stApp h2, .stApp h3, .display {{
  font-family: Fraunces, Georgia, serif;
  letter-spacing: -0.03em;
  {_paint("color", light["ink"], dark["ink"])}
}}
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stCaptionContainer"] p {{
  {_paint("color", light["ink"], dark["ink"])}
}}
.stApp [data-testid="stCaptionContainer"] {{
  opacity: 1;
  font-size: 0.875rem;
  {_paint("color", light["footer"], dark["footer"])}
}}
.muted {{ {_paint("color", light["pitch"], dark["pitch"])} }}
.sr-only {{
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}}
.site-header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  min-height: 4.5rem;
  margin-bottom: 0.5rem;
  {_line("border-bottom", light["border"], dark["border"])}
}}
.brand {{
  display: inline-flex;
  align-items: center;
  gap: 0.6rem;
  font-weight: 700;
  font-size: 1.35rem;
  letter-spacing: -0.04em;
  text-decoration: none;
  {_paint("color", light["ink"], dark["ink"])}
}}
.brand-mark {{
  display: grid;
  place-items: center;
  width: 2.1rem;
  height: 2.1rem;
  border-radius: 0.5rem;
  background: {light["kicker"]};
  background: light-dark({light["kicker"]}, {dark["kicker"]});
  color: {light["app"]};
  color: light-dark({light["app"]}, {dark["app"]});
}}
.st-key-topbar [data-testid="stRadio"] div[role="radiogroup"] {{
  flex-wrap: wrap;
  gap: 0.35rem 0.85rem;
  justify-content: flex-end;
}}
.st-key-topbar [data-testid="stRadio"] label {{
  min-height: 2.75rem;
  {_paint("color", light["pitch"], dark["pitch"])}
}}
.st-key-topbar [data-testid="stRadioOption"] > div > div:first-child {{
  display: none;
}}
.st-key-topbar [data-testid="stRadio"] label[data-checked="true"],
.st-key-topbar [data-testid="stRadio"] label:has(input:checked) {{
  {_paint("color", light["kicker"], dark["kicker"])}
  font-weight: 700;
  box-shadow: inset 0 -3px 0 {light["kicker"]};
  box-shadow: inset 0 -3px 0 light-dark({light["kicker"]}, {dark["kicker"]});
}}
.intro {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr);
  gap: clamp(1.25rem, 4vw, 4rem);
  align-items: center;
  padding: 1.25rem 0 1.5rem;
}}
.intro h1 {{
  margin: 0 0 0.6rem;
  font-size: clamp(2.6rem, 6vw, 4.5rem);
  line-height: 0.98;
  font-weight: 560;
}}
.intro p {{
  margin: 0;
  max-width: 38ch;
  font-size: 1.05rem;
  line-height: 1.5;
  {_paint("color", light["pitch"], dark["pitch"])}
}}
.intro-art svg {{
  display: block;
  width: 100%;
  height: auto;
  max-width: 100%;
}}
.section-title {{
  margin: 0.2rem 0 0.35rem;
  font-size: clamp(1.7rem, 3vw, 2.5rem);
  line-height: 1.1;
}}
.pick {{
  display: flex;
  gap: 0.85rem;
  align-items: flex-start;
  padding: 0.85rem 0;
  {_line("border-top", light["border"], dark["border"])}
  min-width: 0;
}}
.poster {{
  width: 76px;
  height: 114px;
  object-fit: cover;
  border-radius: 8px;
  flex: 0 0 auto;
  {_paint("background", light["placeholder_bg"], dark["placeholder_bg"])}
}}
.poster.placeholder {{
  display: flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box;
  {_paint("color", light["placeholder_fg"], dark["placeholder_fg"])}
  background-image: linear-gradient(
    155deg,
    hsl(var(--poster-hue) 38% 34%),
    hsl(calc(var(--poster-hue) + 28) 42% 22%)
  );
  font-family: Fraunces, Georgia, serif;
  font-size: 1.35rem;
  letter-spacing: 0.04em;
}}
.pick-body {{ min-width: 0; flex: 1 1 auto; }}
.pick-rank {{
  margin: 0;
  font-family: Fraunces, Georgia, serif;
  font-size: 1.15rem;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.pick-title {{
  margin: 0.1rem 0 0;
  font-size: 1.15rem;
  font-weight: 700;
  {_paint("color", light["ink"], dark["ink"])}
  overflow-wrap: anywhere;
}}
.pick-meta {{
  margin: 0.2rem 0 0.35rem;
  font-size: 0.875rem;
  {_paint("color", light["meta"], dark["meta"])}
}}
.pick-why {{
  margin: 0;
  padding-left: 1.1rem;
  {_paint("color", light["why"], dark["why"])}
}}
.pick-why li {{ margin: 0.15rem 0; }}
.ticket {{
  {_paint("background", light["card"], dark["card"])}
  {_line("border", light["border"], dark["border"])}
  border-top: 3px solid {light["kicker"]};
  border-top-color: {light["kicker"]};
  border-top-color: light-dark({light["kicker"]}, {dark["kicker"]});
  border-radius: 10px;
  padding: 1rem 1rem 1.1rem;
}}
.progress {{
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.35rem;
  margin: 0.7rem 0;
}}
.progress span {{
  display: block;
  height: 0.35rem;
  border-radius: 99px;
  background: {light["border"]};
  background: light-dark({light["border"]}, {dark["border"]});
}}
.progress span.on {{
  background: {light["kicker"]};
  background: light-dark({light["kicker"]}, {dark["kicker"]});
}}
.chart {{ margin: 0.75rem 0 1.25rem; }}
.bar-row {{
  display: grid;
  grid-template-columns: minmax(0, 13rem) minmax(0, 1fr) auto;
  gap: 0.55rem;
  align-items: center;
  margin: 0.45rem 0;
  min-width: 0;
}}
.bar-row.is-best .bar-name {{ font-weight: 700; }}
.bar-name {{
  min-width: 0;
  overflow-wrap: anywhere;
  {_paint("color", light["ink"], dark["ink"])}
}}
.bar-id {{
  display: inline-block;
  margin-left: 0.35rem;
  font-family: Figtree, Arial, sans-serif;
  font-size: 0.75rem;
  font-weight: 600;
  {_paint("color", light["meta"], dark["meta"])}
}}
.bar-track {{
  position: relative;
  height: 0.7rem;
  border-radius: 99px;
  background: {light["card"]};
  background: light-dark({light["card"]}, {dark["card"]});
  {_line("border", light["border"], dark["border"])}
  overflow: hidden;
}}
.bar-fill {{
  display: block;
  height: 100%;
  border-radius: 99px;
  background: {light["kicker"]};
  background: light-dark({light["kicker"]}, {dark["kicker"]});
}}
.bar-row.is-best .bar-fill {{
  background: {light["ink"]};
  background: light-dark({light["ink"]}, {dark["kicker"]});
}}
.bar-ci {{
  position: absolute;
  top: -0.18rem;
  height: calc(100% + 0.36rem);
  border-left: 2px solid {light["ink"]};
  border-right: 2px solid {light["ink"]};
  border-left-color: {light["ink"]};
  border-right-color: {light["ink"]};
  border-left-color: light-dark({light["ink"]}, {dark["ink"]});
  border-right-color: light-dark({light["ink"]}, {dark["ink"]});
  box-sizing: border-box;
}}
.bar-value {{
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  {_paint("color", light["ink"], dark["ink"])}
}}
.chart-note, .score-detail p {{
  {_paint("color", light["pitch"], dark["pitch"])}
  font-size: 0.9rem;
}}
.score-detail {{
  padding: 0.65rem 0;
  {_line("border-top", light["border"], dark["border"])}
}}
.score-detail h3 {{
  margin: 0 0 0.25rem;
  font-size: 1.15rem;
}}
.history-row {{
  display: grid;
  grid-template-columns: 46px minmax(0, 1fr) auto;
  gap: 0.65rem;
  align-items: center;
  padding: 0.45rem 0;
  {_line("border-top", light["border"], dark["border"])}
}}
.history-row .poster {{ width: 46px; height: 69px; }}
.history-stars {{
  font-weight: 700;
  white-space: nowrap;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.tmdb-footer {{
  margin-top: 1.6rem;
  padding-top: 0.9rem;
  {_line("border-top", light["border"], dark["border"])}
  {_paint("color", light["footer"], dark["footer"])}
  font-size: 0.875rem;
}}
.tmdb-footer a {{ {_paint("color", light["ink"], dark["ink"])} }}
.tmdb-mark {{
  display: inline-block;
  background: {light["logo_plate"]};
  {_line("border", light["border"], dark["border"])}
  border-radius: 8px;
  padding: 0.35rem 0.55rem 0.3rem;
  line-height: 0;
}}
.tmdb-logo {{ width: 92px; height: auto; display: block; margin: 0; }}
.stApp [data-testid="stBaseButton-primary"] {{
  background: #392975;
  background: light-dark(#392975, #6b59c4);
  color: #ffffff;
  border: 1px solid #392975;
  border-color: light-dark(#392975, #6b59c4);
  min-height: 2.75rem;
}}
.stApp button:disabled {{
  opacity: 0.45;
  cursor: not-allowed;
}}
.stApp [data-testid="stBaseButton-primary"] p {{ color: #ffffff; }}
.stApp [data-testid="stBaseButton-secondary"] {{
  background: #ffffff;
  background: light-dark(#ffffff, #2a2640);
  color: {light["ink"]};
  color: light-dark({light["ink"]}, {dark["ink"]});
  border: 1px solid {light["border"]};
  border-color: light-dark({light["border"]}, {dark["border"]});
  min-height: 2.75rem;
}}
.stApp [data-testid="stBaseButton-secondary"] p {{
  {_paint("color", light["ink"], dark["ink"])}
}}
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stNumberInput"] input {{
  background: #ffffff;
  background: light-dark(#ffffff, #2a2640);
  {_paint("color", light["ink"], dark["ink"])}
  border-color: {light["border"]};
  border-color: light-dark({light["border"]}, {dark["border"]});
  min-height: 2.75rem;
}}
.stApp [data-testid="stTextInput"] input:focus,
.stApp [data-testid="stNumberInput"] input:focus {{
  border-color: #392975;
  border-color: light-dark(#392975, #d4c6ff);
  box-shadow: 0 0 0 1px #392975;
  box-shadow: 0 0 0 1px light-dark(#392975, #d4c6ff);
}}
.stApp [data-testid="stExpander"] {{
  {_paint("background", light["card"], dark["card"])}
  {_line("border", light["border"], dark["border"])}
}}
.stApp [data-testid="stExpander"] details summary,
.stApp [data-testid="stExpander"] details summary p {{
  {_paint("color", light["kicker"], dark["kicker"])}
}}
[class*="st-key-stars-"] [data-testid="stHorizontalBlock"] {{
  flex-wrap: nowrap;
  gap: 0.3rem;
}}
[class*="st-key-stars-"] [data-testid="stColumn"] {{
  min-width: 0 !important;
}}
[class*="st-key-stars-"] button {{
  min-height: 2.75rem;
  width: 100%;
  padding-inline: 0.1rem;
  overflow: hidden;
}}
[class*="st-key-stars-"] button p {{
  font-size: 0 !important;
  line-height: 0;
  margin: 0;
}}
[class*="st-key-stars-"] [data-testid="stBaseButton-primary"] p::before {{
  content: "★";
  font-size: 1.35rem;
  line-height: 1;
  color: #ffffff;
}}
[class*="st-key-stars-"] [data-testid="stBaseButton-secondary"] p::before {{
  content: "☆";
  font-size: 1.35rem;
  line-height: 1;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.stApp [data-testid="stButtonGroup"] [role="toolbar"] {{
  flex-wrap: wrap !important;
  overflow: visible !important;
  height: auto !important;
  row-gap: 0.45rem;
  max-width: 100%;
}}
.stApp [data-testid="stButtonGroup"] button {{
  background: #ffffff;
  background: light-dark(#ffffff, #2a2640);
  {_paint("color", light["ink"], dark["ink"])}
  border: 1px solid {light["border"]};
  border-color: light-dark({light["border"]}, {dark["border"]});
  border-radius: 999px;
}}
.stApp [data-testid="stButtonGroup"] button[aria-pressed="true"] {{
  background: #392975;
  background: light-dark(#392975, #6b59c4);
  color: #ffffff;
  border-color: #392975;
  border-color: light-dark(#392975, #6b59c4);
}}
.stApp [data-testid="stButtonGroup"] button[aria-pressed="true"] p {{
  color: #ffffff;
}}
.stApp [data-testid="stCheckbox"] label,
.stApp [data-testid="stWidgetLabel"] p {{
  {_paint("color", light["ink"], dark["ink"])}
}}
.stApp [data-testid="stAlert"] {{
  {_paint("background", light["card"], dark["card"])}
  {_paint("color", light["ink"], dark["ink"])}
  {_line("border", light["border"], dark["border"])}
}}
.stApp [data-testid="stAlert"] p {{
  {_paint("color", light["ink"], dark["ink"])}
}}
@media (max-width: 800px) {{
  .block-container {{
    padding-left: 0.75rem;
    padding-right: 0.75rem;
  }}
  .st-key-topbar [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  .st-key-topbar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }}
  .st-key-topbar [data-testid="stRadio"] div[role="radiogroup"] {{
    justify-content: flex-start;
  }}
  .intro {{ grid-template-columns: minmax(0, 1fr); }}
  .intro-art {{ order: -1; }}
  .bar-row {{ grid-template-columns: minmax(0, 1fr) auto; }}
  .bar-name {{ grid-column: 1 / -1; }}
  .st-key-workbench [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  .st-key-workbench [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }}
  .st-key-compare [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  .st-key-compare [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }}
}}
@media (max-width: 800px) {{
  .st-key-workbench [class*="st-key-stars-"] [data-testid="stColumn"],
  .st-key-compare [class*="st-key-stars-"] [data-testid="stColumn"] {{
    width: 20% !important;
    flex: 1 1 20% !important;
  }}
}}
</style>
"""


THEME_CSS = _theme_css()

PITCH = (
    "Rate three films you've seen. MovieLens will find ten to try and say why they made the list."
)

_SCROLL = """
<script>
const doc = window.parent.document;
const el = doc.getElementById("your-next-watch");
if (el) el.scrollIntoView({block: "start", behavior: "smooth"});
const head = doc.getElementById("picks-heading");
if (head) head.focus({preventScroll: true});
</script>
"""


def initials(title: str) -> str:
    """Two letters for a poster placeholder. Articles do not count."""
    words = _WORD.findall(title)
    content = [word for word in words if word.casefold() not in _ARTICLES] or words
    if not content:
        return "•"
    if len(content) == 1:
        return content[0][:2].upper()
    return (content[0][0] + content[1][0]).upper()


def hero_html() -> str:
    art_path = Path(__file__).with_name("screening-room.svg")
    art = art_path.read_text(encoding="utf-8") if art_path.is_file() else ""
    return (
        '<section class="intro">'
        "<div>"
        "<h1>Find your next film.</h1>"
        f'<p>{html.escape(PITCH)}</p>'
        "</div>"
        f'<div class="intro-art" aria-hidden="true">{art}</div>'
        "</section>"
    )


def poster_html(title: str, poster_url: str | None) -> str:
    """An ``<img>`` for a URL, or initials on a gradient. Neither path reads image bytes."""
    if poster_url:
        src = html.escape(poster_url, quote=True)
        alt = html.escape(title, quote=True)
        return f'<img class="poster" alt="" data-title="{alt}" width="76" height="114" src="{src}">'
    label = html.escape(title)
    hue = sum(ord(char) for char in title) % 360
    mark = html.escape(initials(title))
    return (
        f'<div class="poster placeholder" role="img" aria-label="No poster for {label}" '
        f'style="--poster-hue:{hue}"><span>{mark}</span></div>'
    )


def _year_text(year: object) -> str:
    if year in (None, ""):
        return ""
    if isinstance(year, float) and year.is_integer():
        return str(int(year))
    return str(year)


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
    shown_year = _year_text(year)
    if shown_year:
        bits.append(shown_year)
    if genres:
        bits.append(" · ".join(part for part in genres.split("|") if part))
    meta = " · ".join(bits) if bits else "—"
    why = ""
    if show_reasons and reasons:
        items = "".join(f"<li>{html.escape(text)}</li>" for text in reasons)
        why = f'<ul class="pick-why">{items}</ul>'
    rank_html = ""
    if rank is not None:
        rank_html = f'<p class="pick-rank">{int(rank):02d}</p>'
    return (
        '<div class="pick">'
        f"{poster_html(title, poster_url)}"
        '<div class="pick-body">'
        f"{rank_html}"
        f'<p class="pick-title">{html.escape(title)}</p>'
        f'<p class="pick-meta">{html.escape(meta)}</p>'
        f"{why}"
        "</div></div>"
    )


def metric_chart_html(rows: list[dict], *, caption: str) -> str:
    """Bars and error whiskers from stored floats. Labels are those floats, rounded."""
    if not rows:
        return ""
    scale = max(max(float(row["high"]), float(row["ndcg"])) for row in rows)
    if scale <= 0:
        scale = 1.0
    parts = [
        '<div class="chart" role="img" '
        'aria-label="NDCG at 10 for each model, with the stored uncertainty range">'
    ]
    for row in rows:
        ndcg = float(row["ndcg"])
        low = float(row["low"])
        high = float(row["high"])
        width = max(0.0, min(100.0, ndcg / scale * 100.0))
        left = max(0.0, min(100.0, low / scale * 100.0))
        right = max(0.0, min(100.0, high / scale * 100.0))
        best = " is-best" if row.get("best") else ""
        label = html.escape(str(row["label"]))
        raw = html.escape(str(row["model"]))
        value = format(ndcg, ".3f")
        low_s = format(low, ".3f")
        high_s = format(high, ".3f")
        tip = html.escape(
            f"{row['label']} ({row['model']}): {value}, range {low_s} to {high_s}",
            quote=True,
        )
        parts.append(
            f'<div class="bar-row{best}">'
            f'<div class="bar-name">{label} <span class="bar-id">{raw}</span></div>'
            f'<div class="bar-track" title="{tip}">'
            f'<span class="bar-fill" style="width:{width:.2f}%"></span>'
            '<span class="bar-ci" style="left:'
            f'{left:.2f}%;width:{max(0.0, right - left):.2f}%"></span>'
            "</div>"
            f'<div class="bar-value">{value}</div>'
            "</div>"
        )
    parts.append(f'<p class="chart-note">{html.escape(caption)}</p></div>')
    return "".join(parts)


def scroll_script() -> str:
    return _SCROLL


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
