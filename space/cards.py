"""Card HTML and the shared palette.

Colours follow the colour scheme Streamlit sets on ``.stApp``. That scheme is
light or dark from the active theme (Settings, or the system preference when
the theme follows the system). ``light-dark()`` reads it. A light fallback is
declared first for a browser that does not understand ``light-dark()``.
"""

from __future__ import annotations

import html
import math
import os
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
# Masks, so the star colour follows the theme. The graphic fills the 44px target
# and the row gap stays 4px, which keeps the stars contiguous.
_STAR_FILLED = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
    "%3Cpath fill='%23000' d='M12 1.6l2.9 6.7 7.3.7-5.5 4.8 1.7 7.2L12 17.3"
    " 5.6 21l1.7-7.2L1.8 9l7.3-.7z'/%3E%3C/svg%3E"
)
_STAR_EMPTY = (
    "data:image/svg+xml,"
    "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
    "%3Cpath fill='%23000' fill-rule='evenodd' d='M12 1.6l2.9 6.7 7.3.7-5.5 4.8"
    " 1.7 7.2L12 17.3 5.6 21l1.7-7.2L1.8 9l7.3-.7z"
    "M12 5.5l1.6 3.8 4.2.4-3.2 2.8.9 4.1L12 14.8 8.5 16.6l.9-4.1-3.2-2.8 4.2-.4z'"
    "/%3E%3C/svg%3E"
)


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
  max-width: 1560px;
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
.stApp a.brand {{
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
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {{
  position: sticky;
  top: 3.75rem;
  z-index: 40;
  {_paint("background", light["app"], dark["app"])}
}}
[data-testid="stVerticalBlock"].st-key-topbar {{
  {_paint("background", light["app"], dark["app"])}
  padding-bottom: 0.15rem;
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
  margin: 1.5rem 0 0.45rem;
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
    hsl(var(--poster-hue) 38% 30%),
    hsl(var(--poster-hue-b) 44% 16%)
  );
  font-family: Fraunces, Georgia, serif;
  font-size: 1.35rem;
  letter-spacing: 0.04em;
  text-shadow: 0 1px 2px rgb(0 0 0 / 45%);
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
  background: transparent;
  border: 0;
  padding: 0;
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
.ui-icon {{
  width: 1.35rem;
  height: 1.35rem;
  flex: 0 0 auto;
  vertical-align: -0.18em;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.icon-heading {{ display: flex; align-items: center; gap: 0.65rem; }}
.icon-heading [data-heading-text] {{
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
}}
.icon-heading .ui-icon {{ width: 1.5rem; height: 1.5rem; }}
.benchmark-summary {{ max-width: 75ch; line-height: 1.6; }}
.chart {{ margin: 1rem 0 1.75rem; max-width: 66rem; }}
.chart-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1.25rem;
  align-items: center;
  margin-bottom: 0.65rem;
  font-size: 0.82rem;
  {_paint("color", light["meta"], dark["meta"])}
}}
.chart-legend span {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
.chart-legend .ui-icon {{ width: 1.2rem; height: 1.2rem; }}
.chart-legend .legend-range .ui-icon {{
  {_paint("color", light["ink"], dark["ink"])}
}}
.chart-axis, .bar-row {{
  display: grid;
  grid-template-columns: minmax(0, 14rem) minmax(0, 1fr) 4rem;
  column-gap: 1rem;
}}
.chart-axis {{ font-size: 0.75rem; {_paint("color", light["meta"], dark["meta"])} }}
.axis-ticks {{ display: flex; justify-content: space-between; font-variant-numeric: tabular-nums; }}
.bar-row {{
  align-items: center;
  padding: 0.75rem 0;
  {_line("border-bottom", light["border"], dark["border"])}
  min-width: 0;
}}
.bar-row.is-best .bar-name {{ font-weight: 700; }}
.best-label {{
  display: block;
  margin-top: 0.15rem;
  font-size: 0.75rem;
  font-weight: 600;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.bar-name {{
  min-width: 0;
  overflow-wrap: anywhere;
  font-size: 0.95rem;
  line-height: 1.4;
  {_paint("color", light["ink"], dark["ink"])}
}}
.stApp .chart .chart-group {{
  margin: 1.1rem 0 0.15rem;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  {_paint("color", light["meta"], dark["meta"])}
}}
.bar-id {{
  display: inline-block;
  margin-left: 0.35rem;
  padding: 0.05rem 0.4rem;
  border-radius: 99px;
  font-family: Figtree, Arial, sans-serif;
  font-size: 0.9rem;
  font-weight: 600;
  {_line("border", light["border"], dark["border"])}
  {_paint("color", light["meta"], dark["meta"])}
}}
.poster.is-broken {{ display: none; }}
.because-line {{
  display: flex;
  align-items: center;
  gap: 0.45rem;
}}
.pick .because-line .poster.evidence-thumb {{
  width: 28px;
  min-width: 28px;
  max-width: 28px;
  height: 42px;
  flex: 0 0 28px;
  object-fit: cover;
  border-radius: 3px;
  font-size: 0.65rem;
}}
.neighbour-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(8.4rem, 1fr));
  gap: 0.55rem 0.7rem;
}}
.neighbour p {{
  margin: 0.25rem 0 0;
  font-size: 0.75rem;
  line-height: 1.25;
}}
.funnel-pad {{ height: 0.35rem; }}
[data-testid="stPopoverBody"] {{
  width: min(26rem, calc(100vw - 2rem));
}}
.funnel {{
  display: flex;
  flex-wrap: nowrap;
  align-items: stretch;
  gap: 0.75rem;
  margin: 1rem 0 1.25rem;
  padding: 0;
  list-style: none;
}}
.stApp .funnel > .funnel-step {{
  flex: 1 1 0;
  margin: 0;
  padding: 1rem;
  border-radius: 10px;
  min-width: 0;
  overflow-wrap: anywhere;
  background: {light["card"]};
  background: light-dark({light["card"]}, {dark["card"]});
  {_line("border", light["border"], dark["border"])}
}}
.stApp .funnel > .funnel-arrow {{
  flex: 0 0 auto;
  align-self: center;
  margin: 0;
  padding: 0;
  border: 0;
  background: transparent;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.funnel-heading {{ display: flex; align-items: center; gap: 0.5rem; }}
.funnel strong {{ font-size: 1rem; line-height: 1.3; }}
.stApp .funnel .funnel-description {{
  margin: 0.65rem 0 0;
  font-size: 0.9rem;
  line-height: 1.5;
  {_paint("color", light["meta"], dark["meta"])}
}}
.stApp .funnel .funnel-count {{
  margin: 0.6rem 0 0;
  font-size: 0.9rem;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.latency {{
  margin: 0.15rem 0;
  cursor: help;
  {_paint("color", light["pitch"], dark["pitch"])}
}}
.beyond-table {{
  overflow-x: auto;
  margin: 0.4rem 0 1rem;
}}
.beyond-table table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.92rem;
}}
.beyond-table th,
.beyond-table td {{
  padding: 0.4rem 0.55rem;
  text-align: left;
  white-space: nowrap;
  {_line("border-bottom", light["border"], dark["border"])}
}}
@media (max-width: 800px) {{
  .beyond-table table {{ font-size: 0.78rem; }}
  .beyond-table th,
  .beyond-table td {{ padding: 0.3rem 0.3rem; }}
  .beyond-table th {{ white-space: normal; }}
  .beyond-table th:first-child,
  .beyond-table td:first-child {{ white-space: normal; }}
}}
.beyond-table th {{
  font-weight: 600;
}}
.list-readout {{
  margin: 0.2rem 0 0.6rem;
  cursor: help;
  {_paint("color", light["meta"], dark["meta"])}
}}
.st-key-familiarity [data-testid="stSliderThumbValue"] {{
  border: 0;
  clip: rect(0 0 0 0);
  clip-path: inset(50%);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
  white-space: nowrap;
}}
.st-key-familiarity [data-testid="stSliderTickBar"] {{
  opacity: 1 !important;
  color: {light["ink"]};
  color: light-dark({light["ink"]}, {dark["ink"]});
}}
.st-key-familiarity [data-testid="stSlider"] div:has(> [data-testid="stSliderThumbValue"]) {{
  background-color: #392975 !important;
  background-color: light-dark(#392975, #6b59c4) !important;
}}
.stApp:has(.picks-stale) .st-key-picks {{
  outline: 1px dashed {light["meta"]};
  outline: 1px dashed light-dark({light["meta"]}, {dark["meta"]});
  outline-offset: 6px;
  border-radius: 12px;
}}
.bar-track {{
  position: relative;
  height: 1.5rem;
  overflow: visible;
}}
.bar-track::before {{
  content: "";
  position: absolute;
  top: 8px;
  left: 0;
  right: 0;
  height: 8px;
  border-radius: 99px;
  {_paint("background", light["border"], dark["border"])}
  opacity: 0.2;
}}
.bar-fill {{
  position: absolute;
  top: 8px;
  left: 0;
  display: block;
  height: 8px;
  border-radius: 99px;
  background: {light["kicker"]};
  background: light-dark({light["kicker"]}, {dark["kicker"]});
  opacity: 0.4;
}}
.bar-row.is-best .bar-fill {{ opacity: 0.85; }}
.bar-ci {{
  position: absolute;
  top: 11px;
  height: 2px;
  {_paint("background", light["ink"], dark["ink"])}
  z-index: 2;
  pointer-events: none;
}}
.bar-ci::before, .bar-ci::after {{
  content: "";
  position: absolute;
  top: -4px;
  height: 10px;
  width: 2px;
  background: inherit;
}}
.bar-ci::before {{ left: 0; }}
.bar-ci::after {{ right: 0; }}
.bar-point {{
  position: absolute;
  top: 7px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  transform: translateX(-50%);
  {_paint("background", light["kicker"], dark["kicker"])}
  z-index: 3;
}}
.bar-value {{
  font-variant-numeric: tabular-nums;
  font-weight: 700;
  text-align: right;
  {_paint("color", light["ink"], dark["ink"])}
}}
.chart-note, .score-detail p {{
  {_paint("color", light["pitch"], dark["pitch"])}
  font-size: 0.9rem;
}}
.stApp .chart .chart-note {{
  margin: 0.65rem 0 0;
  max-width: 75ch;
  font-size: 0.82rem;
  line-height: 1.5;
  {_paint("color", light["meta"], dark["meta"])}
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
.stApp [data-testid="stTextInput"]:focus-within [data-testid="stTextInputRootElement"],
.stApp [data-testid="stNumberInput"]:focus-within [data-testid="stNumberInputRootElement"] {{
  border-color: #392975 !important;
  border-color: light-dark(#392975, #d4c6ff) !important;
}}
.stApp [data-testid="stExpander"] {{
  {_paint("background", light["card"], dark["card"])}
  {_line("border", light["border"], dark["border"])}
}}
.stApp [data-testid="stExpander"] details summary,
.stApp [data-testid="stExpander"] details summary p {{
  {_paint("color", light["kicker"], dark["kicker"])}
}}
.pick-poster {{
  flex-direction: column;
  align-items: stretch;
  gap: 0.65rem;
}}
.stApp .pick-poster .poster {{
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 2 / 3;
  min-height: 0;
  margin: 0;
  object-fit: cover;
}}
.stApp .pick-poster .poster.placeholder {{
  display: flex;
}}
.pick-rich .poster,
.pick-rich .poster.placeholder {{
  width: 132px;
  height: 198px;
}}
.pick-compact {{
  align-items: center;
}}
.pick-compact .poster,
.pick-compact .poster.placeholder {{
  width: 108px;
  height: 162px;
}}
.stApp .pick-poster .pick-title {{
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  font-size: 1rem;
  line-height: 1.35;
  min-height: 2.7em;
  margin: 0;
}}
.stApp .pick-poster .pick-meta {{
  font-size: 0.8125rem;
  line-height: 1.5;
  margin: 0.4rem 0 0;
  {_paint("color", light["meta"], dark["meta"])}
}}
.pick-compact .pick-why {{
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
}}
.rated-line {{
  display: grid;
  grid-template-columns: 40px minmax(0, 1fr);
  gap: 0.55rem;
  align-items: center;
  min-width: 0;
}}
.rated-line .poster {{
  width: 40px;
  height: 60px;
  font-size: 0.85rem;
}}
.rated-line .pick-title {{
  font-size: 0.95rem;
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.rated-line .history-stars {{
  margin: 0.1rem 0 0;
}}
[class*="st-key-film-"],
[class*="st-key-rec-"] {{
  {_paint("background", light["card"], dark["card"])}
  {_line("border", light["border"], dark["border"])}
  border-radius: 12px;
  padding: 0.5rem 0 0.3rem;
  height: 100%;
  box-sizing: border-box;
  container-type: inline-size;
  container-name: film;
  min-width: 0;
}}
[class*="st-key-rec-"] {{
  padding: 0.65rem 0.7rem 0.45rem;
}}
[class*="st-key-film-"] {{
  display: flex !important;
  flex-direction: column !important;
  padding: 0.75rem;
  gap: 0.65rem;
}}
[class*="st-key-film-"] > [class*="st-key-rate-"] {{
  margin-top: auto;
}}
[class*="st-key-film-"] .pick,
[class*="st-key-rec-"] .pick {{
  border-top: 0;
  padding: 0.15rem 0.15rem 0.35rem;
}}
[class*="st-key-film-"] .pick {{
  padding: 0;
}}
.stApp [class*="st-key-film-"] [data-testid="stPopoverButton"] {{
  min-height: 2.5rem;
  padding: 0.4rem 0.65rem;
  border-radius: 8px;
}}
[class*="st-key-rate-"] {{
  width: max-content;
  max-width: 100%;
  margin-left: auto;
  margin-right: auto;
}}
[class*="st-key-rate-"] [data-testid="stButtonGroup"] {{
  width: max-content;
  max-width: 100%;
}}
[class*="st-key-rate-"] [role="radiogroup"] {{
  gap: 4px !important;
  column-gap: 4px !important;
  width: max-content !important;
  max-width: 100%;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
}}
[class*="st-key-rate-"] button {{
  position: relative;
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
  width: 44px !important;
  min-width: 44px !important;
  max-width: 44px !important;
  height: 44px !important;
  min-height: 44px !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 8px !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: hidden;
}}
[class*="st-key-rate-"] button [data-testid="stMarkdownContainer"] {{
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip-path: inset(50%);
  white-space: nowrap !important;
}}
[class*="st-key-rate-"] button::before {{
  content: "";
  display: block;
  width: 28px;
  height: 28px;
  background: {light["meta"]};
  background: light-dark({light["meta"]}, {dark["meta"]});
  -webkit-mask: url("{_STAR_EMPTY}") center / contain no-repeat;
  mask: url("{_STAR_EMPTY}") center / contain no-repeat;
}}
[class*="st-key-rate-"] button:has(~ button[data-selected="true"])::before,
[class*="st-key-rate-"] button[data-selected="true"]::before,
[class*="st-key-rate-"] button:hover::before,
[class*="st-key-rate-"] button:has(~ button:hover)::before,
[class*="st-key-rate-"] button[data-hovered="true"]::before,
[class*="st-key-rate-"] button:has(~ button[data-hovered="true"])::before {{
  background: {light["kicker"]};
  background: light-dark({light["kicker"]}, {dark["kicker"]});
  -webkit-mask-image: url("{_STAR_FILLED}");
  mask-image: url("{_STAR_FILLED}");
}}
[class*="st-key-rate-"] button:focus-visible {{
  outline: 2px solid {light["kicker"]};
  outline: 2px solid light-dark({light["kicker"]}, {dark["kicker"]});
  outline-offset: 2px;
}}
@container film (max-width: 220px) {{
  [class*="st-key-film-"] > [class*="st-key-rate-"],
  [class*="st-key-rate-"],
  [class*="st-key-rate-"] [data-testid="stButtonGroup"],
  [class*="st-key-rate-"] [role="radiogroup"] {{
    width: 100% !important;
    max-width: 100% !important;
  }}
  [class*="st-key-rate-"] [role="radiogroup"] {{
    display: flex !important;
    justify-content: center !important;
  }}
  [class*="st-key-rate-"] button {{
    flex: 1 1 0 !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: none !important;
    height: 44px !important;
    min-height: 44px !important;
  }}
  [class*="st-key-rate-"] button::before {{
    width: 26px;
    height: 26px;
    -webkit-mask-size: contain;
    mask-size: contain;
  }}
}}
[class*="st-key-why-"] {{
  width: max-content;
  max-width: 100%;
}}
[class*="st-key-why-"] [data-testid="stExpander"] {{
  border: 0;
  background: transparent;
}}
[class*="st-key-why-"] details summary {{
  min-height: 0 !important;
  padding: 0.1rem 0 !important;
  font-size: 0.85rem;
}}
[data-testid="stVerticalBlock"][class*="st-key-rated-"] {{
  flex-direction: row !important;
  align-items: center !important;
  gap: 0.35rem !important;
  flex-wrap: nowrap !important;
  width: 100%;
  max-width: 100%;
  min-width: 0;
}}
[data-testid="stVerticalBlock"][class*="st-key-rated-"]
  > [data-testid="stElementContainer"]:first-child {{
  flex: 1 1 auto;
  min-width: 0;
}}
[data-testid="stVerticalBlock"][class*="st-key-rated-"]
  > [data-testid="stElementContainer"]:last-child {{
  flex: 0 0 auto;
  width: auto !important;
}}
[class*="st-key-rated-"] button {{
  width: 44px !important;
  min-width: 44px !important;
  max-width: 44px !important;
  height: 44px !important;
  min-height: 44px !important;
  padding: 0 !important;
  overflow: hidden;
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}}
[class*="st-key-rated-"] button p {{
  font-size: 0 !important;
  line-height: 0;
  margin: 0;
}}
[class*="st-key-rated-"] button p::before {{
  content: "×";
  font-size: 1.35rem;
  line-height: 1;
  color: {light["ink"]};
  color: light-dark({light["ink"]}, {dark["ink"]});
}}
[class*="st-key-rated-"] button:hover p::before {{
  color: {light["kicker"]};
  color: light-dark({light["kicker"]}, {dark["kicker"]});
}}
[data-testid="stVerticalBlock"].st-key-ticket {{
  {_paint("background", light["card"], dark["card"])}
  {_line("border", light["border"], dark["border"])}
  border-top: 3px solid {light["kicker"]};
  border-top-color: {light["kicker"]};
  border-top-color: light-dark({light["kicker"]}, {dark["kicker"]});
  border-radius: 10px;
  padding: 0.9rem 0.95rem 1rem;
  position: sticky;
  top: 7.5rem;
  flex: 0 0 auto !important;
  align-self: flex-start !important;
  max-height: calc(100vh - 8.5rem);
  overflow: auto;
  width: 100%;
  box-sizing: border-box;
}}
.st-key-workbench [data-testid="stColumn"]:has(.st-key-ticket) {{
  order: 2;
}}
.st-key-workbench [data-testid="stColumn"]:has(.st-key-ticket)
  [data-testid="stLayoutWrapper"]:has(> .st-key-ticket) {{
  flex: 1 1 auto !important;
  align-items: flex-start !important;
  width: 100%;
}}
.st-key-workbench [data-testid="stHorizontalBlock"] {{
  align-items: stretch !important;
}}
.st-key-workbench [data-testid="stColumn"] {{
  overflow: visible !important;
}}
.st-key-viewer-bar [data-testid="stHorizontalBlock"] {{
  justify-content: flex-start !important;
  align-items: flex-end !important;
  gap: 0.75rem !important;
  flex-wrap: wrap !important;
}}
.st-key-viewer-bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
  width: 240px !important;
  max-width: 100% !important;
  flex: 0 1 240px !important;
  min-width: 0 !important;
}}
.st-key-viewer-bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
  width: auto !important;
  min-width: 0 !important;
  flex: 0 0 auto !important;
}}
.st-key-starters [data-testid="stHorizontalBlock"],
.st-key-tries [data-testid="stHorizontalBlock"] {{
  justify-content: flex-start !important;
  gap: 0.5rem !important;
  flex-wrap: wrap !important;
}}
.st-key-starters [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
.st-key-tries [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
  width: auto !important;
  flex: 0 0 auto !important;
  min-width: 0 !important;
}}
.st-key-compare [data-testid="stHorizontalBlock"] {{
  align-items: flex-start !important;
}}
.st-key-compare [class*="st-key-rec-"] {{
  height: auto;
}}
.st-key-compare-switch {{
  display: none;
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
.stApp [data-testid="stButtonGroup"] button[aria-pressed="true"],
.stApp [data-testid="stButtonGroup"] button[data-selected="true"] {{
  background: #392975;
  background: light-dark(#392975, #6b59c4);
  color: #ffffff;
  border-color: #392975;
  border-color: light-dark(#392975, #6b59c4);
}}
.stApp [data-testid="stButtonGroup"] button[aria-pressed="true"] p,
.stApp [data-testid="stButtonGroup"] button[data-selected="true"] p {{
  color: #ffffff;
}}
.stApp [data-testid="stCheckbox"] label > div:not([data-testid]):has(svg) {{
  background: #392975 !important;
  background: light-dark(#392975, #6b59c4) !important;
  border-color: #392975 !important;
  border-color: light-dark(#392975, #6b59c4) !important;
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
.st-key-shelf-frame,
.st-key-matches-frame {{
  container-type: inline-size;
  container-name: gridframe;
}}
[data-testid="stVerticalBlock"].st-key-shelf,
[data-testid="stVerticalBlock"].st-key-matches,
.st-key-shelf > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
.st-key-matches > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
.st-key-shelf > [data-testid="stVerticalBlock"],
.st-key-matches > [data-testid="stVerticalBlock"] {{
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  align-items: stretch;
}}
@container gridframe (min-width: 680px) {{
  [data-testid="stVerticalBlock"].st-key-shelf,
  [data-testid="stVerticalBlock"].st-key-matches {{
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }}
}}
@container gridframe (max-width: 480px) {{
  [data-testid="stVerticalBlock"].st-key-shelf,
  [data-testid="stVerticalBlock"].st-key-matches {{
    grid-template-columns: minmax(0, 1fr);
  }}
  .pick-poster {{ flex-direction: row; }}
  .stApp .pick-poster .poster {{
    width: 76px;
    height: 114px;
    flex: 0 0 76px;
  }}
  .stApp .pick-poster .pick-title {{ min-height: 0; }}
}}
@container gridframe (min-width: 920px) {{
  [data-testid="stVerticalBlock"].st-key-shelf,
  [data-testid="stVerticalBlock"].st-key-matches {{
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }}
}}
@container gridframe (min-width: 1160px) {{
  [data-testid="stVerticalBlock"].st-key-shelf,
  [data-testid="stVerticalBlock"].st-key-matches {{
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }}
}}
[data-testid="stVerticalBlock"].st-key-picks,
[data-testid="stVerticalBlock"].st-key-top10,
.st-key-picks > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
.st-key-top10 > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
.st-key-picks > [data-testid="stVerticalBlock"],
.st-key-top10 > [data-testid="stVerticalBlock"] {{
  display: grid !important;
  grid-template-columns: minmax(0, 1fr);
  gap: 0.85rem;
  align-items: stretch;
}}
@media (min-width: 900px) {{
  [data-testid="stVerticalBlock"].st-key-picks,
  [data-testid="stVerticalBlock"].st-key-top10,
  .st-key-picks > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
  .st-key-top10 > [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"],
  .st-key-picks > [data-testid="stVerticalBlock"],
  .st-key-top10 > [data-testid="stVerticalBlock"] {{
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }}
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
  .st-key-viewer-bar [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; }}
  .st-key-viewer-bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
    width: 11.5rem !important;
    min-width: 11.5rem !important;
    flex: 0 0 11.5rem !important;
  }}
  .st-key-workbench [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; }}
  .st-key-workbench [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }}
  .st-key-workbench [data-testid="stColumn"]:has(.st-key-ticket) {{
    order: 0;
  }}
  .st-key-starters [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
  .st-key-tries [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    width: auto !important;
    flex: 0 0 auto !important;
    min-width: max-content !important;
  }}
  .st-key-ticket,
  [data-testid="stVerticalBlock"].st-key-ticket {{
    position: static !important;
    top: auto !important;
    max-height: none !important;
  }}
  .intro {{ grid-template-columns: minmax(0, 1fr); }}
  .intro-art {{ display: none; }}
  .bar-row {{ grid-template-columns: minmax(0, 1fr) auto; row-gap: 0.25rem; }}
  .bar-track {{ grid-column: 1 / -1; grid-row: 2; }}
  .bar-value {{ grid-column: 2; grid-row: 1; }}
  .chart-axis {{ display: block; }}
  [data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {{ z-index: 60; }}
  .st-key-ticket,
  [data-testid="stVerticalBlock"].st-key-ticket {{ z-index: 0 !important; }}
  .funnel {{ flex-direction: column; align-items: stretch; }}
  .stApp .funnel > .funnel-step {{ flex: 1 1 auto; }}
  .funnel-arrow {{ transform: rotate(90deg); align-self: center; }}
  .neighbour-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .because-line {{ align-items: flex-start; }}
  .bar-name {{ grid-column: 1; }}
  .st-key-compare-switch {{ display: block; }}
  .st-key-compare [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
    display: none !important;
    width: 100% !important;
    flex: 1 1 100% !important;
    min-width: 0 !important;
  }}
  .stApp:has(.compare-pick[data-mode="production"])
    .st-key-compare [data-testid="stColumn"]:has(.st-key-col-production),
  .stApp:has(.compare-pick[data-mode="item_item"])
    .st-key-compare [data-testid="stColumn"]:has(.st-key-col-item-item),
  .stApp:has(.compare-pick[data-mode="two_tower"])
    .st-key-compare [data-testid="stColumn"]:has(.st-key-col-two-tower) {{
    display: block !important;
  }}
}}
</style>
"""


THEME_CSS = _theme_css()


def section_scroll_script() -> str:
    """A section change starts at its beginning, even after a long pick list."""
    return """<script>
const main = window.parent.document.querySelector('[data-testid="stMain"]');
if (main) main.scrollTo({top: 0, behavior: "auto"});
</script>"""

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


def _poster_hue(title: str) -> int:
    """Same mixing as ``posters.poster_hue``. Neighbours do not share a gradient."""
    value = 17
    for char in str(title):
        value = (value * 33 + ord(char)) & 0xFFFFFFFF
    return value % 360


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


def poster_html(title: str, poster_url: str | None, *, extra_class: str = "") -> str:
    """An ``<img>`` for a URL, or initials on a gradient. Neither path reads image bytes."""
    classes = "poster" + (f" {extra_class}" if extra_class else "")
    if poster_url:
        src = html.escape(poster_url, quote=True)
        alt = html.escape(title, quote=True)
        # A failed load drops its src once. The browser does not request it again.
        return (
            f'<img class="{classes}" alt="" data-title="{alt}" width="76" height="114" '
            f'loading="lazy" decoding="async" referrerpolicy="no-referrer" src="{src}" '
            "onerror=\"this.onerror=null;this.removeAttribute('src');"
            "this.classList.add('is-broken');\">"
        )
    label = html.escape(title)
    hue = _poster_hue(title)
    hue_b = (hue * 5 + 83) % 360
    mark = html.escape(initials(title))
    return (
        f'<div class="{classes} placeholder" role="img" aria-label="No poster for {label}" '
        f'style="--poster-hue:{hue};--poster-hue-b:{hue_b}"><span>{mark}</span></div>'
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
    kind: str = "row",
    evidence_html: str = "",
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
        items = []
        for index, text in enumerate(reasons):
            thumb = evidence_html if index == 0 and text.startswith("Because you liked") else ""
            items.append(f'<li class="because-line">{thumb}{html.escape(text)}</li>')
        why = f'<ul class="pick-why">{"".join(items)}</ul>'
    rank_html = ""
    if rank is not None:
        rank_html = f'<p class="pick-rank">{int(rank):02d}</p>'
    style = kind if kind in {"poster", "rich", "compact", "row"} else "row"
    return (
        f'<div class="pick pick-{style}">'
        f"{poster_html(title, poster_url)}"
        '<div class="pick-body">'
        f"{rank_html}"
        f'<p class="pick-title">{html.escape(title)}</p>'
        f'<p class="pick-meta">{html.escape(meta)}</p>'
        f"{why}"
        "</div></div>"
    )


def rated_line_html(
    *,
    title: str,
    rating: int,
    poster_url: str | None,
) -> str:
    """A compact rating row: thumb, title, and the stars already given."""
    filled = max(0, min(5, int(rating)))
    stars = "★" * filled + "☆" * (5 - filled)
    return (
        '<div class="rated-line">'
        f"{poster_html(title, poster_url)}"
        "<div>"
        f'<p class="pick-title">{html.escape(title)}</p>'
        f'<p class="history-stars" aria-hidden="true">{stars}</p>'
        "</div></div>"
    )


def neighbour_grid_html(films: list[dict]) -> str:
    """Small posters for films already chosen. No extra requests beyond those URLs."""
    cells = []
    for film in films:
        title = str(film.get("title") or "")
        year = _year_text(film.get("year"))
        label = f"{title} ({year})" if year else title
        cells.append(
            '<div class="neighbour">'
            f"{poster_html(title, film.get('poster_url'))}"
            f"<p>{html.escape(label)}</p></div>"
        )
    return f'<div class="neighbour-grid">{"".join(cells)}</div>'


_ICONS = {
    "film": '<rect x="3" y="3" width="18" height="18" rx="2"/>'
            '<path d="M7 3v18M17 3v18M3 8h4M3 16h4M17 8h4M17 16h4"/>',
    "search": '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    "rank": '<path d="M4 5h16M4 12h11M4 19h6M18 10v10m-3-3 3 3 3-3"/>',
    "list": '<path d="m3 6 1.5 1.5L7 5M10 6h11M3 13l1.5 1.5L7 12M10 13h11M3 20h4M10 20h11"/>',
    "chart": '<path d="M4 3v18h17M8 16v-5M13 16V7M18 16V4"/>',
    "star": '<path d="m12 3 2.8 5.7 6.3.9-4.6 4.5 1.1 6.3'
            'L12 17.4l-5.6 3 1.1-6.3L2.9 9.6l6.3-.9z"/>',
    "users": '<circle cx="9" cy="7" r="3"/><path d="M3 21v-3a6 6 0 0 1 12 0v3'
             'M16 4a3 3 0 0 1 0 6M21 21v-3a6 6 0 0 0-4-5.7"/>',
    "compass": '<circle cx="12" cy="12" r="9"/><path d="m16 8-2.5 5.5L8 16l2.5-5.5z"/>',
    "route": '<circle cx="5" cy="5" r="2"/><circle cx="19" cy="19" r="2"/>'
             '<path d="M7 5h10a4 4 0 0 1 0 8H7a4 4 0 0 0 0 8h5"/>',
    "history": '<path d="M3 4v5h5M3.5 9a9 9 0 1 1-.2 5M12 7v5l3 2"/>',
    "arrow": '<path d="M4 12h16m-6-6 6 6-6 6"/>',
    "score": '<circle cx="12" cy="12" r="3" fill="currentColor"/>',
    "range": '<path d="M3 12h18M3 7v10M21 7v10"/>',
}


def icon_html(name: str) -> str:
    """Decorative icons share one stroke and inherit the active palette."""
    return (
        '<svg class="ui-icon" viewBox="0 0 24 24" width="24" height="24" '
        'fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{_ICONS[name]}</svg>'
    )


def funnel_html(steps: list[dict[str, str]]) -> str:
    """An explained process. Counts are shown only when supplied by the caller."""
    items: list[str] = []
    icons = ("film", "search", "rank", "list")
    for index, step in enumerate(steps):
        if index:
            items.append(f'<li class="funnel-arrow" aria-hidden="true">{icon_html("arrow")}</li>')
        detail = str(step.get("detail") or "")
        count = f'<p class="funnel-count">{html.escape(detail)}</p>' if detail else ""
        items.append(
            '<li class="funnel-step">'
            f'<div class="funnel-heading">{icon_html(icons[min(index, 3)])}'
            f"<strong>{html.escape(step.get('name') or '')}</strong></div>"
            f'<p class="funnel-description">{html.escape(step.get("description") or "")}</p>'
            f"{count}"
            "</li>"
        )
    return f'<ol class="funnel">{"".join(items)}</ol>'


def metric_chart_html(rows: list[dict], *, caption: str) -> str:
    """Zero-based bars, score dots, and thin interval whiskers from stored floats."""
    if not rows:
        return ""
    highest = max(max(float(row["high"]), float(row["ndcg"])) for row in rows)
    scale = max(0.05, math.ceil(highest / 0.05) * 0.05)
    divisions = round(scale / 0.05) if scale <= 0.2 else 4
    ticks = "".join(
        f"<span>{scale * index / divisions:.3f}</span>" for index in range(divisions + 1)
    )
    parts = [
        '<figure class="chart">'
        '<figcaption class="chart-legend">'
        '<span>Ranking score · higher is better</span>'
        f'<span class="legend-score">{icon_html("score")}Score</span>'
        f'<span class="legend-range">{icon_html("range")}Uncertainty range</span></figcaption>'
        f'<div class="chart-axis" aria-hidden="true"><span></span>'
        f'<div class="axis-ticks">{ticks}</div><span></span></div>'
        '<div role="list" aria-label="Model ranking scores">'
    ]
    last_group = None
    for row in rows:
        group = str(row.get("group") or "")
        if group and group != last_group:
            parts.append(f'<p class="chart-group">{html.escape(group)}</p>')
            last_group = group
        ndcg = float(row["ndcg"])
        low = float(row["low"])
        high = float(row["high"])
        width = max(0.0, min(100.0, ndcg / scale * 100.0))
        left = max(0.0, min(100.0, low / scale * 100.0))
        right = max(0.0, min(100.0, high / scale * 100.0))
        best = " is-best" if row.get("best") else ""
        label = html.escape(str(row["label"]))
        value = format(ndcg, ".3f")
        low_s = format(low, ".3f")
        high_s = format(high, ".3f")
        tip = html.escape(
            f"{row['label']} ({row['model']}): {value}, range {low_s} to {high_s}",
            quote=True,
        )
        parts.append(
            f'<div class="bar-row{best}" role="listitem" aria-label="{tip}">'
            f'<div class="bar-name">{label}'
            + ('<span class="best-label">Highest shown score</span>' if best else '')
            + '</div>'
            f'<div class="bar-track" title="{tip}" aria-hidden="true">'
            f'<span class="bar-fill" style="width:{width:.2f}%"></span>'
            '<span class="bar-ci" style="left:'
            f'{left:.2f}%;width:{max(0.0, right - left):.2f}%"></span>'
            f'<span class="bar-point" style="left:{width:.2f}%"></span>'
            "</div>"
            f'<div class="bar-value">{value}</div>'
            "</div>"
        )
    parts.append(
        f'</div><p class="chart-note">{html.escape(caption)} '
        f'Displayed scale: 0–{scale:.3f}; the metric ranges from 0 to 1.</p></figure>'
    )
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
    elif os.environ.get("MOVIELENS_FAKE_POSTERS", "").strip() == "1":
        extra = (
            "Poster pictures on this page are generated locally for layout checks. "
            "No poster request is sent without it."
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
