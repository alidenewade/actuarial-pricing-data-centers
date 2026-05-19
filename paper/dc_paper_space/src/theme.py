"""Intelligent Actuaries palette + theme-aware Plotly helper.

The IA visual identity: bone background, deep warm near-black headings,
burnt-sienna accents. Used in both the paper PDF and this Space so the
two stay visually consistent.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# IA palette — must match dc_paper.tex \definecolor block.
IA_BONE   = "#FAFAF7"
IA_DARK   = "#1B1815"
IA_ACCENT = "#A04A1F"
IA_TAUPE  = "#F2EDE6"
IA_RULE   = "#3A332D"
IA_GREY   = "#5A524A"

# Supporting palette for multi-series plots.
IA_BLUE   = "#1F4F73"
IA_TEAL   = "#0E7C7B"
IA_OLIVE  = "#6B6E3B"
IA_PLUM   = "#5B3A4A"

SERIES_COLORS = [IA_ACCENT, IA_BLUE, IA_TEAL, IA_OLIVE, IA_PLUM, IA_RULE]


def _is_dark_theme() -> bool:
    """Detect whether Streamlit is in dark mode."""
    try:
        base = st.get_option("theme.base") or ""
    except Exception:
        base = ""
    return base.lower() == "dark"


def apply_theme(fig: go.Figure, *, ytype: str | None = None) -> go.Figure:
    """Stamp the IA look on a Plotly figure.

    - Transparent paper / plot so the Streamlit container colour shows
      through (bone in light mode, near-black in dark mode).
    - Readable text + grid colours per OS theme.
    - Optional yaxis type override (e.g. 'log').
    """
    dark = _is_dark_theme()
    text   = "rgba(250,250,250,0.94)" if dark else IA_DARK
    grid   = "rgba(255,255,255,0.10)" if dark else "rgba(27,24,21,0.10)"
    legend = "rgba(0,0,0,0)"

    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Helvetica, Arial, sans-serif", color=text, size=12),
        legend=dict(bgcolor=legend, bordercolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=text),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid, linecolor=text),
    )
    if ytype is not None:
        fig.update_yaxes(type=ytype)
    return fig
