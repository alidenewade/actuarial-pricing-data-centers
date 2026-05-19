"""Hyperscale Data-Center Risk Simulation — Streamlit Space.

Companion to Denewade (2026), Paper 1 of the Intelligent Actuaries
research series. All adjustment widgets live in the left sidebar,
organised by simulation mode. The main area is a segmented-control
tab strip (Reliability / Compound loss / OEP / Pricing / Compare /
Figures). Every slider you move recomputes a live simulation; no
precomputed results.

UI chrome mirrors intelligentactuaries/nanoeconomics-simulation so the
two Spaces read as siblings of the same research lab — theme-aware CSS
via prefers-color-scheme so the page looks correct under both light and
dark Streamlit themes, IA bone / sienna palette applied to Plotly via
`src.theme.apply_theme`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.pricing import Loadings, all_premiums
from src.reliability import (
    MarkovParams,
    Q_matrix,
    TIER_CALIBRATION,
    availability,
    claim_frequency_per_year,
    mtbf_hours,
    simulate_markov_walk,
    stationary_distribution,
)
from src.severity import (
    gpd_tvar,
    simulate_annual_losses,
)
from src.theme import (
    IA_ACCENT,
    IA_DARK,
    IA_GREY,
    IA_RULE,
    SERIES_COLORS,
    apply_theme,
)


# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Hyperscale DC Risk Simulation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Hyperscale Data-Center Risk Simulation — companion to "
            "Denewade (2026), *A Coupled Power-Thermal-Cyber Framework "
            "for the Actuarial Pricing and Insurance of Hyperscale Data "
            "Centers*. Source: github.com/alidenewade/actuarial-pricing-data-centers"
        ),
    },
)


# ─────────────────────────────────────────────────────────────
# Custom CSS — focused, light-touch. Mirrors the nanoeconomics-
# simulation chrome and the IA bone / sienna palette from the
# paper PDF. Less !important, fewer overlapping selectors.
# ─────────────────────────────────────────────────────────────

st.markdown(
    f"""
<style>
    /* Page chrome */
    .block-container {{
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 1320px;
    }}
    #MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden; }}

    /* Typography — inherit colour from the active theme, only nudge weight */
    h1 {{ font-weight: 700; letter-spacing: -0.02em; }}
    h2 {{ font-weight: 650; letter-spacing: -0.01em; }}
    h3 {{ font-weight: 600; }}
    h5 {{
        font-weight: 600; text-transform: uppercase;
        font-size: 0.74rem; letter-spacing: 0.10em;
        opacity: 0.78; margin: 1.25rem 0 0.35rem;
    }}

    /* Tagline pill below the title */
    .ia-tagline {{
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.28rem 0.7rem; margin-top: 0.35rem;
        border: 1px solid rgba(160, 74, 31, 0.28);
        border-radius: 999px;
        background: rgba(160, 74, 31, 0.06);
        font-size: 0.82rem; color: {IA_ACCENT}; font-weight: 600;
    }}
    .ia-tagline .dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: {IA_ACCENT}; display: inline-block;
    }}

    /* Header link row */
    .ia-headlinks {{ text-align: right; padding-top: 0.7rem; }}
    .ia-headlinks a {{
        color: inherit; opacity: 0.78;
        text-decoration: none; font-size: 0.88rem; font-weight: 500;
        margin-left: 1.1rem;
        border-bottom: 1px solid transparent;
        transition: opacity 0.15s ease, border-color 0.15s ease;
    }}
    .ia-headlinks a:hover {{
        opacity: 1; border-bottom-color: {IA_ACCENT};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{ border-right: 1px solid rgba(58, 51, 45, 0.20); }}
    [data-testid="stSidebar"] .block-container {{
        padding-top: 1.1rem; padding-bottom: 2rem;
    }}
    [data-testid="stSidebar"] hr {{ margin: 0.7rem 0; opacity: 0.5; }}
    [data-testid="stSidebar"] .stSlider {{ padding: 0.05rem 0; }}
    [data-testid="stSidebar"] label p {{
        font-size: 0.84rem; font-weight: 500; opacity: 0.92;
    }}

    /* Metric cards — IA tint, compact */
    [data-testid="stMetric"] {{
        background: rgba(160, 74, 31, 0.06);
        border: 1px solid rgba(160, 74, 31, 0.20);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 0.70rem; letter-spacing: 0.08em;
        text-transform: uppercase; font-weight: 600; opacity: 0.80;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 700; }}
    [data-testid="stMetricDelta"] div {{ font-weight: 500; opacity: 0.95; }}

    /* DataFrame */
    [data-testid="stDataFrame"] thead th {{
        background-color: rgba(160, 74, 31, 0.08);
        font-weight: 700;
    }}

    /* Segmented control as full-width tabs */
    .main [data-testid="stSegmentedControl"] > div {{ width: 100%; gap: 0; }}
    .main [data-testid="stSegmentedControl"] label {{
        flex: 1 1 0; text-align: center;
        padding: 0.5rem 0.8rem;
        font-size: 0.92rem; font-weight: 500;
    }}

    /* Plotly: theme-aware text contrast via OS preference */
    @media (prefers-color-scheme: light) {{
        .plotly svg text,
        .plotly .legendtext, .plotly .gtitle,
        .plotly .annotation-text {{ fill: {IA_DARK} !important; }}
        .plotly .gridlayer path {{ stroke: rgba(27, 24, 21, 0.10) !important; }}
    }}
    @media (prefers-color-scheme: dark) {{
        .plotly svg text,
        .plotly .legendtext, .plotly .gtitle,
        .plotly .annotation-text {{ fill: rgba(250, 250, 250, 0.94) !important; }}
        .plotly .gridlayer path {{ stroke: rgba(255, 255, 255, 0.14) !important; }}
    }}

    /* Plotly animation controls — sit them on a subtle pill so they read
       as a UI element rather than floating text. */
    .plotly .updatemenu-button {{
        font-weight: 600 !important;
    }}
    .plotly .updatemenu-button rect.updatemenu-item-rect {{
        rx: 6px; ry: 6px;
    }}

    /* Plotly iframe wrapper transparent so the page bg shows through */
    iframe[srcdoc] {{ background: transparent !important; }}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
# Main page header
# ─────────────────────────────────────────────────────────────

header_l, header_r = st.columns([5, 2], vertical_alignment="bottom")
with header_l:
    st.title("Hyperscale Data-Center Risk Simulation")
    st.markdown(
        '<span class="ia-tagline"><span class="dot"></span>'
        "Power-thermal-cyber coupled framework · Denewade 2026"
        "</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Intelligent Actuaries research series, Paper 1 · "
        "DOI [10.5281/zenodo.20279225](https://doi.org/10.5281/zenodo.20279225)"
    )
with header_r:
    st.markdown(
        '<div class="ia-headlinks">'
        '<a href="https://intelligentactuaries.com/research/data-centers" '
        'target="_blank">Paper ↗</a>'
        '<a href="https://github.com/alidenewade/actuarial-pricing-data-centers" '
        'target="_blank">GitHub ↗</a>'
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# Mode tabs — segmented control acting as full-width tabs.
# Sidebar controls + main view filter on `mode`.
# ─────────────────────────────────────────────────────────────

MODE_RELIA = "Reliability"
MODE_LOSS = "Compound loss"
MODE_OEP = "OEP curve"
MODE_PRICE = "Pricing"
MODE_ECON = "Economics"
MODE_COMPARE = "Compare"
MODE_FIGS = "Figures"

MODES = [MODE_RELIA, MODE_LOSS, MODE_OEP, MODE_PRICE, MODE_ECON, MODE_COMPARE, MODE_FIGS]
MODE_ICON = {
    MODE_RELIA:   "🏗️",
    MODE_LOSS:    "📉",
    MODE_OEP:     "🌪️",
    MODE_PRICE:   "💰",
    MODE_ECON:    "🏛️",
    MODE_COMPARE: "⚖️",
    MODE_FIGS:    "🖼️",
}

mode = (
    st.segmented_control(
        "Mode",
        MODES,
        default=MODE_RELIA,
        format_func=lambda m: f"{MODE_ICON[m]}  {m}",
        label_visibility="collapsed",
        key="active_mode",
    )
    or MODE_RELIA
)

st.divider()


# ─────────────────────────────────────────────────────────────
# Sidebar — controls conditional on `mode`
# ─────────────────────────────────────────────────────────────

def _section(label: str) -> None:
    st.markdown(f"##### {label}")


with st.sidebar:
    st.markdown(f"### ⚙️  {MODE_ICON[mode]}  {mode}")
    st.caption("Adjust parameters · output recomputes live")
    st.divider()

    if mode == MODE_RELIA:
        _section("Topology")
        tier_name = st.selectbox(
            "Uptime Institute Tier",
            list(TIER_CALIBRATION.keys()),
            index=3,
            key="r_tier",
        )
        tier = TIER_CALIBRATION[tier_name]
        st.caption(f"Target availability A = **{tier['avail']*100:.3f}%**")

        _section("Component rates (per hour)")
        r_lam = st.slider("λ — per-train degradation",
                          1e-6, 5e-3, 5e-4, 5e-6, format="%.1e", key="r_lam")
        r_mu = st.slider("μ — per-train repair",
                         0.005, 0.5, tier["mu"], 0.005, key="r_mu")
        r_lam_f = st.slider("λ_f — degraded → failed",
                            1e-4, 5e-2, 5e-3, 1e-4, format="%.1e", key="r_lam_f")
        r_mu_r = st.slider("μ_r — full-site restore",
                           0.005, 0.5, tier["mu_r"], 0.005, key="r_mu_r")

        _section("Live trajectory")
        r_walk_years = st.slider("horizon (years)", 1, 10, 5, 1, key="r_walk_years")
        r_walk_seed = st.number_input("seed", 0, 999_999, 7, 1, key="r_walk_seed")

    elif mode == MODE_LOSS:
        _section("Frequency  (NB-Gamma)")
        l_lam = st.slider("λ — expected outages / year",
                          0.1, 10.0, 2.5, 0.1, key="l_lam")
        l_nu = st.slider("ν — NB dispersion (Gamma frailty)",
                         0.5, 20.0, 4.0, 0.5, key="l_nu")

        _section("Body  (lognormal)")
        l_body_mu = st.slider("μ_body  (log-scale)",
                              1.0, 5.0, 2.5, 0.1, key="l_body_mu")
        l_body_sig = st.slider("σ_body",
                               0.2, 2.0, 0.9, 0.05, key="l_body_sig")

        _section("Tail  (GPD POT)")
        l_tail_xi = st.slider("ξ — GPD shape", 0.0, 0.9, 0.30, 0.01, key="l_tail_xi")
        l_tail_sig = st.slider("σ_tail — GPD scale", 1.0, 50.0, 5.0, 0.5, key="l_tail_sig")
        l_tail_u = st.slider("u — threshold ($M)", 0.0, 100.0, 25.0, 5.0, key="l_tail_u")
        l_tail_frac = st.slider("π_tail — share of events on tail",
                                0.0, 1.0, 0.10, 0.01, key="l_tail_frac")

        _section("Simulation")
        st.caption(
            "Each simulated draw is one independent annual aggregate loss "
            "from the model, **not** a real calendar year of plant operation. "
            "More draws → less Monte Carlo noise."
        )
        l_n_years = st.select_slider(
            "Number of simulated annual losses",
            options=[500, 1_000, 5_000, 10_000, 25_000, 50_000],
            value=5_000, key="l_n_years",
        )
        l_seed = st.number_input("Random seed", 0, 999_999, 42, 1, key="l_seed")

    elif mode == MODE_OEP:
        _section("Engineering anchor")
        o_lam = st.slider("λ — annual outage frequency",
                          0.1, 5.0, 1.5, 0.1, key="o_lam")
        o_nu = st.slider("ν — dispersion", 0.5, 20.0, 4.0, 0.5, key="o_nu")

        _section("Severity")
        o_body_mu = st.slider("μ_body (log-scale)", 1.0, 5.0, 2.8, 0.1, key="o_body_mu")
        o_body_sig = st.slider("σ_body", 0.2, 2.0, 1.1, 0.05, key="o_body_sig")
        o_xi = st.slider("ξ — GPD shape", 0.0, 0.9, 0.35, 0.01, key="o_xi")
        o_sig = st.slider("σ_tail — GPD scale", 1.0, 80.0, 12.0, 1.0, key="o_sig")
        o_u = st.slider("u — threshold ($M)", 5.0, 200.0, 50.0, 5.0, key="o_u")
        o_pi_tail = st.slider("π_tail", 0.0, 1.0, 0.12, 0.01, key="o_pi_tail")

        _section("OEP simulation")
        st.caption(
            "Each simulated draw is one independent annual aggregate loss, "
            "**not** a real calendar year. More draws → tighter estimate of "
            "the 1-in-200 line."
        )
        o_n_years = st.select_slider(
            "Number of simulated annual losses",
            options=[1_000, 5_000, 10_000, 25_000, 50_000, 100_000],
            value=10_000, key="o_n_years",
        )
        o_seed = st.number_input("Random seed", 0, 999_999, 42, 1, key="o_seed")

    elif mode == MODE_PRICE:
        _section("Underlying loss model")
        st.caption("Pricing reuses the NB-GPD engine from Compound loss.")
        p_lam = st.slider("λ — annual frequency", 0.1, 5.0, 1.5, 0.1, key="p_lam")
        p_nu = st.slider("ν — dispersion", 0.5, 20.0, 4.0, 0.5, key="p_nu")
        p_body_mu = st.slider("μ_body", 1.0, 5.0, 2.8, 0.1, key="p_body_mu")
        p_body_sig = st.slider("σ_body", 0.2, 2.0, 1.0, 0.05, key="p_body_sig")
        p_xi = st.slider("ξ — GPD shape", 0.0, 0.9, 0.30, 0.01, key="p_xi")
        p_sig = st.slider("σ_tail", 1.0, 80.0, 10.0, 1.0, key="p_sig")
        p_u = st.slider("u — threshold ($M)", 5.0, 200.0, 40.0, 5.0, key="p_u")
        p_pi_tail = st.slider("π_tail", 0.0, 1.0, 0.10, 0.01, key="p_pi_tail")
        p_n_years = st.select_slider(
            "Number of simulated annual losses",
            options=[2_000, 5_000, 10_000, 20_000],
            value=5_000, key="p_n_years",
            help=(
                "Independent Monte Carlo draws of the annual loss, not "
                "calendar years. Larger samples → smoother premium estimates."
            ),
        )

        _section("Premium-principle loadings")
        p_a = st.slider("a — SD scalar",        0.0, 1.0, 0.20, 0.01, key="p_a")
        p_b = st.slider("b — Variance scalar (×10⁻⁵)", 0.0, 10.0, 1.0, 0.1, key="p_b")
        p_h = st.slider("h — Esscher tilt (×10⁻⁴)",    0.0, 10.0, 1.0, 0.1, key="p_h")
        p_lw = st.slider("λ_Wang — distortion",          0.0, 1.0, 0.20, 0.01, key="p_lw")

    elif mode == MODE_ECON:
        _section("Capital & cost-of-capital")
        e_r = st.slider(
            "r — cost of capital (% / yr)",
            1.0, 20.0, 10.0, 0.5, key="e_r",
            help="Annualised pre-tax cost of capital used to amortise reliability capex K.",
        )
        e_kmax = st.slider(
            "K_max — redundancy budget ($M)",
            20, 200, 120, 5, key="e_kmax",
            help="Largest reliability capex on the x-axis. The optimum K* will be highlighted within this range.",
        )

        _section("Pre-mitigation premium")
        e_p0 = st.slider(
            "P₀ — technical premium at K = 0 ($k / yr)",
            200, 3000, 1500, 50, key="e_p0",
            help="The unloaded annual technical premium the insurer would charge with no mitigation in place.",
        )
        e_p_floor = st.slider(
            "P_∞ — premium floor ($k / yr)",
            0, 500, 80, 10, key="e_p_floor",
            help="Asymptote of the premium as K → ∞. No amount of capex eliminates risk completely.",
        )
        e_k_half = st.slider(
            "K_½ — premium half-life ($M)",
            5, 80, 40, 5, key="e_k_half",
            help="Capex required to roughly halve the premium above the floor — captures the elasticity ∂P/∂K from §14.2.",
        )

        _section("Retained loss below deductible")
        e_d0 = st.slider(
            "L₀ — retained loss at K = 0 ($k / yr)",
            0, 400, 120, 10, key="e_d0",
            help="Expected E[min(S, d)] at K = 0 — the operator's residual self-insured exposure.",
        )
        e_l_floor = st.slider(
            "L_∞ — retained-loss floor ($k / yr)",
            0, 200, 20, 5, key="e_l_floor",
        )
        e_l_half = st.slider(
            "L_½ — retained-loss half-life ($M)",
            10, 120, 60, 5, key="e_l_half",
        )

    elif mode == MODE_COMPARE:
        _section("Configuration A  (e.g. Tier-IV NA)")
        a_lam = st.slider("λ_A", 0.1, 5.0, 0.5, 0.1, key="c_a_lam")
        a_xi = st.slider("ξ_A — GPD shape", 0.0, 0.9, 0.30, 0.01, key="c_a_xi")
        a_sig = st.slider("σ_tail_A", 1.0, 80.0, 10.0, 1.0, key="c_a_sig")
        a_u = st.slider("u_A ($M)", 5.0, 200.0, 25.0, 5.0, key="c_a_u")
        a_pi = st.slider("π_tail_A", 0.0, 1.0, 0.05, 0.01, key="c_a_pi")
        st.divider()
        _section("Configuration B  (e.g. Tier-III EM)")
        b_lam = st.slider("λ_B", 0.1, 5.0, 1.8, 0.1, key="c_b_lam")
        b_xi = st.slider("ξ_B — GPD shape", 0.0, 0.9, 0.45, 0.01, key="c_b_xi")
        b_sig = st.slider("σ_tail_B", 1.0, 80.0, 22.0, 1.0, key="c_b_sig")
        b_u = st.slider("u_B ($M)", 5.0, 200.0, 35.0, 5.0, key="c_b_u")
        b_pi = st.slider("π_tail_B", 0.0, 1.0, 0.15, 0.01, key="c_b_pi")
        st.divider()
        _section("Shared assumptions")
        c_nu = st.slider("ν — dispersion", 0.5, 20.0, 4.0, 0.5, key="c_nu")
        c_body_mu = st.slider("μ_body", 1.0, 5.0, 2.8, 0.1, key="c_body_mu")
        c_body_sig = st.slider("σ_body", 0.2, 2.0, 1.0, 0.05, key="c_body_sig")
        c_n_years = st.select_slider(
            "Number of simulated annual losses (each side)",
            options=[2_000, 5_000, 10_000, 25_000],
            value=5_000, key="c_n_years",
            help=(
                "Independent Monte Carlo draws — not calendar years. Both "
                "configurations run on the same number of draws under the "
                "same seed so the comparison is apples-to-apples."
            ),
        )
        c_seed = st.number_input("Shared random seed", 0, 999_999, 42, 1, key="c_seed")

    else:  # MODE_FIGS
        _section("Browse")
        st.caption(
            "All 16 figures from the paper are bundled with the Space. "
            "Pick a figure to display below; every figure links to its label "
            "in the paper PDF."
        )


# ─────────────────────────────────────────────────────────────
# Mode body — Reliability
# ─────────────────────────────────────────────────────────────

if mode == MODE_RELIA:
    st.subheader("§5  Markov plant-state reliability")
    st.markdown(
        "Four-state continuous-time Markov chain on "
        "$\\{\\mathrm{OK},\\,\\mathrm{Deg}_1,\\,\\mathrm{Deg}_2,\\,\\mathrm{F}\\}$. "
        "Move the rate sliders to see how component reliability propagates to "
        "site-level availability $A$, mean time between outages, and the "
        "actuarial claim frequency $\\lambda^{\\text{out}}$ derived from "
        "eq. (10) and eq. (17) of the paper."
    )

    params = MarkovParams(lam=r_lam, mu=r_mu, lam_f=r_lam_f, mu_r=r_mu_r)
    Q = Q_matrix(params)
    pi = stationary_distribution(Q)
    A = availability(pi)
    U = 1.0 - A
    mtbf = mtbf_hours(pi, params)
    lam_out = claim_frequency_per_year(pi, params)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Availability A",   f"{A*100:.6f}%", delta=f"target {tier['avail']*100:.3f}%")
    m2.metric("Unavailability U", f"{U*100:.6f}%")
    m3.metric("MTBF (hours)",     f"{mtbf:,.0f}" if np.isfinite(mtbf) else "∞")
    m4.metric("λ^out (per year)", f"{lam_out:.3f}")

    # ── Single 3-row animated figure ────────────────────────
    #
    # All four panels live in ONE Plotly figure driven by ONE play
    # button:
    #
    #   Row 1, Col 1:  Stationary distribution π  (theoretical
    #                  reference faint + empirical bars filling up)
    #   Row 1, Col 2:  Sensitivity U(λ) curve + animated marker at
    #                  (λ, empirical U so far)
    #   Row 2 (full):  State path with cursor
    #   Row 3 (full):  Cumulative residence (hours) bars
    #
    # The slider tracks day-of-the-horizon; every panel updates
    # whenever the slider moves or the play button is pressed.
    st.divider()
    st.markdown("##### ▶  Live plant-state trajectory")
    st.caption(
        f"Direct CTMC sample over a {r_walk_years}-year horizon using "
        "the rates above. The empirical π converges to the theoretical "
        "value (faint background bars); the dot on the sensitivity curve "
        "tracks the realised unavailability."
    )

    rng_walk = np.random.default_rng(int(r_walk_seed))
    walk_states, walk_times = simulate_markov_walk(
        params, hours=float(r_walk_years) * 8760.0, rng=rng_walk,
    )
    walk_days = walk_times / 24.0
    n_outages = int(np.sum(walk_states == 3))

    horizon_days = float(r_walk_years) * 365.0
    N_FRAMES = 90
    cursor_days = np.linspace(0.0, horizon_days, N_FRAMES)
    state_labels = ["OK", "Deg₁", "Deg₂", "F"]
    state_colors = [IA_ACCENT, "#D08862", "#B86238", "#9F1239"]

    # Cumulative residence per frame, plus empirical π and empirical U.
    seg_starts = walk_days[:-1]
    seg_ends = walk_days[1:]
    seg_state = walk_states[:-1]
    residence_by_frame: list[np.ndarray] = []  # hours, length 4
    emp_pi_by_frame: list[np.ndarray] = []     # length 4, sums to ~1
    emp_U_by_frame: list[float] = []           # π_F empirical
    for d in cursor_days:
        clipped_end = np.minimum(seg_ends, d)
        clipped_dur = np.clip(clipped_end - seg_starts, 0.0, None)
        bins = np.zeros(4)
        for s in range(4):
            bins[s] = float(np.sum(clipped_dur[seg_state == s]))
        residence_hours = bins * 24.0
        residence_by_frame.append(residence_hours)
        total = residence_hours.sum()
        emp_pi = residence_hours / total if total > 0 else np.zeros(4)
        emp_pi_by_frame.append(emp_pi)
        emp_U_by_frame.append(float(emp_pi[3]))

    # Theoretical sensitivity curve U(λ) keeping (μ, λ_f, μ_r) at their
    # current sidebar values.
    lam_grid = np.geomspace(1e-6, 5e-3, 100)
    u_grid: list[float] = []
    for L in lam_grid:
        p2 = MarkovParams(lam=L, mu=r_mu, lam_f=r_lam_f, mu_r=r_mu_r)
        pi2 = stationary_distribution(Q_matrix(p2))
        u_grid.append(1.0 - availability(pi2))

    from plotly.subplots import make_subplots

    fig_combo = make_subplots(
        rows=3, cols=2,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],
        row_heights=[0.28, 0.39, 0.33],
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
        subplot_titles=(
            "Stationary distribution π",
            "Sensitivity U(λ)",
            "Plant state vs time",
            "Cumulative residence (hours)",
        ),
    )

    # ── Row 1 Col 1: π reference (faint) + empirical π (live) ────
    fig_combo.add_trace(  # trace 0 — theoretical π (static, faint)
        go.Bar(
            x=state_labels, y=pi.tolist(),
            marker=dict(color=state_colors, opacity=0.22,
                        line=dict(color=state_colors, width=1.2)),
            name="theoretical π",
            showlegend=False,
            hovertemplate="theoretical π(%{x}) = %{y:.4%}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig_combo.add_trace(  # trace 1 — empirical π (animated)
        go.Bar(
            x=state_labels, y=emp_pi_by_frame[0],
            marker=dict(color=state_colors),
            text=[f"{v:.2%}" for v in emp_pi_by_frame[0]],
            textposition="outside",
            name="empirical π",
            showlegend=False,
            hovertemplate="empirical π(%{x}) = %{y:.4%}<extra></extra>",
        ),
        row=1, col=1,
    )

    # ── Row 1 Col 2: U(λ) curve (static) + λ marker (static)
    # ──            + empirical U marker (animated) ─────────
    fig_combo.add_trace(  # trace 2 — theoretical U(λ) curve
        go.Scatter(
            x=lam_grid, y=u_grid, mode="lines",
            line=dict(color=IA_RULE, width=2.0),
            opacity=0.55,
            name="U(λ) theory",
            showlegend=False,
            hovertemplate="λ=%{x:.2e}<br>U=%{y:.4%}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig_combo.add_trace(  # trace 3 — theoretical (λ, U) marker, static
        go.Scatter(
            x=[r_lam], y=[U if U > 0 else 1e-9], mode="markers",
            marker=dict(color=IA_DARK, size=11, symbol="diamond-open",
                        line=dict(color=IA_DARK, width=2)),
            name="theoretical",
            showlegend=False,
            hovertemplate=f"theoretical U @ λ={r_lam:.2e}<br>U={U:.4%}<extra></extra>",
        ),
        row=1, col=2,
    )
    fig_combo.add_trace(  # trace 4 — empirical U so far, animated
        go.Scatter(
            x=[r_lam], y=[max(emp_U_by_frame[0], 1e-9)], mode="markers",
            marker=dict(color=IA_ACCENT, size=14, symbol="circle"),
            name="empirical so far",
            showlegend=False,
            hovertemplate="empirical U=%{y:.4%}<extra></extra>",
        ),
        row=1, col=2,
    )

    # ── Row 2: state path + cursor ─────────────────────────
    fig_combo.add_trace(  # trace 5 — full path, faint
        go.Scatter(
            x=walk_days, y=walk_states,
            mode="lines",
            line=dict(color=IA_RULE, width=1.5, shape="hv"),
            opacity=0.30, name="path", showlegend=False,
            hovertemplate="day %{x:.1f}<br>state=%{y}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig_combo.add_trace(  # trace 6 — animated cursor
        go.Scatter(
            x=[0.0, 0.0], y=[-0.5, 3.5],
            mode="lines",
            line=dict(color=IA_ACCENT, width=2.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ),
        row=2, col=1,
    )

    # ── Row 3: residence bars (animated) ───────────────────
    fig_combo.add_trace(  # trace 7
        go.Bar(
            x=state_labels, y=residence_by_frame[0],
            marker=dict(color=state_colors),
            text=[f"{v:.0f}h" for v in residence_by_frame[0]],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{x}: %{y:.0f} h<extra></extra>",
        ),
        row=3, col=1,
    )

    # Per-frame: update traces 1 (empirical π), 4 (empirical-U marker),
    # 6 (cursor), 7 (residence bars).
    fig_combo.frames = [
        go.Frame(
            name=str(k),
            data=[
                go.Bar(
                    x=state_labels, y=emp_pi_by_frame[k],
                    text=[f"{v:.2%}" for v in emp_pi_by_frame[k]],
                ),
                go.Scatter(x=[r_lam], y=[max(emp_U_by_frame[k], 1e-9)]),
                go.Scatter(x=[cursor_days[k], cursor_days[k]],
                           y=[-0.5, 3.5]),
                go.Bar(
                    x=state_labels, y=residence_by_frame[k],
                    text=[f"{v:.0f}h" for v in residence_by_frame[k]],
                ),
            ],
            traces=[1, 4, 6, 7],
        )
        for k in range(N_FRAMES)
    ]

    # Axes & layout polish
    max_pi = max(max(emp_pi_by_frame[-1]), max(pi)) if pi.size else 1.0
    max_res = max(max(r) for r in residence_by_frame) if residence_by_frame else 1.0

    fig_combo.update_yaxes(
        row=1, col=1, title_text="probability",
        range=[0, max_pi * 1.20],
    )
    fig_combo.update_xaxes(
        row=1, col=2, type="log",
        title_text="λ (per-train, 1/h)",
    )
    fig_combo.update_yaxes(
        row=1, col=2, type="log",
        title_text="U (unavailability)",
    )
    fig_combo.update_xaxes(
        row=2, col=1, title_text="time (days)",
        range=[0, horizon_days],
    )
    fig_combo.update_yaxes(
        row=2, col=1,
        tickmode="array", tickvals=[0, 1, 2, 3], ticktext=state_labels,
        range=[-0.5, 3.5], title_text="state",
    )
    fig_combo.update_yaxes(
        row=3, col=1, title_text="hours",
        range=[0, max_res * 1.18],
    )

    # Make the subplot titles compact and IA-grey.
    for ann in fig_combo.layout.annotations:
        ann.update(
            font=dict(size=12, color=IA_GREY, family="Helvetica, Arial, sans-serif"),
            xanchor="left", x=ann.x,
        )

    fig_combo.update_layout(
        height=820,
        margin=dict(l=60, r=20, t=50, b=110),
        bargap=0.18,
        updatemenus=[
            dict(
                type="buttons", direction="left", showactive=False,
                x=0.0, y=-0.06, xanchor="left", yanchor="top",
                pad=dict(r=10, t=2),
                bgcolor="rgba(160, 74, 31, 0.10)",
                bordercolor="rgba(160, 74, 31, 0.30)",
                buttons=[
                    dict(
                        label="▶  Play", method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=60, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸  Pause", method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0, x=0.18, y=-0.045, len=0.78,
                xanchor="left", yanchor="top",
                currentvalue=dict(
                    prefix="day  ", visible=True, xanchor="right",
                    font=dict(size=11, color=IA_GREY),
                ),
                pad=dict(t=2, b=2),
                bgcolor="rgba(160, 74, 31, 0.05)",
                bordercolor="rgba(160, 74, 31, 0.20)",
                steps=[
                    dict(
                        method="animate",
                        label=f"{d:.0f}",
                        args=[
                            [str(k)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for k, d in enumerate(cursor_days)
                ],
            ),
        ],
    )

    st.plotly_chart(apply_theme(fig_combo), width="stretch")

    plural = "" if r_walk_years == 1 else "s"
    out_plural = "" if n_outages == 1 else "s"
    st.caption(
        f"This realisation logged **{n_outages} site-outage event{out_plural}** "
        f"in {r_walk_years} year{plural}. Reshuffle by changing the seed; "
        f"the long-run rates converge to the theoretical π and λ^out above."
    )


# ─────────────────────────────────────────────────────────────
# Mode body — Compound loss
# ─────────────────────────────────────────────────────────────

elif mode == MODE_LOSS:
    st.subheader("§7 + §8  Compound NB-GPD annual loss")
    st.markdown(
        "Annual aggregate loss $S=\\sum_{j=1}^{N} X_j$ with frequency "
        "$N\\sim\\mathrm{NB}(\\nu, \\nu/(\\nu+\\lambda))$ and a body / tail "
        "severity mixture: lognormal body, generalised-Pareto tail above "
        "threshold $u$. The tail fraction $\\pi_{\\mathrm{tail}}$ is the "
        "probability that a given event lands in the GPD tail rather than "
        "the lognormal body. The simulation below draws a large independent "
        "sample of annual aggregate losses — each draw is one simulated "
        "year, **not** a real calendar year — to estimate the distribution."
    )

    S = simulate_annual_losses(
        n_years=int(l_n_years),
        nu=l_nu, lam=l_lam,
        body_mu=l_body_mu, body_sigma=l_body_sig,
        tail_xi=l_tail_xi, tail_sigma=l_tail_sig, tail_threshold=l_tail_u,
        tail_fraction=l_tail_frac,
        seed=int(l_seed),
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean S", f"${np.mean(S):,.2f}M")
    m2.metric("Std(S)", f"${np.std(S):,.2f}M")
    m3.metric("p99.5",  f"${np.quantile(S, 0.995):,.1f}M")
    m4.metric("Max(S)", f"${np.max(S):,.1f}M")

    chart_l, chart_r = st.columns([1, 1])
    with chart_l:
        st.markdown("##### Annual-loss histogram (linear)")
        fig = go.Figure(
            go.Histogram(
                x=S, nbinsx=80, marker=dict(color=IA_ACCENT, line=dict(width=0)),
                hovertemplate="S=$%{x:.1f}M<br>count=%{y}<extra></extra>",
            )
        )
        fig.add_vline(x=np.mean(S), line=dict(color=IA_DARK, dash="dot"),
                      annotation_text=f"mean ${np.mean(S):.1f}M", annotation_position="top")
        fig.add_vline(x=np.quantile(S, 0.995), line=dict(color=IA_RULE, dash="dash"),
                      annotation_text=f"p99.5 ${np.quantile(S, 0.995):.1f}M",
                      annotation_position="top")
        fig.update_layout(xaxis_title="annual S ($M)",
                          yaxis_title="frequency", height=420)
        st.plotly_chart(apply_theme(fig), width="stretch")

    with chart_r:
        st.markdown("##### Log-scale tail")
        S_pos = S[S > 0]
        if len(S_pos):
            fig2 = go.Figure(
                go.Histogram(
                    x=np.log10(S_pos), nbinsx=60,
                    marker=dict(color=IA_RULE, line=dict(width=0)),
                    hovertemplate="log₁₀(S)=%{x:.2f}<br>count=%{y}<extra></extra>",
                )
            )
            fig2.update_layout(xaxis_title="log₁₀  annual S ($M)",
                               yaxis_title="frequency", height=420)
            st.plotly_chart(apply_theme(fig2), width="stretch")


# ─────────────────────────────────────────────────────────────
# Mode body — OEP curve
# ─────────────────────────────────────────────────────────────

elif mode == MODE_OEP:
    st.subheader("§11  Occurrence-Exceedance-Probability curve")
    st.markdown(
        "We draw a large sample of **independent annual aggregate losses** "
        "from the NB-GPD model, sort them, and plot the empirical survival "
        "function $\\bar F_S(s)$. Each draw is one simulated annual loss — "
        "**not** a calendar year of operating history. More draws shrink "
        "the Monte Carlo noise around the 1-in-200 (0.5%) Solvency II line "
        "drawn in red."
    )

    with st.spinner(f"Drawing {o_n_years:,} annual-loss samples…"):
        S = simulate_annual_losses(
            n_years=int(o_n_years),
            nu=o_nu, lam=o_lam,
            body_mu=o_body_mu, body_sigma=o_body_sig,
            tail_xi=o_xi, tail_sigma=o_sig, tail_threshold=o_u,
            tail_fraction=o_pi_tail,
            seed=int(o_seed),
        )

    S_sorted = np.sort(S)
    n = S_sorted.size
    exceed = 1.0 - np.arange(1, n + 1) / (n + 1)

    p995 = np.quantile(S, 0.995)
    p999 = np.quantile(S, 0.999)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean S",           f"${np.mean(S):,.2f}M")
    m2.metric("1-in-200 (p99.5)", f"${p995:,.1f}M")
    m3.metric("1-in-1000 (p99.9)", f"${p999:,.1f}M")
    m4.metric("TVaR α=99.5%",     f"${np.mean(S[S >= p995]):,.1f}M")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.clip(S_sorted, 1e-3, None), y=exceed, mode="lines",
            line=dict(color=IA_ACCENT, width=2.5),
            name="empirical OEP",
            hovertemplate="S≥$%{x:.1f}M<br>P=%{y:.4%}<extra></extra>",
        )
    )
    fig.add_hline(y=0.005, line=dict(color="#9F1239", dash="dash"),
                  annotation_text="1-in-200", annotation_position="right")
    fig.add_hline(y=0.001, line=dict(color=IA_RULE, dash="dot"),
                  annotation_text="1-in-1000", annotation_position="right")
    fig.update_layout(
        xaxis_title="annual aggregate loss S  ($M)",
        yaxis_title="exceedance probability  P(S ≥ s)",
        xaxis_type="log", yaxis_type="log",
        height=520,
    )
    st.plotly_chart(apply_theme(fig), width="stretch")

    st.caption(
        f"Empirical curve from {n:,} simulated annual-loss draws "
        "(independent Monte Carlo realisations, not calendar years). "
        f"GPD analytical TVaR (no body) = "
        f"${gpd_tvar(0.995, o_xi, o_sig, o_u):,.1f}M  (for comparison)."
    )

    # ── Convergence animation ───────────────────────────────
    st.divider()
    st.markdown("##### ▶  Watch the curve settle as we add more samples")
    st.caption(
        "The OEP curve is the empirical survival function of the simulated "
        "annual losses. Press **Play** to watch it stabilise as we keep "
        "adding independent draws. Each draw is one simulated annual loss "
        "— **not** a calendar year. The deepest tail (the 1-in-200 line) "
        "needs the most samples before the curve stops jittering."
    )

    full_n = int(o_n_years)
    checkpoints = sorted(
        {min(full_n, k) for k in
         [100, 250, 500, 1_000, 2_500, 5_000, 7_500, 10_000,
          15_000, 25_000, 50_000, 75_000, 100_000]
         if k <= max(full_n, 100)}
    )
    if not checkpoints or checkpoints[-1] != full_n:
        checkpoints.append(full_n)

    fig_conv = go.Figure()
    first_n = checkpoints[0]
    init = np.sort(S[:first_n])
    init_ex = 1.0 - np.arange(1, init.size + 1) / (init.size + 1)
    fig_conv.add_trace(
        go.Scatter(
            x=np.clip(init, 1e-3, None), y=init_ex,
            mode="lines",
            line=dict(color=IA_ACCENT, width=2.5),
            hovertemplate="S≥$%{x:.1f}M<br>P=%{y:.4%}<extra></extra>",
            name="empirical OEP",
            showlegend=False,
        )
    )
    fig_conv.add_hline(
        y=0.005, line=dict(color="#9F1239", dash="dash"),
        annotation_text="1-in-200", annotation_position="right",
    )

    fig_conv.frames = [
        go.Frame(
            name=str(k),
            data=[
                go.Scatter(
                    x=np.clip(np.sort(S[:k]), 1e-3, None),
                    y=1.0 - np.arange(1, k + 1) / (k + 1),
                )
            ],
            traces=[0],
        )
        for k in checkpoints
    ]

    fig_conv.update_layout(
        xaxis_title="annual aggregate loss S  ($M)",
        yaxis_title="exceedance probability  P(S ≥ s)",
        xaxis_type="log", yaxis_type="log",
        height=520,
        margin=dict(l=60, r=20, t=20, b=110),
        updatemenus=[
            dict(
                type="buttons", direction="left", showactive=False,
                x=0.0, y=-0.22, xanchor="left", yanchor="top",
                pad=dict(r=10, t=2),
                bgcolor="rgba(160, 74, 31, 0.10)",
                bordercolor="rgba(160, 74, 31, 0.30)",
                buttons=[
                    dict(
                        label="▶  Play", method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=320, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=120, easing="cubic-in-out"),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸  Pause", method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0, x=0.18, y=-0.20, len=0.78,
                xanchor="left", yanchor="top",
                currentvalue=dict(
                    prefix="samples  ", visible=True, xanchor="right",
                    font=dict(size=11, color=IA_GREY),
                ),
                pad=dict(t=2, b=2),
                bgcolor="rgba(160, 74, 31, 0.05)",
                bordercolor="rgba(160, 74, 31, 0.20)",
                steps=[
                    dict(
                        method="animate",
                        label=f"{k:,}",
                        args=[
                            [str(k)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for k in checkpoints
                ],
            )
        ],
    )
    st.plotly_chart(apply_theme(fig_conv), width="stretch")


# ─────────────────────────────────────────────────────────────
# Mode body — Pricing
# ─────────────────────────────────────────────────────────────

elif mode == MODE_PRICE:
    st.subheader("§10  Technical premium under five risk-loading principles")
    st.markdown(
        "Given the same NB-GPD loss model, this tab computes the technical "
        "premium $P^{\\mathrm{tech}}$ under each of the five principles from "
        "the paper:"
    )
    st.markdown(
        "- **Pure**: $P=\\mathbb E[S]$\n"
        "- **SD**: $P=\\mathbb E[S] + a\\,\\sigma(S)$\n"
        "- **Variance**: $P=\\mathbb E[S] + b\\,\\mathrm{Var}(S)$\n"
        "- **Esscher**: $P=\\mathbb E[S e^{hS}]/\\mathbb E[e^{hS}]$\n"
        "- **Wang** (1996, 2000): $P=\\int_0^\\infty "
        "\\Phi(\\Phi^{-1}(\\bar F_S(s)) + \\lambda_W)\\,ds$"
    )

    with st.spinner(f"Drawing {p_n_years:,} annual-loss samples…"):
        S = simulate_annual_losses(
            n_years=int(p_n_years),
            nu=p_nu, lam=p_lam,
            body_mu=p_body_mu, body_sigma=p_body_sig,
            tail_xi=p_xi, tail_sigma=p_sig, tail_threshold=p_u,
            tail_fraction=p_pi_tail,
            seed=42,
        )

    # ── Single animated panel: table on top, horizontal bar chart
    #    underneath, one play button driving both. ─────────
    #
    # Press Play to ramp a single "risk-aversion" scalar α from 0% to
    # 100%. At α = 0 every principle collapses to Pure (no loading);
    # at α = 100% each principle reaches the user-set sidebar value.
    # All four loading parameters (a, b, h, λ_W) scale proportionally
    # so the user sees how each principle responds to risk-aversion
    # strength under the same underlying loss sample.
    mean_S = float(np.mean(S))

    # Pre-compute the premium curves for every frame.
    N_FRAMES_P = 41  # 0%, 2.5%, ..., 100%
    alphas = np.linspace(0.0, 1.0, N_FRAMES_P)

    # Use the user-set sidebar value at α = 1 so the last frame matches
    # exactly what the metrics card and tooltip claim.
    base_loadings = Loadings(
        a=p_a, b=p_b * 1e-5, h=p_h * 1e-4, lambda_w=p_lw,
    )
    principle_names = list(all_premiums(S, base_loadings).keys())
    premium_curves: dict[str, list[float]] = {n: [] for n in principle_names}
    for alpha in alphas:
        scaled = Loadings(
            a=p_a * alpha,
            b=p_b * 1e-5 * alpha,
            h=p_h * 1e-4 * alpha,
            lambda_w=p_lw * alpha,
        )
        prem_k = all_premiums(S, scaled)
        for name in principle_names:
            premium_curves[name].append(prem_k[name])

    # Helper: build the four columns of the loading-vs-Pure table at
    # frame k. Returns parallel lists ready to drop into a go.Table.
    def _table_columns(k: int) -> tuple[list[str], list[str], list[str], list[str]]:
        rows = []
        for name in principle_names:
            P = premium_curves[name][k]
            loading_pct = 0.0 if name == "Pure" else 100.0 * (P - mean_S) / mean_S
            rows.append((name, f"${P:,.2f}M",
                         "0.0%" if name == "Pure" else f"{loading_pct:+.1f}%",
                         f"{int(round(alphas[k] * 100))}%"))
        principles = [r[0] for r in rows]
        premium_strs = [r[1] for r in rows]
        loading_strs = [r[2] for r in rows]
        alpha_strs = [r[3] for r in rows]
        return principles, premium_strs, loading_strs, alpha_strs

    init_p, init_prem, init_load, init_alpha = _table_columns(0)
    init_bar_x = [premium_curves[n][0] for n in principle_names]

    from plotly.subplots import make_subplots

    fig_pricing = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "table"}], [{"type": "xy"}]],
        row_heights=[0.42, 0.58],
        vertical_spacing=0.10,
        subplot_titles=(
            "Loading vs Pure premium",
            "Premium comparison",
        ),
    )

    # Row 1 — go.Table (trace 0)
    header_fill = "rgba(160, 74, 31, 0.10)"
    row_fill_a = "rgba(160, 74, 31, 0.03)"
    row_fill_b = "rgba(160, 74, 31, 0.07)"
    cell_fills = [
        [row_fill_a if i % 2 == 0 else row_fill_b for i in range(len(principle_names))]
    ] * 4
    fig_pricing.add_trace(
        go.Table(
            header=dict(
                values=["<b>Principle</b>", "<b>Premium ($M)</b>",
                        "<b>Loading vs Pure</b>", "<b>Strength α</b>"],
                fill_color=header_fill,
                font=dict(color=IA_DARK, size=12),
                align=["left", "right", "right", "right"],
                height=32,
            ),
            cells=dict(
                values=[init_p, init_prem, init_load, init_alpha],
                fill_color=cell_fills,
                font=dict(color=IA_DARK, size=12),
                align=["left", "right", "right", "right"],
                height=28,
            ),
            columnwidth=[1.2, 1.2, 1.2, 1.0],
        ),
        row=1, col=1,
    )

    # Row 2 — horizontal bar (trace 1). Reverse order so the topmost
    # bar reads first naturally.
    reversed_names = list(reversed(principle_names))
    fig_pricing.add_trace(
        go.Bar(
            y=reversed_names,
            x=[premium_curves[n][0] for n in reversed_names],
            orientation="h",
            marker=dict(color=list(reversed(SERIES_COLORS[: len(principle_names)]))),
            text=[f"${v:,.2f}M" for v in [premium_curves[n][0] for n in reversed_names]],
            textposition="outside",
            textfont=dict(size=12),
            showlegend=False,
            hovertemplate="%{y}<br>premium = $%{x:,.2f}M<extra></extra>",
        ),
        row=2, col=1,
    )
    # Pure baseline as a vertical reference line on the bar plot.
    # We can't use add_vline(row=2, col=1) here because Plotly's
    # auto-resolver walks every trace in the figure looking for an
    # x-axis, and the Table trace in row 1 has none → KeyError. We
    # add the shape + annotation directly against the xy subplot's
    # axes instead.
    n_bars = len(principle_names)
    fig_pricing.add_shape(
        type="line",
        x0=mean_S, x1=mean_S,
        y0=-0.5, y1=n_bars - 0.5,
        xref="x", yref="y",
        line=dict(color=IA_RULE, dash="dot", width=1.5),
    )
    fig_pricing.add_annotation(
        x=mean_S, y=n_bars - 0.5,
        xref="x", yref="y",
        text=f"Pure = ${mean_S:,.2f}M",
        showarrow=False,
        xanchor="left", yanchor="bottom",
        xshift=4, yshift=2,
        font=dict(color=IA_RULE, size=11),
    )

    # x-axis ceiling: tallest bar across the whole sweep + headroom.
    x_max = max(max(premium_curves[n]) for n in principle_names) * 1.20

    # One frame per α step; both traces update simultaneously.
    fig_pricing.frames = [
        go.Frame(
            name=str(k),
            data=[
                go.Table(
                    cells=dict(
                        values=list(_table_columns(k)),
                        fill_color=cell_fills,
                        font=dict(color=IA_DARK, size=12),
                        align=["left", "right", "right", "right"],
                        height=28,
                    ),
                ),
                go.Bar(
                    y=reversed_names,
                    x=[premium_curves[n][k] for n in reversed_names],
                    text=[f"${v:,.2f}M" for v in
                          [premium_curves[n][k] for n in reversed_names]],
                ),
            ],
            traces=[0, 1],
        )
        for k in range(N_FRAMES_P)
    ]

    # Subplot-title styling: compact, IA grey.
    for ann in fig_pricing.layout.annotations:
        # The horizontal-bar's "Pure = $...M" vline annotation also
        # ends up in layout.annotations; only restyle the subplot titles
        # (which we know are at y above the plot region).
        if ann.text in {"Loading vs Pure premium", "Premium comparison"}:
            ann.update(
                font=dict(size=12, color=IA_GREY,
                          family="Helvetica, Arial, sans-serif"),
                xanchor="left", x=ann.x,
            )

    fig_pricing.update_xaxes(
        row=2, col=1,
        title_text="technical premium  ($M)",
        range=[0, x_max],
    )
    fig_pricing.update_yaxes(row=2, col=1, title_text="")
    fig_pricing.update_layout(
        height=720,
        margin=dict(l=60, r=20, t=50, b=110),
        bargap=0.30,
        updatemenus=[
            dict(
                type="buttons", direction="left", showactive=False,
                x=0.0, y=-0.04, xanchor="left", yanchor="top",
                pad=dict(r=10, t=2),
                bgcolor="rgba(160, 74, 31, 0.10)",
                bordercolor="rgba(160, 74, 31, 0.30)",
                buttons=[
                    dict(
                        label="▶  Play", method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=140, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=80, easing="cubic-in-out"),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸  Pause", method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0, x=0.18, y=-0.025, len=0.78,
                xanchor="left", yanchor="top",
                currentvalue=dict(
                    prefix="loading strength  ",
                    visible=True, xanchor="right",
                    font=dict(size=11, color=IA_GREY),
                ),
                pad=dict(t=2, b=2),
                bgcolor="rgba(160, 74, 31, 0.05)",
                bordercolor="rgba(160, 74, 31, 0.20)",
                steps=[
                    dict(
                        method="animate",
                        label=f"{int(round(alpha * 100))}%",
                        args=[
                            [str(k)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for k, alpha in enumerate(alphas)
                ],
            ),
        ],
    )

    st.markdown("##### ▶  Risk-aversion sweep")
    st.caption(
        "Press **Play** to ramp the loading strength from 0% to 100% of "
        "your sidebar settings. At 0% every principle equals the Pure "
        "premium (expected loss); at 100% each principle reaches its full "
        "sidebar value. The table and the horizontal bar chart both "
        "update from the same scalar so you can read the $M premium and "
        "the % loading at any point in the sweep."
    )
    st.plotly_chart(apply_theme(fig_pricing), width="stretch")
    st.caption(
        "Same loss sample throughout; only the loading-parameter scalar "
        "moves. The five principles diverge because each one penalises a "
        "different functional of the loss distribution (mean, SD, "
        "variance, exponential tilt, Wang distortion). Heavy-tailed books "
        "reward the right-hand principles disproportionately."
    )


# ─────────────────────────────────────────────────────────────
# Mode body — Economics (§14 of the paper)
# ─────────────────────────────────────────────────────────────

elif mode == MODE_ECON:
    st.subheader("§14  Operator's optimal investment problem")
    st.markdown(
        "The operator chooses how much capital $K$ to spend on redundancy "
        "(extra UPS strings, an extra cooling loop, off-gas detection, "
        "behind-the-meter generation). Annual cost is the sum"
        "\n\n"
        "$$\\mathcal{C}(K) \\;=\\; r\\,K "
        "\\;+\\; P^{\\mathrm{tech}}\\!\\bigl(\\lambda(K), \\xi(K)\\bigr) "
        "\\;+\\; \\mathbb{E}[\\min(S, d)]"
        "$$"
        "\n"
        "Capital cost rises linearly; the technical premium and the "
        "retained-loss component both fall with $K$. The total has a "
        "unique interior minimum at $K^{*}$ where the first-order "
        "condition is satisfied. Press **Play** to sweep $K$ from 0 to "
        "$K_{\\max}$ and watch the components add up."
    )

    # Build the four cost curves over a fine K-grid using the
    # exponential-asymptote forms from the paper's §14.2 (Fig 9).
    K_grid = np.linspace(0.0, float(e_kmax), 121)
    r_decimal = float(e_r) / 100.0
    capital_cost = r_decimal * K_grid * 1000.0     # $k / yr (K is in $M)
    decay_p = np.log(2.0) / float(e_k_half)
    decay_l = np.log(2.0) / float(e_l_half)
    premium_K = (float(e_p0) - float(e_p_floor)) * np.exp(-decay_p * K_grid) + float(e_p_floor)
    retained_K = (float(e_d0) - float(e_l_floor)) * np.exp(-decay_l * K_grid) + float(e_l_floor)
    total_K = capital_cost + premium_K + retained_K
    k_opt_idx = int(np.argmin(total_K))
    K_opt = float(K_grid[k_opt_idx])
    C_opt = float(total_K[k_opt_idx])

    # Static reference cards (optimum) — sit alongside the animated
    # in-chart indicators that follow the play button below. Colours
    # inherit from Streamlit's foreground so the values stay readable
    # under either light or dark theme.
    ref_l, ref_r = st.columns([1, 1])
    ref_l.markdown(
        '<div style="border:1px solid rgba(160,74,31,0.30); '
        'border-radius:10px; padding:0.7rem 0.9rem; '
        'background:rgba(160,74,31,0.06);">'
        '<div style="font-size:0.70rem; letter-spacing:0.08em; '
        'text-transform:uppercase; opacity:0.78; font-weight:600;">'
        'OPTIMUM K*</div>'
        '<div style="font-size:1.5rem; font-weight:700; color:inherit;">'
        f'${K_opt:,.1f}M</div></div>',
        unsafe_allow_html=True,
    )
    ref_r.markdown(
        '<div style="border:1px solid rgba(160,74,31,0.30); '
        'border-radius:10px; padding:0.7rem 0.9rem; '
        'background:rgba(160,74,31,0.06);">'
        '<div style="font-size:0.70rem; letter-spacing:0.08em; '
        'text-transform:uppercase; opacity:0.78; font-weight:600;">'
        'MIN C(K*)</div>'
        '<div style="font-size:1.5rem; font-weight:700; color:inherit;">'
        f'${C_opt:,.0f}'
        '<span style="font-size:0.85rem; opacity:0.6;"> k / yr</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # Animation frames — cursor walks K_grid
    N_FRAMES_E = 60
    cursor_idx = np.linspace(0, len(K_grid) - 1, N_FRAMES_E).astype(int)

    from plotly.subplots import make_subplots

    component_names = ["Capital  r·K", "Premium  P(K)", "Retained  E[min(S,d)]", "Total  C(K)"]
    component_colors = ["#1F4F73", IA_ACCENT, "#0E7C7B", IA_DARK]

    # 3-row layout: indicators on top, line chart in the middle,
    # stacked bar at the bottom. The four indicators animate alongside
    # the line cursor and the bar segments — one play button drives them all.
    fig_econ = make_subplots(
        rows=3, cols=4,
        specs=[
            [
                {"type": "indicator"},
                {"type": "indicator"},
                {"type": "indicator"},
                {"type": "indicator"},
            ],
            [{"type": "xy", "colspan": 4}, None, None, None],
            [{"type": "xy", "colspan": 4}, None, None, None],
        ],
        row_heights=[0.15, 0.52, 0.33],
        vertical_spacing=0.13,
        horizontal_spacing=0.04,
        subplot_titles=(
            "", "", "", "",
            "Annual cost components vs redundancy capital K",
            "Cost stack at the current K",
        ),
    )

    # ── Row 1: four live indicators ─────────────────────────
    # Each shows the current value (at the cursor's K) + a delta versus
    # the same component evaluated at K*. So as the user sweeps K, the
    # indicators reflect the live operating point and how far it sits
    # from the optimum.
    init_total = float(total_K[0])
    init_premium = float(premium_K[0])
    init_capex = float(capital_cost[0])
    init_K = float(K_grid[0])

    fig_econ.add_trace(  # trace 0 — current K
        go.Indicator(
            mode="number",
            value=init_K,
            number=dict(
                valueformat=",.1f",
                prefix="$", suffix="M",
                font=dict(size=28, color=IA_DARK),
            ),
            title=dict(
                text="<span style='font-size:11px; letter-spacing:0.08em; "
                     "text-transform:uppercase; opacity:0.70;'>"
                     "redundancy capital  K</span>",
            ),
            domain=dict(row=0, column=0),
        ),
        row=1, col=1,
    )
    fig_econ.add_trace(  # trace 1 — total cost C(K)
        go.Indicator(
            mode="number+delta",
            value=init_total,
            number=dict(
                valueformat=",.0f", suffix=" k/yr",
                font=dict(size=26, color=IA_DARK),
            ),
            delta=dict(
                reference=C_opt,
                valueformat=",.0f",
                increasing=dict(color="#DC2626"),
                decreasing=dict(color="#16A34A"),
            ),
            title=dict(
                text="<span style='font-size:11px; letter-spacing:0.08em; "
                     "text-transform:uppercase; opacity:0.70;'>"
                     "total cost  C(K)</span>",
            ),
            domain=dict(row=0, column=1),
        ),
        row=1, col=2,
    )
    fig_econ.add_trace(  # trace 2 — premium P(K)
        go.Indicator(
            mode="number+delta",
            value=init_premium,
            number=dict(
                valueformat=",.0f", suffix=" k/yr",
                font=dict(size=26, color=IA_ACCENT),
            ),
            delta=dict(
                reference=float(premium_K[k_opt_idx]),
                valueformat=",.0f",
                increasing=dict(color="#DC2626"),
                decreasing=dict(color="#16A34A"),
            ),
            title=dict(
                text="<span style='font-size:11px; letter-spacing:0.08em; "
                     "text-transform:uppercase; opacity:0.70;'>"
                     "premium  P(K)</span>",
            ),
            domain=dict(row=0, column=2),
        ),
        row=1, col=3,
    )
    fig_econ.add_trace(  # trace 3 — capital cost r·K
        go.Indicator(
            mode="number+delta",
            value=init_capex,
            number=dict(
                valueformat=",.0f", suffix=" k/yr",
                font=dict(size=26, color="#1F4F73"),
            ),
            delta=dict(
                reference=float(capital_cost[k_opt_idx]),
                valueformat=",.0f",
                increasing=dict(color="#DC2626"),
                decreasing=dict(color="#16A34A"),
            ),
            title=dict(
                text="<span style='font-size:11px; letter-spacing:0.08em; "
                     "text-transform:uppercase; opacity:0.70;'>"
                     "capital cost  r·K</span>",
            ),
            domain=dict(row=0, column=3),
        ),
        row=1, col=4,
    )

    # ── Row 2: four static lines (traces 4-7) + cursor + dot (8, 9) ──
    fig_econ.add_trace(  # trace 4
        go.Scatter(
            x=K_grid, y=capital_cost, name=component_names[0], mode="lines",
            line=dict(color=component_colors[0], width=2),
            hovertemplate="K=$%{x:.0f}M<br>r·K=$%{y:,.0f}k/yr<extra></extra>",
        ),
        row=2, col=1,
    )
    fig_econ.add_trace(  # trace 5
        go.Scatter(
            x=K_grid, y=premium_K, name=component_names[1], mode="lines",
            line=dict(color=component_colors[1], width=2),
            hovertemplate="K=$%{x:.0f}M<br>P(K)=$%{y:,.0f}k/yr<extra></extra>",
        ),
        row=2, col=1,
    )
    fig_econ.add_trace(  # trace 6
        go.Scatter(
            x=K_grid, y=retained_K, name=component_names[2], mode="lines",
            line=dict(color=component_colors[2], width=2),
            hovertemplate="K=$%{x:.0f}M<br>retained=$%{y:,.0f}k/yr<extra></extra>",
        ),
        row=2, col=1,
    )
    fig_econ.add_trace(  # trace 7
        go.Scatter(
            x=K_grid, y=total_K, name=component_names[3], mode="lines",
            line=dict(color=component_colors[3], width=3),
            hovertemplate="K=$%{x:.0f}M<br>total=$%{y:,.0f}k/yr<extra></extra>",
        ),
        row=2, col=1,
    )

    # K* dashed line + label on the line chart
    y_max = float(total_K.max()) * 1.10
    fig_econ.add_shape(
        type="line",
        x0=K_opt, x1=K_opt, y0=0, y1=y_max,
        xref="x", yref="y",
        line=dict(color=IA_GREY, dash="dot", width=1.5),
    )
    fig_econ.add_annotation(
        x=K_opt, y=y_max,
        xref="x", yref="y",
        text=f"K* = ${K_opt:.1f}M",
        showarrow=False, xanchor="left", yanchor="top",
        xshift=4, yshift=-4,
        font=dict(color=IA_GREY, size=11),
    )

    # Animated cursor (trace 8) + animated dot (trace 9)
    fig_econ.add_trace(
        go.Scatter(
            x=[K_grid[0], K_grid[0]], y=[0, y_max],
            mode="lines",
            line=dict(color=IA_ACCENT, width=2.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ),
        row=2, col=1,
    )
    fig_econ.add_trace(
        go.Scatter(
            x=[K_grid[0]], y=[total_K[0]],
            mode="markers",
            marker=dict(color=IA_DARK, size=12, symbol="circle"),
            showlegend=False, hoverinfo="skip",
        ),
        row=2, col=1,
    )

    # ── Row 3: horizontal stacked bar (traces 10-12) ─────────
    init_vals = [capital_cost[0], premium_K[0], retained_K[0]]
    fig_econ.add_trace(  # trace 10
        go.Bar(
            x=[init_vals[0]], y=["C(K)"], orientation="h",
            marker=dict(color=component_colors[0]),
            name=component_names[0],
            text=[f"${init_vals[0]:,.0f}k"], textposition="inside",
            insidetextanchor="middle",
            hovertemplate="r·K = $%{x:,.0f} k / yr<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )
    fig_econ.add_trace(  # trace 11
        go.Bar(
            x=[init_vals[1]], y=["C(K)"], orientation="h",
            marker=dict(color=component_colors[1]),
            name=component_names[1],
            text=[f"${init_vals[1]:,.0f}k"], textposition="inside",
            insidetextanchor="middle",
            hovertemplate="P(K) = $%{x:,.0f} k / yr<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )
    fig_econ.add_trace(  # trace 12
        go.Bar(
            x=[init_vals[2]], y=["C(K)"], orientation="h",
            marker=dict(color=component_colors[2]),
            name=component_names[2],
            text=[f"${init_vals[2]:,.0f}k"], textposition="inside",
            insidetextanchor="middle",
            hovertemplate="retained = $%{x:,.0f} k / yr<extra></extra>",
            showlegend=False,
        ),
        row=3, col=1,
    )

    x_stack_overall_max = float(total_K.max()) * 1.05

    # Frames update everything: 4 indicators + cursor + dot + 3 stack bars.
    fig_econ.frames = [
        go.Frame(
            name=str(f_idx),
            data=[
                # 0 — K indicator
                go.Indicator(
                    mode="number",
                    value=float(K_grid[k]),
                    number=dict(
                        valueformat=",.1f", prefix="$", suffix="M",
                        font=dict(size=28, color=IA_DARK),
                    ),
                ),
                # 1 — total C(K) indicator
                go.Indicator(
                    mode="number+delta",
                    value=float(total_K[k]),
                    number=dict(
                        valueformat=",.0f", suffix=" k/yr",
                        font=dict(size=26, color=IA_DARK),
                    ),
                    delta=dict(
                        reference=C_opt,
                        valueformat=",.0f",
                        increasing=dict(color="#DC2626"),
                        decreasing=dict(color="#16A34A"),
                    ),
                ),
                # 2 — premium indicator
                go.Indicator(
                    mode="number+delta",
                    value=float(premium_K[k]),
                    number=dict(
                        valueformat=",.0f", suffix=" k/yr",
                        font=dict(size=26, color=IA_ACCENT),
                    ),
                    delta=dict(
                        reference=float(premium_K[k_opt_idx]),
                        valueformat=",.0f",
                        increasing=dict(color="#DC2626"),
                        decreasing=dict(color="#16A34A"),
                    ),
                ),
                # 3 — capital cost indicator
                go.Indicator(
                    mode="number+delta",
                    value=float(capital_cost[k]),
                    number=dict(
                        valueformat=",.0f", suffix=" k/yr",
                        font=dict(size=26, color="#1F4F73"),
                    ),
                    delta=dict(
                        reference=float(capital_cost[k_opt_idx]),
                        valueformat=",.0f",
                        increasing=dict(color="#DC2626"),
                        decreasing=dict(color="#16A34A"),
                    ),
                ),
                # 8 — cursor
                go.Scatter(x=[K_grid[k], K_grid[k]], y=[0, y_max]),
                # 9 — dot on total cost
                go.Scatter(x=[K_grid[k]], y=[total_K[k]]),
                # 10 — stack: capital_cost
                go.Bar(
                    x=[capital_cost[k]], y=["C(K)"],
                    text=[f"${capital_cost[k]:,.0f}k"],
                ),
                # 11 — stack: premium
                go.Bar(
                    x=[premium_K[k]], y=["C(K)"],
                    text=[f"${premium_K[k]:,.0f}k"],
                ),
                # 12 — stack: retained
                go.Bar(
                    x=[retained_K[k]], y=["C(K)"],
                    text=[f"${retained_K[k]:,.0f}k"],
                ),
            ],
            traces=[0, 1, 2, 3, 8, 9, 10, 11, 12],
        )
        for f_idx, k in enumerate(cursor_idx)
    ]

    # Subplot title styling (only the two real titles; the four empty
    # strings on row 1 don't produce annotations).
    for ann in fig_econ.layout.annotations:
        if ann.text in {
            "Annual cost components vs redundancy capital K",
            "Cost stack at the current K",
        }:
            ann.update(
                font=dict(size=12, color=IA_GREY,
                          family="Helvetica, Arial, sans-serif"),
                xanchor="left", x=ann.x,
            )

    fig_econ.update_xaxes(
        row=2, col=1,
        title_text="redundancy capital  K  ($M)",
        range=[0, float(e_kmax)],
    )
    fig_econ.update_yaxes(
        row=2, col=1,
        title_text="annual cost  ($k / yr)",
        range=[0, y_max],
    )
    fig_econ.update_xaxes(
        row=3, col=1,
        title_text="$ / yr (stacked)",
        range=[0, x_stack_overall_max],
    )
    fig_econ.update_yaxes(row=3, col=1, title_text="")

    fig_econ.update_layout(
        height=860,
        margin=dict(l=60, r=20, t=50, b=110),
        barmode="stack",
        bargap=0.50,
        legend=dict(
            orientation="h",
            x=0.0, y=0.74, xanchor="left", yanchor="bottom",
            bgcolor="rgba(0, 0, 0, 0)",
        ),
        updatemenus=[
            dict(
                type="buttons", direction="left", showactive=False,
                x=0.0, y=-0.04, xanchor="left", yanchor="top",
                pad=dict(r=10, t=2),
                bgcolor="rgba(160, 74, 31, 0.10)",
                bordercolor="rgba(160, 74, 31, 0.30)",
                buttons=[
                    dict(
                        label="▶  Play", method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=80, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸  Pause", method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0, x=0.18, y=-0.025, len=0.78,
                xanchor="left", yanchor="top",
                currentvalue=dict(
                    prefix="K  $",
                    visible=True, xanchor="right",
                    font=dict(size=11, color=IA_GREY),
                ),
                pad=dict(t=2, b=2),
                bgcolor="rgba(160, 74, 31, 0.05)",
                bordercolor="rgba(160, 74, 31, 0.20)",
                steps=[
                    dict(
                        method="animate",
                        label=f"{K_grid[k]:.0f}M",
                        args=[
                            [str(f_idx)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for f_idx, k in enumerate(cursor_idx)
                ],
            ),
        ],
    )

    st.plotly_chart(apply_theme(fig_econ), width="stretch")

    # Below the chart: a one-line economic story
    underspend = max(0.0, K_opt - 0.0)
    overspend_pct = 100.0 * (total_K[0] - C_opt) / max(C_opt, 1e-9)
    far_right_pct = 100.0 * (total_K[-1] - C_opt) / max(C_opt, 1e-9)
    st.caption(
        f"At **K = 0** (no mitigation) the total cost is "
        f"${total_K[0]:,.0f}k / yr — ${overspend_pct:.0f}% above the "
        f"optimum. Pushing all the way to **K = K_max = "
        f"${e_kmax}M** also overshoots by ${far_right_pct:.0f}% because "
        f"each marginal dollar of capex now buys very little premium "
        f"reduction. The minimum at **K* = ${K_opt:.1f}M** is the "
        f"Pigouvian-correct level of self-investment in reliability "
        f"under the current premium curve and cost of capital."
    )

    with st.expander("How is each curve specified?"):
        st.markdown(
            f"""
- **Capital cost** is linear in K with slope **r = {e_r:.1f}% / yr** of K
  (K is in USD millions, output is in USD thousands per year, so
  capital cost = r · K · 1000).
- **Premium** decays exponentially from **P₀ = {e_p0} k/yr** at K = 0
  toward the floor **P∞ = {e_p_floor} k/yr** with half-life
  **K½ = {e_k_half} M**. This captures the empirical observation that
  the first few M of redundancy spend buy the biggest premium discount,
  with sharply diminishing returns thereafter.
- **Retained loss** follows the same exponential form:
  **L₀ = {e_d0} k/yr**, **L∞ = {e_l_floor} k/yr**,
  **L½ = {e_l_half} M**.
- **Total** is the pointwise sum. The first-order condition in §14.2
  (∂C/∂K = 0) is where the increasing capital-cost slope exactly cancels
  the decreasing premium-plus-retained-loss slope — that is **K\\***.
"""
        )


# ─────────────────────────────────────────────────────────────
# Mode body — Compare two configurations under the same seed
# ─────────────────────────────────────────────────────────────

elif mode == MODE_COMPARE:
    st.subheader("Side-by-side: two configurations under the same seed")
    st.markdown(
        "Same compound NB-GPD engine, two sets of frequency / tail / "
        "threshold / tail-fraction inputs, identical seed. Useful for "
        "underwriting comparisons: Tier-IV NA versus Tier-III emerging-market, "
        "pre- versus post-cooling retrofit, baseline versus climate-uplifted "
        "wet-bulb. The OEP curves overlay on a shared log-log axis."
    )

    def _sim(lam_x, xi_x, sig_x, u_x, pi_x):
        return simulate_annual_losses(
            n_years=int(c_n_years),
            nu=c_nu, lam=lam_x,
            body_mu=c_body_mu, body_sigma=c_body_sig,
            tail_xi=xi_x, tail_sigma=sig_x, tail_threshold=u_x,
            tail_fraction=pi_x,
            seed=int(c_seed),
        )

    with st.spinner(f"Drawing 2 × {c_n_years:,} annual-loss samples…"):
        S_A = _sim(a_lam, a_xi, a_sig, a_u, a_pi)
        S_B = _sim(b_lam, b_xi, b_sig, b_u, b_pi)

    def _summary(S):
        p995 = float(np.quantile(S, 0.995))
        return {
            "mean":   float(np.mean(S)),
            "std":    float(np.std(S)),
            "p99.5":  p995,
            "p99.9":  float(np.quantile(S, 0.999)),
            "tvar":   float(np.mean(S[S >= p995])) if (S >= p995).any() else p995,
        }

    sa, sb = _summary(S_A), _summary(S_B)

    st.markdown("##### Summary statistics  ($M)")
    summary_df = pd.DataFrame(
        {
            "Configuration A":  [sa["mean"], sa["std"], sa["p99.5"], sa["p99.9"], sa["tvar"]],
            "Configuration B":  [sb["mean"], sb["std"], sb["p99.5"], sb["p99.9"], sb["tvar"]],
            "B / A":            [sb["mean"]/sa["mean"] if sa["mean"] else np.nan,
                                 sb["std"]/sa["std"]   if sa["std"]  else np.nan,
                                 sb["p99.5"]/sa["p99.5"] if sa["p99.5"] else np.nan,
                                 sb["p99.9"]/sa["p99.9"] if sa["p99.9"] else np.nan,
                                 sb["tvar"]/sa["tvar"]  if sa["tvar"]  else np.nan],
        },
        index=["Mean", "Std", "p99.5", "p99.9", "TVaR α=99.5%"],
    )
    fmt = {"Configuration A": "${:,.2f}M",
           "Configuration B": "${:,.2f}M",
           "B / A": "{:.2f}×"}
    st.dataframe(summary_df.style.format(fmt), width="stretch")

    st.markdown("##### OEP curves (overlay)")
    fig = go.Figure()
    for label, S, colour in (
        ("Configuration A", S_A, IA_ACCENT),
        ("Configuration B", S_B, IA_RULE),
    ):
        sorted_S = np.sort(S)
        nN = sorted_S.size
        ex = 1.0 - np.arange(1, nN + 1) / (nN + 1)
        fig.add_trace(
            go.Scatter(
                x=np.clip(sorted_S, 1e-3, None), y=ex, mode="lines",
                line=dict(color=colour, width=2.4), name=label,
                hovertemplate=f"{label}<br>S≥$%{{x:.1f}}M<br>P=%{{y:.4%}}<extra></extra>",
            )
        )
    fig.add_hline(y=0.005, line=dict(color="#9F1239", dash="dash"),
                  annotation_text="1-in-200", annotation_position="right")
    fig.update_layout(
        xaxis_title="annual aggregate loss S  ($M)",
        yaxis_title="exceedance probability  P(S ≥ s)",
        xaxis_type="log", yaxis_type="log",
        height=520,
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(apply_theme(fig), width="stretch")

    st.caption(
        "Both configurations share the body parameters (μ_body, σ_body), "
        "the dispersion ν, and the simulation seed, so the difference at "
        "every percentile is driven only by the frequency and the tail."
    )


# ─────────────────────────────────────────────────────────────
# Mode body — Figures gallery
# ─────────────────────────────────────────────────────────────

else:  # MODE_FIGS
    st.subheader("Paper figures")
    st.markdown(
        "All sixteen figures from the paper, rendered as standalone PNGs. "
        "Each carries the IA bone background and burnt-sienna accent. The "
        "vector PDFs and standalone TeX sources ship with the companion "
        "GitHub repo under `paper/dc_paper_figures/`."
    )

    FIG_DIR = Path(__file__).parent / "figures"
    FIGURES = [
        ("Fig01_fig_singleline-1.png",
         "Fig 1 · Single-line wireframe of a Tier-IV 2N hyperscale data center",
         "§3 — solid lines are power feeders; dashed lines are SCADA telemetry."),
        ("Fig02_fig_fta-1.png",
         "Fig 2 · Fault tree for the top event 'IT load lost'",
         "§5.1 — each E_• is itself a series chain of elementary component events."),
        ("Fig03_fig_markov-1.png",
         "Fig 3 · Continuous-time Markov chain on plant states",
         "§5.2 — λ degradation, μ repair, λ_f degraded→failed, μ_r restoration."),
        ("Fig04_fig_arr-1.png",
         "Fig 4 · Arrhenius temperature acceleration of the baseline failure rate",
         "§15 — ΔT = 10 K doubles to quadruples component hazard depending on E_a."),
        ("Fig05_fig_oep-1.png",
         "Fig 5 · Annual Occurrence Exceedance Probability curve",
         "§16 — Solvency II 1-in-200 anchor (red dashed)."),
        ("Fig06_fig_cooling-topology-1.png",
         "Fig 6 · Three cooling regimes (air / D2C / immersion)",
         "§4.2 — qualitatively different actuarial signatures per architecture."),
        ("Fig07_fig_cool-hazard-1.png",
         "Fig 7 · Cooling hazard vs wet-bulb temperature",
         "§4.4 — saturated plant on a hot day faces double-digit hazard multipliers."),
        ("Fig08_fig_cooling-faulttree-1.png",
         "Fig 8 · Cooling-system fault tree",
         "§4.6 — the liquid-electrical AND branch is rare but catastrophic."),
        ("Fig09_fig_micro-1.png",
         "Fig 9 · Operator's optimisation min C(K)",
         "§14.2 — total cost has a unique interior minimum at K*."),
        ("Fig10_fig_supdem-1.png",
         "Fig 10 · Insurance market equilibrium",
         "§14.3 — capacity shortfall between hyperscale TIV and single-event ceiling."),
        ("Fig11_fig_supdem-shift-1.png",
         "Fig 11 · Comparative statics: shifts in actuarial parameters",
         "§14.3 — frequency / tail-shape / loading / TIV shocks move equilibrium."),
        ("Fig12_fig_compute-mkt-1.png",
         "Fig 12 · Compute-services market equilibrium",
         "§14.4 — passthrough of insurance hard market into cloud / AI prices."),
        ("Fig13_fig_contagion-1.png",
         "Fig 13 · Macro-financial contagion network",
         "§14.10 — capital-market feedback raises operator's cost of capital."),
        ("Fig14_fig_thevenin-1.png",
         "Fig 14 · Thévenin equivalent at the LV bus",
         "§A.3 — source-side impedance stack-up determining I_sc."),
        ("Fig15_fig_ups-1.png",
         "Fig 15 · Double-conversion UPS topology",
         "§A.5 — total η_UPS ≈ 0.96, dominant PUE-overhead contributor."),
        ("Fig16_fig_itic-1.png",
         "Fig 16 · ITIC / CBEMA voltage tolerance envelope",
         "§A.7 — sags below the lower curve trip the UPS to battery."),
    ]

    if not FIG_DIR.exists():
        st.warning(
            f"`{FIG_DIR}` not found. Copy the 16 PNGs from "
            "`paper/dc_paper_figures/png/` into `figures/` next to `app.py` "
            "(or symlink the folder)."
        )

    col_l, col_r = st.columns(2, gap="medium")
    for i, (fname, title, cap) in enumerate(FIGURES):
        col = col_l if i % 2 == 0 else col_r
        path = FIG_DIR / fname
        with col, st.container(border=True):
            st.markdown(
                f'<div style="font-family: ui-monospace, SFMono-Regular, '
                f'Menlo, monospace; font-size: 0.74rem; '
                f'letter-spacing: 0.08em; text-transform: uppercase; '
                f'color: {IA_ACCENT}; font-weight: 700; '
                f'margin-bottom: 0.4rem;">{title}</div>',
                unsafe_allow_html=True,
            )
            if path.exists():
                st.image(str(path), width="stretch")
            else:
                st.info(f"missing: `figures/{fname}`")
            st.caption(cap)


# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Source paper: *A Coupled Power-Thermal-Cyber Framework for the Actuarial "
    "Pricing and Insurance of Hyperscale Data Centers* (Denewade 2026, "
    "Intelligent Actuaries research series, Paper 1). "
    "DOI [10.5281/zenodo.20279225](https://doi.org/10.5281/zenodo.20279225)."
)
