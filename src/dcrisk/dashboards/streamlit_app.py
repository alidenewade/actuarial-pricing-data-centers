"""Streamlit dashboard — interactive exploration of the paper's loss model.

Launch with:
    streamlit run src/dcrisk/dashboards/streamlit_app.py
or:
    make app

Sidebar sliders control (lambda, xi, sigma, rho, theta). Three live plots:
    (a) OEP curve (1 - F_S) on log-log axes
    (b) Operator total cost C(K) with optimum marker
    (c) Supply-demand equilibrium with comparative-statics overlay
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from dcrisk.copula.gumbel import GumbelCopula
from dcrisk.econ.market import equilibrium, plot_equilibrium
from dcrisk.econ.operator import find_optimum, plot_operator_cost
from dcrisk.frequency.nb_gamma import sample_NB
from dcrisk.severity.gpd import GPD


def main() -> None:
    st.set_page_config(page_title="dcrisk dashboard", layout="wide")
    st.title("dcrisk — interactive loss & pricing model")
    st.caption("Companion to *A Coupled Power-Thermal-Cyber Framework for the "
               "Actuarial Pricing and Insurance of Hyperscale Data Centers* "
               "(Denewade, 2026).")

    with st.sidebar:
        st.header("Model parameters")
        lam = st.slider("lambda (events/year)",   0.1, 10.0, 2.5,  0.1)
        xi  = st.slider("xi (GPD shape)",        -0.5, 0.95, 0.30, 0.01)
        sig = st.slider("sigma (GPD scale)",      0.1, 50.0, 5.0,  0.1)
        rho = st.slider("rho (Gaussian copula)", -0.99, 0.99, 0.5, 0.01)  # noqa: F841 - reserved
        the = st.slider("theta (Gumbel copula)",  1.0, 10.0, 2.0,  0.1)
        nu  = st.slider("nu (NB shape)",          0.5, 20.0, 4.0,  0.5)
        n_sims = st.select_slider("Simulations", options=[1_000, 5_000, 10_000, 25_000, 50_000], value=10_000)
        st.divider()
        st.header("Operator cost")
        r        = st.slider("r (capital cost)",   0.0, 0.5,  0.10, 0.01)
        lam_K    = st.slider("lambda_K (ceded EV)", 1.0, 2.0,  1.20, 0.01)
        xi_K     = st.slider("xi_K (risk load)",    0.0, 1.0,  0.20, 0.01)

    rng = np.random.default_rng(42)
    gpd = GPD(xi=xi, sigma=sig, threshold=0.0)
    counts = sample_NB(nu, lam, size=n_sims, rng=rng)
    S = np.array([gpd.sample(int(c), rng).sum() if c > 0 else 0.0 for c in counts])

    col1, col2, col3 = st.columns(3)

    # --- (a) OEP curve ---
    with col1:
        st.subheader("OEP curve")
        S_sorted = np.sort(S)
        n = S_sorted.size
        exceed = 1.0 - np.arange(1, n + 1) / (n + 1)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(np.clip(S_sorted, 1e-9, None), exceed, lw=1.5)
        ax.set_xlabel("Aggregate loss S")
        ax.set_ylabel("Exceedance probability")
        ax.grid(which="both", alpha=0.3)
        st.pyplot(fig, clear_figure=True)

    # --- (b) Operator C(K) ---
    with col2:
        st.subheader("Operator cost C(K)")
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_operator_cost(r, lam_K, xi_K, S, ax=ax)
        st.pyplot(fig, clear_figure=True)
        K_star, C_star = find_optimum(r, lam_K, xi_K, S)
        st.metric("K*", f"{K_star:,.2f}")
        st.metric("C(K*)", f"{C_star:,.2f}")

    # --- (c) Market equilibrium ---
    with col3:
        st.subheader("Market equilibrium")
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_equilibrium(lam=lam, xi=xi, theta=the, shifts={"xi": min(xi + 0.2, 0.95)}, ax=ax)
        st.pyplot(fig, clear_figure=True)
        P_star, Q_star = equilibrium(lam=lam, xi=xi, theta=the)
        st.metric("P*", f"{P_star:,.2f}")
        st.metric("Q*", f"{Q_star:,.2f}")

    st.divider()
    st.subheader("Gumbel upper-tail dependence (closed form)")
    st.latex(r"\lambda_U = 2 - 2^{1/\theta}")
    st.metric(f"lambda_U at theta={the:.2f}", f"{GumbelCopula(the).upper_tail_dependence():.4f}")


if __name__ == "__main__":
    main()
