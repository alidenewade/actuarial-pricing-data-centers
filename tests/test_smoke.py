"""Smoke tests — every subpackage must import and produce sane values.

Each test is fast (< 100 ms) and uses fixed seeds so failures are deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------

def test_top_level_import():
    import dcrisk
    assert dcrisk.__version__ == "0.1.0"
    assert dcrisk.__author__  == "Ali Denewade"


def test_all_subpackages_import():
    from dcrisk import copula, econ, frequency, pricing, reliability, severity, sde  # noqa
    # touch one symbol from each so flake-8 doesn't strip them
    for mod in (reliability, severity, frequency, copula, sde, pricing, econ):
        assert hasattr(mod, "__all__")


# ---------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------

def test_markov_availability_in_unit_interval():
    from dcrisk.reliability.markov import MarkovParams, build_Q, availability, mtbf_from_Q
    Q = build_Q(MarkovParams(lam=1/8760, lam_f=1/4380, mu=1/24, mu_r=1/8))
    a = availability(Q)
    assert 0.0 < a < 1.0
    assert a > 0.99  # well-managed DC
    assert mtbf_from_Q(Q) > 0


def test_arrhenius_monotone_in_T():
    from dcrisk.reliability.arrhenius import arrhenius_hazard
    Ts = np.array([300.0, 320.0, 340.0])
    h = arrhenius_hazard(Ts)
    assert np.all(np.diff(h) > 0)


def test_fault_tree_or_vs_and():
    from dcrisk.reliability.fault_tree import AndGate, FaultTree, Leaf, OrGate
    leaves = [Leaf(name=f"L{i}", p=0.1) for i in range(3)]
    or_p  = FaultTree(OrGate(name="or",  children=leaves)).top_event_probability()
    and_p = FaultTree(AndGate(name="and", children=leaves)).top_event_probability()
    assert and_p < or_p
    assert abs(or_p - (1 - 0.9**3)) < 1e-12
    assert abs(and_p - 0.1**3) < 1e-12


# ---------------------------------------------------------------------------
# severity
# ---------------------------------------------------------------------------

def test_lognormal_fit_and_sample():
    from dcrisk.severity.lognormal import Lognormal
    rng = np.random.default_rng(0)
    x = rng.lognormal(1.0, 0.7, size=2000)
    fit = Lognormal.fit(x)
    assert abs(fit.mu - 1.0) < 0.05
    assert abs(fit.sigma - 0.7) < 0.05


def test_gpd_mle_recovers_parameters():
    from dcrisk.severity.gpd import GPD
    rng = np.random.default_rng(1)
    true = GPD(xi=0.3, sigma=2.0, threshold=0.0)
    sample = true.sample(5000, rng)
    fit = GPD.fit_mle(sample, threshold=0.0)
    assert abs(fit.xi    - 0.3) < 0.05
    assert abs(fit.sigma - 2.0) < 0.2


# ---------------------------------------------------------------------------
# frequency
# ---------------------------------------------------------------------------

def test_nb_mean_matches_lambda_bar():
    from dcrisk.frequency.nb_gamma import sample_NB
    counts = sample_NB(nu=4.0, lam_bar=5.0, size=20_000, rng=np.random.default_rng(2))
    assert abs(counts.mean() - 5.0) < 0.15


def test_cox_intensity_increases_with_temperature():
    from dcrisk.frequency.nb_gamma import Cox_intensity
    X1 = np.array([[1.0, 310.0, 0.0]])
    X2 = np.array([[1.0, 340.0, 0.0]])
    assert Cox_intensity(0.0, X2)[0] > Cox_intensity(0.0, X1)[0]


# ---------------------------------------------------------------------------
# copula
# ---------------------------------------------------------------------------

def test_gumbel_upper_tail_dependence_closed_form():
    from dcrisk.copula.gumbel import GumbelCopula
    assert abs(GumbelCopula(theta=2.0).upper_tail_dependence() - (2 - 2**0.5)) < 1e-12


def test_gaussian_zero_upper_tail():
    from dcrisk.copula.gaussian import GaussianCopula
    assert GaussianCopula(rho=0.8).upper_tail_dependence() == 0.0


def test_copula_samples_in_unit_square():
    from dcrisk.copula.gaussian import GaussianCopula
    from dcrisk.copula.gumbel import GumbelCopula
    rng = np.random.default_rng(3)
    for cop in (GaussianCopula(rho=0.5), GumbelCopula(theta=1.8)):
        U = cop.sample(1000, rng)
        assert U.shape == (1000, 2)
        assert U.min() >= 0.0 and U.max() <= 1.0


# ---------------------------------------------------------------------------
# sde
# ---------------------------------------------------------------------------

def test_ptcyber_paths_finite_and_shape_correct():
    from dcrisk.sde.ptcyber import PTCyberParams, simulate_ptcyber
    t, X = simulate_ptcyber(T_max=24.0, dt=1/12, n_paths=2,
                            params=PTCyberParams(),
                            rng=np.random.default_rng(4))
    assert X.shape == (2, t.size, 3)
    assert np.isfinite(X).all()
    # cyber index stays in [0, 1]
    assert X[:, :, 2].min() >= 0.0 and X[:, :, 2].max() <= 1.0


# ---------------------------------------------------------------------------
# pricing
# ---------------------------------------------------------------------------

def test_pure_premium_and_loadings_order():
    from dcrisk.pricing.loadings import esscher_premium, sd_premium, variance_premium
    from dcrisk.pricing.pure import pure_premium
    rng = np.random.default_rng(5)
    S = rng.gamma(shape=2.0, scale=3.0, size=10_000)
    pp = pure_premium(S)
    assert sd_premium(S, 0.5) > pp
    assert variance_premium(S, 0.01) > pp
    assert esscher_premium(S, 0.05) > pp


def test_xol_monte_carlo_and_closed_form_close():
    from dcrisk.severity.gpd import GPD
    from dcrisk.pricing.xol import xol_expected_ceded, xol_expected_ceded_gpd
    rng = np.random.default_rng(6)
    g = GPD(xi=0.25, sigma=2.0, threshold=0.5)
    sample = g.sample(50_000, rng)
    p_u = (sample > g.threshold).mean()
    mc = xol_expected_ceded(sample, d=1.5, ell=np.inf)
    cf = xol_expected_ceded_gpd(g, d=1.5, p_u=p_u, ell=np.inf)
    assert abs(mc - cf) / cf < 0.10  # within 10% on 50k draws


# ---------------------------------------------------------------------------
# econ
# ---------------------------------------------------------------------------

def test_operator_optimum_within_grid():
    from dcrisk.econ.operator import find_optimum
    rng = np.random.default_rng(7)
    S = rng.gamma(2.0, 3.0, size=10_000)
    K_star, C_star = find_optimum(r=0.10, lambda_K=1.20, xi_K=0.20, S_samples=S)
    assert 0.0 <= K_star <= np.quantile(S, 0.995)
    assert C_star > 0


def test_market_equilibrium_positive():
    from dcrisk.econ.market import equilibrium
    P, Q = equilibrium(lam=1.0, xi=0.2, theta=1.5, theta_op=1.0)
    assert P > 0 and Q > 0


def test_incidence_shares_sum_to_one():
    from dcrisk.econ.incidence import passthrough_table
    df = passthrough_table({"xi": [0.1, 0.3]})
    for _, row in df.iterrows():
        assert abs(row["share_operator"] + row["share_insurer"] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# dashboards (import-only — full Streamlit run requires its own runtime)
# ---------------------------------------------------------------------------

def test_streamlit_module_imports():
    pytest.importorskip("streamlit")
    from dcrisk.dashboards import streamlit_app
    assert hasattr(streamlit_app, "main")


# ---------------------------------------------------------------------------
# compound aggregator (regression guard against the climate-drift OOM)
# ---------------------------------------------------------------------------

def test_compound_simulate_long_horizon_stays_finite():
    # Regression for the per-year-resample fix: with the old code,
    # simulate(n_years=1000) drifted T_wb to thousands of degrees, hazard
    # exp()-overflowed, NB returned ~1e15 events, and the kernel was killed
    # before this assertion could even run. The fix decouples per-year
    # scenarios from a single multi-decade calendar.
    from dcrisk.monte_carlo.compound import simulate
    s = simulate(n_years=1000, use_gpu=False, seed=42)
    arr = np.asarray(s)
    assert arr.shape == (1000,)
    assert np.all(np.isfinite(arr)), "S must be finite at long horizons"
    assert arr.min() >= 0.0, "annual aggregate losses are non-negative"
    # Loose physical bound — pre-fix this would have been ~1e15+; with the
    # fix p99.5 stays in the low hundreds under the default parameters.
    assert np.quantile(arr, 0.995) < 1e4, (
        f"p99.5 = {np.quantile(arr, 0.995):.2e} — climate drift may be back"
    )
