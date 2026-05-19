"""Severity model — paper §8: lognormal body, GPD tail (POT).

Implements only the CPU operations that the Space needs (sample, quantile,
TVaR). The full statistics+inference machinery lives in dcrisk.severity.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------- Lognormal body ----------------------------------------------------

def lognormal_sample(mu: float, sigma: float, size: int, rng) -> np.ndarray:
    """Draw size samples from LogN(mu, sigma^2)."""
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def lognormal_quantile(p: float, mu: float, sigma: float) -> float:
    return float(stats.lognorm.ppf(p, s=sigma, scale=np.exp(mu)))


# ---------- Generalised Pareto tail (POT) ------------------------------------

def gpd_quantile(p, xi: float, sigma: float, threshold: float = 0.0):
    """Inverse CDF of GPD(ξ, σ) shifted to `threshold`. Vectorised over p."""
    p = np.asarray(p)
    if abs(xi) < 1e-9:
        return threshold + sigma * (-np.log1p(-p))
    return threshold + (sigma / xi) * ((1 - p) ** (-xi) - 1)


def gpd_sample(xi: float, sigma: float, threshold: float, size: int, rng) -> np.ndarray:
    """Inverse-CDF sampling from GPD."""
    u = rng.uniform(size=size)
    return gpd_quantile(u, xi, sigma, threshold)


def gpd_tvar(alpha: float, xi: float, sigma: float, threshold: float = 0.0) -> float:
    """Tail-VaR at level α (closed form for GPD with ξ < 1)."""
    if xi >= 1.0:
        return float("inf")
    var_alpha = gpd_quantile(alpha, xi, sigma, threshold)
    return float((var_alpha + sigma - xi * threshold) / (1.0 - xi))


# ---------- Compound NB-GPD aggregate (single year, no climate uplift) -------

def compound_one_year(
    nu: float, lam: float,
    body_mu: float, body_sigma: float,
    tail_xi: float, tail_sigma: float, tail_threshold: float,
    tail_fraction: float,
    rng,
) -> float:
    """One realisation of S = Σ X_j with frequency NB(ν, ν/(ν+λ)) and
    body/tail mixture severity (with prob `tail_fraction` we draw GPD).
    """
    p = nu / (nu + lam)
    n_events = int(rng.negative_binomial(nu, p))
    if n_events == 0:
        return 0.0
    is_tail = rng.uniform(size=n_events) < tail_fraction
    n_tail  = int(is_tail.sum())
    n_body  = n_events - n_tail
    body    = lognormal_sample(body_mu, body_sigma, n_body, rng) if n_body else np.empty(0)
    tail    = gpd_sample(tail_xi, tail_sigma, tail_threshold, n_tail, rng) if n_tail else np.empty(0)
    return float(body.sum() + tail.sum())


def simulate_annual_losses(
    n_years: int,
    nu: float, lam: float,
    body_mu: float, body_sigma: float,
    tail_xi: float, tail_sigma: float, tail_threshold: float,
    tail_fraction: float,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_years, dtype=np.float64)
    for y in range(n_years):
        out[y] = compound_one_year(
            nu, lam, body_mu, body_sigma,
            tail_xi, tail_sigma, tail_threshold, tail_fraction,
            rng,
        )
    return out
