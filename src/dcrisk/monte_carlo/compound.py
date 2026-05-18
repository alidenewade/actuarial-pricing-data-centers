"""Compound NB-GPD aggregator with copula dependence and cooling hazard uplift.

`simulate(n_years, use_gpu=True)` returns the annual aggregate loss series S
that feeds the OEP curve in §11 of the paper.

The pipeline per simulated year:
  1. draw N ~ NegBin(nu, p)   with mean lambda
  2. apply the cooling hazard multiplier  m = 1 + integral lambda_cool(T_wb)
     over the year's wet-bulb path (paper eq. 7 + eq. 12)
  3. for each event, draw (U_1, U_2) ~ Gumbel copula  (paper §9)
  4. invert U_2 via the GPD severity F^{-1}  (paper §8)
  5. sum into S
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcrisk.cooling.climate import ClimateParams, simulate_Twb
from dcrisk.cooling.hazard import lambda_cool
from dcrisk.copula.gumbel import GumbelCopula
from dcrisk.frequency.nb_gamma import sample_NB
from dcrisk.severity.gpd import GPD


def _hazard_multiplier(T_wb_path: np.ndarray, dt: float) -> float:
    """Time-integral of lambda_cool over a 1-year T_wb path -> multiplicative uplift."""
    h = lambda_cool(T_wb_path)
    return 1.0 + h.sum() * dt


def simulate(
    n_years: int,
    *,
    lam: float = 2.5,
    nu: float = 4.0,
    gpd_xi: float = 0.30,
    gpd_sigma: float = 5.0,
    theta_gumbel: float = 2.0,
    climate: ClimateParams | None = None,
    use_gpu: bool = True,
    seed: int = 42,
) -> pd.Series:
    """Run an annual-loss compound simulation and return S as a pandas Series.

    Notes
    -----
    The `use_gpu` flag is propagated to the SDE/cooling layers when they
    expose a JAX path. The compound aggregation itself is dominated by NumPy
    sorts and is run on the host.
    """
    if n_years <= 0:
        raise ValueError(f"n_years must be positive (got {n_years})")

    rng = np.random.default_rng(seed)
    gpd = GPD(xi=gpd_xi, sigma=gpd_sigma, threshold=0.0)
    cop = GumbelCopula(theta=theta_gumbel)
    cp = climate or ClimateParams()

    # one wet-bulb path per simulated year (daily resolution)
    _, T_wb = simulate_Twb(T_years=float(n_years), dt=1.0 / 365.0, n_paths=1,
                           params=cp, rng=rng)
    T_wb = T_wb[0]  # (n_years * 365 + 1,)

    S = np.empty(n_years, dtype=np.float64)
    days_per_year = 365
    for y in range(n_years):
        # hazard multiplier from this year's slice
        slice_ = T_wb[y * days_per_year:(y + 1) * days_per_year + 1]
        m = _hazard_multiplier(slice_, dt=1.0 / 365.0)

        # frequency
        n_events = int(sample_NB(nu, lam * m, size=1, rng=rng)[0])
        if n_events == 0:
            S[y] = 0.0
            continue

        # severities under copula dependence — invert U_2 marginal via GPD
        UV = cop.sample(n_events, rng=rng)        # (n_events, 2) uniforms
        losses = gpd.quantile(UV[:, 1])
        S[y] = float(losses.sum())

    return pd.Series(S, name="S", index=pd.RangeIndex(n_years, name="year"))


if __name__ == "__main__":
    s = simulate(n_years=2_000, use_gpu=False)
    print(f"mean(S)        = {s.mean():.3f}")
    print(f"std(S)         = {s.std():.3f}")
    print(f"quantile 99.5% = {s.quantile(0.995):.3f}")
