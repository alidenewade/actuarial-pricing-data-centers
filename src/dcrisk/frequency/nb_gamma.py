"""Negative-Binomial frequency model + Cox-process state-dependent intensity.

Parameterisation
----------------
We use the Gamma-mixed Poisson form:

    N | Lambda ~ Poisson(Lambda),   Lambda ~ Gamma(nu, nu / lam_bar)

so that  N ~ NegBin(nu, p)  with  p = nu / (nu + lam_bar)  and
E[N] = lam_bar, Var[N] = lam_bar * (1 + lam_bar / nu).

The Cox-process intensity is a deterministic function of the state vector
X_t coming out of :mod:`dcrisk.sde.ptcyber`, multiplied by Arrhenius and
voltage-stress hazards from :mod:`dcrisk.reliability.arrhenius`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dcrisk.reliability.arrhenius import arrhenius_hazard, voltage_stress_hazard


def sample_NB(
    nu: float,
    lam_bar: float,
    size: int = 1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Sample from NegBin(nu, nu/(nu+lam_bar))."""
    if nu <= 0 or lam_bar < 0:
        raise ValueError("nu must be positive and lam_bar non-negative.")
    rng = np.random.default_rng() if rng is None else rng
    p = nu / (nu + lam_bar)
    # numpy uses (n, p) with E = n*(1-p)/p ; matches our parameterisation
    return rng.negative_binomial(nu, p, size=size).astype(np.int64)


def Cox_intensity(
    t: float | ArrayLike,
    X_t: ArrayLike,
    *,
    lam0: float = 1.0,
    Ea: float = 0.7,
    n_volt: float = 3.0,
    cyber_coef: float = 0.5,
) -> NDArray[np.float64]:
    """Cox-process intensity as a function of the (V, T, C) state vector.

    Parameters
    ----------
    t       : time(s) — kept in the signature for nonstationary extensions.
    X_t     : array of shape (..., 3) holding (V, T, C):
                V — bus voltage [p.u., 1.0 = nominal]
                T — junction temperature [K]
                C — cyber-threat index [unitless, 0 = quiet, 1 = active]
    lam0    : baseline intensity at nominal V, reference T, C=0.
    Ea      : Arrhenius activation energy (eV).
    n_volt  : voltage-stress exponent.
    cyber_coef : multiplicative factor exp(cyber_coef * C).

    Returns
    -------
    Array of shape (...,) with state-conditional intensities.
    """
    X = np.asarray(X_t, dtype=np.float64)
    if X.shape[-1] != 3:
        raise ValueError("X_t last axis must be 3 (V, T, C).")
    V = X[..., 0]
    T = X[..., 1]
    C = X[..., 2]
    therm = arrhenius_hazard(T, Ea=Ea, lam_ref=1.0)
    volt = voltage_stress_hazard(V, n=n_volt, lam_ref=1.0)
    cyber = np.exp(cyber_coef * C)
    return lam0 * therm * volt * cyber
