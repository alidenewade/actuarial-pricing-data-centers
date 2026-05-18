"""Coupled power-thermal-cyber SDE (paper §10, eq. 11).

State vector :math:`X_t = (V_t, T_t, C_t)` where

    V_t : bus voltage [p.u.],
    T_t : junction temperature [K],
    C_t : cyber-threat index [unitless, in [0, 1]].

Dynamics (Euler-Maruyama integration):

    dV_t = kappa_V (V_bar - V_t) dt + sigma_V dW^V_t  +  J^V dN_t
    dT_t = ( kappa_T (T_bar - T_t)  +  beta_T (V_t - V_bar)^2 ) dt  +  sigma_T dW^T_t
    dC_t = kappa_C (C_bar - C_t) dt + sigma_C dW^C_t

with optional voltage-sag jumps from a Poisson process N_t of intensity
lambda_jump and jump size J^V ~ N(mu_J, sigma_J^2).

The thermal drift is **coupled** to voltage (resistive heating proxy: bus
deviations dissipate energy ∝ (V - V_bar)^2). The Cox-process intensity in
:mod:`dcrisk.frequency.nb_gamma` consumes these paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PTCyberParams:
    # mean-reversion levels
    V_bar: float = 1.00
    T_bar: float = 320.0     # 47 deg C — warm-aisle baseline
    C_bar: float = 0.10

    # mean-reversion speeds (1/hr)
    kappa_V: float = 4.0
    kappa_T: float = 0.6
    kappa_C: float = 0.5

    # diffusion volatilities
    sigma_V: float = 0.02
    sigma_T: float = 1.5
    sigma_C: float = 0.05

    # voltage->temperature coupling
    beta_T: float = 25.0

    # voltage-sag jump process
    lambda_jump: float = 0.05   # jumps per hour
    mu_J: float = -0.10         # mean jump in V (sag)
    sigma_J: float = 0.04


def simulate_ptcyber(
    T_max: float,
    dt: float = 1 / 60.0,
    n_paths: int = 1,
    params: PTCyberParams | None = None,
    X0: tuple[float, float, float] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Euler-Maruyama integrator for the coupled (V, T, C) SDE.

    Returns
    -------
    t  : 1-D array of shape (N+1,) of time points
    X  : 3-D array of shape (n_paths, N+1, 3) of state values
    """
    p = params or PTCyberParams()
    rng = np.random.default_rng() if rng is None else rng

    N = int(np.ceil(T_max / dt))
    t = np.linspace(0.0, N * dt, N + 1)
    X = np.empty((n_paths, N + 1, 3), dtype=np.float64)
    X[:, 0, 0] = p.V_bar if X0 is None else X0[0]
    X[:, 0, 1] = p.T_bar if X0 is None else X0[1]
    X[:, 0, 2] = p.C_bar if X0 is None else X0[2]

    sqrt_dt = np.sqrt(dt)
    # pre-draw all noise
    Z = rng.standard_normal((n_paths, N, 3))
    jumps_count = rng.poisson(p.lambda_jump * dt, size=(n_paths, N))
    jump_sizes = rng.normal(p.mu_J, p.sigma_J, size=(n_paths, N))

    for k in range(N):
        V = X[:, k, 0]
        T = X[:, k, 1]
        C = X[:, k, 2]

        # drifts
        muV = p.kappa_V * (p.V_bar - V)
        muT = p.kappa_T * (p.T_bar - T) + p.beta_T * (V - p.V_bar) ** 2
        muC = p.kappa_C * (p.C_bar - C)

        X[:, k + 1, 0] = V + muV * dt + p.sigma_V * sqrt_dt * Z[:, k, 0] + jumps_count[:, k] * jump_sizes[:, k]
        X[:, k + 1, 1] = T + muT * dt + p.sigma_T * sqrt_dt * Z[:, k, 1]
        X[:, k + 1, 2] = np.clip(C + muC * dt + p.sigma_C * sqrt_dt * Z[:, k, 2], 0.0, 1.0)

    return t, X
