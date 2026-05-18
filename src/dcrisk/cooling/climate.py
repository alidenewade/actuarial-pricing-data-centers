"""Non-stationary wet-bulb climate process — paper eq. (12).

    dT_wb(t) = ( mu + beta * t ) dt + sigma * dW_t

with beta the long-run climate-warming drift (degrees C per year).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClimateParams:
    mu: float = 0.0          # baseline drift [C / year]
    beta: float = 0.03       # climate-warming drift [C / year^2]
    sigma: float = 1.2       # annual volatility [C / sqrt(year)]
    T0: float = 22.0         # initial wet-bulb [C]


def simulate_Twb(
    T_years: float,
    dt: float = 1.0 / 365.0,
    n_paths: int = 1,
    params: ClimateParams | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Euler-Maruyama simulation of the non-stationary T_wb(t) process.

    Returns
    -------
    t : (N+1,) time grid in years
    X : (n_paths, N+1) wet-bulb trajectories
    """
    p = params or ClimateParams()
    rng = np.random.default_rng() if rng is None else rng

    N = int(np.ceil(T_years / dt))
    t = np.linspace(0.0, N * dt, N + 1)
    X = np.empty((n_paths, N + 1), dtype=np.float64)
    X[:, 0] = p.T0

    sqrt_dt = np.sqrt(dt)
    Z = rng.standard_normal((n_paths, N))
    for k in range(N):
        drift = (p.mu + p.beta * t[k]) * dt
        X[:, k + 1] = X[:, k] + drift + p.sigma * sqrt_dt * Z[:, k]
    return t, X


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t, X = simulate_Twb(T_years=10.0, n_paths=200, rng=rng)
    print(f"mean T_wb at t=0  : {X[:, 0].mean():.2f} C")
    print(f"mean T_wb at t=10 : {X[:, -1].mean():.2f} C   (drift expected ~+1.5 C with beta=0.03)")
