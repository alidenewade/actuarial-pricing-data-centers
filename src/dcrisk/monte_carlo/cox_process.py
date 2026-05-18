"""Doubly-stochastic Poisson (Cox) process via thinning — paper §6.

Given a stochastic intensity path lambda(t) sampled on a uniform grid, the
ordinary-thinning algorithm draws candidate event times from a homogeneous
Poisson process with rate lambda_max and keeps each candidate t with
probability lambda(t)/lambda_max.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sample_cox(
    t_grid: NDArray[np.float64],
    lambda_path: NDArray[np.float64],
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Sample event times of a Cox process by thinning.

    Parameters
    ----------
    t_grid : (N+1,) float
        Monotone increasing time grid.
    lambda_path : (N+1,) float
        Non-negative intensity at each grid point.
    """
    if t_grid.shape != lambda_path.shape:
        raise ValueError("t_grid and lambda_path must have the same shape")
    if np.any(lambda_path < 0):
        raise ValueError("lambda_path must be non-negative")

    rng = np.random.default_rng() if rng is None else rng
    T0, T1 = float(t_grid[0]), float(t_grid[-1])
    lam_max = float(lambda_path.max())
    if lam_max <= 0.0:
        return np.empty(0, dtype=np.float64)

    # candidate count from a homogeneous Poisson(lam_max, T1 - T0)
    n_cand = rng.poisson(lam_max * (T1 - T0))
    if n_cand == 0:
        return np.empty(0, dtype=np.float64)
    cand = np.sort(rng.uniform(T0, T1, size=n_cand))

    # interpolate intensity at candidate times
    lam_at = np.interp(cand, t_grid, lambda_path)
    keep = rng.uniform(size=n_cand) < (lam_at / lam_max)
    return cand[keep]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, 1001)
    lam = 5.0 * (1.0 + 0.8 * np.sin(2.0 * np.pi * t))    # sinusoidal intensity
    times = sample_cox(t, lam, rng=rng)
    print(f"realised events: {len(times)}  (expected ~5)")
