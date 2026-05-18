"""Cooling-loss hazard — paper §4 eq. (7) (cooling block).

The cooling-loss hazard rate is

    lambda_cool(T_wb) = lambda_0 * exp(alpha * (T_wb - T_wb_design)) * 1[T_wb > T_wb_design]

so the hazard switches on only once the wet-bulb temperature exceeds the
plant's design point (the indicator gate). alpha controls how sharply the
hazard grows once the threshold is breached.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def lambda_cool(
    T_wb: ArrayLike,
    *,
    lambda_0: float = 1.0e-3,
    alpha: float = 0.25,
    T_wb_design: float = 24.0,
) -> NDArray[np.float64]:
    """Cooling-loss hazard rate (per hour).

    Parameters
    ----------
    T_wb : array-like
        Wet-bulb temperature [degrees C].
    lambda_0 : float
        Baseline hazard at the design point.
    alpha : float
        Exponential slope per degree of overshoot.
    T_wb_design : float
        Design wet-bulb temperature; below this the gate is closed (hazard = 0).
    """
    Tw = np.asarray(T_wb, dtype=np.float64)
    gate = (Tw > T_wb_design).astype(np.float64)
    return lambda_0 * np.exp(alpha * (Tw - T_wb_design)) * gate


if __name__ == "__main__":
    T_wb_grid = np.linspace(18.0, 32.0, 8)
    h = lambda_cool(T_wb_grid)
    for t, val in zip(T_wb_grid, h):
        print(f"T_wb = {t:5.1f} C  ->  lambda_cool = {val:.4e}/hr")
