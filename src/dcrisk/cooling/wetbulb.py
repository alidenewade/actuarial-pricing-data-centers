"""Wet-bulb temperature — Stull (2011) empirical approximation.

Stull, R. (2011). "Wet-Bulb Temperature from Relative Humidity and Air
Temperature." J. Appl. Meteor. Climatol. 50, 2267-2269.

Valid range:  T in [-20, 50] C, RH in [5, 99] %.  Accuracy ~0.3 C.
This is paper eq. (5).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def wetbulb_stull(T_C: ArrayLike, RH_pct: ArrayLike) -> NDArray[np.float64]:
    """Stull-2011 wet-bulb temperature [degrees C].

    Parameters
    ----------
    T_C : array-like
        Dry-bulb air temperature in degrees Celsius.
    RH_pct : array-like
        Relative humidity in percent (0-100).
    """
    T = np.asarray(T_C, dtype=np.float64)
    RH = np.asarray(RH_pct, dtype=np.float64)
    Tw = (
        T * np.arctan(0.151_977 * np.sqrt(RH + 8.313_659))
        + np.arctan(T + RH)
        - np.arctan(RH - 1.676_331)
        + 0.003_918_38 * (RH ** 1.5) * np.arctan(0.023_101 * RH)
        - 4.686_035
    )
    return Tw


if __name__ == "__main__":
    # Smoke test — Las Vegas summer afternoon, dry: T=40 C, RH=15 %.
    print(f"T_wb(40 C, 15 %)  = {wetbulb_stull(40.0, 15.0):.2f} C   (expected ~19)")
    # Singapore — humid, warm: T=30 C, RH=85 %.
    print(f"T_wb(30 C, 85 %)  = {wetbulb_stull(30.0, 85.0):.2f} C   (expected ~28)")
