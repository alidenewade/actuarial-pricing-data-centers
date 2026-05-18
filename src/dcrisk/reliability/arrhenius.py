"""Arrhenius temperature-acceleration and voltage-stress hazard models.

Used to translate operating-condition deviations (junction temperature,
bus voltage) into instantaneous failure-rate multipliers that feed the
Cox-process intensity in :mod:`dcrisk.frequency.nb_gamma`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Boltzmann constant in eV/K
K_B_EV_PER_K = 8.617_333_262e-5


def arrhenius_hazard(
    T: ArrayLike,
    *,
    Ea: float = 0.7,
    T_ref: float = 298.15,
    lam_ref: float = 1.0,
) -> NDArray[np.float64]:
    """Arrhenius acceleration factor times a reference failure rate.

    .. math::
       \\lambda(T) = \\lambda_{\\text{ref}} \\,
                     \\exp\\!\\Big(\\tfrac{E_a}{k_B}\\big(\\tfrac{1}{T_{\\text{ref}}} - \\tfrac{1}{T}\\big)\\Big)

    Parameters
    ----------
    T      : junction temperature(s) in Kelvin.
    Ea     : activation energy in eV (silicon devices: ~0.5-0.9 eV).
    T_ref  : reference temperature in Kelvin (default 25 deg C).
    lam_ref: baseline failure rate at T_ref.
    """
    T_arr = np.asarray(T, dtype=np.float64)
    factor = np.exp((Ea / K_B_EV_PER_K) * (1.0 / T_ref - 1.0 / T_arr))
    return lam_ref * factor


def voltage_stress_hazard(
    V: ArrayLike,
    *,
    V_ref: float = 1.0,
    n: float = 3.0,
    lam_ref: float = 1.0,
) -> NDArray[np.float64]:
    """Inverse power-law voltage acceleration (Coffin-Manson form).

    .. math::
       \\lambda(V) = \\lambda_{\\text{ref}} \\, (V / V_{\\text{ref}})^{n}
    """
    V_arr = np.asarray(V, dtype=np.float64)
    return lam_ref * np.power(V_arr / V_ref, n)
