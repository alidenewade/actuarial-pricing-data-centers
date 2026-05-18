"""Cooling thermodynamics — COP, Carnot limit, PUE decomposition.

Paper §4, eq. (4):  PUE = 1 + 1/COP + epsilon_other
where COP <= COP_Carnot = T_cold / (T_hot - T_cold).
"""

from __future__ import annotations

from dataclasses import dataclass


def carnot_cop(T_cold: float, T_hot: float) -> float:
    """Carnot upper bound on the coefficient of performance.

    Parameters
    ----------
    T_cold, T_hot : float
        Absolute (Kelvin) temperatures of the cold and hot reservoirs.
        T_hot > T_cold is required.
    """
    if T_hot <= T_cold:
        raise ValueError(f"T_hot must exceed T_cold (got T_hot={T_hot}, T_cold={T_cold})")
    return T_cold / (T_hot - T_cold)


def cop(T_cold: float, T_hot: float, eta_II: float = 0.55) -> float:
    """Second-law efficiency adjusted COP.

    eta_II is the fraction of the Carnot limit achievable by real chillers;
    0.45-0.60 is typical for hyperscale water-cooled plants (paper §4.2).
    """
    if not 0.0 < eta_II <= 1.0:
        raise ValueError(f"eta_II must lie in (0, 1] (got {eta_II})")
    return eta_II * carnot_cop(T_cold, T_hot)


@dataclass(frozen=True)
class PUEDecomposition:
    cop_value: float
    overhead: float            # epsilon_other (lights, UPS losses, distribution)
    pue: float                 # 1 + 1/COP + overhead


def pue(T_cold: float, T_hot: float, eta_II: float = 0.55, overhead: float = 0.08) -> PUEDecomposition:
    """PUE decomposition consistent with paper eq. (4)."""
    cop_value = cop(T_cold, T_hot, eta_II)
    return PUEDecomposition(cop_value=cop_value, overhead=overhead, pue=1.0 + 1.0 / cop_value + overhead)


if __name__ == "__main__":
    # Smoke test — ASHRAE A1 envelope: chilled water 7 C (280 K), hot side 35 C (308 K).
    decomp = pue(T_cold=280.0, T_hot=308.0)
    print(f"COP = {decomp.cop_value:.2f}, PUE = {decomp.pue:.3f} (overhead {decomp.overhead:.2f})")
