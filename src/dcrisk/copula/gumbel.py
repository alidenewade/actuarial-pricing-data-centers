"""Bivariate Gumbel copula sampler.

Sampling uses the Marshall-Olkin frailty representation: if M is a positive
stable variable with stability parameter alpha = 1 / theta, then

    U_i = exp(-(-log V_i / M) ** (1 / theta))

with independent V_i ~ U(0, 1) gives a sample from the Gumbel(theta) copula.

Upper tail dependence is closed form:

    lambda_U = 2 - 2 ** (1 / theta)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GumbelCopula:
    theta: float  # dependence parameter, theta >= 1

    def __post_init__(self) -> None:
        if self.theta < 1.0:
            raise ValueError("Gumbel theta must be >= 1 (theta=1 -> independence).")

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng() if rng is None else rng
        if self.theta == 1.0:
            return rng.random((n, 2))
        alpha = 1.0 / self.theta
        # Positive alpha-stable via Chambers-Mallows-Stuck
        M = _positive_stable(alpha, n, rng)
        V = rng.random((n, 2))
        U = np.exp(-((-np.log(V).T / M).T) ** (1.0 / self.theta))
        # numerical safety
        return np.clip(U, 1e-12, 1.0 - 1e-12)

    def upper_tail_dependence(self) -> float:
        return float(2.0 - 2.0 ** (1.0 / self.theta))


def _positive_stable(alpha: float, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Chambers-Mallows-Stuck generator for a positive alpha-stable RV (alpha in (0,1))."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1) for a positive stable.")
    u = rng.uniform(0.0, np.pi, size=n)
    w = rng.exponential(1.0, size=n)
    term1 = np.sin(alpha * u) / (np.sin(u) ** (1.0 / alpha))
    term2 = (np.sin((1.0 - alpha) * u) / w) ** ((1.0 - alpha) / alpha)
    return term1 * term2
