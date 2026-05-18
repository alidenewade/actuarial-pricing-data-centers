"""Bivariate Gaussian copula sampler.

The Gaussian copula has **zero** upper-tail dependence for any rho < 1
(this is a well-known weakness and the reason the paper also uses Gumbel).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass(frozen=True)
class GaussianCopula:
    rho: float

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        """Return n x 2 uniform marginals with rank correlation rho."""
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1).")
        rng = np.random.default_rng() if rng is None else rng
        cov = np.array([[1.0, self.rho], [self.rho, 1.0]])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n, 2)) @ L.T
        return stats.norm.cdf(z)

    def upper_tail_dependence(self) -> float:
        """Analytical: lambda_U = 0 for the Gaussian copula whenever rho < 1."""
        return 0.0 if self.rho < 1.0 else 1.0
