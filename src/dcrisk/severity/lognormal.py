"""Lognormal severity distribution for the body of the loss curve.

Thin convenience wrapper around scipy.stats with explicit fit /
sample / quantile / TVaR helpers, so the API matches the GPD class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats


@dataclass(frozen=True)
class Lognormal:
    """Lognormal(mu, sigma) on the natural-log scale."""

    mu: float
    sigma: float

    # ---- factory ----------------------------------------------------------

    @classmethod
    def fit(cls, x: ArrayLike) -> "Lognormal":
        """MLE fit on a non-negative sample."""
        x_arr = np.asarray(x, dtype=np.float64)
        if (x_arr <= 0).any():
            raise ValueError("Lognormal fit requires strictly positive observations.")
        logs = np.log(x_arr)
        return cls(mu=float(logs.mean()), sigma=float(logs.std(ddof=1)))

    # ---- distribution interface ------------------------------------------

    def _frozen(self):
        return stats.lognorm(s=self.sigma, scale=np.exp(self.mu))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng() if rng is None else rng
        return rng.lognormal(self.mu, self.sigma, size=n)

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        return self._frozen().ppf(np.asarray(p))

    def mean(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma**2))

    def tvar(self, alpha: float, *, n: int = 200_000, seed: int | None = None) -> float:
        """Tail-VaR (a.k.a. Expected Shortfall) at level alpha in [0, 1).

        Monte-Carlo estimator; cheap and good enough for the body distribution.
        """
        rng = np.random.default_rng(seed)
        x = self.sample(n, rng)
        q = np.quantile(x, alpha)
        tail = x[x >= q]
        return float(tail.mean()) if tail.size else float("nan")
