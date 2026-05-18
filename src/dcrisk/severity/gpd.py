"""Generalised Pareto Distribution (GPD) for tail severity.

Fits via Maximum Likelihood Estimation (MLE) and the Method of Moments (MoM),
returns shape (xi) and scale (sigma) with 95% **parametric-bootstrap**
confidence intervals, and exposes quantile / Tail-VaR closed forms.

Convention
----------
Excesses Y = X - u | X > u  for threshold u, with density

    f(y; xi, sigma) = (1/sigma) * (1 + xi * y / sigma) ** (-(1 + 1/xi))   if xi != 0
                    = (1/sigma) * exp(-y / sigma)                          if xi == 0

defined on y >= 0 when xi >= 0, and on 0 <= y <= -sigma/xi when xi < 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------

def _neg_loglik(params: tuple[float, float], y: NDArray[np.float64]) -> float:
    xi, sigma = params
    if sigma <= 0:
        return np.inf
    if xi >= 0:
        z = 1.0 + xi * y / sigma
        if (z <= 0).any():
            return np.inf
        return float(y.size * np.log(sigma) + (1.0 + 1.0 / xi) * np.log(z).sum())
    # xi < 0 -> bounded support
    upper = -sigma / xi
    if (y > upper).any():
        return np.inf
    z = 1.0 + xi * y / sigma
    return float(y.size * np.log(sigma) + (1.0 + 1.0 / xi) * np.log(z).sum())


# ---------------------------------------------------------------------------
# GPD class
# ---------------------------------------------------------------------------

@dataclass
class GPD:
    """Generalised Pareto fit object."""

    xi: float
    sigma: float
    threshold: float = 0.0
    ci: dict[str, tuple[float, float]] | None = None

    # ---- factories --------------------------------------------------------

    @classmethod
    def fit_mle(
        cls,
        x: ArrayLike,
        threshold: float = 0.0,
        *,
        n_boot: int = 0,
        rng: np.random.Generator | None = None,
    ) -> "GPD":
        """MLE fit using the POT (peaks-over-threshold) sample.

        If ``n_boot > 0``, attaches 95% parametric-bootstrap CIs for xi and sigma.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y = x_arr[x_arr > threshold] - threshold
        if y.size < 5:
            raise ValueError(f"Need at least 5 exceedances above u={threshold}; got {y.size}.")

        x0 = _mom_init(y)
        res = optimize.minimize(
            _neg_loglik, x0, args=(y,), method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 2000},
        )
        xi_hat, sigma_hat = float(res.x[0]), float(res.x[1])
        obj = cls(xi=xi_hat, sigma=sigma_hat, threshold=threshold)

        if n_boot > 0:
            obj.ci = obj._parametric_bootstrap(y.size, n_boot, rng=rng)
        return obj

    @classmethod
    def fit_mom(cls, x: ArrayLike, threshold: float = 0.0) -> "GPD":
        """Method-of-moments fit (Hosking & Wallis, 1987)."""
        x_arr = np.asarray(x, dtype=np.float64)
        y = x_arr[x_arr > threshold] - threshold
        xi_hat, sigma_hat = _mom_estimate(y)
        return cls(xi=float(xi_hat), sigma=float(sigma_hat), threshold=threshold)

    # ---- distribution interface ------------------------------------------

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        """Threshold + GPD quantile at probability p."""
        p_arr = np.asarray(p, dtype=np.float64)
        if self.xi == 0:
            q = -self.sigma * np.log1p(-p_arr)
        else:
            q = (self.sigma / self.xi) * ((1.0 - p_arr) ** (-self.xi) - 1.0)
        return self.threshold + q

    def tvar(self, alpha: float) -> float:
        """Closed-form Tail-VaR for GPD (valid only when xi < 1)."""
        if self.xi >= 1.0:
            return float("inf")
        var = self.quantile(alpha)
        # TVaR_alpha = VaR_alpha / (1 - xi) + (sigma - xi * u) / (1 - xi)
        return float((var + self.sigma - self.xi * self.threshold) / (1.0 - self.xi))

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng() if rng is None else rng
        u = rng.random(n)
        return self.quantile(u)

    # ---- bootstrap --------------------------------------------------------

    def _parametric_bootstrap(
        self,
        n_excess: int,
        n_boot: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, tuple[float, float]]:
        rng = np.random.default_rng() if rng is None else rng
        xis = np.empty(n_boot)
        sigmas = np.empty(n_boot)
        ok = 0
        for k in range(n_boot):
            y_boot = self.sample(n_excess, rng) - self.threshold
            try:
                xi0 = _mom_init(y_boot)
                res = optimize.minimize(_neg_loglik, xi0, args=(y_boot,), method="Nelder-Mead",
                                        options={"maxiter": 1000})
                if res.success or res.fun < np.inf:
                    xis[ok] = res.x[0]
                    sigmas[ok] = res.x[1]
                    ok += 1
            except Exception:  # noqa: BLE001 - bootstrap noise tolerated
                continue
        xis = xis[:ok]
        sigmas = sigmas[:ok]
        return {
            "xi":    (float(np.percentile(xis, 2.5)),    float(np.percentile(xis, 97.5))),
            "sigma": (float(np.percentile(sigmas, 2.5)), float(np.percentile(sigmas, 97.5))),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mom_estimate(y: NDArray[np.float64]) -> tuple[float, float]:
    """Method-of-moments for GPD: matches sample mean and variance."""
    m = y.mean()
    v = y.var(ddof=1)
    if v <= 0:
        return 0.0, max(m, 1e-9)
    xi_hat = 0.5 * (1.0 - m**2 / v)
    sigma_hat = 0.5 * m * (1.0 + m**2 / v)
    return xi_hat, sigma_hat


def _mom_init(y: NDArray[np.float64]) -> tuple[float, float]:
    xi, sigma = _mom_estimate(y)
    # nudge xi away from boundary for numerical stability of MLE
    return (float(np.clip(xi, -0.49, 0.99)), float(max(sigma, 1e-6)))
