"""Premium loading principles (paper §11).

Each function takes a Monte-Carlo aggregate-loss sample S and one parameter,
returning the loaded premium P(S, .).

    sd_premium(S, a)         P = E[S] + a * sigma_S
    variance_premium(S, a)   P = E[S] + a * Var(S)
    esscher_premium(S, h)    P = E[S e^{h S}] / E[e^{h S}]
    wang_premium(S, lam)     g(u) = Phi(Phi^{-1}(u) + lam)  applied to the survival fn
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def sd_premium(S: ArrayLike, a: float) -> float:
    S_arr = np.asarray(S, dtype=np.float64)
    return float(S_arr.mean() + a * S_arr.std(ddof=1))


def variance_premium(S: ArrayLike, a: float) -> float:
    S_arr = np.asarray(S, dtype=np.float64)
    return float(S_arr.mean() + a * S_arr.var(ddof=1))


def esscher_premium(S: ArrayLike, h: float) -> float:
    """Esscher transform with parameter h > 0.  Numerically stable via log-sum-exp."""
    S_arr = np.asarray(S, dtype=np.float64)
    z = h * S_arr
    z_max = z.max()
    w = np.exp(z - z_max)
    return float(np.sum(S_arr * w) / np.sum(w))


def wang_premium(S: ArrayLike, lam: float) -> float:
    """Wang transform premium with distortion parameter lam.

    Distorted survival G(x) = Phi(Phi^{-1}(F_bar(x)) + lam) integrated over x >= 0.
    Implemented via the empirical survival function on a sorted sample.
    """
    S_arr = np.sort(np.asarray(S, dtype=np.float64))
    n = S_arr.size
    # empirical survival at each x_(k): F_bar(x_(k)) = 1 - k / n
    k = np.arange(1, n + 1)
    surv = 1.0 - k / n
    surv = np.clip(surv, 1e-12, 1.0 - 1e-12)  # numerical safety
    z = stats.norm.ppf(surv) + lam
    g = stats.norm.cdf(z)
    # premium = integral over x of g(F_bar(x)) dx; using sorted-sample trapezoid
    # Append 0 at the high end to close the integral and a 0-start at S_(1)
    x = np.concatenate(([S_arr[0]], S_arr))
    g_full = np.concatenate(([1.0], g))
    return float(np.trapezoid(g_full, x))
