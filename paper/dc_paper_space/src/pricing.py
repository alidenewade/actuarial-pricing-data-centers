"""Premium principles — paper §10.1–§10.5.

Five classical principles applied to a sample of annual losses S:
    pure    : P = E[S]
    sd      : P = E[S] + a · σ(S)
    var     : P = E[S] + b · Var(S)
    esscher : P = E[S exp(hS)] / E[exp(hS)]   (h > 0)
    wang    : P = E[g_λ(F̄_S(s))]              (Wang transform, λ > 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.integrate import trapezoid  # NumPy 2.x compatible


@dataclass(frozen=True)
class Loadings:
    a:        float = 0.20   # SD-principle scalar
    b:        float = 1e-5   # Variance-principle scalar (depends on S units)
    h:        float = 1e-4   # Esscher tilt parameter
    lambda_w: float = 0.20   # Wang distortion parameter (Φ⁻¹ shift)


def pure(S: np.ndarray) -> float:
    return float(np.mean(S))


def sd_principle(S: np.ndarray, a: float) -> float:
    return float(np.mean(S) + a * np.std(S, ddof=1))


def variance_principle(S: np.ndarray, b: float) -> float:
    return float(np.mean(S) + b * np.var(S, ddof=1))


def esscher(S: np.ndarray, h: float) -> float:
    # E[S e^{hS}] / E[e^{hS}], computed in a numerically stable way.
    h_S = h * S
    # subtract max for stability
    m = float(h_S.max())
    w = np.exp(h_S - m)
    return float(np.sum(S * w) / np.sum(w))


def wang_transform(S: np.ndarray, lambda_w: float) -> float:
    """Wang(1996, 2000) distortion premium.

    P = ∫_0^∞ g_λ(F̄_S(s)) ds, with g_λ(u) = Φ(Φ⁻¹(u) + λ).
    Approximated via the empirical survival function on the sorted sample.
    """
    s_sorted = np.sort(S)
    n = s_sorted.size
    surv = 1.0 - np.arange(1, n + 1) / n
    # avoid log(0) when surv reaches 0 at the very last sample
    surv = np.clip(surv, 1e-12, 1 - 1e-12)
    distorted = stats.norm.cdf(stats.norm.ppf(surv) + lambda_w)
    # Trapezoidal integration over the sorted loss axis (works on NumPy 1 and 2).
    return float(trapezoid(distorted, s_sorted))


def all_premiums(S: np.ndarray, loadings: Loadings) -> dict[str, float]:
    return {
        "Pure":          pure(S),
        "SD":            sd_principle(S, loadings.a),
        "Variance":      variance_principle(S, loadings.b),
        "Esscher":       esscher(S, loadings.h),
        "Wang":          wang_transform(S, loadings.lambda_w),
    }
