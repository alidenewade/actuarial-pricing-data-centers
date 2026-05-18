"""Operator-side optimal retention / insurance limit (paper §13, eq. 50).

The operator chooses retained-loss limit K (per occurrence) to minimise

    C(K) = E[ min(S, K) ]              # retained pure loss
         + r * K                       # opportunity cost of capital backing K
         + lambda_K * E[ (S - K)_+ ]   # ceded premium (expected value loading)
         + xi_K * sqrt( Var[(S - K)_+] )    # additional risk loading

where S is the aggregate annual loss sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class OperatorCost:
    K: float
    retained_mean: float
    capital_cost: float
    ceded_premium: float
    risk_loading: float

    @property
    def total(self) -> float:
        return self.retained_mean + self.capital_cost + self.ceded_premium + self.risk_loading


def total_cost(
    K: float | ArrayLike,
    r: float,
    lambda_K: float,
    xi_K: float,
    S_samples: ArrayLike,
) -> NDArray[np.float64]:
    """Vectorised evaluation of C(K) for one or many K values."""
    S = np.asarray(S_samples, dtype=np.float64)
    K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
    out = np.empty_like(K_arr)
    for i, k in enumerate(K_arr):
        retained = np.minimum(S, k).mean()
        ceded_excess = np.maximum(S - k, 0.0)
        ceded_mean = ceded_excess.mean()
        ceded_sd = ceded_excess.std(ddof=1) if S.size > 1 else 0.0
        out[i] = retained + r * k + lambda_K * ceded_mean + xi_K * ceded_sd
    return out if out.size > 1 else out[0]


def find_optimum(
    r: float,
    lambda_K: float,
    xi_K: float,
    S_samples: ArrayLike,
    K_grid: ArrayLike | None = None,
) -> tuple[float, float]:
    """Grid search for K* that minimises C(K). Returns (K_star, C_star)."""
    S = np.asarray(S_samples, dtype=np.float64)
    if K_grid is None:
        # default grid: 0 to 99.5th percentile, 200 points
        hi = float(np.quantile(S, 0.995))
        K_grid = np.linspace(0.0, hi, 200)
    K_grid = np.asarray(K_grid, dtype=np.float64)
    C = total_cost(K_grid, r, lambda_K, xi_K, S)
    idx = int(np.argmin(C))
    return float(K_grid[idx]), float(C[idx])


def plot_operator_cost(
    r: float,
    lambda_K: float,
    xi_K: float,
    S_samples: ArrayLike,
    ax=None,
):
    """Reproduces paper Figure 4 — total cost vs retention K."""
    import matplotlib.pyplot as plt

    S = np.asarray(S_samples, dtype=np.float64)
    hi = float(np.quantile(S, 0.995))
    K_grid = np.linspace(0.0, hi, 200)
    C = total_cost(K_grid, r, lambda_K, xi_K, S)
    K_star, C_star = find_optimum(r, lambda_K, xi_K, S, K_grid)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(K_grid, C, lw=2, label="C(K)")
    ax.axvline(K_star, color="C3", ls="--", label=f"K* = {K_star:,.2f}")
    ax.set_xlabel("Retention K")
    ax.set_ylabel("Operator total cost  C(K)")
    ax.set_title("Operator optimal retention (paper Fig. 4)")
    ax.grid(alpha=0.3)
    ax.legend()
    return ax
