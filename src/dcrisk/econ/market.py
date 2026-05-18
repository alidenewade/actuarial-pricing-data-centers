"""Supply-demand equilibrium for data-center insurance (paper §13, eq. 52-53).

Demand side (operator) — willingness to pay falls with premium and rises with
the operator's loss volatility (theta_op):

    Q_d(P; theta_op) = a - b * P + c * theta_op

Supply side (insurer) — quantity offered rises with premium and falls with
tail risk parameters (xi, theta) and frequency lambda:

    Q_s(P; lam, xi, theta) = -d + e * P - f * lam - g * xi - h * theta

Equilibrium: solve Q_d = Q_s -> closed-form linear system.
Comparative statics drive Figures 5-6 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DemandParams:
    a: float = 50.0
    b: float = 0.40
    c: float = 5.0


@dataclass(frozen=True)
class SupplyParams:
    d: float = 5.0
    e: float = 0.30
    f: float = 2.0
    g: float = 8.0
    h: float = 1.0


def demand(P: NDArray | float, theta_op: float = 1.0, params: DemandParams | None = None):
    p = params or DemandParams()
    return p.a - p.b * np.asarray(P) + p.c * theta_op


def supply(
    P: NDArray | float,
    lam: float = 1.0,
    xi: float = 0.2,
    theta: float = 1.5,
    params: SupplyParams | None = None,
):
    s = params or SupplyParams()
    return -s.d + s.e * np.asarray(P) - s.f * lam - s.g * xi - s.h * theta


def equilibrium(
    lam: float = 1.0,
    xi: float = 0.2,
    theta: float = 1.5,
    theta_op: float = 1.0,
    d_params: DemandParams | None = None,
    s_params: SupplyParams | None = None,
) -> tuple[float, float]:
    """Closed-form (P*, Q*)."""
    dp = d_params or DemandParams()
    sp = s_params or SupplyParams()
    # solve dp.a - dp.b P + dp.c theta_op = -sp.d + sp.e P - sp.f lam - sp.g xi - sp.h theta
    rhs = dp.a + dp.c * theta_op + sp.d + sp.f * lam + sp.g * xi + sp.h * theta
    P_star = rhs / (dp.b + sp.e)
    Q_star = demand(P_star, theta_op=theta_op, params=dp)
    return float(P_star), float(Q_star)


def plot_equilibrium(
    lam: float = 1.0,
    xi: float = 0.2,
    theta: float = 1.5,
    theta_op: float = 1.0,
    shifts: dict[str, float] | None = None,
    ax=None,
):
    """Reproduces paper Figures 5-6 — supply-demand curves with comparative statics.

    Pass e.g. ``shifts={'xi': 0.4}`` to overlay a second equilibrium under
    a different tail-risk shape parameter.
    """
    import matplotlib.pyplot as plt

    P_grid = np.linspace(1.0, 200.0, 400)
    Qd = demand(P_grid, theta_op=theta_op)
    Qs = supply(P_grid, lam=lam, xi=xi, theta=theta)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(P_grid, Qd, label="Demand (baseline)", color="C0", lw=2)
    ax.plot(P_grid, Qs, label="Supply (baseline)", color="C1", lw=2)
    P_star, Q_star = equilibrium(lam, xi, theta, theta_op)
    ax.scatter([P_star], [Q_star], color="C3", s=80, zorder=5, label=f"P*={P_star:,.1f}, Q*={Q_star:,.1f}")

    if shifts:
        kw = dict(lam=lam, xi=xi, theta=theta, theta_op=theta_op)
        kw.update(shifts)
        Qd2 = demand(P_grid, theta_op=kw["theta_op"])
        Qs2 = supply(P_grid, lam=kw["lam"], xi=kw["xi"], theta=kw["theta"])
        label_shift = ", ".join(f"{k}={v}" for k, v in shifts.items())
        ax.plot(P_grid, Qd2, color="C0", lw=1.2, ls="--", alpha=0.7,
                label=f"Demand ({label_shift})")
        ax.plot(P_grid, Qs2, color="C1", lw=1.2, ls="--", alpha=0.7,
                label=f"Supply ({label_shift})")
        P2, Q2 = equilibrium(**kw)
        ax.scatter([P2], [Q2], color="C3", marker="x", s=80, zorder=5,
                   label=f"P*={P2:,.1f}, Q*={Q2:,.1f}")

    ax.set_xlabel("Premium P")
    ax.set_ylabel("Quantity Q")
    ax.set_title("Insurance market equilibrium (paper Figs. 5-6)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    return ax
