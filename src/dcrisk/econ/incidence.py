"""Premium passthrough / tax-incidence elasticity (paper §14).

Standard result: the share of a premium increase borne by **operators**
is the ratio of supply elasticity to the sum of supply and (absolute)
demand elasticities at equilibrium.

    share_operator = eps_s / (eps_s + |eps_d|)
    share_insurer  = |eps_d| / (eps_s + |eps_d|)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dcrisk.econ.market import (
    DemandParams,
    SupplyParams,
    demand,
    equilibrium,
    supply,
)


def passthrough_table(
    grid: dict[str, list[float]],
    d_params: DemandParams | None = None,
    s_params: SupplyParams | None = None,
) -> pd.DataFrame:
    """Build a comparative-statics table of premium, quantity and incidence shares.

    Parameters
    ----------
    grid : dict mapping any of {"lam", "xi", "theta", "theta_op"} to a list of values.
           All other parameters take baseline values.
    """
    dp = d_params or DemandParams()
    sp = s_params or SupplyParams()

    rows: list[dict[str, float]] = []
    keys = list(grid.keys())
    if not keys:
        keys = ["lam"]
        grid = {"lam": [1.0]}

    # cartesian-style sweep (single dimension only for simplicity)
    for key in keys:
        for v in grid[key]:
            kw = dict(lam=1.0, xi=0.2, theta=1.5, theta_op=1.0)
            kw[key] = v
            P_star, Q_star = equilibrium(**kw, d_params=dp, s_params=sp)
            # local elasticities (point estimate at equilibrium)
            eps_d = -dp.b * P_star / Q_star
            eps_s = sp.e * P_star / Q_star
            share_op = eps_s / (eps_s + abs(eps_d))
            share_ins = abs(eps_d) / (eps_s + abs(eps_d))
            rows.append(
                {
                    "swept":            key,
                    "value":            v,
                    "P*":               P_star,
                    "Q*":               Q_star,
                    "eps_demand":       eps_d,
                    "eps_supply":       eps_s,
                    "share_operator":   share_op,
                    "share_insurer":    share_ins,
                }
            )
    return pd.DataFrame(rows)
