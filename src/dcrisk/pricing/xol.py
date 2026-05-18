"""Per-risk excess-of-loss (XoL) reinsurance pricing.

For each loss X the cedant retains  min(X, d)  and cedes
    L^ced = min((X - d)_+, ell)
where d is the retention and ell the limit. The expected ceded loss
follows in closed form for the GPD tail (paper §12.1):

    Let Y = X - u | X > u  ~  GPD(xi, sigma).
    For retention d > u:
        E[(Y - (d - u))_+ ^ ell] = sigma_d / (1 - xi)
                                  - (sigma_d + xi * ell) / (1 - xi) * (1 + xi * ell / sigma_d) ** (-1/xi)
        with  sigma_d = sigma + xi * (d - u)
    multiplied by the exceedance probability p_u * (1 + xi * (d - u) / sigma) ** (-1/xi).

For ell = +inf (and xi < 1) this collapses to  E[Y - (d - u)]  scaled by the
exceedance probability.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from dcrisk.severity.gpd import GPD


def xol_expected_ceded(losses: ArrayLike, d: float, ell: float = np.inf) -> float:
    """Monte-Carlo estimator of E[min((X - d)_+, ell)] from a loss sample."""
    x = np.asarray(losses, dtype=np.float64)
    ceded = np.minimum(np.maximum(x - d, 0.0), ell)
    return float(ceded.mean())


def xol_expected_ceded_gpd(
    gpd: GPD,
    d: float,
    p_u: float,
    ell: float = np.inf,
) -> float:
    """Closed-form expected ceded loss using the GPD tail above threshold u = gpd.threshold.

    Parameters
    ----------
    gpd : fitted GPD object (provides xi, sigma, threshold u).
    d   : retention (d >= u required for the closed form to apply).
    p_u : exceedance probability P(X > u) (estimated from the sample).
    ell : XoL limit. Use np.inf for unlimited.
    """
    u, xi, sigma = gpd.threshold, gpd.xi, gpd.sigma
    if d < u:
        raise ValueError("Closed form requires retention d >= GPD threshold u.")
    if xi >= 1.0:
        return float("inf")

    sigma_d = sigma + xi * (d - u)
    P_exceed_d = p_u * (1.0 + xi * (d - u) / sigma) ** (-1.0 / xi) if xi != 0 else p_u * np.exp(-(d - u) / sigma)

    if np.isinf(ell):
        # E[Y - (d-u) | X > d] = sigma_d / (1 - xi)
        cond_mean = sigma_d / (1.0 - xi)
        return float(P_exceed_d * cond_mean)

    # finite limit -- limited expected value of GPD(xi, sigma_d) up to ell
    if xi != 0:
        z = 1.0 + xi * ell / sigma_d
        E_min = (sigma_d / (1.0 - xi)) * (1.0 - z ** (1.0 - 1.0 / xi))
    else:
        E_min = sigma_d * (1.0 - np.exp(-ell / sigma_d))
    return float(P_exceed_d * E_min)
