"""dcrisk — actuarial pricing and insurance simulation for hyperscale data centers.

The package is organised by the conceptual layers of the paper:

    reliability   — Markov / fault-tree / Arrhenius hazard
    severity      — lognormal body, GPD tail, body-tail mixture
    frequency     — Negative-Binomial and Cox-process counts
    copula        — Gaussian + Gumbel dependence
    sde           — coupled power-thermal-cyber state SDE
    pricing       — pure premium, loading principles, XoL
    econ          — operator optimum and market equilibrium
    dashboards    — interactive Streamlit front-end

Author: Ali Denewade (2026), MIT License.
"""

__version__ = "0.1.0"
__author__ = "Ali Denewade"

__all__ = [
    "reliability",
    "severity",
    "frequency",
    "copula",
    "sde",
    "pricing",
    "econ",
    "dashboards",
]
