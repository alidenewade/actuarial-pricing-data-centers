"""Copula layer: Gaussian (radial) and Gumbel (upper-tail) dependence."""

from dcrisk.copula.gaussian import GaussianCopula
from dcrisk.copula.gumbel import GumbelCopula

__all__ = ["GaussianCopula", "GumbelCopula"]
