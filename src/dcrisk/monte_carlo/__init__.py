"""Top-level Monte Carlo aggregator + Cox-process simulation + parallel dispatch.

Combines frequency (NB / Cox), severity (lognormal-GPD mixture), copula
dependence and the cooling-loss hazard into the annual loss S used in §11 of
the paper.
"""

from dcrisk.monte_carlo.compound import simulate
from dcrisk.monte_carlo.cox_process import sample_cox
from dcrisk.monte_carlo.parallel import detect_backend, run_parallel

__all__ = ["simulate", "sample_cox", "detect_backend", "run_parallel"]
