"""Cooling thermodynamics, climate, and cooling-loss hazard.

Implements §4 of the paper:
  - Carnot-limited COP and PUE decomposition (eq. 4)
  - Stull (2011) wet-bulb approximation     (eq. 5)
  - Cooling-loss hazard with indicator gate (eq. 7, cooling block)
  - Non-stationary T_wb(t) climate process  (eq. 12)
"""

from dcrisk.cooling.climate import simulate_Twb
from dcrisk.cooling.hazard import lambda_cool
from dcrisk.cooling.thermo import carnot_cop, cop, pue
from dcrisk.cooling.wetbulb import wetbulb_stull

__all__ = [
    "carnot_cop",
    "cop",
    "pue",
    "wetbulb_stull",
    "lambda_cool",
    "simulate_Twb",
]
