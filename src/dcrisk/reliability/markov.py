"""4-state continuous-time Markov chain for hyperscale-DC reliability.

States (paper §4, eq. 7)
------------------------
    0 : OK     — nominal operation
    1 : Deg1   — first-level degradation (one redundant unit failed)
    2 : Deg2   — second-level degradation (two redundant units failed)
    3 : F      — failed / outage

Transition rates
----------------
    OK   -> Deg1  : 2*lambda    (two parallel failure modes; either triggers)
    Deg1 -> Deg2  :   lambda    (only one failure mode left exposed)
    Deg2 -> F     :   lambda_f  (final unit failure -> outage)
    Deg1 -> OK    :   mu        (in-place repair of one component)
    Deg2 -> Deg1  :   mu        (in-place repair of one component)
    F    -> OK    :   mu_r      (full restoration)

The diagonal of Q is set so each row sums to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

STATE_NAMES: tuple[str, ...] = ("OK", "Deg1", "Deg2", "F")


@dataclass(frozen=True)
class MarkovParams:
    """Failure / repair rates for the 4-state CTMC (units: 1 / hour)."""

    lam: float          # base component failure rate
    lam_f: float        # final-stage failure rate (Deg2 -> F)
    mu: float           # in-place repair rate
    mu_r: float         # full restoration rate from F

    def __post_init__(self) -> None:
        for name in ("lam", "lam_f", "mu", "mu_r"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be non-negative, got {v}")


def build_Q(p: MarkovParams) -> NDArray[np.float64]:
    """Build the 4x4 generator matrix Q from the failure/repair rates."""
    Q = np.zeros((4, 4), dtype=np.float64)
    # off-diagonals
    Q[0, 1] = 2.0 * p.lam
    Q[1, 2] = p.lam
    Q[2, 3] = p.lam_f
    Q[1, 0] = p.mu
    Q[2, 1] = p.mu
    Q[3, 0] = p.mu_r
    # diagonals so rows sum to zero
    for i in range(4):
        Q[i, i] = -Q[i].sum()
    return Q


def stationary_distribution(Q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve pi Q = 0 subject to sum(pi) = 1 via the augmented linear system."""
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def availability(Q: NDArray[np.float64]) -> float:
    """Steady-state availability = 1 - pi(F)."""
    pi = stationary_distribution(Q)
    return float(1.0 - pi[-1])


def mtbf_from_Q(Q: NDArray[np.float64]) -> float:
    """Mean time between failures = expected time from OK until first hit of F.

    Computed via the fundamental matrix of transient states (everything except F).
    """
    transient = np.array([0, 1, 2])
    Qt = Q[np.ix_(transient, transient)]
    # E[time to absorption | start = state i] = (-Qt^{-1} 1)_i
    ones = np.ones(len(transient))
    times = np.linalg.solve(-Qt, ones)
    return float(times[0])  # starting from OK


def simulate_chain(
    Q: NDArray[np.float64],
    T_max: float,
    start_state: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Gillespie-style exact simulation of the CTMC up to time T_max.

    Returns
    -------
    times  : 1-D array of jump epochs, starting at 0
    states : 1-D array of states held on each interval [times[k], times[k+1])
    """
    if rng is None:
        rng = np.random.default_rng()

    times: list[float] = [0.0]
    states: list[int] = [int(start_state)]

    t = 0.0
    s = int(start_state)
    while t < T_max:
        rate = -Q[s, s]
        if rate <= 0:  # absorbing
            break
        dt = rng.exponential(1.0 / rate)
        t += dt
        if t >= T_max:
            break
        # choose next state proportional to off-diagonal entries
        probs = Q[s].copy()
        probs[s] = 0.0
        probs = probs / probs.sum()
        s = int(rng.choice(Q.shape[0], p=probs))
        times.append(t)
        states.append(s)

    return np.asarray(times), np.asarray(states, dtype=np.int64)
