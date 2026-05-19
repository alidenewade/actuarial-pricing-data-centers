"""4-state plant Markov chain — paper §5.

States: OK -> Deg1 -> Deg2 -> F. Generator Q from eq. (10):

    Q =  [-2λ    2λ           0                 0        ]
         [ μ    -(μ+λ)         λ                 0        ]
         [ 0     μ            -(μ+λ_f)          λ_f      ]
         [ 0     0             μ_r              -μ_r     ]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Uptime Institute Tier → typical (μ, μ_r) calibration in 1/hour.
# Values from Table 1 of dc_paper and §5.3 of the paper.
TIER_CALIBRATION = {
    "Tier I  (N)":                  {"avail": 0.99671, "mu": 1/24,  "mu_r": 1/72},
    "Tier II (N+1)":                {"avail": 0.99741, "mu": 1/12,  "mu_r": 1/48},
    "Tier III (concurrent maint.)": {"avail": 0.99982, "mu": 1/6,   "mu_r": 1/24},
    "Tier IV (2N fault-tolerant)":  {"avail": 0.99995, "mu": 1/4,   "mu_r": 1/12},
}


@dataclass(frozen=True)
class MarkovParams:
    lam:    float    # per-train degradation rate, 1/hour
    mu:     float    # per-train repair rate, 1/hour
    lam_f:  float    # degraded -> failed rate, 1/hour
    mu_r:   float    # full-site restoration rate, 1/hour


def Q_matrix(p: MarkovParams) -> np.ndarray:
    return np.array([
        [-2 * p.lam,        2 * p.lam,            0,                       0],
        [p.mu,             -(p.mu + p.lam),       p.lam,                   0],
        [0,                 p.mu,                -(p.mu + p.lam_f),        p.lam_f],
        [0,                 0,                    p.mu_r,                 -p.mu_r],
    ], dtype=np.float64)


def stationary_distribution(Q: np.ndarray) -> np.ndarray:
    """Solve π Q = 0 subject to π · 1 = 1."""
    n = Q.shape[0]
    A = np.vstack([Q.T, np.ones(n)])
    b = np.concatenate([np.zeros(n), [1.0]])
    pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return pi


def availability(pi: np.ndarray) -> float:
    """A = 1 - π_F = 1 - π[3]."""
    return float(1.0 - pi[3])


def claim_frequency_per_year(pi: np.ndarray, p: MarkovParams) -> float:
    """λ^out = π_2 · λ_f, converted from /hour to /year (eq. 17)."""
    return float(pi[2] * p.lam_f * 8760)


def mtbf_hours(pi: np.ndarray, p: MarkovParams) -> float:
    """Mean time between site outages, in hours."""
    rate = pi[2] * p.lam_f
    return float(np.inf if rate <= 0 else 1.0 / rate)


def simulate_markov_walk(
    p: MarkovParams,
    hours: float,
    rng: np.random.Generator,
    start_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Direct (Gillespie-style) CTMC sample on {OK, Deg1, Deg2, F}.

    Returns
    -------
    states : np.ndarray of int in {0, 1, 2, 3}
    times  : np.ndarray of cumulative hours at which each state began.

    The trajectory is a right-continuous step function: at time `times[i]`
    the chain entered `states[i]` and stays there until `times[i+1]`.
    """
    Q = Q_matrix(p)
    exit_rates = -np.diag(Q)
    n = Q.shape[0]
    # Per-row transition probabilities (off-diagonal renormalised).
    P_jump = np.zeros_like(Q)
    for i in range(n):
        if exit_rates[i] > 0:
            for j in range(n):
                if i != j:
                    P_jump[i, j] = Q[i, j] / exit_rates[i]

    s = int(start_state)
    t = 0.0
    states: list[int] = [s]
    times: list[float] = [t]
    while t < hours:
        rate = exit_rates[s]
        if rate <= 0:
            break
        dt = float(rng.exponential(1.0 / rate))
        t += dt
        if t >= hours:
            break
        probs = P_jump[s].copy()
        total = probs.sum()
        if total <= 0:
            break
        probs /= total
        s = int(rng.choice(n, p=probs))
        states.append(s)
        times.append(t)
    # cap at the horizon so the step plot extends visibly past the last jump
    states.append(states[-1])
    times.append(hours)
    return np.asarray(states, dtype=int), np.asarray(times, dtype=float)
