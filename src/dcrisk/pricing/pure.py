"""Pure premium and aggregate-loss Monte Carlo."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dcrisk.frequency.nb_gamma import sample_NB
from dcrisk.severity.mixture import BodyTailMixture


def pure_premium(S_samples: ArrayLike) -> float:
    """Pure premium = E[S]. Trivial but useful API symmetry."""
    return float(np.mean(np.asarray(S_samples, dtype=np.float64)))


def aggregate_loss_NB_mixture(
    n_sims: int,
    nu: float,
    lam_bar: float,
    severity: BodyTailMixture,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """Monte Carlo aggregate loss S = sum_{i=1}^{N} X_i with N ~ NegBin, X_i ~ severity."""
    rng = np.random.default_rng() if rng is None else rng
    counts = sample_NB(nu, lam_bar, size=n_sims, rng=rng)
    S = np.empty(n_sims, dtype=np.float64)
    for i, n in enumerate(counts):
        S[i] = severity.sample(int(n), rng).sum() if n > 0 else 0.0
    return S
