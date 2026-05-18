"""Body-tail mixture severity model.

The body of the loss distribution is modelled by a Lognormal up to a
splice point u (the POT threshold). Above u, losses follow a GPD tail.
Mass above u is calibrated so that the splice probability matches the
empirical exceedance rate p_u.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dcrisk.severity.gpd import GPD
from dcrisk.severity.lognormal import Lognormal


@dataclass
class BodyTailMixture:
    body: Lognormal
    tail: GPD
    threshold: float
    p_tail: float

    @classmethod
    def fit(
        cls,
        x: ArrayLike,
        threshold: float,
        *,
        n_boot: int = 0,
    ) -> "BodyTailMixture":
        """Fit Lognormal on x <= u, GPD-MLE on excesses x > u."""
        x_arr = np.asarray(x, dtype=np.float64)
        body_sample = x_arr[(x_arr > 0) & (x_arr <= threshold)]
        tail_sample = x_arr[x_arr > threshold]
        if body_sample.size < 5:
            raise ValueError("Need >=5 body observations.")
        if tail_sample.size < 5:
            raise ValueError("Need >=5 tail observations.")
        body = Lognormal.fit(body_sample)
        tail = GPD.fit_mle(x_arr, threshold=threshold, n_boot=n_boot)
        p_tail = float(tail_sample.size / x_arr.size)
        return cls(body=body, tail=tail, threshold=threshold, p_tail=p_tail)

    def sample(self, n: int, rng: np.random.Generator | None = None) -> NDArray[np.float64]:
        rng = np.random.default_rng() if rng is None else rng
        in_tail = rng.random(n) < self.p_tail
        out = np.empty(n)
        n_tail = int(in_tail.sum())
        n_body = n - n_tail
        if n_body > 0:
            out[~in_tail] = self.body.sample(n_body, rng)
        if n_tail > 0:
            out[in_tail] = self.tail.sample(n_tail, rng)
        return out

    def quantile(self, p: ArrayLike) -> NDArray[np.float64]:
        """Quantile of the splice: use body below 1-p_tail, GPD above."""
        p_arr = np.asarray(p, dtype=np.float64)
        out = np.empty_like(p_arr)
        cutoff = 1.0 - self.p_tail
        body_mask = p_arr <= cutoff
        out[body_mask] = self.body.quantile(p_arr[body_mask] / cutoff)
        tail_p = (p_arr[~body_mask] - cutoff) / self.p_tail
        out[~body_mask] = self.tail.quantile(tail_p)
        return out
