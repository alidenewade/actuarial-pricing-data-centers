"""Parallel dispatch — joblib for CPU, JAX device selection for GPU.

The intent is that callers wrap a year of simulation in `run_parallel(fn, args)`
and the module picks the right backend at runtime.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Literal

from joblib import Parallel, delayed

Backend = Literal["jax-gpu", "cpu"]


def detect_backend() -> Backend:
    """Pick a backend by probing for a usable JAX GPU; fall back to CPU."""
    try:
        import jax  # noqa: F401
        if any(d.platform == "gpu" for d in jax.devices()):
            return "jax-gpu"
    except Exception:
        pass
    return "cpu"


def run_parallel(fn: Callable[..., Any], items: Iterable[Any], n_jobs: int | None = None) -> list[Any]:
    """Run `fn(item)` over `items` using joblib loky backend.

    n_jobs defaults to physical-core count - 1 (leaves one core free for the
    OS and Tailscale daemon). Override with the JOBLIB_N_JOBS env var.
    """
    if n_jobs is None:
        n_jobs = int(os.environ.get("JOBLIB_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(x) for x in items)


if __name__ == "__main__":
    print(f"detected backend: {detect_backend()}")
    out = run_parallel(lambda x: x * x, range(8), n_jobs=4)
    print(f"squares: {out}")
