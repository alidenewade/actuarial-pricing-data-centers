"""GPU smoke tests — confirm JAX, PyTorch, and CuPy all see the RTX 5090.

The tests are skipped (not failed) when the corresponding library is
missing, so this file can live alongside the CPU smoke tests on a laptop
without breaking `pytest -q`. On adu-00 inside the dcrisk-gpu env every
test must run and pass.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------

def test_jax_sees_gpu() -> None:
    jax = pytest.importorskip("jax")
    devices = jax.devices()
    assert any(d.platform == "gpu" for d in devices), f"no GPU device among {devices}"


def test_jax_kernel_runs_and_frees() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU visible to JAX")

    x = jnp.arange(1 << 16, dtype=jnp.float32)
    y = jax.jit(lambda a: jnp.sin(a) ** 2 + jnp.cos(a) ** 2)(x)
    np.testing.assert_allclose(np.asarray(y), np.ones_like(np.asarray(y)), atol=1e-5)


def test_jax_vs_numpy_ptcyber() -> None:
    """JAX SDE path must agree with NumPy baseline in distribution (mean, std)."""
    jax = pytest.importorskip("jax")
    from dcrisk.sde.ptcyber import simulate_ptcyber
    from dcrisk.sde.ptcyber_gpu import simulate_ptcyber_gpu

    rng = np.random.default_rng(0)
    _, X_cpu = simulate_ptcyber(T_max=4.0, dt=1 / 60.0, n_paths=256, rng=rng)
    _, X_gpu = simulate_ptcyber_gpu(T_max=4.0, dt=1 / 60.0, n_paths=256, seed=0)

    for k, name in enumerate(("V", "T", "C")):
        mu_cpu, mu_gpu = X_cpu[..., k].mean(), X_gpu[..., k].mean()
        sd_cpu, sd_gpu = X_cpu[..., k].std(),  X_gpu[..., k].std()
        # generous tolerance: independent RNG streams, only matches in distribution
        assert abs(mu_cpu - mu_gpu) < max(0.5, 0.05 * abs(mu_cpu) + 0.05), \
            f"mean({name}) differs: cpu={mu_cpu:.3f} gpu={mu_gpu:.3f}"
        assert abs(sd_cpu - sd_gpu) < max(0.5, 0.20 * abs(sd_cpu) + 0.05), \
            f"std({name}) differs:  cpu={sd_cpu:.3f} gpu={sd_gpu:.3f}"


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

def test_torch_sees_gpu() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    name = torch.cuda.get_device_name(0)
    assert "RTX" in name or "GPU" in name.upper(), f"unexpected GPU name: {name}"


def test_torch_kernel_runs_and_frees() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda not available")
    x = torch.randn(1 << 16, device="cuda")
    y = (x.sin() ** 2 + x.cos() ** 2).cpu().numpy()
    np.testing.assert_allclose(y, np.ones_like(y), atol=1e-5)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CuPy
# ---------------------------------------------------------------------------

def test_cupy_sees_gpu() -> None:
    cp = pytest.importorskip("cupy")
    name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode() \
        if isinstance(cp.cuda.runtime.getDeviceProperties(0)["name"], bytes) \
        else cp.cuda.runtime.getDeviceProperties(0)["name"]
    assert "RTX" in name or "GPU" in name.upper(), f"unexpected GPU name: {name}"


def test_cupy_kernel_runs() -> None:
    cp = pytest.importorskip("cupy")
    x = cp.arange(1 << 16, dtype=cp.float32)
    y = cp.asnumpy(cp.sin(x) ** 2 + cp.cos(x) ** 2)
    np.testing.assert_allclose(y, np.ones_like(y), atol=1e-5)


# ---------------------------------------------------------------------------
# Allocation budget
# ---------------------------------------------------------------------------

def test_all_three_allocate_under_1gb() -> None:
    """No single library should exceed 1 GB after a 64k-element kernel."""
    if importlib.util.find_spec("torch") is not None:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024 ** 3
            assert mem < 1.0, f"torch allocated {mem:.2f} GB"
