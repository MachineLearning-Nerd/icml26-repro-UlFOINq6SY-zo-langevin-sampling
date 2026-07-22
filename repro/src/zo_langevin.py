"""Clean-room zeroth-order Langevin sampling from
"Zeroth-Order Non-Log-Concave Sampling with Variance Reduction" (arXiv 2605.30573).
numpy, CPU. ZO gradient via finite differences; VR via SARAH-style recursive estimator.
c1: Fisher info convergence. c2: VR estimator. c3: SGM prior. c6: O(1) batch complexity.
"""
from __future__ import annotations
import numpy as np


def log_density(x, target='gaussian'):
    if target == 'gaussian':
        return -0.5 * np.sum(x ** 2)
    elif target == 'mixture':
        d = len(x); c = np.ones(d) * 2.0
        return np.log(np.exp(-0.5 * np.sum((x - c) ** 2)) + np.exp(-0.5 * np.sum((x + c) ** 2)))


def zo_gradient(f, x, h=0.05, n_dirs=10, rng=None):
    """Zeroth-order gradient estimate via Gaussian smoothing."""
    if rng is None: rng = np.random.default_rng()
    d = len(x); g = np.zeros(d)
    for _ in range(n_dirs):
        u = rng.standard_normal(d); u /= np.linalg.norm(u)
        g += (f(x + h * u) - f(x - h * u)) / (2 * h) * u
    return g / n_dirs * d


def zo_langevin(f, x0, T, step_size, h=0.05, n_dirs=10, seed=0):
    """Standard ZO Langevin: x += (step/2) * zo_grad(log p) + sqrt(step) * noise."""
    rng = np.random.default_rng(seed); x = x0.copy(); d = len(x0)
    samples = [x.copy()]
    for t in range(T):
        neg_logp = lambda x: -f(x)
        g = zo_gradient(neg_logp, x, h, n_dirs, rng)
        x = x - step_size / 2 * g + np.sqrt(step_size) * rng.standard_normal(d)
        samples.append(x.copy())
    return np.array(samples)


def zo_langevin_vr(f, x0, T, step_size, h=0.05, p=0.1, b=10, seed=0):
    """Variance-reduced ZO Langevin: SARAH-style recursive gradient estimator (Eq 8)."""
    rng = np.random.default_rng(seed); x = x0.copy(); d = len(x0)
    neg_logp = lambda x: -f(x)
    g_full = zo_gradient(neg_logp, x, h, b * 2, rng)  # large-batch estimate
    samples = [x.copy()]
    for t in range(T):
        if rng.random() < p:
            g_full = zo_gradient(neg_logp, x, h, b * 2, rng)  # refresh
        # small-batch correction
        g_small_new = zo_gradient(neg_logp, x, h, 2, rng)
        g_small_old = zo_gradient(neg_logp, x - step_size / 2 * g_full, h, 2, rng)
        g_vr = g_full + g_small_new - g_small_old
        x = x - step_size / 2 * g_vr + np.sqrt(step_size) * rng.standard_normal(d)
        samples.append(x.copy())
    return np.array(samples)


def fisher_info(samples, target='gaussian'):
    """Estimate Fisher information E[||grad log p||^2] from samples."""
    d = samples.shape[1]; fi = 0.0
    for x in samples[::max(1, len(samples)//100)]:
        if target == 'gaussian':
            fi += np.sum(x ** 2)
    return float(fi / max(len(samples[::max(1, len(samples)//100)]), 1))
