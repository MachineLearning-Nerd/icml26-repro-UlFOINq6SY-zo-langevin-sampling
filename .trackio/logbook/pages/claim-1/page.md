# Claim 1 — Theorem 1 convergence rate

> **Paper claim.** Under Assumptions 1–2, VR-ZO-LMC attains relative Fisher information accuracy ε after `O(d⁷ Lₘ⁴ / ε⁴)` iterations with `O(1)` expected function evaluations per iteration.

## Status: VERIFIED

The primary evidence is an independent symbolic reconstruction, appropriate for a universally quantified theorem. A finite Gaussian sweep is disclosed only as corroboration.

## Contract and implementation

The reconstructed schedule is `γ=Lₘ⁻¹N⁻³ᐟ⁴d⁻⁷ᐟ⁴`, `p=LₘN⁻¹ᐟ⁴d⁻¹ᐟ⁴`, `b=⌈1/p⌉`, and `μ²=Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴`. The derivation telescopes the paper’s Lyapunov inequality and reduces every N-dependent term to `O(Lₘd⁷ᐟ⁴N⁻¹ᐟ⁴)`, hence `N=O(d⁷Lₘ⁴/ε⁴)`.

- **Derivation:** [`theorem1_derivation.md`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/theorem1_derivation.md)
- **Implementation:** [`zo_langevin.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/zo_langevin.py)
- **Command:** `uv run python repro/src/verify_zo.py`
- **Seed base:** `1729`; median across four seeds for each plotted point.

## Observed corroboration

| Dimension | FI at N=1,000 | FI at N=3,000 | FI at N=9,000 |
|---:|---:|---:|---:|
| 2 | 1.354 | 0.124 | 0.035 |
| 4 | 8.434 | 0.672 | 0.237 |
| 8 | 13.550 | 3.346 | 0.887 |
| 16 | 70.486 | 12.817 | 3.292 |

![Relative FI decreases with N across dimensions](images/claim1_fi_vs_N.png)

## Controls and release evidence

- `sympy` independently solves the leading term for `N` as `Lₘ⁴d⁷/ε⁴`.
- Every displayed dimension decreases as N grows.
- Raw output: [`verdict.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/outputs/verdict.json), key `c1_theorem1`.

## Scope

The numerical Gaussian test does **not** prove the theorem and is not represented as doing so. The theorem evidence is the derivation under its stated assumptions; the finite sweep checks only the predicted direction of convergence.
