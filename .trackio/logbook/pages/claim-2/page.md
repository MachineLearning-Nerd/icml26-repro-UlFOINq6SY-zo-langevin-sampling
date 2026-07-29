# Claim 2 — Eq. 8 variance-reduced ZO estimator

> **Paper claim.** The recursive estimator refreshes a large ZO batch with probability p and otherwise reuses the same directions at consecutive iterates to reduce estimation error without a dimension-growing batch.

## Status: VERIFIED

The clean-room estimator matches the released implementation’s large-batch / same-direction recursive-control-variate structure. At matched average batch budget, it has lower trajectory gradient MSE than the standard estimator in every tested dimension.

## Contract and implementation

For a large batch, `gₖ=(1/b)Σ∇̃f(xₖ,uᵢ)`. For a recursive update, `gₖ=gₖ₋₁+(1/b′)Σ[∇̃f(xₖ,uᵢ)-∇̃f(xₖ₋₁,uᵢ)]`, using the **same** `uᵢ` in both terms.

- **Implementation:** [`zo_langevin.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/zo_langevin.py)
- **Released-code audit:** [`source_audit.md`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/source_audit.md)
- **Command:** `uv run python repro/src/verify_zo.py`

## Observed evidence

| d | Standard p=1,b=6 | VR p=.4,b=9 | VR p=.2,b=14 | Best VR / standard |
|---:|---:|---:|---:|---:|
| 2 | 1.01 | 0.54 | 0.59 | 0.53 |
| 4 | 2.56 | 2.08 | 1.64 | 0.64 |
| 8 | 12.19 | 9.10 | 6.51 | 0.53 |
| 16 | 43.67 | 30.80 | 25.40 | 0.58 |
| 32 | 173.92 | 124.04 | 107.94 | 0.62 |
| 64 | 703.61 | 537.90 | 454.82 | 0.65 |

![VR estimator error is below the standard estimator](images/claim2_vr_vs_naive.png)

## Controls and release evidence

- Standard `p=1,b=6` is the no-recursion control.
- Both VR settings use an average gradient-approximation budget near six; the complete configurations are printed by the fixed verifier.
- Three fixed-seed trajectories are averaged per dimension.
- Raw output: [`verdict.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/outputs/verdict.json), key `c2_vr_estimator`.

## Scope

This validates the mechanism and a matched-budget variance comparison. It is not a direct reproduction of a full MRI or black-hole reconstruction.
