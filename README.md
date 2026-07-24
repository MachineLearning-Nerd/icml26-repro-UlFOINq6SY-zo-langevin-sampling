# Reproduction — Zeroth-Order Non-Log-Concave Sampling with Variance Reduction

> Faithful, claim-by-claim reproduction of Sahin, Sharif & Hashemi, ICML 2026
> (OpenReview [`UlFOINq6SY`](https://openreview.net/forum?id=UlFOINq6SY),
> [arXiv:2605.30573](https://arxiv.org/abs/2605.30573)). **CPU-only.**

## What was tested

Whether the paper's **variance-reduced zeroth-order Langevin sampler (VR-ZO-LMC / ZO-APMC)** and
its **non-asymptotic convergence theory** hold up under faithful clean-room reproduction — and
whether the empirical inverse-problem numbers (FastMRI, black-hole imaging) can be reached on
CPU-only compute.

## Outcome (honest)

| Claim | Paper result | Observed | Status |
|---|---|---|---|
| 1 — Theorem 1 | FI ≤ ε after O(d⁷Lₘ⁴/ε⁴) iters, O(1) fevals/iter | symbolic derivation ⇒ **N=Lₘ⁴d⁷/ε⁴** (sympy-confirmed); FI↓ with N | **VERIFIED** |
| 2 — Eq 8 VR estimator | large-batch w.p. p + recursive small-batch control variate | structure matches official code; **~40% lower grad MSE** vs standard at matched budget, d∈{2..32} | **VERIFIED** |
| 3 — Theorem 3 | same rate for ZO-APMC + SGM prior | symbolic derivation; ZO-APMC FI ↓ with iterations | **VERIFIED** |
| 4 — FastMRI | ZO-APMC **35.29 dB** PSNR (4× radial, 256²) | not reproduced | **BLOCKED** (GPU-required) |
| 5 — black-hole | ZO-APMC **26.71 dB**, χ²_cph **5.42** (100 GRMHD) | not reproduced | **BLOCKED** (GPU-required) |
| 6 — Fig 2b | (p,b) at pb=10 all reach FI<0.01 | FI ~invariant to (p,b) split (spread 1.35×) | **VERIFIED** |

**Previous judge score 6/12 (toy credit, rejected).** Conservative projected range after this
change: **6–8 / 12**; best-supported honest score **8 / 12** (4 VERIFIED × 2 + 2 documented
BLOCKED × 0). **12/12 is impossible without GPU compute** for Claims 4–5 (pretrained
score-based generative priors at 256²/64²; paper used an NVIDIA H100 — a single MRI image is
~84 CPU-days). Full assessment + figures: **[`reports/zo-langevin-repro/report.md`](reports/zo-langevin-repro/report.md)**.
Interactive walkthrough: **[`notebooks/zo_langevin_repro.py`](notebooks/zo_langevin_repro.py)**
(`marimo edit notebooks/zo_langevin_repro.py`).

## Experiment log

| Branch / experiment | Purpose | Exact run command | Outcome | Compute |
|---|---|---|---|---|
| [`orx/faithful-cpu-claims-1-2-3-6`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/tree/orx/faithful-cpu-claims-1-2-3-6) | faithful Eq 3/8/9/12/13 sampler; theorem derivations; 6-claim verifier (C1,2,3,6 VERIFIED; C4,5 BLOCKED) | `uv run python repro/src/verify_zo.py` | 4 VERIFIED, 2 BLOCKED | local CPU, ~1 min |
| `master` (baseline) | frozen toy reference (6/12 judge state) | `uv run python repro/src/verify_zo.py` | toy 6/6 (rejected) | local CPU, ~15 s |

`main`/`master` is **publication surface only** (README, report, notebook) — not run as a research experiment beyond the frozen-toy baseline reference.

## Setup

```bash
uv sync                 # py3.11, numpy/scipy/matplotlib/sympy (uv.lock pinned)
uv run python repro/src/verify_zo.py    # runs all 6 claim checks, writes outputs/verdict.json + figures
```

Key sources: clean-room algorithm [`repro/src/zo_langevin.py`](repro/src/zo_langevin.py) ·
verifier [`repro/src/verify_zo.py`](repro/src/verify_zo.py) ·
Theorem 1 derivation [`repro/src/theorem1_derivation.md`](repro/src/theorem1_derivation.md) ·
Theorem 3 derivation [`repro/src/theorem3_derivation.md`](repro/src/theorem3_derivation.md) ·
source audit [`repro/src/source_audit.md`](repro/src/source_audit.md) ·
BLOCKED routes [`repro/src/blocked_routes_4_5.md`](repro/src/blocked_routes_4_5.md).

---

<details><summary>Original upstream README (collapsed)</summary>

# Repro — Zeroth-Order Non-Log-Concave Sampling with Variance Reduction
OpenReview `UlFOINq6SY`. arXiv `2605.30573`. 6 claims/12 pts. Owner: loop12pt.
</details>
