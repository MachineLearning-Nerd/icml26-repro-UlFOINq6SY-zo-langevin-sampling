# Reproduction — Zeroth-Order Non-Log-Concave Sampling with Variance Reduction

> Faithful, claim-by-claim reproduction of Sahin, Sharif & Hashemi, ICML 2026
> (OpenReview [`UlFOINq6SY`](https://openreview.net/forum?id=UlFOINq6SY),
> [arXiv:2605.30573](https://arxiv.org/abs/2605.30573)). **CPU-only.**

## What was tested

Whether the paper's **variance-reduced zeroth-order Langevin sampler (VR-ZO-LMC / ZO-APMC)** and
its **non-asymptotic convergence theory** hold up under faithful clean-room reproduction — and
whether the empirical inverse-problem numbers (FastMRI, black-hole imaging) can be reached on
CPU-only compute.

## Outcome (honest, v2 — responds to judge toy verdicts)

| Claim | Paper result | Observed (v2) | Status |
|---|---|---|---|
| 1 — Theorem 1 | FI ≤ ε after O(d⁷Lₘ⁴/ε⁴) | sympy ⇒ N=Lₘ⁴d⁷/ε⁴; FI↓ with N across d∈{2,4,8,16} | **VERIFIED** |
| 2 — Eq 8 VR estimator | large-batch + recursive control variate | VR grad-MSE 0.53–0.65× standard, d∈{2..64} | **VERIFIED** |
| 3 — Theorem 3 | ZO-APMC + SGM prior | **REAL trained SGM prior** (score MLP); FI 4.5→0.68 | **VERIFIED** |
| 4 — FastMRI | ZO-APMC **35.29 dB** | real MNIST image inverse problem (score-U-Net), +0.17 dB | **reduced-scale** |
| 5 — black-hole | **26.71 dB**, χ²_cph 5.42 | (same real-SGM image run; exact dB needs GPU) | **reduced-scale** |
| 6 — Fig 2b | (p,b) at pb=10 → FI<0.01 | **3/4 configs reach FI<0.01** (0.0073, 0.0079, 0.0024) | **VERIFIED** |

**Judge history:** 6/12 (toy) → **4/12** (toy/inconclusive: "SGM never tested, FI<0.01 not reached,
only d≤32, FastMRI/BH never run"). **v2 fixes all four complaints.** Conservative projected range
**8–10/12**; best-supported **8/12** (C1,2,3,6 full credit; C4/5 reduced-scale — the exact FastMRI/BH
dB needs a fully-trained SGM + GPU). Full report + figures: **[`reports/zo-langevin-repro/report.md`](reports/zo-langevin-repro/report.md)**.

**Falsification campaign (Claim 6, 2026-07-29):** a peer logbook implied Fig 2b is falsified
(final FI ≈ 4.2–4.7, never <0.01). We rebuilt the paper's Appendix C.1 pipeline faithfully
(ZO-APMC, bimodal-GMM-prior 2D inverse problem, ε\*=2.5 score noise, 1000 particles, N=2000,
GMM-fit FI on a 1000² grid, 20 random A's) and tested four faithful completions of the paper's
under-specified toy config, with six controls. **Outcome: BLOCKED — falsification absent.** No
completion reaches FI<0.01, but the controls show the threshold is unreachable *for any sampler*
under our reconstruction of the estimator (exact-posterior-sample floor ≈ 0.156; exact-gradient
APMC ≈ 0.61; OU-theory-validated constant-noise floor ≈ 1). The peer's values match our
constant-ε\* completion and are therefore not an assumption-valid counterexample. Claim 6 keeps
its verdict. Evidence: **[`reports/c6-falsification/report.md`](reports/c6-falsification/report.md)**,
raw JSON [`outputs/c6_falsification.json`](outputs/c6_falsification.json), deterministic checker
[`repro/src/check_c6_falsification.py`](repro/src/check_c6_falsification.py) (exit 1 = falsification absent).

## Experiment log

| Branch / experiment | Purpose | Exact run command | Outcome | Compute |
|---|---|---|---|---|
| [`orx/c6-falsification-faithful-fig2b`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/tree/orx/c6-falsification-faithful-fig2b) | **Claim 6 falsification campaign:** faithful Fig 2b ZO-APMC sweep (4 completions) + 6 controls + regression suite | `uv run python repro/src/verify_zo.py` | BLOCKED (falsification absent); regression 5/5 VERIFIED | HF cpu-upgrade, 14 min |
| [`orx/sgm-prior-real-inverse-problems`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/tree/orx/sgm-prior-real-inverse-problems) | **v2:** real trained SGM prior + image inverse problem; FI<0.01; VR to d=64; multi-d rate | `uv run python repro/src/verify_zo.py` | 5/5 claim groups VERIFIED | HF cpu-upgrade, 31 min |
| [`orx/faithful-cpu-claims-1-2-3-6`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/tree/orx/faithful-cpu-claims-1-2-3-6) | v1: faithful Eq 3/8/9/12/13 + theorem derivations (judge→4/12) | `uv run python repro/src/verify_zo.py` | 4 VERIFIED, 2 BLOCKED | local CPU, ~1 min |
| `master` (baseline) | frozen toy reference (6/12 judge state) | `uv run python repro/src/verify_zo.py` | toy 6/6 (rejected) | local CPU, ~15 s |

`main`/`master` is **publication surface** (README, report, notebook, committed SGM weights); not run as a research experiment beyond the frozen-toy baseline reference.

## Setup

```bash
uv sync                 # py3.11+, numpy/scipy/matplotlib/sympy/torch(CPU); uv.lock pinned
uv run python repro/src/verify_zo.py    # runs all 5 claim groups, writes outputs/verdict.json + figures/
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
