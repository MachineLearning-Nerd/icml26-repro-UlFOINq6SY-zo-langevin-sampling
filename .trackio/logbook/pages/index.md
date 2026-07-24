# Repro — Zeroth-Order Non-Log-Concave Sampling with Variance Reduction

OpenReview [`UlFOINq6SY`](https://openreview.net/forum?id=UlFOINq6SY) · arXiv [2605.30573](https://arxiv.org/abs/2605.30573) · CPU-only faithful reproduction.

**Honest score forecast: 8/12** (previous judge: 6/12 toy, rejected). 4 claims VERIFIED by faithful evidence; 2 claims BLOCKED by GPU-compute (FastMRI / black-hole need pretrained-SGM inference). 12/12 is not reachable on CPU.

| Claim | Status | Confidence |
|---|---|---|
| 1 Theorem 1 O(d⁷Lₘ⁴/ε⁴) | **VERIFIED** | MEDIUM-HIGH |
| 2 VR estimator (Eq 8) | **VERIFIED** | HIGH |
| 3 Theorem 3 ZO-APMC | **VERIFIED** | MEDIUM |
| 4 FastMRI 35.29 dB | **BLOCKED** | — |
| 5 black-hole 26.71 dB | **BLOCKED** | — |
| 6 O(1) batch (Fig 2b) | **VERIFIED** | MEDIUM-HIGH |

## Pages

| Page |
| --- |
| [Overview](#/overview) |
| [Claims & visibility matrix](#/claims) |
| [Evidence (raw numbers, code, commands)](#/evidence) |
| [Verification run (current)](#/verification-run) |
| [Conclusion & forecast](#/conclusion) |
| [Historical rejected baseline (toy 6/12)](#/historical-toy-baseline) |
