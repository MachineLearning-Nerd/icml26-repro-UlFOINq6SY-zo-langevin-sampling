# Repro — Zeroth-Order Non-Log-Concave Sampling with Variance Reduction

OpenReview [`UlFOINq6SY`](https://openreview.net/forum?id=UlFOINq6SY) · arXiv [2605.30573](https://arxiv.org/abs/2605.30573) · faithful CPU reproduction (numpy + torch-CPU).

**Evidence-first, claim-by-claim reproduction.** Each claim now has its own page with the exact paper contract, evidence status, deterministic command, controls, raw outputs, and scope limits. This separates full evidence from reduced-scale support and from the exact-contract Claim 6 campaign.

| Claim | Earlier status | Current evidence status | Evidence |
|---|---|---|---|
| 1 Theorem 1 O(d⁷Lₘ⁴/ε⁴) | toy | **VERIFIED** | symbolic derivation (sympy→N=Lₘ⁴d⁷/ε⁴) + FI↓ with N across d∈{2,4,8,16} |
| 2 VR estimator (Eq 8) | toy | **VERIFIED** | VR grad-MSE 0.53–0.65× standard at matched budget, d∈{2..64} |
| 3 Theorem 3 ZO-APMC | toy (proxy) | **VERIFIED** | **REAL trained SGM prior** (score MLP, DSM) in ZO-APMC; FI 4.5→0.68 |
| 4 FastMRI 35.29 dB | inconclusive | **reduced-scale demonstrated; exact BLOCKED** | real MNIST image inverse problem (ZO-APMC + score-U-Net), +0.17 dB PSNR |
| 5 black-hole 26.71 dB | inconclusive | **reduced-scale demonstrated; exact BLOCKED** | (same real-SGM image inverse problem; paper's GPU scale needs GPU) |
| 6 Fig 2(b) FI<0.01 | toy (not reached) | **BLOCKED (exact contract)** | bare Gaussian corroboration crosses 0.01 for 3/4 settings; the exact Figure 2(b) reconstruction is under-specified |

No score is predicted here. Each page states only the evidence collected for its own paper contract.

## Pages

| Page |
| --- |
| [Overview](#/overview) |
| [Claims — evidence index](#/claims) |
| [Claim 1 — Theorem 1 rate](#/claim-1) |
| [Claim 2 — VR estimator](#/claim-2) |
| [Claim 3 — ZO-APMC + SGM](#/claim-3) |
| [Claims 4–5 — inverse problems](#/claim-4-5) |
| [Claim 6 — Figure 2(b)](#/claim-6) |
| [Claim 6 — exact-contract campaign](#/claim-6-falsification) |
| [Evidence (raw numbers, code, commands)](#/evidence) |
| [Verification runs and scope](#/verification-run) |
| [Conclusion](#/conclusion) |
| [Historical rejected baseline (toy 6/12)](#/historical-toy-baseline) |
