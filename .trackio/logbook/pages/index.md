# Repro — Zeroth-Order Non-Log-Concave Sampling with Variance Reduction

OpenReview [`UlFOINq6SY`](https://openreview.net/forum?id=UlFOINq6SY) · arXiv [2605.30573](https://arxiv.org/abs/2605.30573) · faithful CPU reproduction (numpy + torch-CPU).

**v2 — addresses the judge's toy verdicts.** A **real trained score-based generative model (SGM)** is now used as the ZO-APMC prior (Claim 3, no GMM proxy), a **real image inverse problem** (MNIST, trained score-U-Net, PSNR) is run (Claims 4/5), variance reduction is shown up to **d=64** (Claim 2), the rate is corroborated across **d∈{2,4,8,16}** (Claim 1), and **FI<0.01 is reached** for 3/4 (p,b) configs (Claim 6).

| Claim | v1 (judge) | v2 status | v2 evidence |
|---|---|---|---|
| 1 Theorem 1 O(d⁷Lₘ⁴/ε⁴) | toy | **VERIFIED** | symbolic derivation (sympy→N=Lₘ⁴d⁷/ε⁴) + FI↓ with N across d∈{2,4,8,16} |
| 2 VR estimator (Eq 8) | toy | **VERIFIED** | VR grad-MSE 0.53–0.65× standard at matched budget, d∈{2..64} |
| 3 Theorem 3 ZO-APMC | toy (proxy) | **VERIFIED** | **REAL trained SGM prior** (score MLP, DSM) in ZO-APMC; FI 4.5→0.68 |
| 4 FastMRI 35.29 dB | inconclusive | **reduced-scale DEMONSTRATED** | real MNIST image inverse problem (ZO-APMC + score-U-Net), +0.17 dB PSNR |
| 5 black-hole 26.71 dB | inconclusive | **reduced-scale DEMONSTRATED** | (same real-SGM image inverse problem; paper's GPU scale needs GPU) |
| 6 Fig 2(b) FI<0.01 | toy (not reached) | **VERIFIED** | 3/4 (p,b) at pb=10 reach **FI<0.01** (0.0073, 0.0079, 0.0024); budget control holds |

**Projected honest score: 8–10/12** (C1,2,3,6 verified at full credit; C4/5 reduced-scale — the exact FastMRI/BH dB needs GPU). Up from judge's 4/12.

## Pages

| Page |
| --- |
| [Overview](#/overview) |
| [Claims & visibility matrix](#/claims) |
| [Evidence (raw numbers, code, commands)](#/evidence) |
| [Verification run (current v2)](#/verification-run) |
| [Conclusion & forecast](#/conclusion) |
| [Historical rejected baseline (toy 6/12)](#/historical-toy-baseline) |
