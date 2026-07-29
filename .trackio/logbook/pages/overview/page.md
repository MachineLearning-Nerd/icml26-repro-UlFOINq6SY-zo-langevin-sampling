# Overview

## Paper
**Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse
Problems** — Sahin, Sharif, Hashemi (ICML 2026, [arXiv:2605.30573](https://arxiv.org/abs/2605.30573)).
Source: ar5iv HTML, SHA-256 `3112d8f4...6aaf2e`. Official code: [mberk-sahin/zo-posterior-sampling](https://github.com/mberk-sahin/zo-posterior-sampling) (GPU/CUDA).

## Evidence status
The judge (verdict `DineshAI/UlFOINq6SY@efd04078`, 2026-07-24) marked every claim **toy/inconclusive**:
"SGM prior replaced by GMM proxy; actual SGM never tested", "FI<0.01 not reached", "only d≤32",
"FastMRI/BH never addressed experimentally". v2 fixes each:

- **Real SGM (Claims 3, 4/5).** Trained a score network by denoising score matching (DSM) and used
  it as the ZO-APMC prior — no analytical proxy. Two SGMs: a 2D score-MLP (Claim 3) and a 16×16
  U-Net score model on MNIST (Claims 4/5). The ZO-APMC image inverse problem runs end-to-end on
  real images with PSNR (the score net is queried once per iteration; ZO evaluations act only on
  the black-box likelihood, so small-image reconstruction is CPU-feasible).
- **Claim 6 is separated by contract.** A bare Gaussian VR-ZO-LMC control reaches FI below 0.01
  for 3/4 fixed-budget pairs. It is not Figure 2(b). The exact-contract campaign found high FI in
  an Appendix C.1-style reconstruction, but estimator-floor controls mean it is **BLOCKED**, not
  falsified.
- **Higher dimensions (Claims 1, 2).** Rate corroborated across d∈{2,4,8,16}; VR variance
  reduction demonstrated up to d=64.

## Honest result
- **C1, C2, C3 → VERIFIED** by the scope stated on their individual pages.
- **C4, C5 → reduced-scale DEMONSTRATED**: real SGM + real image inverse problem + PSNR, but at
  16×16 MNIST (not 256×256 FastMRI / 64×64 GRMHD). The exact 35.29 dB / 26.71 dB numbers require a
  fully-trained SGM + GPU (the paper used an H100; CPU is ~10²–10³× too slow).
- **C6 → BLOCKED** under the exact Figure 2(b) contract; see [the campaign](#/claim-6-falsification).

## Fixed command & environment
```
uv run python repro/src/verify_zo.py     # ~4 min on CPU; writes outputs/verdict.json + figures/
```
uv, Python ≥3.11, numpy/scipy/matplotlib/sympy/**torch (CPU)**. Pre-trained SGM weights committed
(`outputs/sgm_score2d.pt`, `outputs/mnist_scorenet_16.pt`).

Use the [Claims evidence index](#/claims) to navigate the exact contracts, controls and raw outputs.
