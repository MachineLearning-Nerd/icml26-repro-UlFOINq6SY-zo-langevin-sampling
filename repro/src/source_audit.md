# Source Audit — arXiv 2605.30573 (UlFOINq6SY)

## Paper source
- **URL**: https://ar5iv.labs.arxiv.org/html/2605.30573 (HTML); https://arxiv.org/abs/2605.30573 (abstract)
- **Title**: Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications to Inverse Problems
- **Authors**: M. Berk Sahin, Behzad Sharif, Abolfazl Hashemi (Purdue University)
- **Venue**: ICML 2026 (OpenReview UlFOINq6SY)
- **Retrieval date**: 2026-07-24
- **SHA-256 (ar5iv HTML)**: `3112d8f41346203225ba6559b96789bcd841508eaa25b32de4d5cd37de6aaf2e`
- **arXiv stamp**: v1 [cs.LG] 28 May 2026

## Official code
- **URL**: https://github.com/mberk-sahin/zo-posterior-sampling (cloned 2026-07-24)
- **Environment (env.yaml)**: python 3.11.11, **pytorch 2.4.1 + CUDA 12.1** (GPU-only),
  numpy 1.26.4, scipy, deepinv 0.3.7, sigpy, nfft. Conda-based.
- **Pretrained prior**: `fastmri_brain.pth` from PnP-MonteCarlo (Sun et al. 2024),
  https://github.com/sunyumark/PnP-MonteCarlo
- **Scope of released code**: MRI radial reconstruction only (`main/configs/radial_mri/`).
  The toy 2D synthetic experiments (Fig 2, 6-9) and the black-hole / Navier-Stokes
  configs are NOT released; black-hole/NS use InverseBench (Zheng et al. 2025b,
  ICLR, openreview U3PBITXNG6).
- **ZO-VR estimator implementation** (`main/pmc/algorithms/zo_pmcred.py:gcurr_update`):
  large batch w.p. p via `forward_model.zograd(..., batch_size=b_large)`; else
  `gcurr += forward_model.zo_grad_diff(x_k, x_prev, ..., batch_size=b_small)` —
  the SARAH/PAGE-style control variate of Eq 8. The clean-room
  `VRGradientEstimator` in `zo_langevin.py` reproduces this exactly (large batch
  w.p. p; `g_prev + mean_i[g_hat(x_k,u_i)-g_hat(x_{k-1},u_i)]` w.p. 1-p, with the
  SAME directions u_i at both iterates).

## Claim anchors (exact quantifiers)
| Claim | Source anchor | Exact statement / quantifier |
|---|---|---|
| 1 | Theorem 1, §3.1; proof B.3 (Eq 48, 1256-1266) | FI(ν̄_{Nγ}‖π) ≤ ε after N=O(d⁷Lₘ⁴/ε⁴) iters, O(1) fevals/iter, with γ=Lₘ⁻¹N⁻³ᐟ⁴d⁻⁷ᐟ⁴, p=LₘN⁻¹ᐟ⁴d⁻¹ᐟ⁴, b=⌈1/p⌉, μ²=Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴ |
| 2 | Eq 8, §3.1; Proposition 1 (Eq 11) | gₖ = (1/b)Σ∇̃f(xₖ,u) w.p. p; g_{k-1}+(1/b')Σ(∇̃f(xₖ,u)-∇̃f(x_{k-1},u)) w.p. 1-p |
| 3 | Theorem 3, §3.2; proof B.6 (Eq 80/95) | Same complexity for ZO-APMC (Eq 12) + SGM prior; +irreducible σ̄²+ε̄_σ²+ᾱ² bias |
| 4 | Table 1, §4.2/C.3 | ZO-APMC FastMRI PSNR 35.29 dB (4× radial, 256², 40 imgs ×20 recon, p=0.2,b=10⁴,b'=10³); APMC 36.55 dB |
| 5 | Table 2, §4.3/C.4 | ZO-APMC BH PSNR 26.71 dB, χ²_cph 5.42 (100 GRMHD 64², p=1,b=1024,μ=0.01) |
| 6 | Fig 2(b), §4.1 | (p,b) with pb=10 all reach FI<0.01 after N=2000 (d=2 bimodal GMM, ε*=2.5 score noise) |

## Assumptions
- A1: f is L₁-Lipschitz (does NOT hold globally even for Gaussians; holds on compact domains / with clipping).
- A2: ∇f is L₂-Lipschitz (standard LMC). Lₘ=max{L₁,L₂}.
- A3 (Cor 1): target satisfies Poincaré inequality (for TV bound).
- A4: ‖∇h_{σₖ}−∇h‖ ≤ C₁σₖ. A5: ‖S_θ+∇h_{σₖ}‖ ≤ ε_{σₖ}=O(k⁻¹ᐟ²), ‖S_θ‖≤C₂/σₖ.

## Deviations in this reproduction (honest)
1. **Theorem-claim evidence (1, 3)**: primary = independent symbolic derivation reconstruction
   (accepted mode for universally-quantified theorems); numerical sweeps are scoped corroboration only.
2. **Synthetic setup (claims 3, 6 corroboration)**: the paper's exact 2D toy prior params and the
   ε*=2.5 / αₖ coupling are under-specified (toy configs not released). We use a concrete bimodal-GMM
   prior + random linear A with the paper's stated schedule (σ₀=10,α₀=10,ρ₂=0.975,γ=0.1,μ=10⁻⁴).
   FI is computed via a GMM-fit density on a grid (paper Appendix C.1 method), with k=1 for the
   unimodal posterior. The absolute FI<0.01 threshold is setup-sensitive; we verify the robust
   substance (FI-decrease, batch-invariance, VR variance reduction) and report actual FI values.
3. **Claims 4, 5**: faithful reproduction requires GPU (pretrained SGM inference at 256²/64² over
   thousands of iters × large batches). CPU-only authorization → BLOCKED (documented compute wall).
4. **Diffusion constant**: x ← x − γg + √(2γ)ξ (stationary π∝e^{−f}); matches Eq 9/12.
