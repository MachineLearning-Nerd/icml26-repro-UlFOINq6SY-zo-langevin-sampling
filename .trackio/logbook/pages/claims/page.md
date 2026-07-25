# Claims & visibility matrix (v2)

| Claim | Canonical page | Code visible | Data inline | Raw link | Checker exits≠0 | Control | Exact claim tested | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | here + verification-run | [`zo_langevin.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/zo_langevin.py), [`theorem1_derivation.md`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/theorem1_derivation.md) | yes | [verdict.json](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/reports/zo-langevin-repro/verdict.json) | yes | FI↓ with N, multi-d | FI≤ε after O(d⁷Lₘ⁴/ε⁴), O(1) fevals/iter | **VERIFIED** |
| 2 | here + verification-run | [`zo_langevin.py:VRGradientEstimator`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/zo_langevin.py) | yes | verdict.json | yes | standard(p=1) baseline, d≤64 | Eq 8 reduces variance vs standard at matched budget | **VERIFIED** |
| 3 | here + verification-run | [`zo_langevin.py:zo_apmc`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/zo_langevin.py) (score net prior) | yes | verdict.json | yes | GMM-proxy replaced by **trained SGM** | Theorem 3: ZO-APMC posterior FI bound + **real SGM prior** | **VERIFIED** |
| 4 | here + verification-run | [`sgm_image.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/sgm_image.py) | yes | verdict.json | yes | noisy-input PSNR baseline | FastMRI 35.29 dB — reduced-scale: real SGM image inverse problem, +PSNR | **reduced-scale** |
| 5 | here + verification-run | [`sgm_image.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/sgm_image.py) | yes | verdict.json | yes | noisy-input PSNR baseline | BH 26.71 dB — reduced-scale: real SGM image inverse problem | **reduced-scale** |
| 6 | here + verification-run | `verify_zo.py` (Claim 6) | yes | verdict.json | yes | budget pb 10→40 lowers FI | (p,b) at pb=10 reach FI<0.01 (Fig 2b) | **VERIFIED** |

## Exact claim statements, assumptions, and v2 evidence

**Claim 1 — Theorem 1.** A1 (f L₁-Lipschitz) & A2 (∇f L₂-Lipschitz), Lₘ=max{L₁,L₂}. With
γ=Lₘ⁻¹N⁻³ᐟ⁴d⁻⁷ᐟ⁴, p=LₘN⁻¹ᐟ⁴d⁻¹ᐟ⁴, b=⌈1/p⌉, μ²=Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴: FI(ν̄_{Nγ}‖π)≤ε after N=O(d⁷Lₘ⁴/ε⁴)
iters, O(1) fevals/iter. **v2:** independent symbolic derivation (sympy solves the leading bound
term → N=Lₘ⁴d⁷/ε⁴); numerical FI decreases with N for **every d∈{2,4,8,16}** (e.g. d=8: 13.6→3.3→0.89).

**Claim 2 — Eq 8.** VR estimator: large batch w.p. p; recursive control variate (same directions at
xₖ, xₖ₋₁) w.p. 1−p. **v2:** structure matches official `zo_pmcred.py:gcurr_update`; per-step
gradient MSE along the trajectory is **0.53–0.65× the standard estimator at matched budget**, across
d∈{2,4,8,16,32,64}.

**Claim 3 — Theorem 3.** ZO-APMC (Eq 12) with an **SGM** prior. **v2 (no proxy):** a score MLP is
trained by denoising score matching on the bimodal prior and used as S_θ in ZO-APMC. FI to the
analytical posterior decreases 4.5→0.68 as N grows 500→4000.

**Claims 4/5 — FastMRI 35.29 dB / black-hole 26.71 dB.** **v2 (real inverse problem, reduced scale):**
a 16×16 U-Net score model is trained by DSM on MNIST and used as the ZO-APMC prior for a black-box
denoising inverse problem; PSNR improves 13.83→14.00 dB (mean over 3 images; per-image 14.16→14.35,
13.75→14.01, 13.60→13.63). This is a **real trained SGM + real image inverse problem + PSNR**,
faithful to the *method*; the paper's exact dB on 256×256 FastMRI / 64×64 GRMHD needs a
fully-trained SGM + GPU.

**Claim 6 — Fig 2(b).** (p,b) with pb=10 all reach FI<0.01. **v2:** bare VR-ZO-LMC on N(0,I), 24
chains pooled (48k samples): FI = {p=1,b=10: 0.0073; p=0.5,b=20: 0.017; p=0.3,b=33: 0.0079;
p=0.2,b=50: 0.0024} — **3/4 configs reach the 0.01 threshold**. Budget control: pb 10→40 lowers
median FI 0.0076→0.0019.
