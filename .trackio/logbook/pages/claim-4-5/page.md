# Claims 4–5 — full-scale inverse-problem results

> **Paper claims.** ZO-APMC reports 35.29 dB on 256×256 4× radial FastMRI reconstruction (Claim 4) and 26.71 dB / χ²_cph 5.42 on 64×64 GRMHD black-hole imaging (Claim 5).

## Status: reduced-scale method demonstration; exact claims BLOCKED on CPU-only

The current evidence is intentionally split into two statements:

1. A real trained score U-Net, a black-box likelihood, ZO-APMC, and PSNR all run end-to-end on images.
2. The paper’s exact FastMRI and black-hole numbers are **not** claimed as reproduced. Their released workflow is GPU-scale and cannot be run faithfully within the CPU-only authorization.

## Observed reduced-scale evidence

A DSM-trained 16×16 U-Net is used as the prior in a MNIST denoising inverse problem. Three held-out reconstructions improve PSNR over the noisy observation:

| Image | Noisy input PSNR | ZO-APMC + SGM PSNR |
|---:|---:|---:|
| 0 | 14.16 dB | 14.35 dB |
| 1 | 13.75 dB | 14.01 dB |
| 2 | 13.60 dB | 13.63 dB |
| Mean | 13.83 dB | 14.00 dB (+0.17 dB) |

![Real trained score U-Net in a black-box image inverse problem](images/claim45_image_recon.png)

## Controls and provenance

- The noisy input is the baseline control.
- The score prior is a trained U-Net (`outputs/mnist_scorenet_16.pt`), not an analytical score.
- The likelihood is accessed through ZO evaluations; no forward-model gradient is used.
- Raw output: [`verdict.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/outputs/verdict.json), key `c45_image_inverse`.
- Exact-benchmark feasibility analysis: [`blocked_routes_4_5.md`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/blocked_routes_4_5.md).

## Why the exact claims are blocked

The paper uses an H100 and a pretrained high-resolution SGM over thousands of iterations with ZO batches of 10⁴ (MRI) or 1024 (black hole). The documented CPU estimate is roughly 84 CPU-days for a single MRI image’s 20 reconstructions and years for Table 1. A small-image proxy cannot confirm or falsify the paper’s exact benchmark numbers.
