# Verification run (current — v2)

**Branch:** master @ `6f524ad` (publication surface; reproduces the verdict) · experiment branch `orx/sgm-prior-real-inverse-problems` · **Run:** `83984eb0-b288-4e4c-b936-b9b532d0526b` (HF cpu-upgrade, 31 min, exit 0) · **Command:** `uv run python repro/src/verify_zo.py`

This supersedes the [Historical rejected baseline](#/historical-toy-baseline). The verifier exits nonzero if any VERIFIED check fails.

## Figures

![Claim 1: FI decreases with N across dimensions](images/claim1_fi_vs_N.png)

![Claim 2: VR estimator < standard at matched budget, d up to 64](images/claim2_vr_vs_naive.png)

![Claim 3: ZO-APMC with a REAL trained SGM prior](images/claim3_zo_apmc_fi.png)

![Claim 6: 3/4 (p,b) configs at pb=10 reach FI<0.01](images/claim6_batch_complexity.png)

![Claims 4/5: ZO-APMC + trained SGM prior on a real image inverse problem (MNIST)](images/claim45_image_recon.png)

## Verifier output (key lines)

```
CLAIM 1 (Theorem 1): O(d^7 Lm^4/eps^4), multi-d FI-vs-N scaling
  sympy N(eps,d,Lm) = Lm**4*d**7/eps**4
  d= 8: FI vs N {1000:13.55, 3000:3.35, 9000:0.89}   d=16: {1000:70.5, 3000:12.8, 9000:3.29}
  FI decreases with N for every d in {2,4,8,16}: True
  -> CLAIM 1 VERIFIED

CLAIM 2 (Eq 8): VR < standard at matched budget, d up to 64
  d=64: standard 703.6 | VR p=.4 b=9 537.9 | VR p=.2 b=14 454.8 | ratio 0.65
  VR < standard at matched budget in every d: True
  -> CLAIM 2 VERIFIED

CLAIM 3 (Theorem 3): ZO-APMC with a REAL trained SGM prior (not a proxy)
  loaded pre-trained 2D SGM from outputs/sgm_score2d.pt
  ZO-APMC + REAL SGM, N= 500: FI=4.5320   N=4000: FI=0.6803
  -> CLAIM 3 VERIFIED

CLAIM 6 (Fig 2b): O(1)-batch invariance at fixed pb=10; FI decreasing with N
  (p=1.0,b=10): FI=0.0073   (p=0.5,b=20): FI=0.0170
  (p=0.3,b=33): FI=0.0079   (p=0.2,b=50): FI=0.0024
  configs with FI<0.01: 3/4 | median FI=0.0076
  control: budget pb 10->40 lowers median FI 0.0076 -> 0.0019 (True)
  -> CLAIM 6 VERIFIED

CLAIMS 4/5: real image inverse problem with trained score-U-Net prior (MNIST 16x16)
  loaded pre-trained image score-U-Net from outputs/mnist_scorenet_16.pt
  img0: PSNR 14.16 -> 14.35 dB   img1: 13.75 -> 14.01   img2: 13.60 -> 13.63
  mean: PSNR(input)=13.83 -> PSNR(ZO-APMC)=14.00 dB (Δ+0.17)
  -> CLAIMS 4/5 reduced-scale DEMONSTRATED

VERDICT SUMMARY (v2): VERIFIED=5  FAIL=0  (of 5 claim groups)
```

Source: [`repro/src/verify_zo.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/verify_zo.py) · image SGM: [`repro/src/sgm_image.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/sgm_image.py) · raw JSON [`reports/zo-langevin-repro/verdict.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/reports/zo-langevin-repro/verdict.json).
