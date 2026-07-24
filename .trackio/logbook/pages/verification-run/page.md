# Verification run (current)

**Branch:** master @ `60f1823` (publication surface; reproduces the verdict) · experiment branch `orx/faithful-cpu-claims-1-2-3-6` @ `2130bce` · **Run:** `40d1957d-0c1f-4b6f-9f6a-12d8465bf65c` (local CPU, ~67 s, exit 0) · **Command:** `uv run python repro/src/verify_zo.py`

This page supersedes the [Historical rejected baseline](#/historical-toy-baseline) (toy 6/12). The
verifier below is the current one — it exits nonzero if any VERIFIED-eligible check fails.

## Headline figures

![Claim 1: FI decreases with N](images/claim1_fi_vs_N.png)

![Claim 2: VR estimator < standard at matched budget](images/claim2_vr_vs_naive.png)

![Claim 3: ZO-APMC FI convergence](images/claim3_zo_apmc_fi.png)

![Claim 6: FI ~invariant to (p,b) at fixed pb=10](images/claim6_batch_complexity.png)

## Verifier output (verbatim tail)

```
CLAIM 1 (Theorem 1): FI convergence, O(d^7 Lm^4/eps^4), O(1) fevals/iter
  Independent symbolic derivation present: True
  Solving leading bound term 1/(N*gamma)=eps for N gives: N = Lm**4*d**7/eps**4
  => complexity O(d^7 Lm^4 / eps^4)  [VERIFIED by symbolic reconstruction]
  bare VR-ZO-LMC  d=2  N= 500: median FI = 0.9946
  bare VR-ZO-LMC  d=2  N=4000: median FI = 0.0946
  FI monotonically decreases with N: True
  -> CLAIM 1 VERIFIED  (derivation + scaling corroboration)

CLAIM 2 (Eq 8): VR estimator structure + variance reduction at matched budget
  Per-step gradient MSE e_k^2 along trajectory (gamma=0.01, matched avg budget ~6):
    d= 2 | naive b=6: 0.940 | VR p=.4 b=9: 0.598 | VR p=.2 b=14: 0.621 | 0.613
    d= 8 | naive b=6: 13.51 | VR p=.4 b=9: 8.84  | VR p=.2 b=14: 7.11  | 0.526
    d=32 | naive b=6: 181.6 | VR p=.4 b=9: 132.7 | VR p=.2 b=14: 116.1 | 0.591
  VR gradient-MSE < standard at matched budget in every d: True
  -> CLAIM 2 VERIFIED

CLAIM 3 (Theorem 3): ZO-APMC posterior FI convergence + derivation
  Independent symbolic derivation present: True
  ZO-APMC  N= 500: median relative FI to posterior = 0.6626
  ZO-APMC  N=4000: median relative FI to posterior = 0.1651
  -> CLAIM 3 VERIFIED

CLAIM 6 (Fig 2b): O(1) per-iteration batch complexity
  (p=1.0, b= 10, pb=10): median FI = 0.1654
  (p=0.5, b= 20, pb=10): median FI = 0.2154
  (p=0.2, b= 50, pb=10): median FI = 0.2002
  (p=0.1, b=100, pb=10): median FI = 0.1591
  FI spread (max/min) across (p,b) at pb=10: 1.35
  -> CLAIM 6 VERIFIED  (O(1)-batch invariance)

CLAIM 4 (FastMRI 35.29 dB): BLOCKED -- GPU-required
CLAIM 5 (black-hole 26.71 dB, chi2_cph 5.42): BLOCKED -- GPU-required

VERDICT SUMMARY
  [VERIFIED] c1_theorem1
  [VERIFIED] c2_vr_estimator
  [VERIFIED] c3_theorem3
  [VERIFIED] c6_batch_complexity
  [BLOCKED]  c4_fastmri
  [BLOCKED]  c5_blackhole
  VERIFIED=4  BLOCKED=2  FAIL=0
```

Source: [`repro/src/verify_zo.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/orx/faithful-cpu-claims-1-2-3-6/repro/src/verify_zo.py). Full raw JSON: [`reports/zo-langevin-repro/verdict.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/reports/zo-langevin-repro/verdict.json).
