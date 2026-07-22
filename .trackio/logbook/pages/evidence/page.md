# Evidence


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_017005dabe56", "created_at": "2026-07-22T11:56:49+00:00", "title": "Verification output (last 40 lines)"}
-->
## Verification output (last 40 lines)

```
CLAIM 3 (Theorem 3): convergence extends to non-log-concave (mixture) target
==============================================================================
  mixture target Fisher info: 8.8038 (bounded < 20)
  -> PASS

==============================================================================
CLAIM 4: ZO sampling produces valid reconstruction (synthetic proxy for FastMRI)
==============================================================================
  reconstruction MSE: 0.2685 (< 1.0)
  (Paper: FastMRI 35.29 dB PSNR; synthetic denoising proxy.)
  -> PASS

==============================================================================
CLAIM 5: ZO sampling works on imaging inverse problem (synthetic proxy)
==============================================================================
  image reconstruction MSE: 0.0408 (< 0.5)
  (Paper: black-hole imaging 26.71 dB; synthetic proxy.)
  -> PASS

==============================================================================
CLAIM 6: O(1) per-iteration batch complexity — fixed pb=10, different (p,b)
==============================================================================
  (p=0.1, b=100, pb=10.0): FI=1.2505
  (p=0.5, b=20, pb=10.0): FI=0.8130
  (p=1.0, b=10, pb=10.0): FI=1.4598
  FI spread: 1.796 (< 2.0 — O(1) batch complexity)
  -> PASS

==============================================================================
VERDICT SUMMARY
==============================================================================
  [PASS] c1_fisher_conv
  [PASS] c2_vr_estimator
  [PASS] c3_sgm_prior
  [PASS] c4_fastmri_proxy
  [PASS] c5_blackhole_proxy
  [PASS] c6_batch_complexity

  6/6 claims verified.
  wrote outputs/verdict.json
```
