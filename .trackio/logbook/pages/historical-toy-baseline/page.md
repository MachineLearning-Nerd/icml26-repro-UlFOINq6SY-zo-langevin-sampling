# Historical rejected baseline (toy 6/12)

> **Status: REJECTED.** This is the evidence from judged Space revision
> `e5ffe005d9ed871afd36dcdf5b1dda186b673a2b` (2026-07-24). It is preserved unchanged for
> provenance. It is **superseded** by the [current verification run](#/verification-run).
> Do not treat these numbers as current evidence.

The judge (2026-07-24) assessed this as 6/12 — toy credit only:

> "This logbook runs only tiny 2-4D synthetic numpy experiments on CPU in under 3 seconds, using
> arbitrary thresholds and proxy targets instead of any real datasets, models, or scales
> described in the paper. None of the paper's actual numerical results (FastMRI PSNR, black-hole
> imaging metrics, FI<0.01) are reproduced; all claims are addressed only on clearly simplified
> toy setups."

## The toy verifier output (preserved for reference)
```
c1: Fisher info vs T [200,800,3200]: [7.147, 3.435, 2.022]  (threshold: FI < FI[0]*1.5)
c2: std var=1.1272, VR var=0.7530                            (threshold: VR <= std*1.5)
c3: mixture target Fisher info: 8.8038                        (threshold: < 20)
c4: reconstruction MSE: 0.2685                                (threshold: < 1.0; "synthetic proxy")
c5: image reconstruction MSE: 0.0408                          (threshold: < 0.5; "synthetic proxy")
c6: FI spread 1.796 for (p,b) at pb=10                       (threshold: < 2.0); FI values 1.25/0.81/1.46
```

**Why it was rejected:** arbitrary thresholds (e.g. FI<FI[0]·1.5, MSE<1.0) unrelated to the
paper's quantifiers; 2-4D synthetic proxies for FastMRI/black-hole; no real datasets, no
pretrained SGM, no PSNR measurement; FI values (1.0–1.5) far above the paper's 0.01 threshold.

The [current verification run](#/verification-run) replaces this with faithful clean-room
evidence for claims 1, 2, 3, 6 and honest BLOCKED documentation for claims 4, 5.
