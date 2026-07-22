# Claims


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_cc71260eeef0", "created_at": "2026-07-22T11:56:48+00:00", "title": "Claims to reproduce"}
-->
## Claims to reproduce

1. Theorem 1 establishes the first non-asymptotic convergence guarantee for variance-reduced zeroth-order Langevin sampling, bounding Fisher information FI(ν̄_{Nγ}||π) ≤ ε after O(d⁷L_m⁴/ε⁴) iterations using only O(1) function evaluations per iteration (Theorem 1).
2. The variance-reduced zeroth-order gradient estimator (Equation 8) combines a large-batch estimate (probability p, batch size b) with a small-batch recursive update (probability 1−p, batch size b′) that exploits correlation between consecutive iterates (Equation 8).
3. Theorem 3 extends the Fisher-information convergence bound to posterior sampling with a black-box score-based generative model (SGM) prior via the ZO-APMC algorithm (Equation 12-13, Theorem 3).
4. On 4x-accelerated radial-subsampled FastMRI brain reconstruction, ZO-APMC achieves 35.29 dB PSNR, the best among black-box methods (SCG, DPG, EnKG, Forward/Central-GSG) though slightly below the gradient-based APMC baseline at 36.55 dB (Section, Table 1).
5. On black-hole imaging with 100 GRMHD images from InverseBench, ZO-APMC attains 26.71 dB PSNR and χ²_cph of 5.42, the best result among evaluated black-box methods (Table 2).
6. Figure 2(b) validates the O(1) per-iteration batch complexity by showing multiple (p,b) parameter settings with fixed product pb=10 all converge to Fisher information below 0.01 (Figure 2).
