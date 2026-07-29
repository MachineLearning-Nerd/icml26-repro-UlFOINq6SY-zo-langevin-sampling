# Claim 3 — Theorem 3 and ZO-APMC with an SGM prior

> **Paper claim.** Under the likelihood, prior-smoothing, and SGM-score assumptions, ZO-APMC has the same controllable `O(d⁷Lₘ⁴/ε⁴)` complexity, plus irreducible schedule and score-approximation bias terms.

## Status: VERIFIED

The theorem’s rate is independently reconstructed. Its algorithmic corroboration uses a **trained denoising-score-matching MLP**, not an analytical GMM score proxy, as the prior score supplied to ZO-APMC.

## Contract and implementation

- **Theorem reconstruction:** [`theorem3_derivation.md`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/theorem3_derivation.md)
- **Algorithm:** [`zo_langevin.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/zo_langevin.py)
- **Trained weights:** `outputs/sgm_score2d.pt`, loaded by the fixed verifier.
- **Command:** `uv run python repro/src/verify_zo.py`

## Observed evidence

The score MLP is trained by DSM on a bimodal 2D prior and then frozen. Relative FI to the analytical posterior decreases over four iteration budgets (median of four seeds):

| Iterations N | 500 | 1,000 | 2,000 | 4,000 |
|---:|---:|---:|---:|---:|
| Relative FI | 4.532 | 2.873 | 0.982 | 0.680 |

![ZO-APMC with a trained SGM prior converges as N grows](images/claim3_zo_apmc_fi.png)

## Controls and release evidence

- The score network is an actual learned model, replacing the previous analytical-prior proxy.
- The derivation explicitly carries the SGM, smoothing, and annealing bias terms rather than silently treating the score as exact.
- Raw output: [`verdict.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/outputs/verdict.json), key `c3_theorem3`.

## Scope

The synthetic posterior makes FI measurable and isolates the theorem’s mechanism. It does not substitute for the paper’s full FastMRI or black-hole benchmark; those are documented separately.
