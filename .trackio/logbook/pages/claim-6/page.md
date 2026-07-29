# Claim 6 — Figure 2(b) and O(1) batch complexity

> **Paper claim.** In the 2D toy inverse problem, all Figure 2(b) parameter settings with `pb=10` reach relative FI below `0.01` after 2,000 iterations.

## Status: BLOCKED for the exact contract

The paper gives the broad toy setting and FI estimator, but does not release the Figure 2 configuration or specify enough of the prior, initialisation, score-noise coupling, and estimator details to make one reconstruction uniquely authoritative. No assumption-valid falsification has been established, so the current claim verdict is preserved.

## Corroborating mechanism result (not an exact Figure 2 reproduction)

A bare VR-ZO-LMC Gaussian test bed with a closed-form Gaussian FI estimator and 24 pooled chains gives:

| (p,b) | pb | FI |
|---|---:|---:|
| (1.0,10) | 10 | 0.0073 |
| (0.5,20) | 10 | 0.0170 |
| (0.3,33) | 10 | 0.0079 |
| (0.2,50) | 10 | 0.0024 |

Three of four settings cross 0.01, and increasing the budget from `pb=10` to `pb=40` lowers median FI from 0.0076 to 0.0019. This supports the variance-reduction mechanism but cannot establish the paper’s universal “all Figure 2(b) settings” statement.

![Corroborating fixed-budget Gaussian test](images/claim6_batch_complexity.png)

## Exact-contract campaign

The dedicated reconstruction uses ZO-APMC, the stated bimodal-GMM setting, 1,000 particles, 2,000 iterations, 20 random forward operators, and the paper’s grid-GMM FI method. All reconstructed completions fail to cross 0.01, but the controls show that the chosen estimator itself has a floor above 0.01 even for exact posterior samples. That prevents those observations from being a valid counterexample.

- [Exact-contract campaign](#/claim-6-falsification)
- Raw output: [`c6_falsification.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/outputs/c6_falsification.json)
- Deterministic checker: [`check_c6_falsification.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/check_c6_falsification.py)
