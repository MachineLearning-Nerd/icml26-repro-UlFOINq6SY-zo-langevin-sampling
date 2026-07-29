# Claim 6 — exact-contract falsification campaign

## Status: BLOCKED — no assumption-valid falsification

The paper says that, in Figure 2(b), **all** tested parameter pairs with fixed
`p b = 10` converge below Fisher information (FI) `0.01`. A valid falsification
must therefore use the Figure 2(b) problem, the stated sampler regime and FI
protocol, and exhibit a qualifying pair above the threshold. The result below
does not meet that release-quality bar because essential toy-problem details are
not specified sufficiently to establish that the reconstruction is Figure 2(b).

### Hypothesis tested

A peer logbook suggested that the fixed-budget pairs might finish far above
`0.01`. We treated that as a hypothesis, not as evidence, and independently
implemented an Appendix C.1-style reconstruction.

### Reconstruction result

The reconstruction uses the stated two-dimensional bimodal-GMM setting,
random linear operator, 1,000 particles, 2,000 iterations, 20 operator draws,
the paper's annealing schedule, and a GMM density evaluated on the Appendix C.1
grid for relative FI. With constant score-noise `eps*=2.5` and refresh batch
`b'=5`, none of the four fixed-budget pairs reached `0.01`:

| `p` | `b` | `p b` | mean final FI | runs below `0.01` |
|---:|---:|---:|---:|---:|
| 1.0 | 10 | 10 | 1.805 | 0 / 20 |
| 0.5 | 20 | 10 | 3.059 | 0 / 20 |
| 0.2 | 50 | 10 | 9.387 | 0 / 20 |
| 0.1 | 100 | 10 | 24.169 | 0 / 20 |

![Faithful-reconstruction FI by fixed-budget pair](images/claim6_falsification.png)

### Reachability controls prevent a falsification verdict

The same FI estimator was tested on exact samples from the known posterior and
on an exact-gradient APMC control. It returned FI `0.156` and `0.609`,
respectively—both above the paper's `0.01` threshold. The empirical result
therefore cannot distinguish a sampler failure from an estimator floor and
discretization bias. Replacing the interpretation with a falsification would
silently assume unspecified prior parameters, estimator choices, and the
score-noise-to-annealing coupling.

![Estimator and exact-gradient controls](images/claim6_falsify_controls.png)

### Executable evidence

| Item | Location |
|---|---|
| Reproduction command | `uv run python repro/src/c6_falsify.py` |
| Deterministic checker | `uv run python repro/src/check_c6_falsification.py` |
| Checker contract | exit `0` only for a valid, assumption-complete falsification; exit `1` for absent/BLOCKED; exit `2` for missing or malformed evidence |
| Raw summary | [`outputs/c6_falsification.json`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/outputs/c6_falsification.json) |
| Code | [`repro/src/c6_falsify.py`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/repro/src/c6_falsify.py) |
| Full report | [`reports/c6-falsification/report.md`](https://github.com/MachineLearning-Nerd/icml26-repro-UlFOINq6SY-zo-langevin-sampling/blob/master/reports/c6-falsification/report.md) |

The HF `cpu-upgrade` run completed with exit 0 (run `9ea72365-642d-47b0-b593-2cc9c6bc3f5d`,
commit `81fd932`). Its checker intentionally returned `1`, which records the
scientific status above rather than an execution failure.

### Interpretation

**BLOCKED is the rigorous outcome.** The reconstruction is a useful diagnostic
and its high FI values are preserved as raw evidence, but they do not satisfy
the paper claim's exact, fully specified contract. The separate
[Claim 6 evidence page](#/claim-6) preserves the earlier bare-Gaussian
fixed-budget corroboration without presenting it as Figure 2(b).
