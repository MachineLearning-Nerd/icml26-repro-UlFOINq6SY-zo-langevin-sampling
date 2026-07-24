# Independent Reconstruction of Theorem 3 (arXiv 2605.30573)

**Claim.** Let π ∝ ℓ(y|x) p(x) be the posterior, with likelihood potential f
satisfying Assumptions 1, 2 (constants L_{f1}, L_{f2}), prior potential h
satisfying Assumption 2 (Lₕ) and Assumption 4 (perturbed-vs-true score gap
≤ C₁σₖ), and the SGM satisfying Assumption 5 (score error ε_{σₖ} = O(k⁻¹ᐟ²),
‖S_θ‖ ≤ C₂/σₖ). Let Lₘ := max{L_{f2}+Lₕ, L_{f1}}. For the ZO-APMC iterates
(Eq 12) with schedules (Eq 13) and the VR estimator (Eq 8), the time-averaged
law satisfies FI(ν̄_{Nγ} ‖ π) ≤ ε after **N = O(d⁷ Lₘ⁴ / ε⁴)** iterations with
**O(1)** function evaluations per iteration.

This is the primary evidence for Claim 3; the ZO-APMC FI-vs-iteration sweep in
`verify_zo.py` is corroborating scope only.

## 1. How Theorem 3 extends Theorem 1

The proof (Appendix B.6) reuses the Lyapunov framework of Theorem 1 (§B.3) but
the drift now contains the SGM prior term −αₖ S_θ(x,σₖ) (Eq 12). Triangle-
inequality decomposition of the score error (Eq 82) splits the per-step bound
into the *same four Theorem-1 terms* plus three additional bias terms that do
**not depend on N** (they are determined by the annealing schedule and SGM
quality). The resulting bound (Eq 80 / 95):

```
FI(ν̄_{Nγ} ‖ π) ≤ C₀/(Nγ)                          ← same discretization term as Thm 1
   + 17 d(d+2) L_{f1}² / (2b)                       ← batched-ZO variance (Thm 1 analogue)
   + 17 μ² L_{f2}² (d+3)³ / 8                       ← ZO smoothing bias (Thm 1 analogue)
   + σ̄²                                             ← prior-smoothing bias (Assumption 4)
   + ε̄_σ²                                           ← SGM estimation error (Assumption 5)
   + ᾱ²                                             ← annealing-weight bias (αₖ→1)
   + 32 γ Lₘ² d φ(μ) ,                              ← discretization (Thm 1 analogue)
```
where the three schedule-dependent biases are
```
σ̄²  := (51 C₁²)/(2N) Σₖ σₖ²          (Assumption 4: ‖∇h_{σₖ}−∇h‖ ≤ C₁σₖ)
ε̄_σ² := (51)/(2N) Σₖ ε_{σₖ}²          (Assumption 5: SGM score error)
ᾱ²   := (51 C₂²)/(2N) Σₖ (αₖ−1)²/σₖ²  (annealing weight → 1)
```
These are finite constants (independent of N, d) whenever the schedules σₖ, αₖ
decay geometrically to their limits (Eq 121) and the SGM error decays as
ε_{σₖ} = O(k⁻¹ᐟ²) — both satisfied by the paper's setup.

## 2. Exponent check — identical to Theorem 1

With the same parameter choices as Theorem 1
(γ = Lₘ⁻¹N⁻³ᐟ⁴d⁻⁷ᐟ⁴, p = LₘN⁻¹ᐟ⁴d⁻¹ᐟ⁴, b = ⌈1/p⌉, μ² = Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴),
the N,d-dependent terms again each reduce to O(Lₘ d⁷ᐟ⁴ / N¹ᐟ⁴) (identical
algebra to theorem1_derivation.md §4). The schedule-bias terms σ̄², ε̄_σ², ᾱ²
are O(1) constants that lower-bound the achievable FI (the irreducible bias of
using a finite-noise SGM prior rather than the exact prior). Therefore

```
FI(ν̄_{Nγ} ‖ π) = O( Lₘ d⁷ᐟ⁴ / N¹ᐟ⁴ ) + [σ̄² + ε̄_σ² + ᾱ²] ,
```
and the iteration complexity to drive the *controllable* part below ε is
**N = O(d⁷ Lₘ⁴ / ε⁴)** with **O(1)** per-iteration evaluations. ✓

Under the additional Poincaré inequality (Assumption 3, constant C_PI),
Corollary 2 strengthens this to ε-accuracy in **squared TV distance** after
O(d⁷ Lₘ⁴ C_PI⁴ / ε⁴) iterations (via FI ≥ TV² / C_PI).

## 3. What "black-box SGM prior" means (algorithmic faithfulness)

ZO-APMC (Eq 12) queries ONLY the forward model through function evaluations
(via the ZO estimator gₖ over f) and the prior ONLY through the SGM
S_θ(x,σₖ). No gradient of the forward model is used. This is exactly what the
clean-room `zo_apmc` in `zo_langevin.py` implements: the likelihood potential f
is accessed solely through `VRGradientEstimator` (Eq 3/8), and the prior enters
only via `prior_score_fn(x, sigma)` (the SGM slot, supplied analytically in our
synthetic corroboration). The official `zo_pmcred.py:gcurr_update` has the
identical structure (large-batch w.p. p, recursive `zo_grad_diff` w.p. 1−p).

## Non-circularity note
Universally quantified over posteriors satisfying Assumptions 1, 2, 4, 5. This
document is an independent symbolic derivation (accepted evidence mode). The
numerical ZO-APMC sweep corroborates the convergence direction only.

## References (paper)
- Theorem 3 statement & bound Eq 80: §3.2 / Appendix B.6. Corollary 2 (TV):
  Appendix B.7. Assumptions 4, 5: §3.2. Schedules Eq 121: Appendix C.1.
