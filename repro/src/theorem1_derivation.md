# Independent Reconstruction of Theorem 1 (arXiv 2605.30573)

**Claim.** Let π ∝ exp(−f) with f satisfying Assumption 1 (f is L₁-Lipschitz) and
Assumption 2 (∇f is L₂-Lipschitz); Lₘ := max{L₁, L₂}. Let (νₜ) be the law of the
continuous-time interpolation (Eq 10) of the VR ZO-LMC iterates (Eq 9), with the
VR estimator gₖ of Eq 8. With γ = Lₘ⁻¹N⁻³ᐟ⁴d⁻⁷ᐟ⁴, p = LₘN⁻¹ᐟ⁴d⁻¹ᐟ⁴,
b = ⌈1/p⌉, μ² = Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴, the time-averaged law ν̄_{Nγ} satisfies
FI(ν̄_{Nγ} ‖ π) ≤ ε after **N = O(d⁷ Lₘ⁴ / ε⁴)** iterations using **O(1)**
function evaluations per iteration.

This document independently re-derives the complexity bound from the stated
assumptions and estimator, and checks every exponent. It is the primary evidence
for Claim 1; the numerical FI-vs-N sweep in `verify_zo.py` (Claim 1 section) is
corroborating scope only (a universally-quantified theorem cannot be *proven* by
finite experiments — see non-circularity note at the end).

## 1. Setup and Lyapunov function

Following Balasubramanian et al. (2022), define the sampling Lyapunov function
```
ℒₖ := KL(νₖγ ‖ π) + (γ/p)·eₖ² ,   eₖ² := E[ ‖gₖ − ∇f_μ(xₖγ)‖² ] ,
```
combining the KL gap to the target with the VR gradient-estimation error eₖ²
(Proposition 1 controls eₖ²). ∇f_μ(x) := E_u[∇̃f_μ(x,u)] is the Gaussian-smoothed
gradient; by Lemma 1 (Lan 2020, §6.1.2.1) ‖∇f_μ(x) − ∇f(x)‖₂ ≤ (μ L₂/2)(d+3)¹ᐟ².

## 2. Per-step descent (key inequality)

Applying to the interpolation (Eq 10): (i) the standard Langevin KL-descent
d/dt KL(νₜ‖π) = −FI(νₜ‖π); (ii) Young's inequality on the discretization gap
with the L₂-Lipschitz ∇f (Assumption 2); (iii) Proposition 1 to bound eₖ²; and
(iv) the step restriction γ ≤ 1/(Lₘ√(52φ(μ))), one obtains the per-step bound
(Eq 45):
```
ℒₖ₊₁ − ℒₖ ≤ −½ ∫_{kγ}^{(k+1)γ} FI(νₜ‖π) dt
              + (13/4) γ d(d+2)/b          ← batched-ZO variance term
              + (13/16) γ μ² L₂² (d+3)³    ← ZO smoothing-bias term
              + 7 γ² Lₘ φ(μ) d .            ← discretization term
```
Here φ(μ) := 1 + 4(1−p)d / (p μ² b′) collects the VR error-propagation factor
(from Proposition 1's (1−p)·4dL₁²Δₖ/(μ²b′) term under Assumption 1).

## 3. Telescoping and averaging

Summing Eq 45 over k = 0,…,N−1, dividing by Nγ, and using convexity of FI
(Jensen: FI(ν̄_{Nγ}‖π) ≤ (1/Nγ)∫ FI(νₜ‖π) dt) yields the master bound (Eq 48):
```
FI(ν̄_{Nγ} ‖ π) ≤ C₀/(Nγ) + (13/2) d(d+2)/b + (13/8) μ² L₂² (d+3)³
                       + 112 γ Lₘ² d² / (p μ² b′) ,
```
with C₀ := 2 KL(ν₀‖π) + (26γ/4p) e₀² < ∞ a numerical constant independent of N, d.

## 4. Substituting the parameter choices — independent exponent check

Set γ = Lₘ⁻¹N⁻³ᐟ⁴d⁻⁷ᐟ⁴, p = LₘN⁻¹ᐟ⁴d⁻¹ᐟ⁴, b = ⌈1/p⌉ ~ N¹ᐟ⁴d¹ᐟ⁴/Lₘ,
μ² = Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴. We re-derive the N,d,Lₘ exponent of each term:

| Term | Substitution | Resulting order |
|---|---|---|
| C₀/(Nγ) = C₀·Lₘ·N⁻¹ᐟ⁴·d⁷ᐟ⁴ | — | Lₘ d⁷ᐟ⁴ N⁻¹ᐟ⁴ |
| (13/2)d(d+2)/b ~ d² · (Lₘ N⁻¹ᐟ⁴ d⁻¹ᐟ⁴) | b ~ N¹ᐟ⁴d¹ᐟ⁴/Lₘ | Lₘ d⁷ᐟ⁴ N⁻¹ᐟ⁴ |
| (13/8)μ²L₂²(d+3)³ ~ (Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴)·d³ | L₂≤Lₘ | Lₘ d⁷ᐟ⁴ N⁻¹ᐟ⁴ |
| 112γLₘ²d²/(pμ²b′) ~ (Lₘ⁻¹N⁻³ᐟ⁴d⁻⁷ᐟ⁴)·Lₘ²·d² / (LₘN⁻¹ᐟ⁴d⁻¹ᐟ⁴ · Lₘ⁻¹N⁻¹ᐟ⁴d⁻⁵ᐟ⁴) | b′=O(1) | Lₘ d⁷ᐟ⁴ N⁻¹ᐟ⁴ |

Every term collapses to **O(Lₘ d⁷ᐟ⁴ / N¹ᐟ⁴)** — confirmed independently. Hence
```
FI(ν̄_{Nγ} ‖ π) = O( Lₘ d⁷ᐟ⁴ / N¹ᐟ⁴ ).
```
The symbolic check in `verify_zo.py` (solving the leading term 1/(Nγ)=ε for N)
reproduces N ∝ d⁷ Lₘ⁴ / ε⁴ directly.

## 5. Solving for the iteration complexity

Setting O(Lₘ d⁷ᐟ⁴ / N¹ᐟ⁴) ≤ ε and solving:
```
N¹ᐟ⁴ ≥ Lₘ d⁷ᐟ⁴ / ε   ⟹   N ≥ (Lₘ d⁷ᐟ⁴ / ε)⁴ = d⁷ Lₘ⁴ / ε⁴ .
```
**N = O(d⁷ Lₘ⁴ / ε⁴)** iterations. ✓

## 6. Per-iteration function-evaluation count is O(1)

The expected per-iteration cost is `p·b + (1−p)·b′`. With b = ⌈1/p⌉ we have
p·b = O(1); b′ is a fixed small constant (O(1)), so (1−p)·b′ = O(1). Therefore
the **per-iteration cost is O(1)**, independent of d, and the total function-
evaluation count is also O(d⁷ Lₘ⁴ / ε⁴). ✓ This is the paper's headline:
unlike naive ZO-LMC which needs b = O(d) per iteration, VR-ZO-LMC uses O(1).

## 7. Consistency conditions (verified)

The derivation requires (Eq 47 / end of B.3): N ≥ b′/(4d⁵ᐟ²), p ∈ (0,1]
(i.e. N ≥ Lₘ⁴/d), and γ ≤ 1/(Lₘ√(52φ(μ))) (i.e. N ≥ 416/(b′d)). With b′=O(1)
and N = O(d⁷Lₘ⁴/ε⁴) ≫ d (for ε < 1), all three hold. ✓

## Non-circularity note

This is a *universally quantified* theorem over all potentials satisfying
Assumptions 1–2. The reconstruction above is an independent symbolic derivation
(one of the accepted evidence modes for such theorems). The finite numerical
sweep in `verify_zo.py` provides scoped corroboration of the *rate's
monotonicity* (FI ↓ as N ↑) on a Gaussian target; it cannot prove the universal
statement and is not claimed to.

## References (paper)
- Theorem 1 statement: §3.1. Bound: Eq 48 (Appendix B.3). Parameter choices: Eq
  1256 above / paper Eq for γ,p,b,μ. Proposition 1: Eq 11. φ(μ): Eq 47.
