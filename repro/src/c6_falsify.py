"""Claim 6 falsification campaign: faithful Fig 2(b) / Appendix C.1 pipeline.

CLAIM (Fig 2b): every (p,b) setting with fixed product pb=10 converges to
relative Fisher information < 0.01 after 2000 ZO-APMC iterations.

Faithful spec recovered from the paper (Sec 4.1 + Fig 2 caption + App C.1):
  * problem: 2D synthetic linear inverse problem, bimodal Gaussian-mixture
    prior, random forward model A, y = A x_true + xi, xi ~ N(0, I)
  * score: ANALYTICAL smoothed prior score corrupted by additive Gaussian
    noise with std eps* = 2.5 (mimics SGM estimation error, Assumption 5)
  * sampler: ZO-APMC (Eq 12) with schedules (Eq 121): sigma_k = max(sigma0*
    rho2^k, sigma_min), alpha_k = max(alpha0*sigma_k^2, 1);
    sigma0=10, alpha0=10, rho2=0.975, sigma_min=0, gamma=0.1, mu=1e-4
  * 1000 particles initialized U[-50,50]^2, N=2000 iterations
  * metric: fit a GMM to the particle cloud, compute relative FI to the
    ANALYTICAL posterior on a 1000x1000 grid over [-50,50]^2; average the
    per-run FI over 20 random forward operators (seeds)

Under-specified in the paper (treated as completions, all tested):
  * exact GMM prior parameters      -> primary means +-(2.5,0), cov I;
                                       variant +-(8,0)
  * b' for Fig 2b                   -> primary b'=5 (Fig 2a); variant b'=2
                                       (the peer's choice)
  * eps* constant vs decaying       -> primary constant 2.5 (literal
                                       reading); variant linear 2.5 -> 0
                                       (App C.1 weak-convergence protocol,
                                       consistent with Assumption 5 /
                                       Theorem 3's decreasing eps_sigma_k)

Controls:
  * POSITIVE  gradient-based APMC, no score noise  -> must reach the
    estimator floor, else the completion/pipeline is broken
  * FLOOR     GMM fit to 1000 EXACT posterior samples -> what the FI
    estimator reports for perfect sampling at n=1000
  * ANALYTIC  gradient APMC + constant score noise vs OU-theory FI floor
    prediction (per-seed, unimodal-dominant seeds only)
  * NEGATIVE  ZO-APMC p=0.3, b=10 (pb=3, the paper's own unstable case in
    Fig 2a must NOT satisfy the criterion)
  * REFERENCE FI(GMM fit of prior samples || posterior) -- magnitude of a
    non-converged cloud (peer reported 4.2-4.7)
  * X-IMPL    sequential single-chain zo_apmc (zo_langevin.py) vs the
    vectorized implementation here, same config

Peer hypothesis under test (algorise logbook, claim-6 page): a faithful
pb=10 sweep lands at final FI ~ 4.2-4.7, nowhere near 0.01.

Everything is deterministic given SEED0. numpy/CPU only.
"""
from __future__ import annotations
import json
import numpy as np

import zo_langevin as ZO

SEED0 = 20260729

# --- paper Appendix C.1 constants (fixed contract) ------------------------- #
SIGMA0, ALPHA0, RHO2, SIGMA_MIN = 10.0, 10.0, 0.975, 0.0
GAMMA, MU = 0.1, 1e-4
NPART, NITER = 1000, 2000
EPS_STAR = 2.5
GRID_N, GRID_LIM = 1000, 50.0
PAIRS = [(1.0, 10), (0.5, 20), (0.2, 50), (0.1, 100)]   # pb = 10 (peer's set)
CLIP = 1e7          # overflow guard; any clipped run is marked diverged
ZO_SEQ_CHAINS = 100  # chains for the sequential-vs-vectorized cross-check


# --------------------------------------------------------------------------- #
# Problem: bimodal GMM prior (+-m, cov I), y = A x + xi -> exact GMM posterior
# --------------------------------------------------------------------------- #
class Fig2Problem:
    def __init__(self, seed, prior_mean=(2.5, 0.0), a_shape=(2, 2)):
        rng = np.random.default_rng(seed)
        self.m = np.asarray(prior_mean, float)
        self.A = rng.standard_normal(a_shape)
        comp = rng.integers(0, 2)
        x_true = (self.m if comp == 0 else -self.m) + rng.standard_normal(2)
        self.y = self.A @ x_true + rng.standard_normal(a_shape[0])
        self.AtA = self.A.T @ self.A
        self.Aty = self.A.T @ self.y
        # exact posterior: 2-component GMM, shared covariance
        P = np.eye(2) + self.AtA                     # prior cov = I
        self.post_cov = np.linalg.inv(P)
        self.post_means = np.stack([self.post_cov @ (sgn * self.m + self.Aty)
                                    for sgn in (+1.0, -1.0)])
        S_y = self.A @ self.A.T + np.eye(a_shape[0])  # cov of y | component
        iS_y = np.linalg.inv(S_y)
        lw = []
        for sgn in (+1.0, -1.0):
            r = self.y - self.A @ (sgn * self.m)
            lw.append(-0.5 * r @ iS_y @ r)
        lw = np.asarray(lw)
        lw -= lw.max()
        w = np.exp(lw)
        self.post_w = w / w.sum()
        self.post_icov = np.linalg.inv(self.post_cov)

    # likelihood potential f(x) = 0.5||Ax-y||^2, batched over rows of X
    def f_batch(self, X):
        R = X @ self.A.T - self.y
        return 0.5 * np.einsum("ij,ij->i", R, R)

    def grad_f_batch(self, X):
        return X @ self.AtA.T - self.Aty

    # smoothed prior score  grad log (p * N(0, sigma^2 I)), batched
    def prior_score_batch(self, X, sigma):
        cs = 1.0 + sigma * sigma                     # isotropic: I + sigma^2 I
        d1 = X - self.m
        d2 = X + self.m
        lp1 = -0.5 * np.einsum("ij,ij->i", d1, d1) / cs
        lp2 = -0.5 * np.einsum("ij,ij->i", d2, d2) / cs
        mx = np.maximum(lp1, lp2)
        e1, e2 = np.exp(lp1 - mx), np.exp(lp2 - mx)
        w1 = e1 / (e1 + e2)
        return (w1[:, None] * (-d1) + (1 - w1)[:, None] * (-d2)) / cs

    # exact posterior score, batched (2-component GMM, shared cov)
    def posterior_score_batch(self, X):
        X = np.asarray(X, float)
        lps, ss = [], []
        for j in range(2):
            d = X - self.post_means[j]
            lps.append(np.log(self.post_w[j] + 1e-300)
                       - 0.5 * np.einsum("ij,jk,ik->i", d, self.post_icov, d))
            ss.append(-(d @ self.post_icov.T))
        lps = np.stack(lps, 1)
        lps -= lps.max(1, keepdims=True)
        r = np.exp(lps)
        r /= r.sum(1, keepdims=True)
        return r[:, 0, None] * ss[0] + r[:, 1, None] * ss[1]

    def posterior_sample(self, rng, n):
        comp = rng.random(n) < self.post_w[0]
        L = np.linalg.cholesky(self.post_cov)
        z = rng.standard_normal((n, 2))
        means = np.where(comp[:, None], self.post_means[0], self.post_means[1])
        return means + z @ L.T

    def prior_sample(self, rng, n):
        comp = rng.integers(0, 2, n)
        return np.where(comp[:, None] == 0, self.m, -self.m) \
            + rng.standard_normal((n, 2))


# --------------------------------------------------------------------------- #
# FI estimator: GMM(k=2) fit + 1000x1000 grid quadrature (paper App C.1),
# with a grid-free Monte-Carlo cross-check and a mass-in-box validity guard.
# --------------------------------------------------------------------------- #
def _gmm_score_and_density(pts, w, means, covs):
    N, k = len(pts), len(w)
    icovs = np.linalg.inv(covs)
    const = 2 * np.log(2 * np.pi)
    logterms = np.empty((N, k))
    for j in range(k):
        d = pts - means[j]
        _, ld = np.linalg.slogdet(covs[j])
        maha = np.einsum("ij,jk,ik->i", d, icovs[j], d)
        logterms[:, j] = np.log(w[j] + 1e-300) - 0.5 * (const + ld + maha)
    mx = logterms.max(1, keepdims=True)
    dens = np.exp(mx[:, 0]) * np.exp(logterms - mx).sum(1)
    r = np.exp(logterms - mx)
    r /= r.sum(1, keepdims=True)
    score = np.zeros((N, 2))
    for j in range(k):
        score += r[:, j, None] * (-((pts - means[j]) @ icovs[j].T))
    return score, np.maximum(dens, 1e-300)


def fi_of_samples(samples, prob, ngrid=None, lim=GRID_LIM, k=2, seed=0,
                  n_mc=20000):
    """Fit GMM(k) to samples; return grid FI (paper estimator), MC FI
    (grid-free cross-check), and mass_in_box."""
    ngrid = GRID_N if ngrid is None else ngrid
    s = np.asarray(samples)
    s = s[np.all(np.isfinite(s), axis=1)]
    out = {"fi_grid": float("inf"), "fi_mc": float("inf"),
           "mass_in_box": 0.0, "valid": False}
    if len(s) < 4 * k:
        return out
    out["mass_in_box"] = float(np.mean(np.all(np.abs(s) <= lim, axis=1)))
    w, means, covs = ZO._gmm_em(s, k=k, reg=1e-4,
                                rng=np.random.default_rng(seed))
    gx = np.linspace(-lim, lim, ngrid)
    XX, YY = np.meshgrid(gx, gx)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    gs, nu = _gmm_score_and_density(pts, w, means, covs)
    gp = prob.posterior_score_batch(pts)
    dx = 2 * lim / (ngrid - 1)
    fi = float(np.sum(nu * np.sum((gs - gp) ** 2, 1)) * dx * dx)
    out["fi_grid"] = fi if np.isfinite(fi) else float("inf")
    # grid-free MC cross-check: E_{x~fitted GMM}||score_hat - score_pi||^2
    rng = np.random.default_rng(seed + 1)
    comp = rng.choice(k, n_mc, p=w / w.sum())
    z = rng.standard_normal((n_mc, 2))
    xs = np.empty((n_mc, 2))
    for j in range(k):
        idx = comp == j
        if idx.any():
            L = np.linalg.cholesky(covs[j] + 1e-12 * np.eye(2))
            xs[idx] = means[j] + z[idx] @ L.T
    gs2, _ = _gmm_score_and_density(xs, w, means, covs)
    gp2 = prob.posterior_score_batch(xs)
    out["fi_mc"] = float(np.mean(np.sum((gs2 - gp2) ** 2, 1)))
    out["valid"] = bool(out["mass_in_box"] > 0.99 and np.isfinite(out["fi_grid"]))
    return out


# --------------------------------------------------------------------------- #
# Vectorized ZO-APMC / APMC over NPART particles (Eq 12 + Eq 121 schedules)
# --------------------------------------------------------------------------- #
def run_apmc(prob, seed, p, b, b_prime, eps_mode, n=None, niter=None,
             gradient=False, eps0=EPS_STAR, checkpoints=None):
    """eps_mode: 'const' (eps0 all steps), 'decay' (linear eps0 -> 0), 'none'.
    gradient=True -> exact likelihood gradient (APMC positive control).
    Returns dict with final particles, diverged count, checkpoint FI list."""
    n = NPART if n is None else n
    niter = NITER if niter is None else niter
    rng = np.random.default_rng(seed)
    X = rng.uniform(-50.0, 50.0, size=(n, 2))
    Xprev = X.copy()
    G = np.zeros((n, 2))
    diverged = np.zeros(n, bool)
    if not gradient:                                   # VR init: large batch
        U = rng.standard_normal((b, n, 2))
        fx = prob.f_batch(X)
        fp = prob.f_batch((X[None] + MU * U).reshape(-1, 2)).reshape(b, n)
        G = np.mean(((fp - fx[None]) / MU)[:, :, None] * U, axis=0)
    cps = {}
    sqrt2g = np.sqrt(2.0 * GAMMA)
    for k in range(niter):
        sigma_k = max(SIGMA0 * RHO2 ** k, SIGMA_MIN)
        alpha_k = max(ALPHA0 * sigma_k * sigma_k, 1.0)
        if gradient:
            G = prob.grad_f_batch(X)
        else:
            refresh = rng.random(n) < p
            idx_r = np.flatnonzero(refresh)
            idx_c = np.flatnonzero(~refresh)
            if len(idx_r):
                Xr = X[idx_r]
                U = rng.standard_normal((b, len(idx_r), 2))
                fx = prob.f_batch(Xr)
                fp = prob.f_batch((Xr[None] + MU * U).reshape(-1, 2)) \
                    .reshape(b, -1)
                G[idx_r] = np.mean(((fp - fx[None]) / MU)[:, :, None] * U, 0)
            if len(idx_c):
                Xc, Xp = X[idx_c], Xprev[idx_c]
                U = rng.standard_normal((b_prime, len(idx_c), 2))
                fxc, fxp = prob.f_batch(Xc), prob.f_batch(Xp)
                fpc = prob.f_batch((Xc[None] + MU * U).reshape(-1, 2)) \
                    .reshape(b_prime, -1)
                fpp = prob.f_batch((Xp[None] + MU * U).reshape(-1, 2)) \
                    .reshape(b_prime, -1)
                dlt = ((fpc - fxc[None]) - (fpp - fxp[None])) / MU
                G[idx_c] = G[idx_c] + np.mean(dlt[:, :, None] * U, 0)
        Xprev = X.copy()
        s = prob.prior_score_batch(X, sigma_k)
        if eps_mode == "const":
            s = s + eps0 * rng.standard_normal((n, 2))
        elif eps_mode == "decay":
            s = s + eps0 * (1.0 - k / niter) * rng.standard_normal((n, 2))
        X = X - GAMMA * (G - alpha_k * s) + sqrt2g * rng.standard_normal((n, 2))
        bad = ~np.all(np.isfinite(X), 1) | (np.max(np.abs(X), 1) > CLIP)
        if bad.any():
            diverged |= bad
            X[bad] = np.clip(np.nan_to_num(X[bad], nan=0.0,
                                           posinf=CLIP, neginf=-CLIP),
                             -CLIP, CLIP)
        if checkpoints and (k + 1) in checkpoints:
            cps[k + 1] = fi_of_samples(X, prob)
    return {"X": X, "n_diverged": int(diverged.sum()), "checkpoints": cps}


def ou_fi_prediction(prob, eps=EPS_STAR, gamma=GAMMA):
    """OU-theory FI floor for gradient APMC + constant score noise, valid when
    one posterior mode dominates: per eigenmode lam of the posterior precision,
    stationary var v = (2g + g^2 eps^2) / (g lam (2 - g lam)); FI contribution
    lam^2 v - 2 lam + 1/v."""
    lam = np.linalg.eigvalsh(prob.post_icov)
    if np.any(gamma * lam >= 2):
        return float("inf")
    v = (2 * gamma + gamma ** 2 * eps ** 2) / (gamma * lam * (2 - gamma * lam))
    return float(np.sum(lam ** 2 * v - 2 * lam + 1.0 / v))


# --------------------------------------------------------------------------- #
# Campaign driver
# --------------------------------------------------------------------------- #
def _sweep(name, eps_mode, b_prime, prior_mean, seeds, log,
           checkpoints_seed0=False):
    res = {}
    for (p, b) in PAIRS:
        fis, nb, ndiv, fi_mcs = [], 0, 0, []
        cps = None
        for si in range(seeds):
            prob = Fig2Problem(SEED0 + si, prior_mean=prior_mean)
            want_cp = checkpoints_seed0 and si == 0
            r = run_apmc(prob, SEED0 + 1000 * si + int(1000 * p) + b,
                         p, b, b_prime, eps_mode,
                         checkpoints={250, 500, 1000, 1500, 2000}
                         if want_cp else None)
            fi = fi_of_samples(r["X"], prob, seed=si)
            fis.append(fi["fi_grid"])
            fi_mcs.append(fi["fi_mc"])
            nb += int(fi["fi_grid"] < 0.01 and fi["valid"])
            ndiv += r["n_diverged"]
            if want_cp:
                cps = {str(k): round(v["fi_grid"], 6)
                       for k, v in r["checkpoints"].items()}
        arr = np.asarray(fis)
        entry = {"p": p, "b": b, "b_prime": b_prime,
                 "FI_mean": float(np.mean(arr)), "FI_min": float(np.min(arr)),
                 "FI_max": float(np.max(arr)),
                 "FI_mc_mean": float(np.mean(fi_mcs)),
                 "seeds_below_0p01": nb, "n_seeds": seeds,
                 "n_diverged_particles": ndiv}
        if cps:
            entry["FI_curve_seed0"] = cps
        res[f"p={p},b={b}"] = entry
        log(f"    [{name}] (p={p}, b={b:>3}, b'={b_prime}) "
            f"mean FI={entry['FI_mean']:.4f} "
            f"[{entry['FI_min']:.4f},{entry['FI_max']:.4f}] "
            f"mc={entry['FI_mc_mean']:.4f} "
            f"<0.01: {nb}/{seeds} div={ndiv}")
    means = [v["FI_mean"] for v in res.values()]
    return {"configs": res,
            "claim_all_below_0p01": bool(max(means) < 0.01),
            "n_configs_below_0p01": int(sum(m < 0.01 for m in means))}


def run_campaign(log=print, n_seeds_primary=20, n_seeds_variant=5):
    out = {"spec": {
        "claim": "all (p,b) with pb=10 reach FI<0.01 after 2000 iters",
        "pairs": PAIRS, "sigma0": SIGMA0, "alpha0": ALPHA0, "rho2": RHO2,
        "sigma_min": SIGMA_MIN, "gamma": GAMMA, "mu": MU,
        "n_particles": NPART, "n_iters": NITER, "eps_star": EPS_STAR,
        "grid": f"{GRID_N}x{GRID_N} on [-{GRID_LIM},{GRID_LIM}]^2",
        "seed0": SEED0}}

    log("  -- faithful completions (claim satisfied iff ALL 4 configs mean "
        "FI < 0.01) --")
    out["primary_const_bp5"] = _sweep(
        "PRIMARY const-eps b'=5", "const", 5, (2.5, 0.0),
        n_seeds_primary, log, checkpoints_seed0=True)
    out["variant_decay_bp5"] = _sweep(
        "VARIANT decay-eps b'=5", "decay", 5, (2.5, 0.0), n_seeds_variant, log)
    out["variant_const_bp2"] = _sweep(
        "VARIANT const-eps b'=2 (peer)", "const", 2, (2.5, 0.0),
        n_seeds_variant, log)
    out["variant_const_sep8"] = _sweep(
        "VARIANT const-eps prior +-(8,0)", "const", 5, (8.0, 0.0),
        n_seeds_variant, log)
    out["variant_noeps_bp5"] = _sweep(
        "CONTROL eps=0 (ZO isolation)", "none", 5, (2.5, 0.0),
        n_seeds_variant, log)

    log("  -- controls --")
    # POSITIVE: gradient APMC, no score noise -> should hit the estimator floor
    fis = []
    for si in range(n_seeds_primary):
        prob = Fig2Problem(SEED0 + si)
        r = run_apmc(prob, SEED0 + 7000 + si, 1, 1, 1, "none", gradient=True)
        fis.append(fi_of_samples(r["X"], prob, seed=si)["fi_grid"])
    out["control_apmc_clean"] = {
        "FI_mean": float(np.mean(fis)), "FI_min": float(np.min(fis)),
        "FI_max": float(np.max(fis)),
        "n_below_0p01": int(np.sum(np.asarray(fis) < 0.01)),
        "n_seeds": n_seeds_primary}
    log(f"    [POSITIVE] gradient APMC eps=0: mean FI="
        f"{out['control_apmc_clean']['FI_mean']:.4f} "
        f"<0.01: {out['control_apmc_clean']['n_below_0p01']}/{n_seeds_primary}")

    # FLOOR: GMM fit to exact posterior samples (paper's n=1000)
    fis = []
    for si in range(n_seeds_primary):
        prob = Fig2Problem(SEED0 + si)
        s = prob.posterior_sample(np.random.default_rng(SEED0 + 8000 + si),
                                  NPART)
        fis.append(fi_of_samples(s, prob, seed=si)["fi_grid"])
    out["control_estimator_floor"] = {
        "FI_mean": float(np.mean(fis)), "FI_min": float(np.min(fis)),
        "FI_max": float(np.max(fis)),
        "n_below_0p01": int(np.sum(np.asarray(fis) < 0.01)),
        "n_seeds": n_seeds_primary}
    log(f"    [FLOOR] exact posterior samples (n={NPART}): mean FI="
        f"{out['control_estimator_floor']['FI_mean']:.4f} "
        f"<0.01: {out['control_estimator_floor']['n_below_0p01']}"
        f"/{n_seeds_primary}")

    # ANALYTIC: gradient APMC + const noise vs OU prediction
    rows = []
    for si in range(n_seeds_variant):
        prob = Fig2Problem(SEED0 + si)
        if prob.post_w.max() < 0.95:      # OU prediction needs unimodal dom.
            continue
        r = run_apmc(prob, SEED0 + 9000 + si, 1, 1, 1, "const", gradient=True)
        meas = fi_of_samples(r["X"], prob, seed=si)["fi_grid"]
        rows.append({"seed": si, "FI_measured": float(meas),
                     "FI_ou_predicted": ou_fi_prediction(prob)})
        log(f"    [ANALYTIC] seed {si}: grad-APMC + const eps=2.5 measured "
            f"FI={meas:.4f} vs OU-predicted {rows[-1]['FI_ou_predicted']:.4f}")
    out["control_analytic_noise_floor"] = rows

    # NEGATIVE: paper's own unstable case p=0.3, b=10 (pb=3)
    fis = []
    for si in range(n_seeds_variant):
        prob = Fig2Problem(SEED0 + si)
        r = run_apmc(prob, SEED0 + 9500 + si, 0.3, 10, 5, "const")
        fis.append(fi_of_samples(r["X"], prob, seed=si)["fi_grid"])
    out["control_negative_pb3"] = {
        "FI_mean": float(np.mean(fis)),
        "n_below_0p01": int(np.sum(np.asarray(fis) < 0.01)),
        "n_seeds": n_seeds_variant}
    log(f"    [NEGATIVE] p=0.3,b=10 (pb=3): mean FI="
        f"{out['control_negative_pb3']['FI_mean']:.4f}")

    # REFERENCE: FI of a GMM fit to PRIOR samples (non-converged magnitude)
    fis = []
    for si in range(n_seeds_variant):
        prob = Fig2Problem(SEED0 + si)
        s = prob.prior_sample(np.random.default_rng(SEED0 + 9900 + si), NPART)
        fis.append(fi_of_samples(s, prob, seed=si)["fi_grid"])
    out["reference_prior_fit"] = {"FI_mean": float(np.mean(fis)),
                                  "FI_values": [round(f, 3) for f in fis]}
    log(f"    [REFERENCE] FI(prior fit || posterior): mean="
        f"{out['reference_prior_fit']['FI_mean']:.3f} "
        f"(peer reported 4.2-4.7 as 'converged')")

    # X-IMPL: sequential single-chain zo_apmc vs vectorized (p=0.5, b=20)
    prob = Fig2Problem(SEED0)
    n_x = ZO_SEQ_CHAINS
    rngx = np.random.default_rng(SEED0 + 12345)
    seq = np.empty((n_x, 2))
    for i in range(n_x):
        x0 = rngx.uniform(-50, 50, 2)
        s = ZO.zo_apmc(lambda x: float(prob.f_batch(x[None])[0]),
                       lambda x, sig: prob.prior_score_batch(
                           np.asarray(x, float)[None], sig)[0],
                       x0, N=NITER, gamma=GAMMA, mu=MU, p=0.5, b=20, b_prime=5,
                       sigma0=SIGMA0, alpha0=ALPHA0, rho2=RHO2,
                       sigma_min=SIGMA_MIN, seed=SEED0 + i,
                       score_noise_std=EPS_STAR)
        seq[i] = s[-1]
    fi_seq = fi_of_samples(seq, prob, seed=0)["fi_grid"]
    r = run_apmc(prob, SEED0 + 54321, 0.5, 20, 5, "const", n=n_x)
    fi_vec = fi_of_samples(r["X"], prob, seed=0)["fi_grid"]
    out["crosscheck_sequential_vs_vectorized"] = {
        "FI_sequential": float(fi_seq), "FI_vectorized": float(fi_vec),
        "n_particles": n_x}
    log(f"    [X-IMPL] sequential zo_apmc FI={fi_seq:.4f} vs vectorized "
        f"FI={fi_vec:.4f} (n={n_x})")

    # ---- decision ---------------------------------------------------------- #
    faithful = ["primary_const_bp5", "variant_decay_bp5",
                "variant_const_bp2", "variant_const_sep8"]
    controls_ok = (out["control_estimator_floor"]["FI_mean"] < 0.01
                   and out["control_apmc_clean"]["FI_mean"] < 0.01
                   and out["control_negative_pb3"]["FI_mean"] > 0.05)
    claim_holds_somewhere = any(out[k]["claim_all_below_0p01"]
                                for k in faithful)
    all_completions_fail = all(not out[k]["claim_all_below_0p01"]
                               for k in faithful)
    falsified = bool(controls_ok and all_completions_fail)
    out["decision"] = {
        "controls_ok": bool(controls_ok),
        "claim_holds_in_some_faithful_completion": bool(claim_holds_somewhere),
        "all_faithful_completions_fail": bool(all_completions_fail),
        "falsification_established": falsified,
        "evidence_status": "FALSIFIED" if falsified else "BLOCKED",
        "faithful_completions": faithful}
    log(f"  DECISION: controls_ok={controls_ok} | claim holds in some "
        f"faithful completion={claim_holds_somewhere} | "
        f"falsification_established={falsified} -> "
        f"{out['decision']['evidence_status']}")
    return out


if __name__ == "__main__":
    # local smoke test only (the formal run goes through verify_zo.py):
    # shrink every knob so it finishes in well under 5 minutes on 1 core
    import sys
    import time
    NPART, NITER, GRID_N = 200, 400, 200
    ZO_SEQ_CHAINS = 20
    t0 = time.time()
    res = run_campaign(n_seeds_primary=2, n_seeds_variant=1)
    print(f"smoke elapsed {time.time()-t0:.0f}s")
    json.dump(res, open("/tmp/c6_smoke.json", "w"), indent=2, default=str)
    sys.exit(0)
