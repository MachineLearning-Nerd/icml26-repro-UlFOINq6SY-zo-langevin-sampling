"""Faithful clean-room implementation of the variance-reduced zeroth-order
Langevin / ZO-APMC algorithms from
"Zeroth-Order Non-Log-Concave Sampling with Variance Reduction and Applications
to Inverse Problems" (arXiv 2605.30573, UlFOINq6SY).

Conventions match the paper exactly:
  * One-point ZO estimator (Eq 3):   g_hat(x,u) = (f(x+mu*u) - f(x)) / mu * u
  * VR estimator (Eq 8):  large batch w.p. p ; recursive control-variate
                          g_{k-1} + (1/b') Sum_i ( g_hat(x_k,u_i) - g_hat(x_{k-1},u_i) ) w.p. 1-p
                          (the SAME directions u_i are used at x_k and x_{k-1} -- this
                          correlated sharing is what makes the variance reduction work,
                          and matches official `zo_pmcred.py:gcurr_update`).
  * ZO-LMC iterate (Eq 9):   x <- x - gamma*g_k + sqrt(2*gamma)*xi
  * ZO-APMC iterate (Eq 12): x <- x - gamma*(g_k - alpha_k*S_theta(x,sigma_k)) + sqrt(2*gamma)*xi
  * Schedules (Eq 121/C.1):  sigma_k = max(sigma0*rho2^k, sigma_min); alpha_k = max(alpha0*sigma_k^2, 1)

The diffusion uses sqrt(2*gamma)*N(0,I) so that the stationary law of the
gradient-access Langevin diffusion dx=-grad f dt + sqrt(2)dB is pi ∝ exp(-f).
numpy / CPU only.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import gaussian_kde


# --------------------------------------------------------------------------- #
# Zeroth-order primitives (Eq 3)
# --------------------------------------------------------------------------- #
def zo_estimate(f, x, mu, u):
    """One-point ZO estimate (Eq 3) along a single direction u: (f(x+mu u)-f(x))/mu * u."""
    return (f(x + mu * u) - f(x)) / mu * u


def batched_zo(f, x, mu, U):
    """Large-batch ZO gradient (1/b) Sum_i zo_estimate(x, u_i). U has shape (b, d)."""
    if U.ndim == 1:
        return zo_estimate(f, x, mu, U)
    fx = f(x)
    g = np.zeros_like(x, dtype=float)
    for u in U:
        g += (f(x + mu * u) - fx) / mu * u
    return g / len(U)


# --------------------------------------------------------------------------- #
# Variance-reduced ZO gradient estimator (Eq 8)  -- stateful controller
# --------------------------------------------------------------------------- #
class VRGradientEstimator:
    """Implements Eq (8). Holds g_curr and x_prev between iterations.

    On each call at iterate x_k:
      w.p. p   -> g_curr = batched_zo(x_k, b large directions)        [fresh large batch]
      w.p. 1-p -> g_curr += mean_i [ zo_est(x_k,u_i) - zo_est(x_prev,u_i) ]  (b' directions,
                                                                     SAME u_i at both points)
    Returns g_curr and the number of forward evaluations consumed.
    """

    def __init__(self, f, mu, p, b, b_prime, rng):
        self.f = f
        self.mu = mu
        self.p = float(p)
        self.b = int(b)
        self.b_prime = int(b_prime)
        self.rng = rng
        self.g_curr = None
        self.x_prev = None

    def reset(self, x0, U0=None):
        b = self.b
        U = self.rng.standard_normal((b, len(x0))) if U0 is None else U0
        self.g_curr = batched_zo(self.f, x0, self.mu, U)
        self.x_prev = x0.copy()
        return b  # forward evals: b (each zo_estimate uses 1 f-eval beyond f(x) which is shared)

    def update(self, x_k):
        d = len(x_k)
        if self.g_curr is None:
            return self.reset(x_k)
        if self.rng.random() < self.p:
            U = self.rng.standard_normal((self.b, d))
            self.g_curr = batched_zo(self.f, x_k, self.mu, U)
            fevals = self.b
        else:
            U = self.rng.standard_normal((self.b_prime, d))
            xp = self.x_prev
            mu = self.mu
            f = self.f
            # correlated difference: same u at x_k and x_prev
            diff = np.zeros(d, dtype=float)
            for u in U:
                gk = (f(x_k + mu * u) - f(x_k)) / mu * u
                gp = (f(xp + mu * u) - f(xp)) / mu * u
                diff += gk - gp
            diff /= self.b_prime
            self.g_curr = self.g_curr + diff
            fevals = 2 * self.b_prime
        self.x_prev = x_k.copy()
        return fevals


# --------------------------------------------------------------------------- #
# Samplers
# --------------------------------------------------------------------------- #
def zo_lmc_vr(f, x0, N, gamma, mu, p, b, b_prime, seed=0, track_fevals=False):
    """Variance-reduced ZO-Langevin Monte Carlo (Eq 9). Returns (samples[N+1,d], info)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    est = VRGradientEstimator(f, mu, p, b, b_prime, rng)
    est.reset(x)
    samples = np.empty((N + 1, d))
    samples[0] = x
    total_fe = 0
    fe_curve = np.empty(N)
    sqrt2g = np.sqrt(2.0 * gamma)
    for k in range(N):
        fe = est.update(x)
        x = x - gamma * est.g_curr + sqrt2g * rng.standard_normal(d)
        samples[k + 1] = x
        total_fe += fe
        fe_curve[k] = fe
    info = {"fevals": total_fe, "fe_curve": fe_curve, "mean_fe_per_iter": total_fe / N}
    return samples, info


def zo_lmc_naive(f, x0, N, gamma, mu, b, seed=0):
    """Naive (non-VR) ZO-LMC: fresh large batch every iteration (the p=1 special case)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    samples = np.empty((N + 1, d))
    samples[0] = x
    sqrt2g = np.sqrt(2.0 * gamma)
    for k in range(N):
        U = rng.standard_normal((b, d))
        g = batched_zo(f, x, mu, U)
        x = x - gamma * g + sqrt2g * rng.standard_normal(d)
        samples[k + 1] = x
    return samples


def zo_apmc(likelihood_potential, prior_score_fn, x0, N, gamma, mu, p, b, b_prime,
            sigma0, alpha0, rho2, sigma_min, seed=0, score_noise_std=0.0):
    """ZO-APMC posterior sampler (Eq 12) with annealing (Eq 121).

    likelihood_potential: f such that likelihood ∝ exp(-f). Only f is queried (black-box).
    prior_score_fn(x, sigma) -> ∇log p_sigma(x)  (smoothed prior score; in the real method
                            this is the SGM S_theta(x,sigma_k)). Matches official drift.
    score_noise_std: std of additive Gaussian noise on the prior score, modelling SGM error
                     (the paper's epsilon_{k*} = 2.5 for the d=2 numerical validation).
    Returns samples[N+1,d].
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    d = len(x)
    est = VRGradientEstimator(likelihood_potential, mu, p, b, b_prime, rng)
    est.reset(x)
    samples = np.empty((N + 1, d))
    samples[0] = x
    sqrt2g = np.sqrt(2.0 * gamma)
    for k in range(N):
        sigma_k = max(sigma0 * (rho2 ** k), sigma_min)
        alpha_k = max(alpha0 * (sigma_k ** 2), 1.0)
        est.update(x)
        s = prior_score_fn(x, sigma_k)
        if score_noise_std > 0:
            s = s + score_noise_std * rng.standard_normal(d)
        # Eq 12: drift = -gamma*(g_k - alpha_k * S_theta). S_theta ≈ ∇log p_sigma (toward prior mode).
        x = x - gamma * (est.g_curr - alpha_k * s) + sqrt2g * rng.standard_normal(d)
        samples[k + 1] = x
    return samples


# --------------------------------------------------------------------------- #
# Relative Fisher information  FI(nu || pi) = E_nu[ ||grad log nu - grad log pi||^2 ]
# Computed on a grid (paper Appendix C.1: 1000x1000 cells); density of nu estimated
# from samples via a Gaussian KDE (robust equivalent of the paper's GMM fit).
# --------------------------------------------------------------------------- #
def relative_fisher_info(samples, log_posterior_grad, xlim=(-50, 50), ylim=(-50, 50),
                         ngrid=400, burn=0):
    """Relative FI of the sample distribution to the target posterior.

    log_posterior_grad(x) -> ∇log π(x|y)  (analytical, callable).
    samples: (n,d) with d==2. burn: leading samples to drop.
    """
    s = np.asarray(samples)[burn:]
    s = s[np.all(np.isfinite(s), axis=1)]
    if len(s) < 10:
        return float("inf")
    kde = gaussian_kde(s.T)
    gx = np.linspace(*xlim, ngrid)
    gy = np.linspace(*ylim, ngrid)
    XX, YY = np.meshgrid(gx, gy)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    nu = kde(pts.T).reshape(XX.shape)
    nu = np.maximum(nu, 1e-300)
    dx = (xlim[1] - xlim[0]) / (ngrid - 1)
    dy = (ylim[1] - ylim[0]) / (ngrid - 1)
    log_nu = np.log(nu)
    g_nu_y, g_nu_x = np.gradient(log_nu, dy, dx)
    # analytical target score on the grid
    gp = np.array([log_posterior_grad(pt) for pt in pts]).reshape(XX.shape + (2,))
    diff_x = g_nu_x - gp[..., 0]
    diff_y = g_nu_y - gp[..., 1]
    fi_density = nu * (diff_x ** 2 + diff_y ** 2)
    return float(np.sum(fi_density) * dx * dy)


def _gmm_em(samples, k=2, n_iter=100, reg=1e-3, rng=None):
    """Lightweight k-component GMM fit (full covariance) via EM. Returns (w, means, covs)."""
    rng = np.random.default_rng(0) if rng is None else rng
    X = np.asarray(samples, dtype=float)
    n, d = X.shape
    # init: random partition
    idx = rng.integers(0, k, n)
    means = np.array([X[idx == j].mean(0) if np.sum(idx == j) > 0 else X[j % n] for j in range(k)])
    covs = np.array([np.cov(X.T) + reg * np.eye(d) for _ in range(k)])
    w = np.full(k, 1.0 / k)
    for _ in range(n_iter):
        # E-step
        resp = np.zeros((n, k))
        for j in range(k):
            diff = X - means[j]
            _, ld = np.linalg.slogdet(covs[j])
            maha = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(covs[j]), diff)
            resp[:, j] = np.log(w[j] + 1e-300) - 0.5 * (d * np.log(2 * np.pi) + ld + maha)
        resp -= resp.max(1, keepdims=True)
        resp = np.exp(resp)
        resp /= resp.sum(1, keepdims=True)
        # M-step
        Nk = resp.sum(0) + 1e-10
        w = Nk / n
        means = (resp.T @ X) / Nk[:, None]
        for j in range(k):
            diff = X - means[j]
            covs[j] = (resp[:, j, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(0) / Nk[j]
            covs[j] += reg * np.eye(d)
    return w, means, covs


def relative_fisher_info_gmm(samples, target_score_batch, k=2, xlim=(-50, 50),
                             ylim=(-50, 50), ngrid=300, burn=0, reg=1e-3, seed=0):
    """Relative FI using a smooth k-component GMM fit to the samples for the density of nu
    (paper Appendix C.1: 'use a GMM to fit a distribution to the samples'). Analytical
    GMM score => no numerical-differentiation noise. Fully vectorized over the grid.

    target_score_batch(pts) -> (N,2) array of ∇log π(x|y) for pts (N,2).
    """
    s = np.asarray(samples)[burn:]
    s = s[np.all(np.isfinite(s), axis=1)]
    if len(s) < 2 * k:
        return float("inf")
    w, means, covs = _gmm_em(s, k=k, reg=reg, rng=np.random.default_rng(seed))
    icovs = np.linalg.inv(covs)                       # (k,2,2)

    gx = np.linspace(*xlim, ngrid)
    gy = np.linspace(*ylim, ngrid)
    XX, YY = np.meshgrid(gx, gy)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (N,2)
    N = len(pts)

    # GMM density and score (vectorized over components)
    const = 2 * np.log(2 * np.pi)
    logterms = np.empty((N, k))
    for j in range(k):
        diff = pts - means[j]
        _, ld = np.linalg.slogdet(covs[j])
        maha = np.einsum("ij,jk,ik->i", diff, icovs[j], diff)
        logterms[:, j] = np.log(w[j] + 1e-300) - 0.5 * (const + ld + maha)
    lt = logterms - logterms.max(1, keepdims=True)
    r = np.exp(lt)
    r /= r.sum(1, keepdims=True)                       # responsibilities (N,k)
    # proper density: sum_j w_j * N(pt;mean_j,cov_j)
    nu = np.zeros(N)
    for j in range(k):
        diff = pts - means[j]
        _, ld = np.linalg.slogdet(covs[j])
        maha = np.einsum("ij,jk,ik->i", diff, icovs[j], diff)
        nu += w[j] * np.exp(-0.5 * (const + ld + maha))
    nu = np.maximum(nu, 1e-300)
    # GMM score = sum_j resp_j * (-icov_j (pt - mean_j))
    gs = np.zeros((N, 2))
    for j in range(k):
        diff = pts - means[j]
        gs += r[:, j, None] * (-(diff @ icovs[j].T))
    gp = np.asarray(target_score_batch(pts))           # (N,2) target score
    diff_sq = np.sum((gs - gp) ** 2, axis=1)
    dx = (xlim[1] - xlim[0]) / (ngrid - 1)
    dy = (ylim[1] - ylim[0]) / (ngrid - 1)
    return float(np.sum(nu * diff_sq) * dx * dy)


# --------------------------------------------------------------------------- #
# Bimodal Gaussian-mixture prior and a 2D synthetic linear inverse problem
# (paper §4.1 numerical validation setup).
# --------------------------------------------------------------------------- #
class BimodalGMM:
    """Symmetric 2-component GMM prior: 0.5 N(+m, S) + 0.5 N(-m, S)."""

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.icov = np.linalg.inv(self.cov)
        self._gmm_mean()

    def _gmm_mean(self):
        self.d = len(self.mean)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        return _logsumexp_pair(self._logN(x, +self.mean), self._logN(x, -self.mean))

    def score(self, x):
        """∇log p(x) analytically (unsmoothed)."""
        x = np.asarray(x, dtype=float)
        lp1 = self._logN(x, +self.mean)
        lp2 = self._logN(x, -self.mean)
        w1, w2 = _softmax2(lp1, lp2)
        s1 = -self.icov @ (x - self.mean)
        s2 = -self.icov @ (x + self.mean)
        return w1 * s1 + w2 * s2

    def smoothed_score(self, x, sigma):
        """∇log p_sigma(x) for the smoothed prior p_sigma = p * N(0,sigma^2 I),
        i.e. a GMM with covariance cov + sigma^2 I. This is what S_theta(x,sigma_k)
        in Eq (12) estimates. Matches official drift: score_fn(x, sigma)."""
        x = np.asarray(x, dtype=float)
        cov_s = self.cov + (sigma ** 2) * np.eye(self.d)
        icov_s = np.linalg.inv(cov_s)
        lp1 = -0.5 * (x - self.mean) @ icov_s @ (x - self.mean)
        lp2 = -0.5 * (x + self.mean) @ icov_s @ (x + self.mean)
        w1, w2 = _softmax2(lp1, lp2)
        s1 = -icov_s @ (x - self.mean)
        s2 = -icov_s @ (x + self.mean)
        return w1 * s1 + w2 * s2

    def score_batch(self, pts):
        """Vectorized unsmoothed score for (N,d) array -> (N,d)."""
        pts = np.asarray(pts, dtype=float)
        d1 = pts - self.mean
        d2 = pts + self.mean
        lp1 = -0.5 * np.einsum("ij,jk,ik->i", d1, self.icov, d1)
        lp2 = -0.5 * np.einsum("ij,jk,ik->i", d2, self.icov, d2)
        m = np.maximum(lp1, lp2)
        e1, e2 = np.exp(lp1 - m), np.exp(lp2 - m)
        Z = e1 + e2
        w1, w2 = e1 / Z, e2 / Z
        s1 = -(d1 @ self.icov.T)
        s2 = -(d2 @ self.icov.T)
        return w1[:, None] * s1 + w2[:, None] * s2

    def _logN(self, x, m):
        d = self.d
        diff = x - m
        c = -0.5 * (d * np.log(2 * np.pi) + np.log(np.linalg.det(self.cov)))
        return c - 0.5 * diff @ self.icov @ diff


def _logsumexp_pair(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m) + 1e-300)


def _softmax2(a, b):
    m = np.maximum(a, b)
    ea, eb = np.exp(a - m), np.exp(b - m)
    Z = ea + eb
    return ea / Z, eb / Z

class SyntheticInverseProblem:
    """2D linear inverse problem y = A x + xi, xi~N(0,I); bimodal-GMM prior.

    Provides: likelihood potential f(x)=0.5||Ax-y||^2, analytical posterior score
    ∇log π(x|y) = -Aᵀ(Ax-y) + ∇log p(x), and a sampler for the prior (for ground truth).
    """

    def __init__(self, gmm_prior, A, y, seed=0):
        self.prior = gmm_prior
        self.A = np.asarray(A, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.AtA = self.A.T @ self.A
        self.Aty = self.A.T @ self.y

    def f(self, x):
        r = self.A @ x - self.y
        return 0.5 * float(r @ r)

    def posterior_score(self, x):
        return -(self.AtA @ x - self.Aty) + self.prior.score(x)

    def posterior_score_batch(self, pts):
        """Vectorized posterior score for (N,2) array -> (N,2)."""
        pts = np.asarray(pts, dtype=float)
        lik = -(pts @ self.AtA.T - self.Aty)         # (N,2)
        prior = self.prior.score_batch(pts)
        return lik + prior

    def prior_sample(self, rng, n=1):
        comp = rng.integers(0, 2, n)
        z = rng.standard_normal((n, self.prior.d))
        L = np.linalg.cholesky(self.prior.cov)
        out = np.empty((n, self.prior.d))
        for i in range(n):
            m = self.prior.mean if comp[i] == 0 else -self.prior.mean
            out[i] = m + L @ z[i]
        return out if n > 1 else out[0]


def make_synthetic_problem(seed=0, prior_mean=(2.5, 0.0), prior_cov=None,
                           A_shape=(2, 2)):
    """Construct one random instance of the §4.1 synthetic inverse problem."""
    rng = np.random.default_rng(seed)
    cov = np.eye(2) if prior_cov is None else np.asarray(prior_cov)
    prior = BimodalGMM(prior_mean, cov)
    A = rng.standard_normal(A_shape)
    x_true = prior.prior_sample(rng, n=1) if hasattr(prior, "prior_sample") else \
        prior.mean  # fallback
    # sample a true signal from the prior, then observe
    x_true = np.atleast_1d(prior.mean if rng.random() < 0.5 else -np.asarray(prior_mean))
    xi = rng.standard_normal(A_shape[0])
    y = A @ x_true + xi
    prob = SyntheticInverseProblem(prior, A, y, seed=seed)
    return prob, x_true
