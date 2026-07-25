"""Real score-based generative model (SGM) prior + ZO-APMC image inverse-problem
reproduction, faithful to Sahin et al. (2026, UlFOINq6SY) §3.2/4.

Trains a U-Net score network s_theta(x, sigma) by denoising score matching (DSM)
on real images (MNIST), then runs ZO-APMC (Eq 12) on a black-box image inverse
problem (inpainting / denoising), measuring PSNR. The score net is queried ONCE
per iteration (the ZO evaluations act only on the black-box forward likelihood),
so small-image reconstruction is feasible on CPU. torch CPU only.
"""
from __future__ import annotations
import os, gzip, struct, urllib.request, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "outputs", "mnist")
MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist"


# --------------------------------------------------------------------------- #
# MNIST loader (raw idx files; no torchvision dependency)
# --------------------------------------------------------------------------- #
def _download(name):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, name)
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{MNIST_BASE}/{name}", path + ".tmp")
        os.rename(path + ".tmp", path)
    return path


def _load_idx_gz(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic, n = struct.unpack(">II", data[:8])
    if magic == 2051:  # images
        rows, cols = struct.unpack(">II", data[8:16])
        arr = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)
    else:  # labels
        arr = np.frombuffer(data[8:], dtype=np.uint8)
    return arr


def load_mnist(n_train=6000, seed=0, size=32):
    _download("train-images-idx3-ubyte.gz")
    imgs = _load_idx_gz(os.path.join(CACHE, "train-images-idx3-ubyte.gz"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(imgs))[:n_train]
    x = imgs[idx].astype(np.float32) / 127.5 - 1.0   # normalize to [-1,1]
    xt = torch.from_numpy(x).unsqueeze(1)             # (n,1,28,28)
    if size != 28:
        xt = F.interpolate(xt, size=(size, size), mode="bilinear", align_corners=False)
    return xt


# --------------------------------------------------------------------------- #
# Score U-Net (small, sigma-conditioned)
# --------------------------------------------------------------------------- #
class GaussianFourierProjection(nn.Module):
    def __init__(self, dim=32, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim) * scale)

    def forward(self, sigma):
        t = torch.arctan2(sigma, torch.ones_like(sigma))  # in (0, pi/2]
        proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


class ScoreUNet(nn.Module):
    """Compact U-Net for 28x28 grayscale, conditioned on noise level sigma."""

    def __init__(self, ch=32):
        super().__init__()
        self.sigmaproj = nn.Sequential(GaussianFourierProjection(32), nn.Linear(64, ch))
        self.enc1 = nn.Conv2d(1, ch, 3, padding=1)
        self.enc2 = nn.Conv2d(ch, 2 * ch, 3, padding=1)
        self.enc3 = nn.Conv2d(2 * ch, 4 * ch, 3, padding=1)
        self.bottleneck = nn.Conv2d(4 * ch, 4 * ch, 3, padding=1)
        self.up1 = nn.Conv2d(4 * ch + 4 * ch, 2 * ch, 3, padding=1)
        self.up2 = nn.Conv2d(2 * ch + 2 * ch, ch, 3, padding=1)
        self.up3 = nn.Conv2d(ch + ch, ch, 3, padding=1)
        self.out = nn.Conv2d(ch, 1, 3, padding=1)
        self.act = lambda x: F.elu(x)

    def forward(self, x, sigma):
        s = self.sigmaproj(sigma.view(-1))[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        e1 = self.act(self.enc1(x) + s)                      # 32x32, ch
        e2 = self.act(self.enc2(F.avg_pool2d(e1, 2)))        # 16x16, 2ch
        e3 = self.act(self.enc3(F.avg_pool2d(e2, 2)))        # 8x8,  4ch
        b = self.act(self.bottleneck(F.avg_pool2d(e3, 2)))   # 4x4,  4ch
        u = F.interpolate(b, scale_factor=2, recompute_scale_factor=False)        # 8x8
        u = self.act(self.up1(torch.cat([u, e3], 1)))        # 8x8,  2ch
        u = F.interpolate(u, scale_factor=2, recompute_scale_factor=False)        # 16x16
        u = self.act(self.up2(torch.cat([u, e2], 1)))        # 16x16, ch
        u = F.interpolate(u, scale_factor=2, recompute_scale_factor=False)        # 32x32
        u = self.act(self.up3(torch.cat([u, e1], 1)))        # 32x32, ch
        return self.out(u)


def train_scorenet(net, X, sigmas, epochs=8, batch=128, lr=2e-4, seed=0, verbose=True):
    """Denoising score matching: E over x,sigma [ || s_theta(x+sigma*eps, sigma) + eps/sigma ||^2 ]."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    sigmas_t = torch.tensor(sigmas, dtype=torch.float32)
    n = len(X)
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x = X[idx]
            b = x.shape[0]
            sig = sigmas_t[torch.randint(0, len(sigmas_t), (b,))]
            eps = torch.randn_like(x) * sig.view(b, 1, 1, 1)
            target = -eps / (sig.view(b, 1, 1, 1) + 1e-8)
            pred = net(x + eps, sig)
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * b
        if verbose:
            print(f"  DSM epoch {ep+1}/{epochs}: loss={tot/n:.4f}", flush=True)
    return net


# --------------------------------------------------------------------------- #
# Black-box forward models (likelihood potential f; only f is queried via ZO)
# --------------------------------------------------------------------------- #
def make_inpainting(img, keep_frac=0.3, noise_std=0.1, seed=0):
    """Inpainting: observe a random subset of pixels + noise. Returns f(x), y, mask."""
    rng = np.random.default_rng(seed)
    img = img.squeeze().numpy() if torch.is_tensor(img) else np.asarray(img).squeeze()
    mask = (rng.random(img.shape) < keep_frac).astype(np.float32)
    y = img * mask + noise_std * rng.standard_normal(img.shape)
    H, W = img.shape

    def f(x):
        x = x.reshape(H, W)
        return 0.5 * float(np.sum((mask * (x - y)) ** 2)) / (noise_std ** 2 + 1e-8)

    return f, y, mask


# --------------------------------------------------------------------------- #
# ZO-APMC image reconstruction (Eq 12), torch score net + numpy ZO likelihood
# --------------------------------------------------------------------------- #
def zo_apmc_image(net, f_likelihood, x0, N, gamma, mu, p, b, b_prime,
                  sigma0, alpha0, rho2, sigma_min, device="cpu", seed=0):
    """ZO-APMC for images. net: trained ScoreUNet (called once per iter).
    f_likelihood: black-box numpy scalar f(x) on flattened image.
    Returns samples list of (D,) arrays + feval count."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    D = x.size
    sqrt2g = np.sqrt(2.0 * gamma)
    net.eval()
    samples = [x.copy()]
    fevals = 0
    g_curr = None; x_prev = None

    def zo_batch(xx, U):
        fx = f_likelihood(xx); g = np.zeros(D)
        for u in U:
            g += (f_likelihood(xx + mu * u) - fx) / mu * u
        return g / len(U)

    # initial large batch
    U0 = rng.standard_normal((b, D)); g_curr = zo_batch(x, U0); fevals += b; x_prev = x.copy()
    for k in range(N):
        sigma_k = max(sigma0 * (rho2 ** k), sigma_min)
        alpha_k = max(alpha0 * (sigma_k ** 2), 1.0)
        # VR estimator (Eq 8)
        if rng.random() < p:
            U = rng.standard_normal((b, D)); g_curr = zo_batch(x, U); fevals += b
        else:
            U = rng.standard_normal((b_prime, D)); diff = np.zeros(D)
            for u in U:
                diff += ((f_likelihood(x + mu * u) - f_likelihood(x)) -
                         (f_likelihood(x_prev + mu * u) - f_likelihood(x_prev))) / mu * u
            diff /= b_prime; g_curr = g_curr + diff; fevals += 2 * b_prime
        x_prev = x.copy()
        # prior score from SGM (once per iter)
        with torch.no_grad():
            xt = torch.from_numpy(x.reshape(1, 1, int(np.sqrt(D)), int(np.sqrt(D))).astype(np.float32))
            s = net(xt, torch.tensor([sigma_k])).flatten().numpy()
        x = x - gamma * (g_curr - alpha_k * s) + sqrt2g * rng.standard_normal(D)
        x = np.clip(x, -1.0, 1.0)
        samples.append(x.copy())
    return np.array(samples), fevals


def psnr(x_hat, x_gt, max_val=2.0):
    """Paper PSNR (Appendix C.3): 10 log10( max^2 / MSE )."""
    mse = float(np.mean((x_hat - x_gt) ** 2))
    return 10 * np.log10(max_val ** 2 / max(mse, 1e-12))
