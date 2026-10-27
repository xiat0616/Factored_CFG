"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch 

def rbf_kernel(x, sigma=None):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    dist = x_norm + x_norm.t() - 2 * x @ x.t()
    if sigma is None:
        sigma = torch.median(dist.detach())
    gamma = 1.0 / (2 * sigma)
    return torch.exp(-gamma * dist)

def compute_hsic_norm_loss(x, y, eps=1e-5):
    """
    Normalized HSIC (a.k.a. linear CKA with RBF kernels).
    Inputs:
        x: Tensor of shape (B, D1)
        y: Tensor of shape (B, D2)
        eps: small number for numerical stability
    Returns:
        Normalized HSIC value (scalar)
    """
    # Infer device from inputs
    device = x.device

    # Center the inputs
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)

    # Compute RBF kernels
    K = rbf_kernel(x)
    L = rbf_kernel(y)

    # Centered kernels using centering matrix
    B = K.size(0)
    H = torch.eye(B, device=device) - (1.0 / B)
    Kc = H @ K @ H
    Lc = H @ L @ H

    # Compute normalized HSIC
    hsic_xy = (Kc * Lc).sum()
    hsic_xx = (Kc * Kc).sum()
    hsic_yy = (Lc * Lc).sum()

    denom = torch.sqrt(hsic_xx * hsic_yy) + eps
    return hsic_xy / denom
def compute_hsic_loss(x, y, sigma=1.0):
    """
    Unbiased estimator of the squared Hilbert-Schmidt Independence Criterion (HSIC)
    between two sets of embeddings x and y.
    """
    def rbf_kernel(a, sigma):
        norm = (a.unsqueeze(1) - a.unsqueeze(0)).pow(2).sum(2)
        return torch.exp(-norm / (2 * sigma ** 2))

    _x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    _y = (y - y.mean(0)) / (y.std(0) + 1e-6)

    K = rbf_kernel(_x, sigma)
    L = rbf_kernel(_y, sigma)
    n = x.size(0)
    H = torch.eye(n, device=x.device) - (1. / n)
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = (Kc * Lc).sum() / ((n - 1) ** 2)
    return hsic

def compute_cka(x, y, eps=1e-5):
    """
    Computes linear CKA (Centered Kernel Alignment) between x and y.
    Inputs:
        x: Tensor of shape (B, D1)
        y: Tensor of shape (B, D2)
    Returns:
        Scalar value between 0 and 1 indicating similarity.
    """
    # Center x and y
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    # Compute Gram matrices
    K = x @ x.t()
    L = y @ y.t()

    # Center Gram matrices
    B = x.size(0)
    H = torch.eye(B, device=x.device) - (1.0 / B)
    Kc = H @ K @ H
    Lc = H @ L @ H

    # Compute CKA
    hsic = (Kc * Lc).sum()
    norm_k = (Kc * Kc).sum()
    norm_l = (Lc * Lc).sum()
    cka = hsic / (torch.sqrt(norm_k * norm_l) + eps)
    return cka


def kl_divergence_latent(mu, logvar, free_bits=0.0):
    """
    Compute KL divergence between N(mu, sigma^2) and standard normal N(0,1),
    apply free bits per latent dimension.

    Args:
        mu:      [B, D] mean of approximate posterior
        logvar:  [B, D] log variance of approximate posterior
        free_bits: minimum KL contribution per latent dim (float or tensor)

    Returns:
        Scalar: mean KL loss across batch and dimensions with free bits
    """
    # Standard KL per dim
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, D]

    if free_bits > 0.0:
        free_bits = torch.tensor(free_bits, dtype=kl_per_dim.dtype, device=kl_per_dim.device)
        kl_per_dim = torch.maximum(kl_per_dim, free_bits)

    kl_loss = kl_per_dim.mean()  # Average over [B, D]
    return kl_loss


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs