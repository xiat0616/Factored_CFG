"""
APG (x_0-space) + DCFG engine.

This file is meant to live next to `gaussian_diffusion.py` and re-use its
helper utilities (ModelVarType, _extract_into_tensor, etc).

It implements:
    - Standard diffusion infrastructure (betas, q, p, DDIM, etc.)
    - Classifier-Free Guidance with multiple modes:
          cfg_mode = "none" / "global" / "per_attr" / "interv_group"
      exactly mirroring your DCFG implementation in `GaussianDiffusion`.
    - APG (projected guidance in x_0-space) as in Sadat et al. (2024).

Key idea:
    - For each timestep, we compute:
          eps_uncond  = model(x_t, t, cemb=0, ...)
          eps_guided  = DCFG-guided eps (depending on cfg_mode)
      and convert them to:
          x0_uncond   = x_0(eps_uncond)
          x0_guided   = x_0(eps_guided)  (or x_0(cond) in "global" mode)

    - APG is then applied in x_0-space:
          g = x0_guided - x0_uncond
          x0_apg = APG( pred_cond=x0_guided,
                        pred_uncond=x0_uncond,
                        guidance_scale = cfg_w_apg )

      Finally, we convert x0_apg back to epsilon and continue as usual.
"""

import enum
import math
import copy

import numpy as np
import torch as th

from nn_utils import mean_flat
from losses import normal_kl, discretized_gaussian_log_likelihood
from gaussian_diffusion import ModelVarType, _extract_into_tensor


# ---------------------------------------------------------------------
# APG helpers (direction-based, paper-correct)
# ---------------------------------------------------------------------

class MomentumBuffer:
    """
    Reverse-momentum buffer for APG.
    β < 0 sharpens guidance; β = 0 disables momentum.
    """
    def __init__(self, momentum: float):
        self.momentum = float(momentum)
        self.prev = None

    def apply(self, current: th.Tensor) -> th.Tensor:
        if self.prev is None or self.momentum == 0.0:
            self.prev = current.detach()
            return current
        out = current + self.momentum * (current - self.prev)
        self.prev = current.detach()
        return out


def project(v0: th.Tensor, v1: th.Tensor):
    """
    Decompose v0 into components parallel and orthogonal to v1.
    """
    dtype = v0.dtype
    v0d, v1d = v0.double(), v1.double()
    v1n = th.nn.functional.normalize(v1d, dim=[-1, -2, -3])
    v_par = (v0d * v1n).sum(dim=[-1, -2, -3], keepdim=True) * v1n
    v_orth = v0d - v_par
    return v_par.to(dtype), v_orth.to(dtype)


def apg_direction(
    pred_cond: th.Tensor,
    pred_uncond: th.Tensor,
    momentum_buffer: MomentumBuffer | None = None,
    eta: float = 1.0,
    norm_threshold: float = 0.0,   # r in paper
) -> th.Tensor:
    """
    Compute APG-normalized direction u.

    Returns:
        u = d_orth + eta * d_par, optionally clipped to radius r
    """
    # raw CFG direction
    d = pred_cond - pred_uncond

    # reverse momentum
    if momentum_buffer is not None:
        d = momentum_buffer.apply(d)

    # projection
    d_par, d_orth = project(d, pred_cond)
    u = d_orth + eta * d_par

    # norm clipping (paper Eq. 6)
    if norm_threshold > 0:
        u_norm = u.norm(p=2, dim=[-1, -2, -3], keepdim=True).clamp_min(1e-8)
        scale = th.minimum(th.ones_like(u_norm), norm_threshold / u_norm)
        u = u * scale

    return u

def nullify_semb(model_kwargs):
    if "semb" in model_kwargs and model_kwargs["semb"] is not None:
        model_kwargs["semb"] = th.zeros_like(model_kwargs["semb"])


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Same as in gaussian_diffusion.py, duplicated for convenience.
    """
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


# ---------------------------------------------------------------------
# Main APG + DCFG diffusion class
# ---------------------------------------------------------------------


class GaussianDiffusion_apg:
    """
    Diffusion utilities with DCFG + APG in x_0-space.

    This mirrors your `GaussianDiffusion` API, but:
      - p_mean_variance implements DCFG modes:
            cfg_mode ∈ { "none", "global", "per_attr", "interv_group" }
      - Then applies APG *in x_0-space* between:
            x0_uncond   (baseline)
            x0_guided   (DCFG-guided)
    """

    def __init__(self, diffusion_args):
        self.diffusion_args = diffusion_args
        self.model_var_type = self.diffusion_args.model_var_type
        self.loss_type = self.diffusion_args.loss_type
        self.rescale_timesteps = self.diffusion_args.rescale_timesteps

        # For per_attr / interv_group, you still use cfg_kwargs weights to
        # build the DCFG direction, then cfg_w_apg can act as an outer scale.
        self.parents = getattr(self.diffusion_args, "parents", None)

        # Use float64 for accuracy.
        betas = np.array(self.diffusion_args.diffusion_betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # ------------------------------------------------------------------
    # q, p, etc. same as in GaussianDiffusion
    # ------------------------------------------------------------------

    def q_mean_variance(self, x_start, t):
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ------------------------------------------------------------------
    # p_mean_variance with DCFG modes + APG in x_0
    # ------------------------------------------------------------------

    def p_mean_variance(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        use_cfg=True,
        model_kwargs=None,
        cfg_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if cfg_kwargs is None:
            cfg_kwargs = {}

        B = x.shape[0]
        assert t.shape == (B,)

        cfg_mode = cfg_kwargs.get("cfg_mode", "none")
        parents = self.parents

        if not use_cfg:
            cfg_mode = "none"

        has_cond = ("cemb" in model_kwargs) and (model_kwargs["cemb"] is not None)

        # --------------------------------------------------
        # 1) EPS PREDICTIONS (DCFG STAGE)
        # --------------------------------------------------

        if cfg_mode == "none" or not has_cond:
            pred_eps = model(x, self._scale_timesteps(t), **model_kwargs)
            pred_eps_uncond = pred_eps
            pred_eps_uncond_for_out = pred_eps

        else:
            # unconditional
            uncond_kwargs = copy.deepcopy(model_kwargs)
            if "cemb" in uncond_kwargs and uncond_kwargs["cemb"] is not None:
                uncond_kwargs["cemb"] = th.zeros_like(uncond_kwargs["cemb"])
            if cfg_kwargs.get("null_semb_uncond", False):
                nullify_semb(uncond_kwargs)

            pred_eps_uncond = model(x, self._scale_timesteps(t), **uncond_kwargs)

            # shared APG params
            global_mb = cfg_kwargs.get("apg_momentum_buffer", None)
            global_eta = float(cfg_kwargs.get("apg_eta", 1.0))
            global_r = float(cfg_kwargs.get("apg_norm_threshold", 0.0))

            if cfg_mode == "global":
                pred_eps_cond = model(x, self._scale_timesteps(t), **model_kwargs)

            elif cfg_mode == "interv_group":
                assert parents is not None
                cemb = model_kwargs["cemb"]
                pdim = cemb.shape[1] // len(parents)
                chunks = cemb.split(pdim, dim=1)
                ikeys = cfg_kwargs.get("cfg_intervened_keys", [])

                def build(keys):
                    keep = [th.zeros_like(c) for c in chunks]
                    for i, k in enumerate(parents):
                        if k in keys:
                            keep[i] = chunks[i]
                    kw = copy.deepcopy(model_kwargs)
                    kw["cemb"] = th.cat(keep, dim=1)
                    return model(x, self._scale_timesteps(t), **kw)

                pred_eps_int = build(ikeys)
                pred_eps_inv = build([k for k in parents if k not in ikeys])


            elif cfg_mode == "per_attr":
                assert parents is not None
                cemb = model_kwargs["cemb"]
                pdim = cemb.shape[1] // len(parents)
                chunks = cemb.split(pdim, dim=1)
                eps_attr = {}

                for i, key in enumerate(parents):
                    w = cfg_kwargs.get("cfg_attr_weights", {}).get(key, 0.0)
                    if w == 0.0:
                        continue
                    keep = [th.zeros_like(c) for c in chunks] 
                    keep[i] = chunks[i]
                    kw = copy.deepcopy(model_kwargs)
                    kw["cemb"] = th.cat(keep, dim=1)
                    eps_attr[key] = (w, model(x, self._scale_timesteps(t), **kw))

        # --------------------------------------------------
        # 2) APG IN x0-SPACE (DIRECTION-BASED)
        # --------------------------------------------------

        if cfg_mode == "none" or not has_cond:
            pred_eps = pred_eps_uncond
            pred_eps_uncond_for_out = pred_eps_uncond

        else:
            def eps_to_x0(eps):
                return self._predict_xstart_from_eps(x, t, eps)

            x0_uncond = eps_to_x0(pred_eps_uncond)

            if cfg_mode == "global":
                x0_cond = eps_to_x0(pred_eps_cond)
                w = float(self.diffusion_args.cfg_w)

                u = apg_direction(
                    x0_cond, x0_uncond,
                    momentum_buffer=global_mb,
                    eta=global_eta,
                    norm_threshold=global_r,
                )

                x0 = x0_uncond + w * u

            elif cfg_mode == "interv_group":
                x0_int = eps_to_x0(pred_eps_int)
                x0_inv = eps_to_x0(pred_eps_inv)

                w_int = float(cfg_kwargs.get("cfg_w_intervened", 1.0))
                w_inv = float(cfg_kwargs.get("cfg_w_unintervened", 1.0))

                u_int = apg_direction(
                    x0_int, x0_uncond,
                    cfg_kwargs.get("apg_momentum_buffer_intervened", None),
                    cfg_kwargs.get("apg_eta_intervened", global_eta),
                    cfg_kwargs.get("apg_norm_threshold_intervened", global_r),
                )

                u_inv = apg_direction(
                    x0_inv, x0_uncond,
                    cfg_kwargs.get("apg_momentum_buffer_unintervened", None),
                    cfg_kwargs.get("apg_eta_unintervened", global_eta),
                    cfg_kwargs.get("apg_norm_threshold_unintervened", global_r),
                )

                x0 = x0_uncond + w_int * u_int + w_inv * u_inv
                
            elif cfg_mode == "per_attr":
                x0 = x0_uncond
                for _, (w, eps_k) in eps_attr.items():
                    x0_k = eps_to_x0(eps_k)
                    u = apg_direction(x0_k, x0_uncond, None, global_eta, global_r)
                    x0 = x0 + w * u

            pred_eps = self._predict_eps_from_xstart(x, t, x0)
            pred_eps_uncond_for_out = self._predict_eps_from_xstart(x, t, x0_uncond)

        # --------------------------------------------------
        # 3) POSTERIOR (UNCHANGED)
        # --------------------------------------------------

        model_variance, model_log_variance = {
            ModelVarType.FIXED_LARGE: (
                np.append(self.posterior_variance[1], self.betas[1:]),
                np.log(np.append(self.posterior_variance[1], self.betas[1:])),
            ),
            ModelVarType.FIXED_SMALL: (
                self.posterior_variance,
                self.posterior_log_variance_clipped,
            ),
        }[self.model_var_type]

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart = self._predict_xstart_from_eps(x, t, pred_eps)
        pred_xstart_uncond = self._predict_xstart_from_eps(x, t, pred_eps_uncond_for_out)

        if denoised_fn is not None:
            pred_xstart = denoised_fn(pred_xstart)
        if clip_denoised:
            pred_xstart = pred_xstart.clamp(-1, 1)

        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)

        return dict(
            mean=model_mean,
            variance=model_variance,
            log_variance=model_log_variance,
            pred_xstart=pred_xstart,
            pred_xstart_uncond=pred_xstart_uncond,
            pred_eps=pred_eps,
            pred_eps_uncond=pred_eps_uncond_for_out,
        )


    # ------------------------------------------------------------------
    # Misc helpers from GaussianDiffusion
    # ------------------------------------------------------------------

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    # ------------------------------------------------------------------
    # Sampling / DDIM
    # ------------------------------------------------------------------

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        cfg_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        cfg_kwargs should contain:
            - cfg_mode: "none" / "global" / "per_attr" / "interv_group"
            - any DCFG weights (cfg_attr_weights, cfg_intervened_keys, etc.)
            - APG params: apg_momentum_buffer, apg_eta, apg_norm_threshold
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            use_cfg=True if cfg_kwargs is not None else False,
            model_kwargs=model_kwargs,
            cfg_kwargs=cfg_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        cfg_kwargs=None,
    ):
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            cfg_kwargs=cfg_kwargs,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        cfg_kwargs=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cfg_kwargs=cfg_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        use_cfg=True,
        cfg_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model using DDIM with DCFG+APG.

        - If use_cfg=False: unconditional DDIM.
        - If use_cfg=True: use cfg_mode + APG in x_0-space.
        """
        if cfg_kwargs is None:
            cfg_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}

        out = self.p_mean_variance(
            model=model,
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            use_cfg=use_cfg,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cfg_kwargs=cfg_kwargs,
        )

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        use_cfg=True,
        cfg_kwargs=None,
    ):
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            use_cfg=use_cfg,
            cfg_kwargs=cfg_kwargs,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        use_cfg=True,
        cfg_kwargs=None,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    use_cfg=use_cfg,
                    cfg_kwargs=cfg_kwargs,
                )
                yield out
                img = out["sample"]

    # Reverse DDIM: always unconditional (no APG) to keep inversion stable.
    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        assert eta == 0.0, "Reverse ODE only for deterministic path (eta must be 0)."

        if model_kwargs is None:
            model_kwargs = {}

        out = self.p_mean_variance(
            model=model,
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            use_cfg=False,          # unconditional
            model_kwargs=model_kwargs,
            cfg_kwargs=None,
        )

        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample_loop(
        self,
        model,
        image,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        final = None
        for sample in self.ddim_reverse_sample_loop_progressive(
            model,
            image,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_reverse_sample_loop_progressive(
        self,
        model,
        image,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        if device is None:
            device = next(model.parameters()).device
        shape = image.shape
        indices = list(range(self.num_timesteps))

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_reverse_sample(
                    model,
                    image,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                yield out
                image = out["sample"]

    # ------------------------------------------------------------------
    # VLB / training: same as GaussianDiffusion, no APG here
    # ------------------------------------------------------------------

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model=model,
            x=x_t,
            t=t,
            clip_denoised=clip_denoised,
            use_cfg=False,              # training VLB is unconditional
            model_kwargs=model_kwargs,
            cfg_kwargs=None,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            target = noise   # always predict eps
            if not (model_output.shape == target.shape == x_start.shape):
                raise ValueError(
                    f"Shape mismatch detected:\n"
                    f"  model_output.shape: {model_output.shape}\n"
                    f"  target.shape: {target.shape}\n"
                    f"  x_start.shape: {x_start.shape}\n"
                    f"All three shapes are expected to be identical."
                )
            terms["mse"] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
