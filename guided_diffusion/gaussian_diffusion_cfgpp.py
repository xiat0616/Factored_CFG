"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
import copy
from nn_utils import mean_flat
from losses import normal_kl, discretized_gaussian_log_likelihood
from gaussian_diffusion import ModelVarType, _extract_into_tensor



def nullify_semb(model_kwargs):
    if "semb" in model_kwargs and model_kwargs["semb"] is not None:
        model_kwargs["semb"] = th.zeros_like(model_kwargs["semb"])

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
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
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion_cfgpp:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        diffusion_args,
    ):  
        self.diffusion_args = diffusion_args
        self.model_var_type = self.diffusion_args.model_var_type
        self.loss_type = self.diffusion_args.loss_type
        self.rescale_timesteps = self.diffusion_args.rescale_timesteps
        self.cfg_w = self.diffusion_args.cfg_w # Weight for CFG 
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

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
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

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

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
        """
        Compute the mean and variance of p(x_{t-1} | x_t) and estimate x_0.

        This version is **CFG++-only**:
          - Requires use_cfg=True and cfg_mode != "none".
          - Always computes both conditional and unconditional predictions:
              * pred_eps        : guided eps (CFG-style, per cfg_mode)
              * pred_eps_uncond : unconditional eps (null cemb)
              * pred_xstart     : guided x_0 from pred_eps
              * pred_xstart_uncond: unconditional x_0 from pred_eps_uncond

        Returns:
            dict with keys:
                - 'mean': Mean of p(x_{t-1} | x_t)
                - 'variance': Variance of p(x_{t-1} | x_t)
                - 'log_variance': Log variance
                - 'pred_xstart': guided predicted x_0
                - 'pred_xstart_uncond': unconditional predicted x_0
                - 'pred_eps': guided eps
                - 'pred_eps_uncond': unconditional eps
        """
        if model_kwargs is None:
            model_kwargs = {}
        if cfg_kwargs is None:
            cfg_kwargs = {}

        cfg_mode = cfg_kwargs.get("cfg_mode", "none")
        parents = self.diffusion_args.parents

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # -------- enforce CFG++-only usage --------
        # if (not use_cfg) or (cfg_mode == "none"):
        #     raise ValueError(
        #         "CFG++ diffusion: p_mean_variance expects use_cfg=True and "
        #         "cfg_mode in {'global', 'per_attr', 'interv_group'}."
        #         f"but now got use_cfg={use_cfg}, cfg_mode={cfg_mode}."
        #     )

        # ---------------------------------------------------------
        # 1) Compute eps predictions (guided + unconditional)
        # ---------------------------------------------------------
        if (not use_cfg) or (cfg_mode == "none"):
            pred_eps = model(x, self._scale_timesteps(t), **model_kwargs)
            pred_eps_uncond = pred_eps  # not returned, but useful if you keep it
        elif cfg_mode == "global":
            # Standard global CFG: use cemb vs zeroed cemb
            cond_model_kwargs = copy.deepcopy(model_kwargs)
            pred_eps_cond = model(x, self._scale_timesteps(t), **cond_model_kwargs)

            uncond_model_kwargs = copy.deepcopy(model_kwargs)
            if "cemb" in uncond_model_kwargs and uncond_model_kwargs["cemb"] is not None:
                uncond_model_kwargs["cemb"] = th.zeros_like(uncond_model_kwargs["cemb"])
            pred_eps_uncond = model(x, self._scale_timesteps(t), **uncond_model_kwargs)

            pred_eps = pred_eps_uncond + self.cfg_w * (pred_eps_cond - pred_eps_uncond)

        elif cfg_mode == "interv_group":
            # Intervened / unintervened group CFG: two group deltas from same uncond
            cemb = model_kwargs["cemb"]
            parent_dim = cemb.shape[1] // len(parents)
            chunks = cemb.split(parent_dim, dim=1)

            intervened_keys = cfg_kwargs.get("cfg_intervened_keys", [])
            w_intervened = cfg_kwargs.get("cfg_w_intervened", 1.0)
            w_unintervened = cfg_kwargs.get("cfg_w_unintervened", 1.0)

            # Unconditional baseline
            uncond_model_kwargs = copy.deepcopy(model_kwargs)
            if "cemb" in uncond_model_kwargs and uncond_model_kwargs["cemb"] is not None:
                uncond_model_kwargs["cemb"] = th.zeros_like(uncond_model_kwargs["cemb"])
            pred_eps_uncond = model(x, self._scale_timesteps(t), **uncond_model_kwargs)

            # Intervened
            keep_chunks = [th.zeros_like(chunk) for chunk in chunks]
            for i, key in enumerate(parents):
                if key in intervened_keys:
                    keep_chunks[i] = chunks[i]
            intervened_cemb = th.cat(keep_chunks, dim=1)
            model_kwargs_intervened = copy.deepcopy(model_kwargs)
            model_kwargs_intervened["cemb"] = intervened_cemb
            pred_eps_intervened = model(x, self._scale_timesteps(t), **model_kwargs_intervened)

            # Unintervened
            keep_chunks = [th.zeros_like(chunk) for chunk in chunks]
            for i, key in enumerate(parents):
                if key not in intervened_keys:
                    keep_chunks[i] = chunks[i]
            unintervened_cemb = th.cat(keep_chunks, dim=1)
            model_kwargs_unintervened = copy.deepcopy(model_kwargs)
            model_kwargs_unintervened["cemb"] = unintervened_cemb
            pred_eps_unintervened = model(x, self._scale_timesteps(t), **model_kwargs_unintervened)

            delta_intervened = pred_eps_intervened - pred_eps_uncond
            delta_unintervened = pred_eps_unintervened - pred_eps_uncond
            pred_eps = pred_eps_uncond + w_intervened * delta_intervened + w_unintervened * delta_unintervened

        elif cfg_mode == "per_attr":
            # Per-attribute CFG: build guided eps from per-parent chunks
            cfg_attr_weights = cfg_kwargs.get("cfg_attr_weights", {})
            cemb = model_kwargs["cemb"]
            parent_dim = cemb.shape[1] // len(parents)
            chunks = cemb.split(parent_dim, dim=1)

            # Unconditional baseline
            uncond_model_kwargs = copy.deepcopy(model_kwargs)
            uncond_model_kwargs["cemb"] = th.zeros_like(model_kwargs["cemb"])
            pred_eps_uncond = model(x, self._scale_timesteps(t), **uncond_model_kwargs)

            pred_eps = pred_eps_uncond
            for i, key in enumerate(parents):
                if key not in cfg_attr_weights:
                    continue
                w = cfg_attr_weights[key]
                if w == 0.0:
                    continue

                keep_chunks = [th.zeros_like(chunk) for chunk in chunks]
                keep_chunks[i] = chunks[i]
                partial_cemb = th.cat(keep_chunks, dim=1)

                partial_model_kwargs = copy.deepcopy(model_kwargs)
                partial_model_kwargs["cemb"] = partial_cemb

                pred_eps_partial = model(x, self._scale_timesteps(t), **partial_model_kwargs)
                delta = pred_eps_partial - pred_eps_uncond
                pred_eps = pred_eps + w * delta
        # ---------------------------------------------------------
        # 2) Variance selection (unchanged)
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 3) x_0 estimates from eps (guided + uncond), with optional clipping
        # ---------------------------------------------------------
        def process_xstart(x0):
            if denoised_fn is not None:
                x0 = denoised_fn(x0)
            if clip_denoised:
                return x0.clamp(-1, 1)
            return x0

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=pred_eps)
        )
        pred_xstart_uncond = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=pred_eps_uncond)
        )

        # Posterior mean uses guided x_0 (standard practice)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape
            == model_log_variance.shape
            == pred_xstart.shape
            == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "pred_xstart_uncond": pred_xstart_uncond,
            "pred_eps": pred_eps,
            "pred_eps_uncond": pred_eps_uncond,
        }


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
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

        This class implements CFG++-only sampling, so p_sample internally
        calls p_mean_variance with CFG++-style guidance enabled.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param cfg_kwargs: CFG++ control dictionary.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            use_cfg=True,              # CFG++-only
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
        cfg_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model (ancestral sampling, CFG++-only).

        :param cfg_kwargs: dict controlling CFG++ behaviour (must include cfg_mode).
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            cfg_kwargs=cfg_kwargs,
            device=device,
            progress=progress,
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
        cfg_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples and yield intermediate samples from each timestep.

        CFG++-only (p_mean_variance enforces guided sampling).
        """
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
        Sample x_{t-1} from the model using DDIM with CFG++-style guidance.

        CFG++-only diffusion:
          - `use_cfg` must be True.
          - `cfg_mode` (in cfg_kwargs) must be one of
            {'global','per_attr','interv_group'}.

        In CFG++ DDIM:
          - x_0 comes from the guided prediction (pred_xstart)
          - the DDIM direction uses the unconditional eps, derived from
            pred_xstart_uncond.
        """
        if not use_cfg:
            raise ValueError(
                "CFG++ diffusion: ddim_sample expects use_cfg=True "
                "(no unconditional DDIM here)."
            )

        if cfg_kwargs is None:
            cfg_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}

        # p_mean_variance is CFG++-only and will compute
        # pred_xstart (guided) and pred_xstart_uncond (unconditional)
        out = self.p_mean_variance(
            model=model,
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            use_cfg=True,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            cfg_kwargs=cfg_kwargs,
        )

        # guided and unconditional eps from their respective x_0
        eps_guided = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        eps_uncond = self._predict_eps_from_xstart(x, t, out["pred_xstart_uncond"])

        # CFG++: always use unconditional eps in the DDIM direction
        eps_for_ddim = eps_uncond

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
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_for_ddim
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

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
        """
        Generate samples from the model using DDIM with CFG++-style guidance.

        Same usage as p_sample_loop(), but CFG++-only.
        """
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
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM, with CFG++-style guidance.

        CFG++-only diffusion (requires use_cfg=True).
        """
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


    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        use_cfg=True,
        cond_fn=None,
        model_kwargs=None,
        cfg_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using the DDIM reverse ODE (inversion)
        with CFG++-style guidance.

        CFG++-only inversion:
          - Requires use_cfg=True.
          - Always uses unconditional x_0 and guided eps, as in the CFG++
            inversion formulation.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path (eta must be 0)."

        if not use_cfg:
            raise ValueError(
                "CFG++ diffusion: ddim_reverse_sample expects use_cfg=True."
            )

        if cfg_kwargs is None:
            cfg_kwargs = {}
        if model_kwargs is None:
            model_kwargs = {}

        # Get all eps / x0 variants (guided + unconditional)
        out = self.p_mean_variance(
            model=model,
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            use_cfg=True,
            model_kwargs=model_kwargs,
            cfg_kwargs=cfg_kwargs,
        )

        # alphā_{t+1}
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        eps_guided = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        # CFG++ inversion:
        #   x0_term uses unconditional x0 (from ε̂_∅)
        #   eps_term uses guided eps
        x0_term = out["pred_xstart_uncond"]
        eps_term = eps_guided

        mean_pred = (
            x0_term * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps_term
        )

        return {"sample": mean_pred, "pred_xstart": x0_term}



    def ddim_reverse_sample_loop(
        self,
        model,
        image,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        use_cfg=True,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        cfg_kwargs=None,
    ):
        final = None
        for sample in self.ddim_reverse_sample_loop_progressive(
            model,
            image,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            use_cfg=use_cfg,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            cfg_kwargs=cfg_kwargs,
        ):
            final = sample
        return final["sample"]

    def ddim_reverse_sample_loop_progressive(
        self,
        model,
        image,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        use_cfg=True,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        cfg_kwargs=None,
    ):
        """
        DDIM inversion loop (x_0 -> x_T) with CFG++-style guidance.
        """
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
                    use_cfg=use_cfg,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    cfg_kwargs=cfg_kwargs,
                    eta=eta,
                )
                yield out
                image = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model=model, 
            x=x_t, 
            t=t, 
            clip_denoised=clip_denoised, 
            model_kwargs=model_kwargs,
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

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
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
            # Always predict EPSILON
            target = noise
            # assert model_output.shape == target.shape == x_start.shape
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
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
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
