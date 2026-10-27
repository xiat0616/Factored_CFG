"""
Various utilities for neural networks.
"""
from abc import abstractmethod
import math
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def sample_from_latent(mu, logvar):
    std = torch.exp(0.5 * logvar)           # [B, D]
    eps = torch.randn_like(std)             # [B, D] ~ N(0, I)
    return mu + eps * std                   # [B, D] sample from N(mu, sigma^2)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def exists(val):
    return val is not None


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )

def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
    

class EMA(nn.Module):
    """
    Adapted from: https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
        self,
        model,
        beta=0.999,
        update_after_step=100,
        update_every=1,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model

        try:
            self.ema_model = copy.deepcopy(model)
        except:  # noqa E722
            print(
                "Your model was not copyable. Please make sure you are not using any LazyLinear"
            )
            exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.register_buffer("initted", torch.Tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    def copy_params_from_model_to_ema(self):
        for ma_params, current_params in zip(
            list(self.ema_model.parameters()), list(self.online_model.parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(
            list(self.ema_model.buffers()), list(self.online_model.buffers())
        ):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch <= 0:
            return 0.0

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            list(current_model.named_parameters()), list(ma_model.named_parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(
            list(current_model.named_buffers()), list(ma_model.named_buffers())
        ):
            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class EMA(nn.Module):
    """
    Adapted from: https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
        self,
        model,
        beta=0.999,
        update_after_step=100,
        update_every=1,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model

        try:
            self.ema_model = copy.deepcopy(model)
        except:  # noqa E722
            print(
                "Your model was not copyable. Please make sure you are not using any LazyLinear"
            )
            exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.register_buffer("initted", torch.Tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    def copy_params_from_model_to_ema(self):
        for ma_params, current_params in zip(
            list(self.ema_model.parameters()), list(self.online_model.parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(
            list(self.ema_model.buffers()), list(self.online_model.buffers())
        ):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch <= 0:
            return 0.0

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            list(current_model.named_parameters()), list(ma_model.named_parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(
            list(current_model.named_buffers()), list(ma_model.named_buffers())
        ):
            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings and cemb (cemb is None if we do not use it later) as additional arguments.
    """
    @abstractmethod
    def forward(self, x, temb=None, cemb=None, semb=None):
        """
        abstract method
        """

class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x, temb=None, cemb=None, semb=None) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb, semb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(EmbedBlock):
    def __init__(
        self,
        channels,
        dropout,
        tdim,
        cdim=None,
        sdim=None,  # semantic/style dim
        out_channels=None,
        use_conv=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        embedding_mode=None,  # One of: "concat_cemb_semb", "semb_shift_scale", "add_all", "concat_all"
    ):
        super().__init__()
        self.channels = channels
        self.tdim = tdim
        self.cdim = cdim
        self.sdim = sdim
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.embedding_mode = embedding_mode

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # Projections based on embedding_mode
        if embedding_mode == "concat_cemb_semb":
            cat_dim = (cdim or 0) + (sdim or 0)
            if cat_dim > 0:
                self.cemb_semb_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(cat_dim, 2 * self.out_channels),
                )
            if tdim:
                self.temp_proj_temb = nn.Sequential(
                    nn.SiLU(),
                    linear(tdim, 2 * self.out_channels),
                )

        elif embedding_mode == "semb_shift_scale":
            if tdim:
                self.temp_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(tdim, 2 * self.out_channels),
                )
            if cdim:
                self.cemb_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(cdim, 2 * self.out_channels),
                )
            if sdim:
                self.style_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(sdim, self.out_channels),
                )

        elif embedding_mode == "add_all":
            if tdim:
                self.temp_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(tdim, 2 * self.out_channels),
                )
            if cdim:
                self.cemb_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(cdim, 2 * self.out_channels),
                )
            if sdim:
                self.semb_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(sdim, 2 * self.out_channels),
                )

        elif embedding_mode == "concat_all":
            cat_dim = (tdim or 0) + (cdim or 0) + (sdim or 0)
            if cat_dim > 0:
                self.all_emb_proj = nn.Sequential(
                    nn.SiLU(),
                    linear(cat_dim, 2 * self.out_channels),
                )


        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, temb=None, cemb=None, semb=None):
        return checkpoint(self._forward, (x, temb, cemb, semb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, temb=None, cemb=None, semb=None):
        assert (self.cdim is None or cemb is not None), f"Expected cemb when cdim is specified."
        assert (self.sdim is None or semb is not None), f"Expected semb when sdim is specified."

        if self.updown:
            h = self.in_layers[:-1](x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_layers[-1](h)
        else:
            h = self.in_layers(x)

        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]

        if self.embedding_mode == "semb_shift_scale":
            base_emb = 0
            if temb is not None:
                base_emb += self.temp_proj(temb).type(h.dtype)
            if cemb is not None:
                base_emb += self.cemb_proj(cemb).type(h.dtype)
            while len(base_emb.shape) < len(h.shape):
                base_emb = base_emb[..., None]
            scale, shift = torch.chunk(base_emb, 2, dim=1)

            style_scale = self.style_proj(semb).type(h.dtype)
            while len(style_scale.shape) < len(h.shape):
                style_scale = style_scale[..., None]

            h = out_norm(h)
            h = h * (1 + scale) + shift
            h = h * style_scale
            h = out_rest(h)

        elif self.embedding_mode == "concat_cemb_semb":
            emb_out = 0
            if cemb is not None or semb is not None:
                cat = []
                if cemb is not None:
                    cat.append(cemb)
                if semb is not None:
                    cat.append(semb)
                cat_proj = self.cemb_semb_proj(torch.cat(cat, dim=-1)).type(h.dtype)
                while len(cat_proj.shape) < len(h.shape):
                    cat_proj = cat_proj[..., None]
                emb_out += cat_proj
            if temb is not None:
                temb_proj = self.temp_proj_temb(temb).type(h.dtype)
                while len(temb_proj.shape) < len(h.shape):
                    temb_proj = temb_proj[..., None]
                emb_out += temb_proj

            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        elif self.embedding_mode == "add_all":
            emb_out = 0
            if temb is not None:
                temb_out = self.temp_proj(temb).type(h.dtype)
                while len(temb_out.shape) < len(h.shape):
                    temb_out = temb_out[..., None]
                emb_out += temb_out
            if cemb is not None:
                cemb_out = self.cemb_proj(cemb).type(h.dtype)
                while len(cemb_out.shape) < len(h.shape):
                    cemb_out = cemb_out[..., None]
                emb_out += cemb_out
            if semb is not None:
                semb_out = self.semb_proj(semb).type(h.dtype)
                while len(semb_out.shape) < len(h.shape):
                    semb_out = semb_out[..., None]
                emb_out += semb_out

            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        elif self.embedding_mode == "concat_all":
            emb_list = []
            if temb is not None:
                emb_list.append(temb)
            if cemb is not None:
                emb_list.append(cemb)
            if semb is not None:
                emb_list.append(semb)
            if emb_list:
                emb_cat = torch.cat(emb_list, dim=-1)
                emb_proj = self.all_emb_proj(emb_cat).type(h.dtype)
                while len(emb_proj.shape) < len(h.shape):
                    emb_proj = emb_proj[..., None]
                scale, shift = torch.chunk(emb_proj, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = self.out_layers(h)
        elif self.embedding_mode == "none":
            h = self.out_layers(h)
        else:
            raise NotImplementedError(f"Unknown embedding_mode: {self.embedding_mode}")

        return self.skip_connection(x) + h




class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)