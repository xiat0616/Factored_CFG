import torch
from respace_baselines import SpacedDiffusion, space_timesteps
import gaussian_diffusion as gd
import numpy as np
from unet import UNetModel
import random
import os

def load_model(model_args, state_dict):
    """
    Load a UNetModel from a provided state dictionary.

    Args:
        model_args: Arguments required to initialize the model.
        state_dict (dict): The saved state dictionary.

    Returns:
        torch.nn.Module: The loaded UNetModel.
    """
    model = UNetModel(model_args)  # Initialize UNetModel
    model.load_state_dict(state_dict)
    return model

def create_gaussian_diffusion(diffusion_args):
    """
    Create a diffusion model using the provided diffusion_args.
    :param diffusion_args: An object containing all necessary arguments for diffusion.
    :return: A SpacedDiffusion instance.
    """
    steps = diffusion_args.steps
    noise_schedule = diffusion_args.noise_schedule
    use_kl = diffusion_args.use_kl
    rescale_timesteps = diffusion_args.rescale_timesteps
    rescale_learned_sigmas = diffusion_args.rescale_learned_sigmas
    timestep_respacing = diffusion_args.timestep_respacing or [steps]

    # Generate diffusion_betas based on noise schedule
    diffusion_betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # Determine loss type
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # ------------ KEY CHANGE: choose model_var_type based on DIFF_ENGINE ------------    
    model_var_type = (
        gd.ModelVarType.FIXED_LARGE
        if not diffusion_args.sigma_small
        else gd.ModelVarType.FIXED_SMALL
    )
    # -------------------------------------------------------------------------------

    diffusion_args.diffusion_betas = np.array(diffusion_betas)
    diffusion_args.model_var_type = model_var_type
    diffusion_args.loss_type = loss_type
    diffusion_args.rescale_timesteps = rescale_timesteps

    spaced_diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        diffusion_args=diffusion_args,
    )
    return spaced_diffusion


def save_diffusion(diffusion):
    """
    Serialize a SpacedDiffusion into a dictionary format.

    Args:
        diffusion (SpacedDiffusion): An instance of the SpacedDiffusion model to be saved.

    Returns:
        dict: A dictionary containing the serialized diffusion model parameters:
            - "type" (str): Specifies the diffusion model type ("SpacedDiffusion").
            - "use_timesteps" (optional): The timesteps used in the diffusion process.
            - "base_params" (dict): The base parameters of the SpacedDiffusion model.

    Raises:
        ValueError: If the provided diffusion model is not an instance of SpacedDiffusion.
    """
    diffusion_type = diffusion.__class__.__name__
    base_diffusion_params = diffusion.base_diffusion.diffusion_args
    use_timesteps = diffusion.use_timesteps

    if isinstance(diffusion, SpacedDiffusion):
        return {
            "type": "SpacedDiffusion",
            "use_timesteps": use_timesteps,
            "base_params": base_diffusion_params,
        }
    else:
        raise ValueError(f"Unsupported diffusion type: {diffusion_type}")


def load_diffusion(diffusion_dict):
    """
    Load and initialize a SpacedDiffusion model based on the provided configuration.

    Args:
        diffusion_dict (dict): A dictionary containing the required parameters:
            - "type" (str): Must be "SpacedDiffusion".
            - "base_params" (dict): Parameters required to initialize the SpacedDiffusion model.
            - "use_timesteps" (optional): Specifies timesteps to be used in the diffusion process.

    Returns:
        SpacedDiffusion: An instance of the SpacedDiffusion model.

    Raises:
        ValueError: If the diffusion type is not "SpacedDiffusion".
    """
    diffusion_type = diffusion_dict["type"]
    base_diffusion_params = diffusion_dict["base_params"]
    use_timesteps = diffusion_dict["use_timesteps"]
    
    if diffusion_type == "SpacedDiffusion":
        return SpacedDiffusion(diffusion_args=base_diffusion_params, use_timesteps=use_timesteps)
    else:
        raise ValueError(f"Unknown diffusion type: {diffusion_type}")

### For sampler ###
from resample import UniformSampler, LossSecondMomentResampler
def create_named_schedule_sampler(name, diffusion):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    :param diffusion: the diffusion object to sample for.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")

def save_schedule_sampler(schedule_sampler):
    """
    Extracts necessary parameters from a schedule sampler for saving.
    
    :param schedule_sampler: An instance of a schedule sampler (e.g., UniformSampler).
    :return: A dictionary containing the sampler type and parameters.
    """
    sampler_type = schedule_sampler.__class__.__name__

    if isinstance(schedule_sampler, UniformSampler):
        sampler_params = {}  # No extra parameters needed for UniformSampler
    else:
        sampler_params = vars(schedule_sampler)  # Extract relevant parameters

    return {
        "type": sampler_type,
        "params": sampler_params
    }

def load_schedule_sampler(sampler_info, diffusion):
    """
    Reconstructs a schedule sampler object from saved parameters.
    
    :param sampler_info: A dictionary containing the saved sampler type and parameters.
    :param diffusion: The diffusion instance associated with the sampler.
    :return: A new instance of the appropriate schedule sampler class.
    """
    sampler_type = sampler_info["type"]
    sampler_params = sampler_info["params"]

    if sampler_type == "UniformSampler":
        return UniformSampler(diffusion)
    else:
        sampler_class = globals().get(sampler_type)
        if sampler_class:
            return sampler_class(diffusion, **sampler_params)
        else:
            raise ValueError(f"Unknown schedule sampler type: {sampler_type}")


def init_bias(m):
    if isinstance(m, torch.nn.Conv2d):
        try: 
            torch.nn.init.zeros_(m.bias)
        except:
            pass

def init_weights(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # 初始化权重
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)  # 把 bias 初始化为 0


def seed_all(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
