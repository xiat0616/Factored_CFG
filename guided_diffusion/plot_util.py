import os
import torch as th
import numpy as np
import matplotlib.pyplot as plt


def sample_images_single_batch(
    args,
    model,
    diffusion,
    step,
    batch,
    use_ddim=False,
    save_dir="./samples",
    num_plt=4,
    logger=None,
):
    """
    Generate and save example images from a single batch using a diffusion model,
    with visualization of input images.

    Args:
        model (torch.nn.Module): The diffusion model.
        diffusion (object): The diffusion process (e.g., p_sample_loop).
        step (int): The current training step, used for naming the output file.
        batch (dict): A single batch of input data (keys: "x", optional "y").
        use_ddim (bool): Whether to use DDIM sampling instead of default.
        save_dir (str): Directory to save the generated images.
        num_plt (int): Number of images to plot (both input and generated).
    """
    # logger.info("Sampling example images from a single batch...")
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Prepare model_kwargs if class labels are present
    model_kwargs = {}
    if "cemb" in batch.keys():  # Check if class labels are present
        model_kwargs["cemb"] = batch["cemb"].to(args.device)
    if "semb" in batch.keys():
        model_kwargs["semb"] = batch["semb"].to(args.device)
    # print(f"bacth.keys: {batch.keys()}")
    # Load input images
    input_images = batch["x"].to(args.device)  # Input tensor

    # Generate samples
    sample_fn = diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
    samples = sample_fn(
        model,
        input_images.shape,  # Use the shape of the input batch
        clip_denoised=True,
        model_kwargs=model_kwargs,
    )
    samples = ((samples + 1) * 127.5).clamp(0, 255).to(th.uint8)
    samples = samples.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (H, W, C)

    # Prepare input images for visualization
    input_images = ((input_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
    input_images = input_images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (H, W, C)

    # Limit the number of images to plot
    num_plt = min(num_plt, len(samples))
    input_images = input_images[:num_plt]
    samples = samples[:num_plt]

    # Plot input and generated images side by side
    fig, axes = plt.subplots(2, num_plt, figsize=(num_plt * 3, 6))
    for i, (inp_img, gen_img) in enumerate(zip(input_images, samples)):
        # Determine if grayscale based on channel dimension
        is_grayscale = inp_img.shape[-1] == 1

        # Input images
        axes[0, i].imshow(
            inp_img.squeeze(-1) if is_grayscale else inp_img,
            cmap="gray" if is_grayscale else None,
        )
        axes[0, i].axis("off")
        axes[0, i].set_title("Input")

        # Generated images
        axes[1, i].imshow(
            gen_img.squeeze(-1) if is_grayscale else gen_img,
            cmap="gray" if is_grayscale else None,
        )
        axes[1, i].axis("off")
        axes[1, i].set_title("Generated")

    plt.tight_layout()

    # Save the visualization
    save_path = os.path.join(save_dir, f"samples_step_{step}.png")
    plt.savefig(save_path, dpi=600)
    plt.close()
    # logger.info(f"Saved input and generated images to {save_path}")

def reconstruct_images_single_batch(
    args,
    model,
    diffusion,
    step,
    batch,
    save_dir="./reconstructions",
    num_plt=4,
    eta=0.0,
    logger=None,
):
    """
    Reconstruct input images from a single batch using DDIM reverse sampling followed by DDIM sampling.

    Args:
        model (torch.nn.Module): The diffusion model.
        diffusion (object): The diffusion process (e.g., ddim_reverse_sample_loop and ddim_sample_loop).
        step (int): The current training step, used for naming the output file.
        batch (dict): A single batch of input data (keys: "x", optional "y").
        save_dir (str): Directory to save the reconstructed images.
        num_plt (int): Number of images to plot (both input and reconstructed).
        eta (float): Noise scale for DDIM reverse sampling. 0.0 for deterministic reconstruction.
    """
    # logger.info("Reconstructing images from a single batch using DDIM reverse and forward sampling...")
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Prepare model_kwargs if class labels are present
    model_kwargs = {}
    if "cemb" in batch:  # Check if class labels are present
        model_kwargs["cemb"] = batch["cemb"].to(args.device)
    if "semb" in batch:
        model_kwargs["semb"] = batch["semb"].to(args.device)
    # Load input images
    input_images = batch["x"].to(args.device)  # Input tensor
    # logger.info(f"debug plot: input_images min {input_images.min()}, max {input_images.max()}")

    # Perform DDIM reverse sampling to get noise
    reversed_noise = diffusion.ddim_reverse_sample_loop(
        model,
        input_images,
        clip_denoised=True,
        model_kwargs=model_kwargs,
        eta=eta,
        use_cfg=False,
    )

    # Use the noise as input to DDIM sampling to reconstruct images
    reconstructed = diffusion.ddim_sample_loop(
        model,
        input_images.shape,
        noise=reversed_noise,
        clip_denoised=True,
        model_kwargs=model_kwargs,
        eta=eta,
        use_cfg=True,
        cfg_kwargs=None,
    )

    reconstructed = ((reconstructed + 1) * 127.5).clamp(0, 255).to(th.uint8)
    reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (H, W, C)

    # Prepare input images for visualization
    input_images = ((input_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
    input_images = input_images.permute(0, 2, 3, 1).cpu().numpy()  # Convert to (H, W, C)

    # Limit the number of images to plot
    num_plt = min(num_plt, len(reconstructed))
    input_images = input_images[:num_plt]
    reconstructed = reconstructed[:num_plt]

    # Plot input and reconstructed images side by side
    fig, axes = plt.subplots(2, num_plt, figsize=(num_plt * 3, 6))
    for i, (inp_img, rec_img) in enumerate(zip(input_images, reconstructed)):
        # Determine if grayscale based on channel dimension
        is_grayscale = inp_img.shape[-1] == 1

        # Input images
        axes[0, i].imshow(
            inp_img.squeeze(-1) if is_grayscale else inp_img,
            cmap="gray" if is_grayscale else None,
        )
        axes[0, i].axis("off")
        axes[0, i].set_title("Input")

        # Reconstructed images
        axes[1, i].imshow(
            rec_img.squeeze(-1) if is_grayscale else rec_img,
            cmap="gray" if is_grayscale else None,
        )
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    plt.tight_layout()

    # Save the visualization
    save_path = os.path.join(save_dir, f"reconstructions_step_{step}.png")
    plt.savefig(save_path, dpi=600)
    plt.close()
    # logger.info(f"Saved input and reconstructed images to {save_path}")
