import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors
import pandas as pd
from PIL import Image
import time

CLIP_DENOISED = True

def format_cfg_w(cfg_w):
    s = f"{cfg_w:.8f}".rstrip("0")
    if s.endswith("."):
        s = s[:-1]
    decimals = max(1, len(s.split(".")[1]) if "." in s else 0)
    return f"{cfg_w:.{decimals}f}"

def save_image_auto_mode(img_np, path):
    if img_np.ndim == 2:
        Image.fromarray(img_np, mode="L").save(path)
    elif img_np.ndim == 3 and img_np.shape[-1] == 1:
        Image.fromarray(img_np.squeeze(-1), mode="L").save(path)
    else:
        Image.fromarray(img_np).save(path)

def imshow_auto(ax, img, title=None, cmap="gray", norm=None):
    if img.shape[-1] == 1:
        im = ax.imshow(img.squeeze(-1), cmap=cmap, norm=norm)
    else:
        im = ax.imshow(img, norm=norm)
    ax.axis("off")
    if title:
        ax.set_title(title)
    return im

def squeeze_if_gray(img):
    return img.squeeze(-1) if img.shape[-1] == 1 else img

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def generate_random_samples(
    diffusion,
    model,
    args,
    save_path,
    loader=None,
    use_ddim=False,
    cond_embedder=None,
    sem_encoder=None,
    use_cfg=False,
    cfg_kwargs=None,
):
    """
    Generates random samples using the diffusion model and saves them.

    This is agnostic to the specific guidance geometry:
      - If `diffusion` is a GaussianDiffusion:
          * `use_cfg=True` → standard CFG (with cfg_kwargs controlling cfg_mode, etc.)
      - If `diffusion` is a GaussianDiffusion_cfgpp:
          * `use_cfg=True` → CFG++ geometry (guided x0 + uncond eps inside the class)
      - If `diffusion` is a GaussianDiffusion_apg:
          * `use_cfg=True` → APG guidance in x0-space (cfg_kwargs carries APG params)

    Args:
        diffusion: Diffusion object (GaussianDiffusion / GaussianDiffusion_cfgpp / GaussianDiffusion_apg).
        model: The underlying UNet model.
        args: Namespace-like object with at least:
              - num_samples, batch_size, in_channels, image_size, device,
                sem_embed_dim (if sem_encoder is used).
        save_path: Directory to save images and CSV.
        loader: DataLoader yielding batch dicts with parents / labels, used for
                conditional sampling and CSV metadata.
        use_ddim: If True, use DDIM sampling (`diffusion.ddim_sample_loop`),
                  otherwise ancestral sampling (`diffusion.p_sample_loop`).
        cond_embedder: Module mapping `batch -> cemb` (conditional embedding).
        sem_encoder: If not None, we *sample* semantic latents from N(0, I)
                     with shape (batch_size, args.sem_embed_dim) to condition the model.
        use_cfg: Whether to enable guidance in forward sampling (meaning depends
                 on diffusion class: CFG / CFG++ / APG).
        cfg_kwargs: Dict with guidance options:
                    - For CFG: e.g. {"cfg_mode": "global", "cfg_attr_weights": {...}, ...}
                    - For CFG++: e.g. {"cfg_mode": "global", ...}
                    - For APG: e.g. {"apg_eta": 1.0, "apg_norm_threshold": 0.0, ...}
    """
    os.makedirs(save_path, exist_ok=True)
    if cfg_kwargs is None:
        cfg_kwargs = {}

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    progress_bar = tqdm(
        total=args.num_samples,
        desc="Generating Random Samples",
        unit="samples",
    )

    total_saved = 0
    csv_file = os.path.join(save_path, "random_samples.csv")
    write_header = not os.path.exists(csv_file)

    for batch_idx, batch in zip(range(num_batches), loader):
        remaining = args.num_samples - total_saved
        batch = {k: v[:remaining] if isinstance(v, (torch.Tensor, list, tuple)) else v
                 for k, v in batch.items()}
        current_batch_size = len(batch["x"])
        csv_data = []

        model_kwargs = {}
        # ===== Move batch items to device =====
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)

        # ===== Conditional Embedding =====
        if cond_embedder is not None:
            cemb = cond_embedder(batch)
            model_kwargs["cemb"] = cemb

        # ===== Semantic Embedding =====
        if sem_encoder is not None:
            # For random sampling, there is no input x to encode.
            # We therefore sample semantic latents from N(0, I).
            semb_shape = (current_batch_size, args.sem_embed_dim)
            semb = torch.randn(semb_shape, device=args.device)
            model_kwargs["semb"] = semb

        # ===== Sampling =====
        if use_ddim:
            # DDIM (optionally with CFG / CFG++ / APG, depending on diffusion class)
            sample = diffusion.ddim_sample_loop(
                model,
                (current_batch_size, args.in_channels, args.image_size, args.image_size),
                noise=None,
                clip_denoised=CLIP_DENOISED,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                device=args.device,
                progress=False,
                eta=0.0,
                use_cfg=use_cfg,
                cfg_kwargs=cfg_kwargs,
            )
        else:
            # Ancestral sampling (Ho et al.), usually unconditional or CFG-only
            sample = diffusion.p_sample_loop(
                model,
                (current_batch_size, args.in_channels, args.image_size, args.image_size),
                noise=None,
                clip_denoised=CLIP_DENOISED,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                device=args.device,
                progress=False,
            )

        # ===== Post-process and save images =====
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()

        for i, img in enumerate(sample):
            save_file = os.path.join(save_path, f"sample_{total_saved + i}.png")
            # grayscale-friendly save
            plt.imsave(save_file, img.squeeze(), cmap="gray")

            # Build CSV entry using batch metadata
            csv_entry = {
                key: batch[key][i].item()
                if key not in ["race", "scanner"]
                else torch.argmax(batch[key][i]).item()
                for key in batch
                if key not in ["pa", "x", "short_path"]
            }
            csv_entry["file_path"] = save_file
            csv_data.append(csv_entry)

        df_batch = pd.DataFrame(csv_data)
        df_batch.to_csv(csv_file, mode="a", header=write_header, index=False)
        write_header = False

        batch_size_effective = len(sample)
        total_saved += batch_size_effective
        progress_bar.update(batch_size_effective)

        if total_saved >= args.num_samples:
            break

    progress_bar.close()
    print(f"Finished! Saved {total_saved} images to {save_path}.")
    print(f"Saved random samples metadata to {csv_file}")


def reconstruct_images(
    args,
    model,
    diffusion,
    loader,
    save_dir="./reconstructions",
    eta=0.0,
    use_cfg=False,          # generic "use guidance" flag
    cond_embedder=None,
    sem_encoder=None,
    cfg_kwargs=None,        # generic guidance config (cfg_mode, weights, APG params, etc.)
):
    """
    Reconstruct images using DDIM reverse and forward sampling.

    This function is written to be compatible with:
      - GaussianDiffusion           (interprets use_cfg as standard CFG)
      - GaussianDiffusion_cfgpp     (interprets use_cfg as CFG++)
      - GaussianDiffusion_apg       (interprets use_cfg as APG)

    Conventions (by diffusion implementation):

      • Reverse pass (x -> noise):
          - GaussianDiffusion:      typically unconditional (use_cfg=False by default in its API)
          - GaussianDiffusion_cfgpp: CFG++-style inversion (use_cfg=True by default in its API)
          - GaussianDiffusion_apg:  unconditional reverse (no guidance used)

        We *do not* pass use_cfg / cfg_kwargs to the reverse call here to keep a
        common API across all three; each diffusion class uses its own defaults.

      • Forward pass (noise -> reconstruction):
          - We always call diffusion.ddim_sample_loop with:
                use_cfg=use_cfg
                cfg_kwargs=cfg_kwargs
            and let the specific diffusion object decide how to interpret this
            (CFG / CFG++ / APG).

    Args:
        args: Namespace with attributes:
              - num_samples, batch_size, device, in_channels, image_size, hps, use_vae_encoder, etc.
        model: UNet (or similar) denoiser.
        diffusion: One of {GaussianDiffusion, GaussianDiffusion_cfgpp, GaussianDiffusion_apg}.
        loader: DataLoader providing batches with at least:
                batch["x"] (images in [-1,1]) and batch["short_path"].
        save_dir: Directory where reconstructions and visualizations are saved.
        eta: DDIM eta (0.0 for deterministic DDIM).
        use_cfg: Whether to use guidance on the *forward* DDIM pass.
        cond_embedder: Module mapping a batch dict -> cemb.
        sem_encoder: Module mapping images -> semb (or (mu, logvar) if VAE encoder).
        cfg_kwargs: Dict of guidance options:
            - For CFG:      {"cfg_mode": "...", "cfg_attr_weights": {...}, ...}
            - For CFG++:    {"cfg_mode": "...", ...}
            - For APG:      {"apg_guidance_scale": ..., "apg_eta": ..., ...}
    """
    print("Reconstructing images using DDIM reverse and forward sampling...")

    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    if cfg_kwargs is None:
        cfg_kwargs = {}

    total_saved = 0
    csv_file = os.path.join(save_dir, "reconstructions.csv")
    write_header = not os.path.exists(csv_file)

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    progress_bar = tqdm(total=args.num_samples, desc="Reconstructing", unit="samples")

    for batch_idx, batch in zip(range(num_batches), loader):
        remaining = args.num_samples - total_saved
        batch = {k: v[:remaining] if isinstance(v, (torch.Tensor, list, tuple)) else v
                 for k, v in batch.items()}
        current_batch_size = len(batch["x"])
        csv_data = []
        model_kwargs = {}

        # Move all batch tensors to args.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)

        # ===== Conditional Embedding =====
        if cond_embedder is not None:
            cemb = cond_embedder(batch)
            model_kwargs["cemb"] = cemb

        # ===== Semantic Embedding =====
        if sem_encoder is not None:
            if getattr(args, "use_vae_encoder", False):
                mu, _ = sem_encoder(batch["x"])
                semb = mu
            else:
                semb = sem_encoder(batch["x"])
            model_kwargs["semb"] = semb

        input_images = batch["x"]
        input_file_names = batch.get("short_path", [str(total_saved + i) for i in range(len(input_images))])

        # -------------------------------------------------
        # 1) Reverse sampling (encode to noise)
        #    -> API-compatible with all three diffusions
        #       - CFG:      uses its default use_cfg (usually False)
        #       - CFG++:    uses its default use_cfg (True, CFG++ inversion)
        #       - APG:      unconditional reverse
        # -------------------------------------------------
        reversed_noise = diffusion.ddim_reverse_sample_loop(
            model,
            input_images,
            clip_denoised=CLIP_DENOISED,
            denoised_fn=None,
            # NOTE: we deliberately do NOT pass use_cfg / cfg_kwargs here
            # to keep the call valid for GaussianDiffusion_apg as well.
            model_kwargs=model_kwargs,
            device=args.device,
            progress=False,
            eta=eta,
        )

        # -------------------------------------------------
        # 2) Forward sampling (reconstruct) with unified guidance API
        #    - GaussianDiffusion:      use_cfg => CFG, cfg_kwargs => CFG options
        #    - GaussianDiffusion_cfgpp:use_cfg => CFG++, cfg_kwargs => CFG++ options
        #    - GaussianDiffusion_apg:  use_cfg => APG, cfg_kwargs => APG params
        # -------------------------------------------------
        reconstructed = diffusion.ddim_sample_loop(
            model,
            input_images.shape,
            noise=reversed_noise,
            clip_denoised=CLIP_DENOISED,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            device=args.device,
            progress=False,
            eta=eta,
            use_cfg=use_cfg,
            cfg_kwargs=cfg_kwargs,
        )

        # -------------------------------------------------
        # 3) Preprocess images for saving and visualization
        # -------------------------------------------------
        input_images_np = ((input_images + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        input_images_np = input_images_np.permute(0, 2, 3, 1).cpu().numpy()

        reconstructed_np = ((reconstructed + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        reconstructed_np = reconstructed_np.permute(0, 2, 3, 1).cpu().numpy()

        differences = reconstructed_np.astype(float) - input_images_np.astype(float)

        # -------------------------------------------------
        # 4) Visualization (Input / Rec / Diff) per image
        # -------------------------------------------------
        for i, (inp_img, rec_img, diff_img, short_path) in enumerate(
            zip(input_images_np, reconstructed_np, differences, input_file_names)
        ):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Input
            axes[0].imshow(squeeze_if_gray(inp_img), cmap="gray")
            axes[0].axis("off")
            axes[0].set_title("Input")

            # Reconstructed
            axes[1].imshow(squeeze_if_gray(rec_img), cmap="gray")
            axes[1].axis("off")
            axes[1].set_title("Reconstructed")

            # Difference
            norm = MidpointNormalize(
                vmin=diff_img.min(), midpoint=0, vmax=diff_img.max()
            )
            im = axes[2].imshow(squeeze_if_gray(diff_img), cmap="RdBu_r", norm=norm)
            axes[2].axis("off")
            axes[2].set_title("Difference rec - x")

            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
            cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Difference (Reconstructed - Input)", fontsize=12)

            save_path_fig = os.path.join(save_dir, f"rec_viz_{total_saved + i}.png")
            plt.savefig(save_path_fig, dpi=500, bbox_inches="tight")
            plt.close()

        # -------------------------------------------------
        # 5) Save raw input / rec + CSV metadata
        # -------------------------------------------------
        for i, (inp_img, rec_img) in enumerate(zip(input_images_np, reconstructed_np)):
            inp_img = squeeze_if_gray(inp_img)
            rec_img = squeeze_if_gray(rec_img)

            input_path = os.path.join(save_dir, f"input_{i + total_saved}.png")
            rec_path = os.path.join(save_dir, f"rec_{i + total_saved}.png")

            save_image_auto_mode(inp_img, input_path)
            save_image_auto_mode(rec_img, rec_path)

            csv_entry = {
                key: batch[key][i].item()
                if key not in ["race", "scanner"]
                else torch.argmax(batch[key][i]).item()
                for key in batch
                if key not in ["pa", "x", "short_path"]
            }
            csv_entry["input_path"] = input_path
            csv_entry["reconstruction_path"] = rec_path
            csv_data.append(csv_entry)

        pd.DataFrame(csv_data).to_csv(
            csv_file, mode="a", header=write_header, index=False
        )
        write_header = False
        total_saved += len(reconstructed_np)
        progress_bar.update(len(input_images_np))

        if total_saved >= args.num_samples:
            break

    progress_bar.close()
    print(
        f"Finished! Saved {total_saved} reconstructed images with visualizations to {save_dir}."
    )
    print(f"Saved reconstruction metadata to {csv_file}")


def generate_counterfactuals(
    args,
    model,
    diffusion,
    loader,
    save_dir="./counterfactuals",
    eta=0.0,
    do_keys=None,        # list of intervened attributes
    cond_embedder=None,
    sem_encoder=None,
    cfg_kwargs=None,     # generic guidance config (CFG / CFG++ / APG)
    reverse_cf=False,
    use_cfg=True,        # generic "use guidance" flag for forward passes
):
    """
    Generate counterfactual images via:
      1) DDIM reverse ODE to obtain latent noise from original images,
      2) DDIM forward sampling under intervened attributes, with optional guidance,
      3) (Optional) reverse counterfactuals back towards original attributes.

    This function is written to be compatible with:
      - GaussianDiffusion           (interprets use_cfg as standard CFG)
      - GaussianDiffusion_cfgpp     (interprets use_cfg as CFG++)
      - GaussianDiffusion_apg       (interprets use_cfg as APG in x_0-space)

    Conventions by diffusion implementation:

      • Reverse passes (x -> noise):
          - GaussianDiffusion:       default is typically unconditional (use_cfg=False).
          - GaussianDiffusion_cfgpp: default is CFG++ inversion (use_cfg=True inside).
          - GaussianDiffusion_apg:   reverse is unconditional (no guidance).

        We DO NOT pass `use_cfg` / `cfg_kwargs` to reverse calls here, so each
        diffusion class uses its own defaults and signature.

      • Forward passes (noise -> images):
          - We always call `ddim_sample_loop` with:
                use_cfg=use_cfg
                cfg_kwargs=cfg_kwargs
            and let the diffusion class interpret them:
              * CFG:    cfg_kwargs hold cfg_mode, weights, etc.
              * CFG++:  cfg_kwargs hold cfg_mode, interv_group keys, etc.
              * APG:    cfg_kwargs hold apg_eta, apg_guidance_scale, etc.

    Args:
        args: Namespace with at least:
              - parents: ordered list of parent attributes.
              - num_samples, batch_size, device, hps, use_vae_encoder, ...
        model: Denoiser (e.g. UNet).
        diffusion: One of {GaussianDiffusion, GaussianDiffusion_cfgpp, GaussianDiffusion_apg}.
        loader: DataLoader yielding batches with keys such as
                'x', 'short_path', and attribute tensors (e.g. 'age','sex','race',...).
        save_dir: Base directory where CFs and reverse-CFs are saved.
        eta: DDIM eta (0.0 for deterministic, >0 for stochastic).
        do_keys: List of attributes to intervene on (e.g. ['sex'], ['race'], ...).
        cond_embedder: Module mapping batch dict -> cemb.
        sem_encoder: Module mapping x -> semb (or (mu, logvar) if VAE encoder).
        cfg_kwargs: Dict of guidance hyperparameters (CFG / CFG++ / APG).
        reverse_cf: If True, generate reverse counterfactuals (CF -> approx original).
        use_cfg: Whether to use guidance for forward CF / reverse-CF sampling.
    """
    print("Generating counterfactual images...")
    model.eval()

    # --------- Guidance kwargs (local copy) ---------
    cfg_kwargs = dict(cfg_kwargs) if cfg_kwargs is not None else {}
    if do_keys is None:
        do_keys = []
    else:
        # store which attributes are intervened (used e.g. by CFG++ interv_group)
        cfg_kwargs["cfg_intervened_keys"] = do_keys

    cf_suffix = "_".join(do_keys) if do_keys else "no_intervention"

    # --------- Prepare directories ---------
    os.makedirs(save_dir, exist_ok=True)
    sub_dirs = {
        "input": os.path.join(save_dir, "input_images"),
        "cf": os.path.join(save_dir, "cf_images"),
        "cf_vis": os.path.join(save_dir, "cf_visuals"),
        "rev": os.path.join(save_dir, "reversed_images"),
        "rev_vis": os.path.join(save_dir, "reversed_visuals"),
    }
    for path in sub_dirs.values():
        os.makedirs(path, exist_ok=True)

    # --------- CSV bookkeeping ---------
    total_saved = 0
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    progress_bar = tqdm(
        total=args.num_samples,
        desc=f"Generating counterfactuals {cf_suffix}",
        unit="samples",
    )

    cf_csv_file = os.path.join(save_dir, f"counterfactuals_{cf_suffix}.csv")
    rev_csv_file = os.path.join(save_dir, f"reversed_{cf_suffix}.csv")
    write_cf_header = not os.path.exists(cf_csv_file)
    write_rev_header = not os.path.exists(rev_csv_file)

    # --------- Category labels for titles (dataset-dependent) ---------
    sex_categories = ["male", "female"]
    race_categories = ["White", "Asian", "Black"]    # adjust if needed
    finding_categories = ["No disease", "Pleural Effusion"]

    for batch_idx, batch in zip(range(num_batches), loader):
        remaining = args.num_samples - total_saved
        batch = {k: v[:remaining] if isinstance(v, (torch.Tensor, list, tuple)) else v
                 for k, v in batch.items()}
        current_batch_size = len(batch["x"])
        cf_csv_data = []
        rev_csv_data = []

        # Move batch tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(args.device)

        # Build model_kwargs for this batch (shared across passes)
        model_kwargs = {}

        # ----- Conditional embedding -----
        if cond_embedder is not None:
            model_kwargs["cemb"] = cond_embedder(batch)

        # ----- Semantic embedding -----
        if sem_encoder is not None:
            if getattr(args, "use_vae_encoder", False):
                mu, _ = sem_encoder(batch["x"])
                semb = mu
            else:
                semb = sem_encoder(batch["x"])
            model_kwargs["semb"] = semb

        input_images = batch["x"]

        # ==========================
        # 1) Reverse: x -> noise
        #    - GaussianDiffusion:       uncond by default
        #    - GaussianDiffusion_cfgpp: CFG++ inversion by default
        #    - GaussianDiffusion_apg:   uncond reverse
        #
        # We DO NOT pass use_cfg / cfg_kwargs here: each implementation uses
        # its own signature & defaults.
        # ==========================
        reversed_noise = diffusion.ddim_reverse_sample_loop(
            model,
            input_images,
            clip_denoised=CLIP_DENOISED,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            device=args.device,
            progress=False,
            eta=eta,
        )

        # ==========================
        # 2) Build counterfactual parent batch (cf_batch)
        # ==========================
        cf_batch = {key: batch[key].clone() for key in args.parents if key in batch}

        # Apply interventions
        for do_key in do_keys:
            if do_key == "age":
                # re-sample age from uniform in [0,1] (later scaled if needed)
                cf_batch[do_key] = torch.rand_like(batch[do_key])
            elif do_key in [
                "sex",
                "finding",
                "density",
                "Smiling",
                "Male",
                "Young",
                "Eyeglasses",
                "Mouth_Slightly_Open",
                "triangle",
                "circle",
            ]:
                # simple 0/1 flip
                cf_batch[do_key] = 1 - batch[do_key]
            elif do_key == "race":
                # cycle race one-hot to next class
                num_classes = batch[do_key].shape[1]
                current_indices = torch.argmax(batch[do_key], dim=1)
                next_indices = (current_indices + 1) % num_classes
                cf_batch[do_key] = torch.zeros_like(batch[do_key]).scatter_(
                    1, next_indices.unsqueeze(1), 1
                )

        # Concatenate parents into cf_batch["pa"]
        cf_batch["pa"] = torch.cat([cf_batch[key] for key in args.parents], dim=1).detach()

        # Ensure cf_batch tensors are on device
        for k, v in cf_batch.items():
            if isinstance(v, torch.Tensor):
                cf_batch[k] = v.to(args.device)

        # Recompute cemb under intervened parents for forward CF pass
        if cond_embedder is not None:
            model_kwargs["cemb"] = cond_embedder(cf_batch)

        # ==========================
        # 3) Forward: noise -> CF images
        #    - GaussianDiffusion:      use_cfg => CFG
        #    - GaussianDiffusion_cfgpp:use_cfg => CFG++
        #    - GaussianDiffusion_apg:  use_cfg => APG
        # ==========================
        start_time = time.perf_counter()
        counterfactuals = diffusion.ddim_sample_loop(
            model,
            input_images.shape,
            noise=reversed_noise,
            clip_denoised=CLIP_DENOISED,
            denoised_fn=None,
            model_kwargs=model_kwargs,
            device=args.device,
            progress=False,
            eta=eta,
            use_cfg=use_cfg,
            cfg_kwargs=cfg_kwargs,
        )
        elapsed = time.perf_counter() - start_time
        print(f"ddim_sample_loop took {elapsed:.2f} seconds")

        # ==========================
        # 4) Convert to uint8 images + differences
        # ==========================
        input_np = (
            ((input_images + 1) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        cf_np = (
            ((counterfactuals + 1) * 127.5)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        diff_np = cf_np.astype(float) - input_np.astype(float)

        # ==========================
        # 5) Save inputs / CFs + visualisations + metadata
        # ==========================
        for i, (inp, cf, diff) in enumerate(zip(input_np, cf_np, diff_np)):
            # ---- Raw input / CF paths ----
            inp_img = inp if inp.shape[-1] != 1 else inp.squeeze(-1)
            inp_path = os.path.join(
                sub_dirs["input"], f"input_{i + total_saved}_{cf_suffix}.png"
            )
            save_image_auto_mode(img_np=inp_img, path=inp_path)

            cf_img = cf if cf.shape[-1] != 1 else cf.squeeze(-1)
            cf_path = os.path.join(
                sub_dirs["cf"], f"cf_{i + total_saved}_{cf_suffix}.png"
            )
            save_image_auto_mode(img_np=cf_img, path=cf_path)

            # ---- Metadata entry for CF ----
            cf_entry = {
                key: cf_batch[key][i].item()
                if key not in ["race", "scanner"]
                else torch.argmax(cf_batch[key][i]).item()
                for key in cf_batch
                if key != "pa"
            }

            # ---- Build titles ----
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Original image + title
            axes[0].imshow(inp_img, cmap="gray" if inp_img.ndim == 2 else None)
            axes[0].axis("off")

            if "mimic" in args.hps:
                axes[0].set_title(
                    f"age={100 * batch['age'][i].item():.1f}, "
                    f"sex={sex_categories[int(batch['sex'][i].item())]}, "
                    f"race={race_categories[int(torch.argmax(batch['race'][i]).item())]}, "
                    f"finding={finding_categories[int(batch['finding'][i].item())]}"
                )
            elif "embed" in args.hps:
                axes[0].set_title(
                    f"density={int(batch['density'][i].item())}, "
                    f"circle={int(batch['circle'][i].item())}"
                )
            elif "celeba" in args.hps:
                axes[0].set_title(
                    ", ".join(f"{key}={int(batch[key][i].item())}" for key in args.parents)
                )
                    
            # CF title showing interventions
            cf_titles = []
            for key in do_keys:
                val = cf_batch[key][i]
                if key == "sex":
                    cf_titles.append(f"s = {sex_categories[int(val.item())]}")
                elif key == "race":
                    cf_titles.append(
                        f"r = {race_categories[int(torch.argmax(val).item())]}"
                    )
                elif key == "finding":
                    cf_titles.append(
                        f"f = {finding_categories[int(val.item())]}"
                    )
                elif key == "age":
                    cf_titles.append(f"age = {100 * val.item():.1f}")
                else:
                    abbrev = "".join(w[0] for w in key.lower().split("_"))
                    cf_titles.append(f"{abbrev} = {int(val.item())}")

            axes[1].imshow(cf_img, cmap="gray" if cf_img.ndim == 2 else None)
            axes[1].axis("off")
            axes[1].set_title("do(" + ", ".join(cf_titles) + ")")

            # Difference plot
            diff_for_plot = (
                diff.mean(axis=-1)
                if diff.ndim == 3 and diff.shape[-1] == 3
                else diff.squeeze(-1)
            )
            vmin, vmax = diff_for_plot.min(), diff_for_plot.max()
            if abs(vmax - vmin) < 1e-6:
                vmin -= 1
                vmax += 1
            norm = MidpointNormalize(vmin=vmin, midpoint=0, vmax=vmax)
            im = axes[2].imshow(diff_for_plot, cmap="RdBu_r", norm=norm)
            axes[2].axis("off")
            axes[2].set_title("Difference")

            cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
            cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label("Difference (CF - Input)", fontsize=12)

            vis_path = os.path.join(
                sub_dirs["cf_vis"], f"cf_visual_{i + total_saved}_{cf_suffix}.png"
            )
            plt.savefig(vis_path, dpi=300, bbox_inches="tight")
            plt.close()

            cf_entry["file_path"] = cf_path
            cf_entry["input_path"] = inp_path
            cf_csv_data.append(cf_entry)

        # Write CF CSV for this batch
        pd.DataFrame(cf_csv_data).to_csv(
            cf_csv_file, mode="a", header=write_cf_header, index=False
        )
        write_cf_header = False

        # ==========================
        # 6) Optional reverse CF (CF -> original attributes)
        # ==========================
        if reverse_cf:
            rev_kwargs = {}

            # For reverse from CFs, condition on CF parents first
            if cond_embedder is not None:
                rev_kwargs["cemb"] = cond_embedder(cf_batch)
            if sem_encoder is not None:
                if getattr(args, "use_vae_encoder", False):
                    mu_rev, _ = sem_encoder(counterfactuals)
                    semb_rev = mu_rev
                else:
                    semb_rev = sem_encoder(counterfactuals)
                rev_kwargs["semb"] = semb_rev

            # CF -> noise (reverse)
            #   - GaussianDiffusion:       uncond reverse
            #   - GaussianDiffusion_cfgpp: CFG++ inversion reverse
            #   - GaussianDiffusion_apg:   uncond reverse
            reverse_noise = diffusion.ddim_reverse_sample_loop(
                model,
                counterfactuals,
                clip_denoised=CLIP_DENOISED,
                denoised_fn=None,
                model_kwargs=rev_kwargs,
                device=args.device,
                progress=False,
                eta=eta,
            )

            # Now recondition on ORIGINAL parents for forward reverse_x
            if cond_embedder is not None:
                rev_kwargs["cemb"] = cond_embedder(batch)
            if sem_encoder is not None:
                if getattr(args, "use_vae_encoder", False):
                    mu_orig, _ = sem_encoder(batch["x"])
                    semb_orig = mu_orig
                else:
                    semb_orig = sem_encoder(batch["x"])
                rev_kwargs["semb"] = semb_orig

            # Guided forward reverse_x:
            #   - GaussianDiffusion:       CFG
            #   - GaussianDiffusion_cfgpp: CFG++
            #   - GaussianDiffusion_apg:   APG
            reversed_x = diffusion.ddim_sample_loop(
                model,
                counterfactuals.shape,
                noise=reverse_noise,
                clip_denoised=CLIP_DENOISED,
                denoised_fn=None,
                model_kwargs=rev_kwargs,
                device=args.device,
                progress=False,
                eta=eta,
                use_cfg=use_cfg,
                cfg_kwargs=cfg_kwargs,
            )

            reversed_np = (
                ((reversed_x + 1) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            rev_diff_np = reversed_np.astype(float) - input_np.astype(float)

            for i, (rev_img, inp_img, diff_img) in enumerate(
                zip(reversed_np, input_np, rev_diff_np)
            ):
                rev_img_disp = rev_img.squeeze(-1) if rev_img.shape[-1] == 1 else rev_img
                rev_path = os.path.join(
                    sub_dirs["rev"], f"reversed_{i + total_saved}_{cf_suffix}.png"
                )
                save_image_auto_mode(img_np=rev_img_disp, path=rev_path)

                rev_entry = {
                    key: batch[key][i].item()
                    if key not in ["race", "scanner"]
                    else torch.argmax(batch[key][i]).item()
                    for key in batch
                    if key in args.parents and key != "pa"
                }

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(inp_img, cmap="gray" if inp_img.ndim == 2 else None)
                axes[0].axis("off")
                axes[0].set_title("Original")

                axes[1].imshow(rev_img_disp, cmap="gray" if rev_img_disp.ndim == 2 else None)
                axes[1].axis("off")
                axes[1].set_title("Reversed")

                diff_for_plot = (
                    diff_img.mean(axis=-1)
                    if diff_img.ndim == 3 and diff_img.shape[-1] == 3
                    else squeeze_if_gray(diff_img)
                    if diff_img.ndim == 3
                    else diff_img
                )
                vmin, vmax = diff_for_plot.min(), diff_for_plot.max()
                if abs(vmax - vmin) < 1e-6:
                    vmin -= 1
                    vmax += 1
                norm = MidpointNormalize(vmin=vmin, midpoint=0, vmax=vmax)
                im = axes[2].imshow(diff_for_plot, cmap="RdBu_r", norm=norm)
                axes[2].axis("off")
                axes[2].set_title("Difference")

                cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
                cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal")
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label("Difference (Reversed - Input)", fontsize=12)

                vis_path = os.path.join(
                    sub_dirs["rev_vis"],
                    f"reversed_visual_{i + total_saved}_{cf_suffix}.png",
                )
                plt.savefig(vis_path, dpi=300, bbox_inches="tight")
                plt.close()

                rev_entry["file_path"] = rev_path
                rev_entry["input_path"] = os.path.join(
                    sub_dirs["input"], f"input_{i + total_saved}_{cf_suffix}.png"
                )
                rev_csv_data.append(rev_entry)

            pd.DataFrame(rev_csv_data).to_csv(
                rev_csv_file, mode="a", header=write_rev_header, index=False
            )
            write_rev_header = False

        total_saved += len(input_np)
        progress_bar.update(len(input_np))

        if total_saved >= args.num_samples:
            break

    progress_bar.close()
    print(f"Saved {total_saved} counterfactual images to {save_dir}")
    print(
        f"Metadata saved to {cf_csv_file}"
        + (f" and {rev_csv_file}" if reverse_cf else "")
    )
