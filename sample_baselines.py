# Sample code
import ast
import argparse
import os
import json
import sys
import re


from project_paths import setup_import_paths
setup_import_paths()

import torch
from hps import add_sample_arguments
from utils_baselines import (
    seed_all,
    create_gaussian_diffusion,
)

from nn_utils import EMA
from train_setup import (
    setup_results_directories_baselines,
    setup_dataloaders,
    setup_logging,
)
from unet import UNetModel
from sampling_utils import (
    generate_random_samples,
    reconstruct_images,
    generate_counterfactuals,
)
from semantic_encoder import SemanticEncoder, VAESemanticEncoder
from cond_embed import CondEmbedder, GlobalCondEmbedder


def build_cfg_kwargs(args):
    """
    Build cfg_kwargs dictionary from CLI / hparams for conditional sampling.

    For CFG / CFG++:
      - cfg_mode, cfg_choice, cfg_w, cfg_attr_weights, cfg_w_intervened, cfg_w_unintervened

    For APG:
      - uses cfg_w as guidance scale w (same as CFG)
      - additionally uses:
          apg_eta           (η, balance between parallel / orthogonal)
          apg_norm_threshold (r, norm clipping radius; 0 => off)
          apg_momentum      (β, reverse momentum strength; buffer created elsewhere)
    """
    cfg_mode = getattr(args, "cfg_mode", "global")

    cfg_kwargs = {
        "cfg_mode": cfg_mode,
        "cfg_choice": getattr(args, "cfg_choice", "eq1"),
        "cfg_w": None,
        "cfg_attr_weights": None,
        "cfg_w_intervened": None,
        "cfg_w_unintervened": None,
    }

    if cfg_mode == "global":
        cfg_kwargs["cfg_w"] = getattr(args, "cfg_w", None)
    elif cfg_mode == "per_attr":
        cfg_kwargs["cfg_attr_weights"] = getattr(args, "cfg_attr_weights", None)
    elif cfg_mode == "interv_group":
        cfg_kwargs["cfg_w_intervened"] = getattr(args, "cfg_w_intervened", None)
        cfg_kwargs["cfg_w_unintervened"] = getattr(args, "cfg_w_unintervened", None)

    # 🔹 APG hyper-params (no separate APG guidance scale; APG uses cfg_w)
    cfg_kwargs.update(
        dict(
            apg_eta=getattr(args, "apg_eta", 1.0),
            apg_norm_threshold=getattr(args, "apg_norm_threshold", 0.0),
            apg_momentum=getattr(args, "apg_momentum", 0.0),
            # apg_momentum_buffer will be created per sampling chain (if used)
        )
    )

    return cfg_kwargs


def sample():
    # -----------------------------
    # 1) Parse CLI arguments
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser = add_sample_arguments(parser)
    sample_args = parser.parse_known_args()[0]

    print(f"cfg_attr_weights (raw): {sample_args.cfg_attr_weights}, type: {type(sample_args.cfg_attr_weights)}")

    # Parse cfg_attr_weights string into dict if needed
    if isinstance(sample_args.cfg_attr_weights, str) and sample_args.cfg_attr_weights != "":
        try:
            s = sample_args.cfg_attr_weights.strip()

            # Remove outer quotes if present
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]

            # Fix unquoted keys: {sex: 1.2, race: 0.8} -> {"sex": 1.2, "race": 0.8}
            s_fixed = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z_0-9]*)(\s*:\s*)', r'\1"\2"\3', s)
            sample_args.cfg_attr_weights = ast.literal_eval(s_fixed)

            if not isinstance(sample_args.cfg_attr_weights, dict):
                raise ValueError("Parsed cfg_attr_weights is not a dict.")

            print(f"✅ cfg_attr_weights (parsed): {sample_args.cfg_attr_weights}, type: {type(sample_args.cfg_attr_weights)}")

        except Exception as e:
            print(f"❌ Failed to parse cfg_attr_weights: {e}")
            raise ValueError("Invalid cfg_attr_weights string format.")

    print(f"cfg_attr_weights (parsed): {sample_args.cfg_attr_weights}, type: {type(sample_args.cfg_attr_weights)}")

    # -----------------------------
    # 2) Load checkpoint + hparams
    # -----------------------------
    if not os.path.isfile(sample_args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {sample_args.ckpt_path}")
    print(f"\nLoading saved checkpoint: {sample_args.ckpt_path}")
    ckpt = torch.load(sample_args.ckpt_path, weights_only=False, map_location="cpu")
    ckpt_args = ckpt["hparams"]

    # Overwrite training hparams with sampling-specific CLI args
    for k, v in vars(sample_args).items():
        setattr(ckpt_args, k, v)

    if sample_args.cfg_mode in {"per_attr", "interv_group"} and ckpt_args.global_cond:
        raise ValueError("Factored guidance requires an attribute-split checkpoint (global_cond=False).")

    # During inference, we do not do cfg dropout
    ckpt_args.cfg_train = False
    ckpt_args.cfg_dropout = 0.0

    # -----------------------------
    # 3) Build cfg_kwargs (for CFG / CFG++ / APG)
    # -----------------------------
    cfg_kwargs = build_cfg_kwargs(sample_args)
    vars(ckpt_args).update(cfg_kwargs)
    vars(sample_args).update(cfg_kwargs)

    for k, v in cfg_kwargs.items():
        print(f"cfg_kwargs {k}: {v}")

    # -----------------------------
    # 4) Seed + data
    # -----------------------------
    seed_all(ckpt_args.seed, ckpt_args.deterministic)
    dataloaders = setup_dataloaders(ckpt_args, balance_test=True)

    # -----------------------------
    # 5) Model + EMA
    # -----------------------------
    model = UNetModel(ckpt_args)
    ema = EMA(model, beta=ckpt_args.ema_rate)
    model.load_state_dict(ckpt["model_state_dict"])
    ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])

    # -----------------------------
    # 6) Diffusion (CFG / CFG++ / APG decided inside create_gaussian_diffusion)
    # -----------------------------
    diffusion = create_gaussian_diffusion(ckpt_args)

    # -----------------------------
    # 7) Conditional embedder (+ EMA)
    # -----------------------------
    cond_embedder = GlobalCondEmbedder(ckpt_args) if ckpt_args.global_cond else CondEmbedder(ckpt_args)
    cond_embedder.load_state_dict(ckpt["cond_embedder_state_dict"])
    ema_cond_embedder = EMA(cond_embedder, beta=ckpt_args.ema_rate)
    ema_cond_embedder.ema_model.load_state_dict(ckpt["ema_cond_embedder_state_dict"])

    # -----------------------------
    # 8) Semantic encoder (optional) + EMA
    # -----------------------------
    if ckpt_args.use_semantic_encoder:
        sem_encoder = VAESemanticEncoder(ckpt_args) if ckpt_args.use_vae_encoder else SemanticEncoder(ckpt_args)
        sem_encoder.load_state_dict(ckpt["sem_encoder_state_dict"])
        ema_sem_encoder = EMA(sem_encoder, beta=ckpt_args.ema_rate)
        ema_sem_encoder.ema_model.load_state_dict(ckpt["ema_sem_encoder_state_dict"])
    else:
        sem_encoder, ema_sem_encoder = None, None

    # -----------------------------
    # 9) Device setup
    # -----------------------------
    ckpt_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(ckpt_args.device).eval()
    ema.to(ckpt_args.device).eval()
    cond_embedder.to(ckpt_args.device).eval()
    ema_cond_embedder.to(ckpt_args.device).eval()
    if sem_encoder:
        sem_encoder.to(ckpt_args.device).eval()
        ema_sem_encoder.to(ckpt_args.device).eval()

    # -----------------------------
    # 10) Logging + save dir
    # -----------------------------
    assert ckpt_args.exp_name != "", "No experiment name given."
    ckpt_args.save_dir = setup_results_directories_baselines(
        ckpt_args, results_dir=sample_args.results_dir
    )
    logger = setup_logging(ckpt_args, log_file="sample_log.txt")

    logger.info("----- The training paras are -----")
    for k in sorted(vars(ckpt_args)):
        if k not in ["diffusion_betas"]:
            logger.info(f"--{k}={vars(ckpt_args)[k]}")

    logger.info("----- Sampling with the following params -----")
    for k in sorted(vars(sample_args)):
        if k not in ["diffusion_betas"]:
            logger.info(f"--{k}={vars(sample_args)[k]}")

    # -----------------------------
    # 11) Inference
    # -----------------------------
    with torch.no_grad():
        # ----------------- Random sampling -----------------
        if "rs" in ckpt_args.sample_types:
            generate_random_samples(
                diffusion=diffusion,
                model=ema.ema_model,
                args=ckpt_args,
                save_path=os.path.join(ckpt_args.save_dir, "rs"),
                loader=dataloaders["test"],
                use_ddim="ddim" in ckpt_args.timestep_respacing,
                cond_embedder=ema_cond_embedder.ema_model,
                sem_encoder=ema_sem_encoder.ema_model if ckpt_args.use_semantic_encoder else None,
                use_cfg=True,        # interpreted as CFG / CFG++ / APG depending on diffusion
                cfg_kwargs=cfg_kwargs,
            )
            logger.info("Random sample generation is done.")

        # ----------------- Reconstructions -----------------
        if "rec" in ckpt_args.sample_types:
            # with guidance (CFG / CFG++ / APG)
            reconstruct_images(
                args=ckpt_args,
                model=ema.ema_model,
                diffusion=diffusion,
                save_dir=os.path.join(ckpt_args.save_dir, "rec_cfg"),
                loader=dataloaders["test"],
                eta=0.0,
                use_cfg=True,
                cond_embedder=ema_cond_embedder.ema_model,
                sem_encoder=ema_sem_encoder.ema_model if ckpt_args.use_semantic_encoder else None,
                cfg_kwargs=cfg_kwargs,
            )
            logger.info("Reconstruction generation cfg is done.")

            # without guidance (plain reverse + forward DDIM)
            reconstruct_images(
                args=ckpt_args,
                model=ema.ema_model,
                diffusion=diffusion,
                save_dir=os.path.join(ckpt_args.save_dir, "rec_no_cfg"),
                loader=dataloaders["test"],
                eta=0.0,
                use_cfg=False,
                cond_embedder=ema_cond_embedder.ema_model,
                sem_encoder=ema_sem_encoder.ema_model if ckpt_args.use_semantic_encoder else None,
                cfg_kwargs=cfg_kwargs,
            )
            logger.info("Reconstruction generation no cfg is done.")

        # ----------------- Counterfactuals -----------------
        if "cf" in ckpt_args.sample_types:
            if len(ckpt_args.sample_do_keys) > 0:
                for key_str in ckpt_args.sample_do_keys:
                    do_keys, remaining = [], key_str
                    while remaining:
                        key = next((k for k in sorted(ckpt_args.parents, key=len, reverse=True)
                                    if remaining == k or remaining.startswith(k + "_")), None)
                        if key is None:
                            raise ValueError(f"Unknown intervention {key_str!r}; parents: {ckpt_args.parents}")
                        do_keys.append(key)
                        remaining = remaining[len(key):].lstrip("_")
                    do_keys_str = "_".join(do_keys)

                    # Build a fresh cfg_kwargs per do_keys if needed
                    cfg_kwargs_this = build_cfg_kwargs(sample_args)
                    cfg_kwargs_this["cfg_intervened_keys"] = do_keys

                    generate_counterfactuals(
                        args=ckpt_args,
                        model=ema.ema_model,
                        diffusion=diffusion,
                        save_dir=os.path.join(ckpt_args.save_dir, f"cfs_{do_keys_str}"),
                        loader=dataloaders["test"],
                        eta=0.0,
                        do_keys=do_keys,
                        cond_embedder=ema_cond_embedder.ema_model,
                        sem_encoder=ema_sem_encoder.ema_model if ckpt_args.use_semantic_encoder else None,
                        cfg_kwargs=cfg_kwargs_this,
                        reverse_cf=True,   # CF -> approx original
                        use_cfg=True,      # guided forward passes
                    )
                    logger.info(f"Counterfactual generation for {do_keys_str} is done.")


if __name__ == "__main__":
    sample()
