import argparse
import ast
import os
import sys
from project_paths import setup_import_paths
setup_import_paths()

import torch
import numpy as np
from pprint import pprint
from hps import setup_hparams, add_arguments
from utils import (
    init_bias, 
    init_weights,
    seed_all, 
    load_model,  
    save_diffusion, 
    load_diffusion, 
    create_gaussian_diffusion, 
    create_named_schedule_sampler,
    save_schedule_sampler, 
    load_schedule_sampler,
)

from nn_utils import EMA
from train_setup import (
    setup_directories,
    setup_tensorboard,
    setup_optimizer, 
    setup_dataloaders,
    setup_logging
)
from unet import UNetModel
from trainer import Trainer
from semantic_encoder import SemanticEncoder, VAESemanticEncoder
from cond_embed import CondEmbedder, GlobalCondEmbedder

def main():
    # Hparams
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = setup_hparams(parser)
    # Convert the string argument to a proper dictionary
    if isinstance(args.parent_dims, str):
        args.parent_dims = ast.literal_eval(args.parent_dims)
        
    if not args.use_semantic_encoder:
        args.sem_embed_dim=0 
        args.kl_weight=0.
        args.kl_free_bits=0.
    elif not args.use_vae_encoder:
        args.kl_weight=0.
        args.kl_free_bits=0.

    if not args.cfg_train:
        args.cfg_dropout=None
        args.cfg_w=1.0

    seed_all(args.seed, args.deterministic)
    # update hyperparams if resuming from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"\nLoading checkpoint: {args.resume}")
            ckpt = torch.load(args.resume,weights_only=False)
            ckpt_args = {k: v for k, v in vars(ckpt["hparams"]).items() if k != "resume"}
            if args.lr < ckpt_args["lr"]:
                ckpt_args["lr"] = args.lr
            ckpt_args["wd"] = args.wd
            vars(args).update(ckpt_args)
        else:
            print(f"Checkpoint not found at: {args.resume}")
    # load data
    dataloaders = setup_dataloaders(args, balance_train=True)
    # init model
    model = UNetModel(args) 
    model.apply(init_bias)
    ema=EMA(model, beta=args.ema_rate)
    # init diffusion
    diffusion = create_gaussian_diffusion(args)
    # init diffusion schedule sampler
    schedule_sampler = create_named_schedule_sampler(args.sampler_name, diffusion.base_diffusion)
    # setup model save directory and logger
    assert args.exp_name != "", "No experiment name given."
    args.save_dir = setup_directories(args)
    # writer = setup_tensorboard(args, model)  # Do not use tensorboard for now.
    logger = setup_logging(args)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)
    model.to(args.device)
    ema.to(args.device)

    # === Conditional Embedder ===
    cond_embedder = GlobalCondEmbedder(args) if args.global_cond else CondEmbedder(args)
    cond_embedder.apply(init_bias)
    ema_cond_embedder = EMA(cond_embedder, beta=args.ema_rate)
    cond_embedder.to(args.device)
    ema_cond_embedder.to(args.device)
    # === Semantic Encoder ===
    if args.use_semantic_encoder:
        sem_encoder = VAESemanticEncoder(args) if args.use_vae_encoder else SemanticEncoder(args)
        sem_encoder.apply(init_weights)
        ema_sem_encoder = EMA(sem_encoder, beta=args.ema_rate)
        sem_encoder.to(args.device)
        ema_sem_encoder.to(args.device)
    else:
        sem_encoder = None
        ema_sem_encoder = None
    # setup optimizer
    optimizer, scheduler = setup_optimizer(args, model, cond_embedder, sem_encoder)

    # load checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            args.start_epoch = ckpt["epoch"]
            args.iter = ckpt["step"]
            args.best_loss = ckpt["best_loss"]
            model.load_state_dict(ckpt["model_state_dict"])
            ema.ema_model.load_state_dict(ckpt["ema_model_state_dict"])
            cond_embedder.load_state_dict(ckpt["cond_embedder_state_dict"])
            ema_cond_embedder.ema_model.load_state_dict(ckpt["ema_cond_embedder_state_dict"])
            if "sem_encoder_state_dict" in ckpt and sem_encoder is not None:
                sem_encoder.load_state_dict(ckpt["sem_encoder_state_dict"])
            if "ema_sem_encoder_state_dict" in ckpt and ema_sem_encoder is not None:
                ema_sem_encoder.ema_model.load_state_dict(ckpt["ema_sem_encoder_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            diffusion = load_diffusion(ckpt["diffusion"])
            schedule_sampler = load_schedule_sampler(ckpt["schedule_sampler"], diffusion=diffusion.base_diffusion)
            # scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            # update lr of the loaded optimizer
            for g in optimizer.param_groups:
                g["lr"] = args.lr
                g["wd"] = args.wd # Reset wd of the loaded optimizer
                g["initial_lr"] = args.lr  # needed to init the scheduler lr
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: x * 0 + 1
            )
            print(f"Load checkpoint from {args.resume}")
        else:
            print("Checkpoint not found at: {}".format(args.resume))
    else:
        args.start_epoch, args.iter = 0, 0
        args.best_loss = float("inf")

    trainer = Trainer(
        args=args,
        model=model,
        ema=ema,
        cond_embedder=cond_embedder,
        ema_cond_embedder=ema_cond_embedder,
        sem_encoder=sem_encoder,
        ema_sem_encoder=ema_sem_encoder,
        diffusion=diffusion,
        schedule_sampler=schedule_sampler,
        dataloaders=dataloaders,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
    )
    
    stats=trainer.run()

if __name__=="__main__":
    main()
