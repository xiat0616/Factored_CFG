import functools
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from resample import LossAwareSampler
from plot_util import sample_images_single_batch, reconstruct_images_single_batch
from utils import save_diffusion, save_schedule_sampler
from losses import kl_divergence_latent, compute_hsic_loss, compute_hsic_norm_loss
from nn_utils import sample_from_latent
import torch.nn.functional as F

def cfg_drop(cemb, cfg_dropout=0.1):
    if cemb is None or cfg_dropout == 0:
        return cemb # Nothing to do
    mask = (torch.rand(cemb.size(0), device=cemb.device)>=cfg_dropout).float()
    return cemb * mask[:, None]

def cfg_drop_together(cemb, semb, cfg_dropout=0.1):
    if (cemb is None and semb is None) or cfg_dropout == 0:
        return cemb, semb  # Nothing to do

    # Pick whichever tensor exists to determine batch size and device
    ref_tensor = cemb if cemb is not None else semb

    # Generate binary mask: 1 = keep, 0 = drop
    mask = (torch.rand(ref_tensor.size(0), device=ref_tensor.device) >= cfg_dropout).float()

    if cemb is not None:
        cemb = cemb * mask[:, None]
    if semb is not None:
        semb = semb * mask[:, None]

    return cemb, semb


class Trainer:
    def __init__(
        self,
        *,
        args, 
        model, 
        ema, 
        cond_embedder,
        ema_cond_embedder,
        sem_encoder,
        ema_sem_encoder,
        diffusion,
        schedule_sampler,
        dataloaders, 
        optimizer, 
        scheduler, 
        logger,
    ):  
        # Hparams
        self.args = args
        # Model
        self.model = model
        self.ema = ema
        self.cond_embedder = cond_embedder
        self.ema_cond_embedder = ema_cond_embedder
        self.sem_encoder = sem_encoder
        self.ema_sem_encoder = ema_sem_encoder
        # Diffusion
        self.diffusion = diffusion.base_diffusion # This is the diffusion we train
        self.sample_diffusion = diffusion   # This is the diffusion for plot sampling
        self.schedule_sampler = schedule_sampler 
        # Data
        self.dataloaders = dataloaders
        # Training
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Other params
        self.logger = logger
        # self.microbatch = self.args.microbatch if self.args.microbatch > 0 else batch_size

        # Directory for image samples
        self.image_save_dir = os.path.join(self.args.save_dir, "images")
        os.makedirs(self.image_save_dir, exist_ok=True)

    def run(self):
        # === Initialization logging ===
        for k in sorted(vars(self.args)):
            if k not in ["diffusion_betas"]:
                self.logger.info(f"--{k}={vars(self.args)[k]}")
        self.logger.info(f"total params: {sum(p.numel() for p in self.model.parameters()):,}")
        # === Main training loop ===
        for epoch in range(self.args.start_epoch, self.args.epochs):
            self.args.epoch=epoch+1
            # Run one training epoch
            stats = self.run_epoch(dataloader=self.dataloaders['train'], training=True)
            self.logger.info(            
                f'=> train | loss: {stats["loss"]:.4f}'
                + f" - steps: {self.args.iter}"
                + f" --epoch: {epoch}"
            )

            # === Periodic validation ===
            if (self.args.epoch -1) % self.args.eval_freq ==0:
                valid_stats = self.run_epoch(self.dataloaders["valid"], training=False)
                self.logger.info(
                    f'=> valid | loss: {valid_stats["loss"]:.4f}'
                    + f" - steps: {self.args.iter}"
                )
                # Save checkpoint if validation improves
                if valid_stats["loss"] < self.args.best_loss:
                    self.args.best_loss = valid_stats["loss"]
                    self.logger.info(f"current best valid loss: {self.args.best_loss}")
                    ckpt_path = os.path.join(self.args.save_dir, "checkpoint.pt")
                    self.save(ckpt_path=ckpt_path)
                    self.logger.info(f"Model saved: {ckpt_path}")

            # Always save latest checkpoint
            ckpt_path = os.path.join(self.args.save_dir, "last.pt")
            self.save(ckpt_path=ckpt_path)
            self.logger.info(f"Model saved: {ckpt_path}")
        return stats

    def run_epoch(self, dataloader, training=True):
        self.model.train(training)
        self.model.zero_grad(set_to_none=True)
        stats = {k: 0 for k in ["loss", "kl_loss", "loss_q0", "loss_q1", "loss_q2", "loss_q3",
                                "mse", "mse_q0", "mse_q1", "mse_q2", "mse_q3"]}
        # === Initialize statistics for loss tracking ===
        stats_n ={} 
        for k in stats.keys():
            stats_n[k+"_n"] = 0
        
        updates_skipped = 0
        # Set loader
        mininterval = 0.1
        loader = tqdm(
            enumerate(dataloader), total=len(dataloader), mininterval=mininterval
        )

        for i, batch in loader:     
            bs = batch["x"].shape[0]
            update_stats = True                  
            # === Forward/backward training step ===
            if training:
                self.args.iter = i + 1 + (self.args.epoch - 1) * len(dataloader)
                out = self.forward_backward(
                    model=self.model, 
                    batch=batch, 
                    cond_embedder=self.cond_embedder,
                    sem_encoder=self.sem_encoder,
                    cfg_train=self.args.cfg_train,
                )
                out["loss"] = out["loss"] / self.args.accu_steps
                out["loss"].backward()
                
                # === Gradient update with clipping and skipping ===
                if i % self.args.accu_steps == 0:  
                    params = list(self.model.parameters())
                    if self.cond_embedder is not None:
                        params += list(self.cond_embedder.parameters())
                    if self.sem_encoder is not None:
                        params += list(self.sem_encoder.parameters())

                    grad_norm = nn.utils.clip_grad_norm_(
                        params, self.args.grad_clip
                    )
                    
                    loss_nan = torch.isnan(out["loss"]).sum()

                    if (grad_norm < self.args.grad_skip and loss_nan==0) or self.args.iter<500: # Need this as early grad_norm very unstable
                        self.optimizer.step()
                        self.scheduler.step()
                        self.ema.update()
                        if self.ema_sem_encoder is not None:
                            self.ema_sem_encoder.update()
                        if self.ema_cond_embedder is not None:
                            self.ema_cond_embedder.update()
                    else:
                        updates_skipped += 1
                        update_stats = False
                        self.logger.info(
                            f"Updates skipped: {updates_skipped}"
                            + f" - grad_norm: {grad_norm:.3f}"
                            + f" - loss_nan: {loss_nan.item()}"
                            )
                    self.optimizer.zero_grad()
                    self.model.zero_grad(set_to_none=True) # Only reset after optimizer step
                    self.cond_embedder.zero_grad(set_to_none=True) # Only reset after optimizer step
                    if self.sem_encoder is not None:
                        self.sem_encoder.zero_grad(set_to_none=True) # Only reset after optimizer step
                # === Visualize sampled and reconstructed images ===
                if (
                    ((self.args.iter<self.args.plt_freq) and (self.args.iter % 1000==0)) 
                    or (self.args.iter % self.args.plt_freq == 0)
                ):  
                    with torch.no_grad():
                        self.ema.ema_model.train(False)    
                        self.ema_cond_embedder.ema_model.train(False)
                        if self.ema_sem_encoder is not None:
                            self.ema_sem_encoder.ema_model.train(False)  
                        self.process_and_save_images(
                            model=self.ema.ema_model,
                            batch=batch,
                            cond_embedder=self.ema_cond_embedder.ema_model,
                            sem_encoder=self.ema_sem_encoder.ema_model if self.ema_sem_encoder is not None else None,
                        )
            # === EMA-based forward pass for logging and validation ===     
            with torch.no_grad():
                self.ema.ema_model.train(False)
                self.ema_cond_embedder.ema_model.train(False)
                if self.ema_sem_encoder is not None:
                    self.ema_sem_encoder.ema_model.train(False) 
                # Obtain new out for stats and loader description. 
                out = self.forward_backward(
                    model=self.ema.ema_model, 
                    batch=batch,
                    sem_encoder=self.ema_sem_encoder.ema_model if self.ema_sem_encoder is not None else None,
                    cond_embedder=self.ema_cond_embedder.ema_model,
                    cfg_train=self.args.cfg_train,
                ) 
            if update_stats:
                if training:
                    out["loss"] *= self.args.accu_steps

            for k in out.keys():
                if k not in stats.keys():
                    stats[k], stats_n[k+"_n"]=0, 0
                stats_n[k+"_n"] +=bs
                try:
                    stats[k]+=out[k].detach().item() * bs
                except:
                    stats[k]+=out[k] * bs

            split = "train" if training else "valid"

            loader.set_description(
                f' => {split} | loss: {stats["loss"] / (1e-8 + stats_n["loss_n"]):.5f}'
                + f' - kl: {stats["kl_loss"] / (1e-8 + stats_n["kl_loss_n"]):.4f}'
                + f' - mse: {stats["mse"] / (1e-8 + stats_n["mse_n"]):.4f}'
                + f' - loss_q0: {stats["loss_q0"] / (1e-8 + stats_n["loss_q0_n"]):.3f}'
                + f' - loss_q1: {stats["loss_q1"] / (1e-8 + stats_n["loss_q1_n"]):.3f}'
                + f' - loss_q2: {stats["loss_q2"] / (1e-8 + stats_n["loss_q2_n"]):.3f}'
                + f' - loss_q3: {stats["loss_q3"] / (1e-8 + stats_n["loss_q3_n"]):.3f}'
                + f" - lr: {self.scheduler.get_last_lr()[0]:.6g}"
                + (f" - grad norm: {grad_norm:.2f}" if training else ""),
                refresh=False,
            )

        return {k: v /(1e-5+stats_n[k+"_n"]) for k, v in stats.items() }

    def save(self, ckpt_path):
        save_dict = {
            "epoch": self.args.epoch,
            "step": self.args.epoch * len(self.dataloaders['train']),
            "best_loss": self.args.best_loss,
            "hparams": self.args,
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema.ema_model.state_dict(),
            "cond_embedder_state_dict": self.cond_embedder.state_dict(),
            "ema_cond_embedder_state_dict": self.ema_cond_embedder.ema_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "diffusion": save_diffusion(self.sample_diffusion),
            "schedule_sampler": save_schedule_sampler(self.schedule_sampler),
        }

        # Only save sem_encoder state dicts if present
        if self.sem_encoder is not None:
            save_dict["sem_encoder_state_dict"] = self.sem_encoder.state_dict()
        if self.ema_sem_encoder is not None:
            save_dict["ema_sem_encoder_state_dict"] = self.ema_sem_encoder.ema_model.state_dict()

        torch.save(save_dict, ckpt_path)
        
    def process_and_save_images(
            self, 
            model, 
            batch, 
            cond_embedder=None, 
            sem_encoder=None
    ):
        """
        Generate, visualize, and save samples and reconstructions using the current model and input batch.

        Args:
            model: The model to use (typically EMA version)
            batch: A dict containing input image tensor under 'x' and conditioning labels
            cond_embedder: Module to produce conditional embeddings (e.g., from parent labels)
            sem_encoder: Module to produce semantic (identity) embeddings
        """
        image = batch["x"].to(self.args.device)
        # === Conditional Embedding ===
        if cond_embedder is not None:
            y_dict = {k: v.to(self.args.device) for k, v in batch.items() if k in self.args.parents}
            cemb = cond_embedder(y_dict)
        else:
            cemb = None
        
        # === Semantic Embedding ===
        if sem_encoder is not None:
            if self.args.use_vae_encoder:
                mu, logvar = sem_encoder(image)
                semb = sample_from_latent(mu=mu, logvar=logvar)
            else:
                semb = sem_encoder(image)
                mu = semb
        else:
            semb = None
            mu = semb
        # === Sampling & Reconstruction Directories ===
        ddim_dir = os.path.join(self.image_save_dir, "ddim")
        ddpm_dir = os.path.join(self.image_save_dir, "ddpm")
        rec_dir = os.path.join(self.image_save_dir, "ddim_rec")
        # Call sample_images with DDIM
        print(f"DDIM sampling")
        self.sample_images(
            model=model,
            diffusion=self.sample_diffusion,
            image_save_dir=ddim_dir,
            image=image,
            cemb=cemb,
            semb=semb,
            use_ddim=True,
        )

        # Call sample_images with DDPM
        print(f"DDPM sampling")
        self.sample_images(
            model=model,
            diffusion=self.diffusion,
            image_save_dir=ddpm_dir,
            image=image,
            cemb=cemb,
            semb=semb,
            use_ddim=False,
        )
        # Call reconstruct_images
        print(f"DDIM rec sampling")
        self.reconstruct_images(
            model=model,
            diffusion=self.sample_diffusion,
            image_save_dir=rec_dir,
            image=image,
            cemb=cemb,
            semb=mu,
            eta=0.0,  # We can adjust eta as needed
        )

    def sample_images(self, model, diffusion, image_save_dir, image, cemb, semb=None, use_ddim=False):
        """
        Generate and save example images from the current batch with optional spaced sampling.
        """
        batch = {"x": image, "cemb": cemb}
        if semb is not None:
            # Sample semantic embedding from standard normal
            sampled_semb = torch.randn_like(semb)
            batch["semb"] = sampled_semb

        sample_images_single_batch(
            args=self.args,
            model=model,
            diffusion=diffusion,
            step=self.args.iter,
            batch=batch,
            save_dir=image_save_dir,
            use_ddim=use_ddim,
            logger=self.logger,
        )
    
    def reconstruct_images(self, model, diffusion, image_save_dir, image, cemb, semb=None, eta=0.0):
        """
        Perform reconstruction of input images using DDIM reverse and forward sampling.

        Args:
            image_save_dir (str): Directory to save the reconstructed images.
            image (torch.Tensor): Batch of input images.
            cond (torch.Tensor): Conditional labels or other inputs.
            eta (float): Noise scale for DDIM reverse sampling. 0.0 for deterministic reconstruction.
        """
        batch = {"x": image, "cemb": cemb}
        if semb is not None:
            batch["semb"] = semb
        reconstruct_images_single_batch(
            args=self.args,
            model=model,
            diffusion=diffusion,
            step=self.args.iter,
            batch=batch,
            save_dir=image_save_dir,
            eta=eta,
            logger=self.logger,
        )

    def forward_backward(
            self, 
            model, 
            batch, 
            cond_embedder=None,
            sem_encoder=None,
            cfg_train=True,
            ):
        image = batch['x'].to(self.args.device)
        # === Conditional Embedding (e.g., parent labels or context info) ===
        cemb = None
        if cond_embedder is not None:
            y_dict = {k: v.to(self.args.device) for k, v in batch.items() if k in self.args.parents}
            cemb = cond_embedder(y_dict)
        # === Semantic Embedding (identity info) ===
        semb = None
        mu = None
        if sem_encoder is not None:
            if self.args.use_vae_encoder:
                mu, logvar = sem_encoder(image)
                semb = sample_from_latent(mu=mu, logvar=logvar)
                kl_loss = kl_divergence_latent(mu, logvar, free_bits=self.args.kl_free_bits)
            else:
                semb = sem_encoder(image)
                mu = semb
                kl_loss = torch.tensor(0.0, device=image.device)
        else:
            kl_loss = torch.tensor(0.0, device=image.device)
        # === Optional: semantic disentanglement loss ===
        decorrelation_loss = 0.0
        if cemb is not None and semb is not None:
            hsic_loss = compute_hsic_norm_loss(cemb, semb)
            decorrelation_loss = hsic_loss
        else:
            decorrelation_loss = torch.tensor(0.0, device=image.device)

        # === CFG Dropout ===
        if cfg_train and self.args.cfg_dropout is not None and self.args.cfg_dropout > 0:
            style = self.args.cfg_dropout_style

            if style == "together":
                cemb, semb = cfg_drop_together(cemb, semb, cfg_dropout=self.args.cfg_dropout)

            elif style == "independent":
                if cemb is not None:
                    cemb = cfg_drop(cemb, cfg_dropout=self.args.cfg_dropout)
                if semb is not None:
                    semb = cfg_drop(semb, cfg_dropout=self.args.cfg_dropout)

            elif style == "cemb_only":
                if cemb is not None:
                    cemb = cfg_drop(cemb, cfg_dropout=self.args.cfg_dropout)

            else:
                raise ValueError(f"Unknown cfg_dropout_style: {style}")

        # === Main diffusion loss ===
        t, weights = self.schedule_sampler.sample(image.shape[0], self.args.device)
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            model,
            image,
            t,
            model_kwargs={"cemb":cemb, "semb":semb}, # Option 1, use sampled semb for diffusion training.
            # model_kwargs={"cemb":cemb, "semb":mu},   # Option 2, use determnistic mu for diffusion training.
        )

        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        
        loss = (losses["loss"] * weights).mean()
        if kl_loss is not None:
            loss+=kl_loss * self.args.kl_weight
            
        loss += self.args.lambda_decor * decorrelation_loss
        out = self.log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        # === Logging losses ===
        out['loss'] = loss
        out['kl_loss'] = kl_loss.detach().item() if kl_loss is not None else 0.0
        return out 

    def log_loss_dict(self, diffusion, ts, losses):
        log_dict = {}

        for key, values in losses.items():
            log_dict[key] = values.mean().item()  # Store mean loss

            # Store quantile-based loss values
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                log_dict[f"{key}_q{quartile}"] = sub_loss

        return log_dict
    
    