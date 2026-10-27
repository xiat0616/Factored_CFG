import sys
import os
print(os.getcwd())
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import setup_import_paths
setup_import_paths()

from train_setup import (
    setup_dataloaders, 
    setup_directories,
    setup_logging
)
from hps import str2bool
from nn_utils import EMA
from utils import seed_all
import os
import copy
import argparse
import pyro
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from pyro.infer import config_enumerate
from sklearn.metrics import roc_auc_score

from causal_model.pgm.mimic_pgm import MimicFlowPGM
from causal_model.pgm.embed_pgm import EmbedFlowPGM

from utils_pgm import update_stats, calculate_loss
from pgm_layers import TraceStorage_ELBO
from torch.optim.lr_scheduler import ReduceLROnPlateau

def preprocess(batch, device="cuda:0"):
    batch_new = {}
    for k, v in batch.items():
        if k in ['x', 'age', 'race', 'finding', 'sex', 'scanner', 'density', 'Smiling', 'Male', 'Young', 'Eyeglasses', 'Mouth_Slightly_Open']:
            batch_new[k] = batch[k].float().to(device) 
        else:
            try:
                batch_new[k] = batch[k].float().to(device) 
            except:
                continue
    return batch_new

def sup_epoch(args, model, ema, dataloader, elbo_fn, optimizer=None, setup=None, is_train=True, loss_norm="l1"):
    "supervised epoch"
    stats = {'loss': 0, 'n': 0}  # sample counter
    loader = tqdm(enumerate(dataloader), total=len(
        dataloader), miniters=len(dataloader)//100, mininterval=5)
    
    model.train(is_train)
    for i, batch in loader:
        bs = batch['x'].shape[0]
        batch = preprocess(batch, device=args.device)
        
        with torch.set_grad_enabled(is_train):
            if setup == 'sup_aux':
                loss = elbo_fn.differentiable_loss(model.model_anticausal,
                               model.guide_pass, **batch) / bs
            elif setup == 'sup_pgm':
                loss = elbo_fn.differentiable_loss(model.svi_model,
                               model.guide_pass, **batch) / bs
            elif setup == "sup_determ": # Train classifiers in deterministic way
                pred_batch = model.predict_unnorm(**batch)
                loss = calculate_loss(
                    pred_batch=pred_batch, 
                    target_batch=batch, 
                    loss_norm=loss_norm
                )
            else:
                NotImplementedError

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
            optimizer.step()
            ema.update()
        
        stats['loss'] += loss.item() * bs
        stats['n'] += bs
        try:
            stats = update_stats(stats, elbo_fn)
        except:
            pass
        loader.set_description(
            f" => {('train' if is_train else 'eval')} | " +
            f", ".join(f"{k}: {v / stats['n']:.4f}" for k, v in stats.items() if k != 'n') +
            (f', grad_norm: {grad_norm:.3f}' if is_train else "") +
            f", lr: {optimizer.param_groups[0]['lr']:.6f}"  # Add current learning rate
        )     
    return {k: v / stats['n'] for k, v in stats.items() if k != 'n'}

@torch.no_grad()
def eval_epoch(args, model, dataloader):
    model.eval()
    preds = {k: [] for k in model.variables.keys()}
    targets = {k: [] for k in model.variables.keys()}
    all_mae = []

    for batch in tqdm(dataloader):
        for k in targets.keys():
            targets[k].extend(copy.deepcopy(batch[k]))

        # predict
        batch = preprocess(batch, args.device)
        out = model.predict(**batch)

        for k, v in out.items():
            preds[k].extend(v)

        # Optional image-level reconstruction MAE
        if "input_x" in batch:
            mae = F.l1_loss(batch["x"], batch["input_x"], reduction="none")
            mae = mae.view(mae.size(0), -1).mean(dim=1)  # MAE per image
            all_mae.append(mae.cpu())

    for k, v in preds.items():
        preds[k] = torch.stack(v).squeeze().cpu()
        targets[k] = torch.stack(targets[k])

    stats = {}

    for k in model.variables.keys():
        if isinstance(targets[k], torch.Tensor):
            targets_k = targets[k]
        else:
            targets_k = torch.stack(targets[k])

        if k in ['sex', 'finding', 'circle', 'density', 'Smiling', 'Male', 'Young', 'Eyeglasses', 'Mouth_Slightly_Open']:
            stats[k + '_acc'] = (targets_k.squeeze(-1) == torch.round(preds[k])).sum().item() / targets_k.shape[0]
            stats[k + '_rocauc'] = roc_auc_score(
                targets_k.numpy(), preds[k].numpy(), average='macro')
        elif k == 'age':
            preds_k = preds[k].unsqueeze(-1)
            assert targets_k.size() == preds_k.size(), f"{k} size doesn't match, targets {targets_k.size()} preds {preds_k.size()}"
            stats[k + '_mae'] = torch.mean(torch.abs(targets_k - preds_k)).item() * 100
        elif k in ['race', 'scanner']:
            num_corrects = (targets[k].argmax(-1) == preds[k].argmax(-1)).sum()
            stats[k + "_acc"] = num_corrects.item() / targets[k].shape[0]
            stats[k + "_rocauc"] = roc_auc_score(
                targets[k].numpy(),
                preds[k].numpy(),
                multi_class="ovr",
                average="macro",
            )
        else:
            raise ValueError(f"Key '{k}' not recognized in calculate_loss. Please update loss logic accordingly.")

    # Add reconstruction MAE if applicable
    if all_mae:
        mae_all = torch.cat(all_mae)
        stats["reconstruction_mae"] = mae_all.mean().item()

    return stats


def pgm_main():
    parser = argparse.ArgumentParser()
    # Hparams for PGM training
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name.")
    parser.add_argument('--hps', help='Hparam set', type=str, default="mimic")
    # Data
    parser.add_argument('--image_size', help="Image size", type=int, default=256)
    parser.add_argument('--batch_size', help="batch size for training", type=int, default=32)
    parser.add_argument("--parents", help="Parents of x to condition on.", nargs="+", default=["race", "age", "sex", "finding"])
    parser.add_argument("--use_manual", type=str2bool, default=True)
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle data.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader.")
    parser.add_argument("--pin_memory", type=str2bool, default=True, help="Pin memory option for torch dataloder.")
    # Train
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay rate.")
    parser.add_argument("--epochs", type=int, default=5000, help="Total training epochs.")
    parser.add_argument('--loss_norm', help='Loss norm for age.', type=str, default='l1')
    parser.add_argument("--eval_freq", help="Train epochs per validation.", type=int, default=1)
    # PGM setup
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument('--pgm_setup',help='pgm setup, sup_pgm, sup_aux or sup_determ', type=str, default='')
    parser.add_argument('--enc_net',help="pgm enc_net", type=str, default="CNN")
    parser.add_argument('--std_fixed', help='Fix aux dist std value (0 is off).', type=float, default=0)
    # Experiment
    parser.add_argument('--seed', help='Set random seed.', type=int, default=7)
    parser.add_argument('--deterministic', help='Toggle cudNN determinism.', type=str2bool, default=False)
    # args = setup_hparams(parser)
    args = parser.parse_known_args()[0]

    # ===== Temporal setting==========
    # args.pgm_setup = "sup_determ"
    # args.exp_name = "mimic_sup_pgm"
    # ===== Temporal setting==========
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device.type == "cuda":
        torch.cuda.set_device(args.device)

    seed_all(args.seed, args.deterministic)
    # Set up dataloaders
    dataloaders = setup_dataloaders(args, balance_train=False)
    # Init model
    pyro.clear_param_store()
    if "mimic" in args.hps:
        model = MimicFlowPGM(args=args)
    elif "embed" in args.hps:
        model = EmbedFlowPGM(args=args)
    elif "celeba" in args.hps:
        if "Eyeglasses" in args.parents:
            from causal_model.pgm.celeba_pgm_s_e import CelebaFlowPGM
        elif "Mouth_Slightly_Open" in args.parents:
            from causal_model.pgm.celeba_pgm_2 import CelebaFlowPGM
        else:
            from causal_model.pgm.celeba_pgm import CelebaFlowPGM
        model = CelebaFlowPGM(args=args)

    ema = EMA(model, beta=0.999)
    model.to(args.device)
    ema.to(args.device)
     # Init loss & optimizer
    elbo_fn = TraceStorage_ELBO(num_particles=2)
    aux_elbo_fn = TraceStorage_ELBO(num_particles=2)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.wd,
    )
    scheduler=ReduceLROnPlateau(
                optimizer,
                factor=0.1,
                patience=10,
                mode='min',
                min_lr=1e-5,
    )

    args.save_dir = setup_directories(args, ckpt_dir='pgm_checkpoints')
    logger = setup_logging(args)

    for k in sorted(vars(args)):
        logger.info(f'--{k}={vars(args)[k]}')
    
    args.best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # supervised training of PGM or aux models
        if args.pgm_setup == 'sup_pgm' or args.pgm_setup == 'sup_aux' or args.pgm_setup=="sup_determ":
            stats = sup_epoch(
                args=args,
                model=model, 
                ema=ema,
                dataloader=dataloaders['train'],
                elbo_fn=elbo_fn,
                optimizer=optimizer,
                setup=args.pgm_setup,
                is_train=True,
                loss_norm=args.loss_norm,
            )
            steps = (epoch + 1) * len(dataloaders['train'])
            
            if epoch % args.eval_freq == 0:
                valid_stats = sup_epoch(
                    args=args,
                    model=ema.ema_model, 
                    ema=ema,
                    elbo_fn=elbo_fn,
                    optimizer=optimizer,
                    dataloader=dataloaders['valid'],
                    setup=args.pgm_setup,
                    is_train=False,
                    loss_norm=args.loss_norm,
                )
                logger.info(
                    f'loss | train: {stats["loss"]:.4f}' +
                    f' - valid: {valid_stats["loss"]:.4f} - steps: {steps}'
                )

                if not args.pgm_setup == 'sup_pgm':  # eval aux classifiers
                    metrics = eval_epoch(args=args, model=ema.ema_model, dataloader=dataloaders['valid'])
                    logger.info(
                        'valid | '+' - '.join(f'{k}: {v:.4f}' for k, v in metrics.items()))
            
            scheduler.step(valid_stats['loss'])  # Reduce LR based on validation loss

            if valid_stats['loss'] < args.best_loss:
                args.best_loss = valid_stats['loss']
                ckpt_path = os.path.join(args.save_dir, 'checkpoint.pt')
                torch.save(
                            {
                            'epoch': epoch + 1,
                            'step': steps,
                            'best_loss': args.best_loss,
                            'model_state_dict': model.state_dict(),
                            'ema_model_state_dict': ema.ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'hparams': vars(args)
                            }, 
                            ckpt_path,
                )
                logger.info(f'Model saved: {ckpt_path}')
            # Save current checkpoint
            ckpt_path = os.path.join(args.save_dir, 'checkpoint_current.pt')
            torch.save(
                        {
                        'epoch': epoch + 1,
                        'step': steps,
                        'best_loss': args.best_loss,
                        'model_state_dict': model.state_dict(),
                        'ema_model_state_dict': ema.ema_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hparams': vars(args)
                        }, 
                        ckpt_path,
            )
            logger.info(f'Model saved: {ckpt_path}')

if __name__=="__main__":
    pgm_main()