import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os
from pathlib import Path
import pandas as pd
from skimage import io

from project_paths import setup_import_paths
setup_import_paths()

from train_setup import setup_dataloaders
from torchvision.transforms import ToTensor, Resize, CenterCrop
import argparse
import pyro
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from eval_utils import eval_epoch
from causal_model.pgm.mimic_pgm import MimicFlowPGM
from causal_model.pgm.embed_pgm import EmbedFlowPGM
from causal_model.pgm.celeba_pgm import CelebaFlowPGM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None


def sort_cf_results(df):
    df_real = df[df["do_key"] == "real"]
    df_rest = df[df["do_key"] != "real"]
    df_rest_sorted = df_rest.sort_values(by=["do_key", "cfg_mode", "cfg_weight_pattern"])
    return pd.concat([df_real, df_rest_sorted]).reset_index(drop=True)


class CF_Dataset(Dataset):
    def __init__(self, df, args):
        self.df = df
        self.args = args
        self.img_paths = df.file_path.values

        self.image_size = args.image_size
        self.input_paths = df.input_path.values if "input_path" in df.columns else None

    def __len__(self):
        return len(self.df)

    def read_image(self, idx):
        return self.read_image_by_path(self.img_paths[idx])

    def read_image_by_path(self, path):
        as_gray = not ("celeba" in self.args.hps)
        img = io.imread(path, as_gray=as_gray)

        if as_gray:
            img = img / (img.max() + 1e-12)
        else:
            img = img.astype(np.float32) / 255.0

        img = ToTensor()(img)
        img = Resize(self.image_size, antialias=True)(img)
        img = CenterCrop(self.image_size)(img)
        return img * 2 - 1

    def __getitem__(self, idx):
        img = self.read_image(idx)

        sample = {"x": img}
        for key in self.args.parents:
            value = torch.tensor(self.df.iloc[idx][key])
            if key in {"race", "scanner"}:
                value = F.one_hot(value.long(), num_classes=3 if key == "race" else 5)
            else:
                value = value.unsqueeze(-1)
            sample[key] = value.float()

        if self.input_paths is not None:
            inp_img = self.read_image_by_path(self.input_paths[idx])
            sample["input_x"] = inp_img

        return sample


def preprocess_batch_for_predictor(batch, device):
    batch_new = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch_new[k] = v.float().to(device)
    return batch_new


def prepare_fid_images(x):
    """Convert tensors from [-1, 1] to [0, 1] float and ensure 3 channels for FID."""
    x = x.clamp(-1, 1)
    x = (x + 1) / 2
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] > 3:
        x = x[:, :3]
    return x.float()


@torch.no_grad()
def compute_fid_from_loader(loader, device, fid_feature=2048):
    """Compute distribution-level FID between generated images x and input images input_x."""
    if FrechetInceptionDistance is None:
        raise ImportError(
            "FID requires torchmetrics + torch-fidelity. "
            "Install with `pip install torchmetrics[image]` or `pip install torch-fidelity`."
        )

    metric = FrechetInceptionDistance(feature=fid_feature, normalize=True).to(device)
    metric.set_dtype(torch.float64)

    seen = 0
    for batch in loader:
        if "input_x" not in batch:
            return float("nan")

        real = prepare_fid_images(batch["input_x"].to(device, non_blocking=True))
        fake = prepare_fid_images(batch["x"].to(device, non_blocking=True))
        metric.update(real, real=True)
        metric.update(fake, real=False)
        seen += fake.shape[0]

    if seen == 0:
        return float("nan")

    return float(metric.compute().item())


@torch.no_grad()
def compute_f1_from_loader(loader, predictor, device):
    binary_keys = {
        'sex', 'finding', 'density', 'circle',
        'Smiling', 'Male', 'Young', 'Eyeglasses', 'Mouth_Slightly_Open'
    }
    multiclass_keys = {'race', 'scanner'}

    pred_keys = list(getattr(predictor, "variables", []))
    if len(pred_keys) == 0:
        return {}

    preds = {k: [] for k in pred_keys}
    targets = {k: [] for k in pred_keys}

    predictor.eval()

    for batch in loader:
        batch_dev = preprocess_batch_for_predictor(batch, device)
        out = predictor.predict(**batch_dev)

        for k in pred_keys:
            if k not in out or k not in batch:
                continue
            preds[k].append(out[k].detach().cpu())
            targets[k].append(batch[k].detach().cpu())

    metrics = {}

    for k in pred_keys:
        if len(preds[k]) == 0:
            continue

        p = torch.cat(preds[k], dim=0)
        t = torch.cat(targets[k], dim=0)

        if k in binary_keys:
            y_score = p.squeeze().numpy()
            y_true = t.squeeze().numpy().astype(int)
            y_pred = (y_score >= 0.5).astype(int)
            metrics[f"{k}_f1"] = f1_score(y_true, y_pred, zero_division=0)

        elif k in multiclass_keys:
            if t.ndim > 1 and t.shape[-1] > 1:
                y_true = t.argmax(-1).numpy()
            else:
                y_true = t.squeeze().numpy().astype(int)

            if p.ndim > 1 and p.shape[-1] > 1:
                y_pred = p.argmax(-1).numpy()
            else:
                y_pred = (p.squeeze().numpy() >= 0.5).astype(int)

            metrics[f"{k}_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return metrics


def plot_delta_auc(
    df_final,
    save_dir="plots_delta_grouped",
    baseline="real",
    eval_keys=['sex', 'race', 'finding', 'age'],
    eval_reverse=False,
):
    os.makedirs(save_dir, exist_ok=True)

    metric_cols = [col for col in df_final.columns if col not in ["do_key", "cfg_mode", "cfg_weight_pattern"]]
    df_final[metric_cols] = df_final[metric_cols].apply(pd.to_numeric, errors="coerce")

    df_cfg = df_final[df_final["do_key"] != "real"]
    df_real = df_final[df_final["do_key"] == "real"]

    if eval_reverse:
        eval_keys = list(eval_keys) + [f"reversed_{k}" for k in eval_keys]

    metrics_to_plot = []
    for key in eval_keys:
        metrics_to_plot.append(f"{key}_rocauc")
        metrics_to_plot.append(f"{key}_acc")
        if f"{key}_f1" in df_final.columns:
            metrics_to_plot.append(f"{key}_f1")

    metrics_to_plot.append("fid_input")

    if eval_reverse:
        metrics_to_plot.append("rec_mae")
        metrics_to_plot.append("rec_lpips")

    for do_key in eval_keys:
        df_subset = df_cfg[df_cfg["do_key"] == do_key].copy()
        if df_subset.empty:
            continue

        if baseline == "real":
            base_row = df_real.iloc[0]
        else:
            base_rows = df_subset[df_subset["cfg_weight_pattern"] == baseline]
            if base_rows.empty:
                print(f"⚠️ Skipping {do_key} — baseline '{baseline}' not found.")
                continue
            base_row = base_rows.iloc[0]

        delta_data = []
        for metric_col in metrics_to_plot:
            if (metric_col in ["rec_mae", "rec_lpips"]) and not do_key.startswith("reversed_"):
                continue
            if metric_col not in df_subset.columns or metric_col not in base_row:
                continue

            base_val = base_row[metric_col]

            for _, row in df_subset.iterrows():
                delta = row[metric_col] - base_val
                delta_data.append({
                    "cfg_weight_pattern": row["cfg_weight_pattern"],
                    "cfg_mode": row["cfg_mode"],
                    "metric": metric_col,
                    "delta": delta,
                })

        if not delta_data:
            continue

        df_delta = pd.DataFrame(delta_data)

        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df_delta,
            x="cfg_weight_pattern",
            y="delta",
            hue="metric",
            dodge=True,
        )
        plt.axhline(0, color="gray", linestyle="--")
        title_base = f"Δ Metrics for do({do_key}) vs Baseline ({baseline})"
        plt.title(title_base)
        plt.ylabel("Delta (CF - Baseline)")
        plt.xlabel("CFG Weight Pattern")
        plt.xticks(rotation=90)
        plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        fname = os.path.join(save_dir, f"delta_do_{do_key}_baseline_{baseline.replace('.', '_')}.png")
        plt.savefig(fname)
        plt.close()
        print(f"✅ Saved plot: {fname}")


def evaluate_cfg_folder(
    cfg_mode_folder,
    cfg_weight_folder,
    do_keys,
    predictor,
    eval_args
):
    base_dir = Path(eval_args.results_dir) / "_".join(eval_args.parents) / eval_args.exp_name / eval_args.timestep_respacing / eval_args.cfg_choice
    target_dir = base_dir / cfg_mode_folder / cfg_weight_folder

    rows_df = {}

    for do_key in do_keys:
        cf_csv = target_dir / f"cfs_{do_key}" / f"counterfactuals_{do_key}.csv"
        if not cf_csv.exists():
            print(f"⚠️ Missing: {cf_csv}")
            continue

        cf_df = pd.read_csv(cf_csv)
        dataset = CF_Dataset(cf_df, eval_args)
        loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=6)

        with torch.no_grad():
            metrics = eval_epoch(eval_args, predictor, loader)

        f1_metrics = compute_f1_from_loader(loader, predictor, eval_args.device)
        metrics.update(f1_metrics)

        if not eval_args.skip_fid:
            try:
                metrics["fid_input"] = compute_fid_from_loader(
                    loader,
                    device=eval_args.device,
                    fid_feature=eval_args.fid_feature,
                )
            except Exception as e:
                print(f"⚠️ FID failed for {cf_csv}: {e}")
                metrics["fid_input"] = float("nan")

        row = {
            "do_key": do_key,
            "cfg_mode": cfg_mode_folder,
            "cfg_weight_pattern": cfg_weight_folder,
        }
        for k, v in metrics.items():
            row[k] = f"{v:.3f}" if isinstance(v, float) else v

        rows_df[do_key] = pd.DataFrame([row])

        rev_csv = target_dir / f"cfs_{do_key}" / f"reversed_{do_key}.csv"
        if rev_csv.exists():
            rev_df = pd.read_csv(rev_csv)
            rev_dataset = CF_Dataset(rev_df, eval_args)
            rev_loader = DataLoader(rev_dataset, batch_size=10, shuffle=False, num_workers=6)

            with torch.no_grad():
                rev_metrics = eval_epoch(eval_args, predictor, rev_loader)

            rev_f1_metrics = compute_f1_from_loader(rev_loader, predictor, eval_args.device)
            rev_metrics.update(rev_f1_metrics)

            if not eval_args.skip_fid:
                try:
                    rev_metrics["fid_input"] = compute_fid_from_loader(
                        rev_loader,
                        device=eval_args.device,
                        fid_feature=eval_args.fid_feature,
                    )
                except Exception as e:
                    print(f"⚠️ FID failed for {rev_csv}: {e}")
                    rev_metrics["fid_input"] = float("nan")

            row_rev = {
                "do_key": f"reversed_{do_key}",
                "cfg_mode": cfg_mode_folder,
                "cfg_weight_pattern": cfg_weight_folder,
            }
            for k, v in rev_metrics.items():
                row_rev[k] = f"{v:.3f}" if isinstance(v, float) else v

            rows_df[f"reversed_{do_key}"] = pd.DataFrame([row_rev])
        else:
            print(f"⚠️ Missing: {rev_csv}")

    return rows_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predictor checkpoint.")
    parser.add_argument("--results_dir", type=str, default="sampling_results")
    parser.add_argument("--sample_do_keys", nargs="+", default=None, help="Intervention folders to evaluate, including multi-attribute names.")
    parser.add_argument("--exp_name", type=str, help="Diffusion model name.")
    parser.add_argument("--timestep_respacing", default="ddim250", type=str)
    parser.add_argument(
        "--cfg_choice",
        type=str,
        default="eq1",
        choices=["eq1", "eq2", "eq3"],
        help="Which CFG equation to apply during guidance interpolation."
    )
    parser.add_argument(
        "--fid_feature",
        type=int,
        default=2048,
        choices=[64, 192, 768, 2048],
        help="Inception feature dimensionality for FID."
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID computation."
    )
    eval_args = parser.parse_known_args()[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(eval_args.pred_path, map_location=device, weights_only=False)
    ckpt_args = ckpt["hparams"]
    if isinstance(ckpt_args, dict):
        ckpt_args = argparse.Namespace(**ckpt_args)

    ckpt_args.device = device
    eval_args.hps = ckpt_args.hps
    eval_args.device = ckpt_args.device
    eval_args.parents = ckpt_args.parents
    eval_args.image_size = ckpt_args.image_size
    eval_args.do_keys = eval_args.sample_do_keys or list(eval_args.parents)

    pyro.clear_param_store()

    if "mimic" in eval_args.hps:
        predictor = MimicFlowPGM(ckpt_args)
    elif "embed" in eval_args.hps:
        predictor = EmbedFlowPGM(ckpt_args)
    elif "celeba" in eval_args.hps:
        if "Eyeglasses" in ckpt_args.parents:
            from causal_model.pgm.celeba_pgm_s_e import CelebaFlowPGM as Predictor
        elif "Mouth_Slightly_Open" in ckpt_args.parents:
            from causal_model.pgm.celeba_pgm_2 import CelebaFlowPGM as Predictor
        else:
            Predictor = CelebaFlowPGM
        predictor = Predictor(ckpt_args)
    else:
        raise ValueError(f"Unsupported hps: {eval_args.hps}")

    predictor.load_state_dict(ckpt["ema_model_state_dict"])
    predictor.to(device)
    predictor.eval()

    print("\n🔍 Evaluating Real Test Set...")
    dataloaders = setup_dataloaders(ckpt_args, balance_test=True)

    if "mimic" in eval_args.hps:
        limited_test_dataset = Subset(dataloaders["test"].dataset, list(range(min(1500, len(dataloaders["test"].dataset)))))
        dataloaders["test"] = DataLoader(
            limited_test_dataset,
            batch_size=dataloaders["test"].batch_size,
            shuffle=False,
            num_workers=dataloaders["test"].num_workers,
            pin_memory=dataloaders["test"].pin_memory
        )

    with torch.no_grad():
        eval_real_metrics = eval_epoch(eval_args, predictor, dataloaders["test"])

    real_f1_metrics = compute_f1_from_loader(dataloaders["test"], predictor, eval_args.device)
    eval_real_metrics.update(real_f1_metrics)

    formatted_metrics = " | ".join(f"{k}: {v:.4f}" for k, v in eval_real_metrics.items())
    print(f"✅ Real     → {formatted_metrics}")

    real_row = {
        "do_key": "real",
        "cfg_mode": "real",
        "cfg_weight_pattern": "real",
    }
    for k, v in eval_real_metrics.items():
        real_row[k] = f"{v:.3f}" if isinstance(v, float) else v

    df_real = pd.DataFrame([real_row])
    for col in ["rec_mae", "rec_lpips", "fid_input"]:
        if col not in df_real.columns:
            df_real[col] = 0.0
        else:
            df_real[col] = pd.to_numeric(df_real[col], errors="coerce").fillna(0.0)

    base_dir = Path(eval_args.results_dir) / "_".join(eval_args.parents) / eval_args.exp_name / eval_args.timestep_respacing / eval_args.cfg_choice
    all_cfg_rows = []

    for cfg_mode_folder in sorted(base_dir.iterdir()):
        if not cfg_mode_folder.is_dir():
            continue
        for cfg_weight_folder in sorted(cfg_mode_folder.iterdir()):
            if not cfg_weight_folder.is_dir():
                continue
            print(f"🔍 Evaluating {cfg_mode_folder.name}/{cfg_weight_folder.name}")
            rows_df = evaluate_cfg_folder(
                cfg_mode_folder.name,
                cfg_weight_folder.name,
                eval_args.do_keys,
                predictor,
                eval_args,
            )

            for _, df_row in rows_df.items():
                all_cfg_rows.append(df_row)

    if all_cfg_rows:
        df_cfg = pd.concat(all_cfg_rows, ignore_index=True)
        df_final = pd.concat([df_real, df_cfg], ignore_index=True)

        base_cols = ["do_key", "cfg_mode", "cfg_weight_pattern"]
        metric_cols = [col for col in df_final.columns if col not in base_cols]
        df_final = df_final[base_cols + metric_cols]

        df_final = sort_cf_results(df_final)

        save_path = base_dir / "all_cf_eval_results.csv"
        df_final.to_csv(save_path, index=False)
        print(f"✅ Saved results to {save_path}")

        plot_delta_auc(
            df_final=df_final,
            save_dir=base_dir,
            baseline="real",
            eval_keys=eval_args.parents,
            eval_reverse=True,
        )

        plot_delta_auc(
            df_final=df_final,
            save_dir=base_dir,
            baseline="cfg_w_1.0",
            eval_keys=eval_args.parents,
            eval_reverse=True,
        )
    else:
        print("⚠️ No evaluation results collected.")


if __name__ == "__main__":
    main()