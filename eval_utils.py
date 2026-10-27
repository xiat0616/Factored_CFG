import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import copy
import lpips
from collections import defaultdict

def preprocess(batch, device="cuda:0"):
    batch_new = {}
    for k, v in batch.items():
        if k in ['x', 'age', 'race', 'finding', 'sex', 'scanner', 'density', 'circle', 'Smiling', 'Male', 'Young', 'Eyeglasses', 'Mouth_Slightly_Open']:
            batch_new[k] = batch[k].float().to(device) 
        else:
            try:
                batch_new[k] = batch[k].float().to(device) 
            except:
                continue
    return batch_new

@torch.no_grad()
def eval_epoch(args, model, dataloader):
    model.eval()

    # ------------------------------------------------------------
    # containers for original metrics
    # ------------------------------------------------------------
    preds, targets = {k: [] for k in model.variables}, {k: [] for k in model.variables}
    all_mae, all_lpips = [], []

    # ------------------------------------------------------------
    # containers for new AMC / SMC
    # ------------------------------------------------------------
    amc_lists, smc_lists = defaultdict(list), defaultdict(list)
    T = 0.5                                  # decision threshold

    # LPIPS model
    lpips_model = lpips.LPIPS(net="alex").to(args.device).eval()

    # ============================================================
    # main loop
    # ============================================================
    for raw_batch in tqdm(dataloader):
        # ---- keep ground‑truth ----
        for k in targets:
            targets[k].extend(copy.deepcopy(raw_batch[k]))

        # ---- COUNTERFACTUAL pass (what you already used) ----
        batch_cf = preprocess(raw_batch, args.device)            # contains 'x' = CF image
        out_cf   = model.predict(**batch_cf)
        for k, v in out_cf.items(): preds[k].extend(v)

        # ---- ORIGINAL pass (swap in raw image) ---------------
        batch_orig      = {k: v for k, v in batch_cf.items()}
        batch_orig["x"] = batch_cf["input_x"] if "input_x" in batch_cf.keys() else batch_cf['x']                   # original image
        out_orig        = model.predict(**batch_orig)

        # ---- AMC / SMC for every single‑logit head -----------
        for k in out_cf.keys():
            # skip non‑binary heads (multiclass or regression)
            if out_cf[k].ndim > 1 and out_cf[k].shape[-1] != 1:
                continue

            cf_prob   =  out_cf[k].reshape(-1)
            orig_prob =  out_orig[k].reshape(-1)

            amc_lists[k].append(torch.abs(cf_prob - T) - torch.abs(orig_prob - T))
            smc_lists[k].append((cf_prob - T) - (orig_prob - T))

        # ---- optional reconstruction metrics -----------------
        if "input_x" in raw_batch:
            mae = F.l1_loss(batch_cf["x"], batch_cf["input_x"], reduction="none")
            all_mae.append(mae.view(mae.size(0), -1).mean(1).cpu())

            def norm(t): return (t * 2.0) - 1.0 if t.min() >= 0 else t
            lpips_vals = lpips_model(norm(batch_cf["x"]), norm(batch_cf["input_x"])).view(-1)
            all_lpips.append(lpips_vals.cpu())

    # ------------------------------------------------------------
    # stack predictions / targets
    # ------------------------------------------------------------
    for k in preds:
        preds[k]   = torch.stack(preds[k]).squeeze().cpu()
        targets[k] = torch.stack(targets[k])

    # ============================================================
    #   METRIC AGGREGATION
    # ============================================================
    stats = {}

    for k in model.variables:
        t = targets[k]
        p = preds[k]

        # ------------ binary heads --------------------------------
        if k in ['sex', 'finding', 'density', 'circle',
                 'Smiling', 'Male', 'Young', 'Eyeglasses', 'Mouth_Slightly_Open']:
            stats[f"{k}_acc"]    = (t.squeeze(-1) == torch.round(p)).float().mean().item()
            stats[f"{k}_rocauc"] = roc_auc_score(t.numpy(), p.numpy(), average="macro")

        # ------------ regression head -----------------------------
        elif k == 'age':
            p = p.unsqueeze(-1)
            stats[f"{k}_mae"] = torch.mean(torch.abs(t - p)).item() * 100

        # ------------ multiclass heads ----------------------------
        elif k in ['race', 'scanner']:
            num_corrects = (t.argmax(-1) == p.argmax(-1)).sum()
            stats[f"{k}_acc"]    = num_corrects.item() / t.shape[0]
            stats[f"{k}_rocauc"] = roc_auc_score(t.numpy(), p.numpy(),
                                                 multi_class="ovr", average="macro")

    # ------------ reconstruction metrics -------------------------
    if all_mae:   stats["rec_mae"]   = torch.cat(all_mae).mean().item()
    if all_lpips: stats["rec_lpips"] = torch.cat(all_lpips).mean().item()

    # ------------ NEW AMC / SMC ---------------------------------

    for k, lst in amc_lists.items():
        stats[f"{k}_amc"] = torch.cat(lst).mean().item()          # >0 ⇒ amplification
    for k, lst in smc_lists.items():
        stats[f"{k}_smc"] = torch.cat(lst).mean().item()          # signed, direction‑aware

    return stats