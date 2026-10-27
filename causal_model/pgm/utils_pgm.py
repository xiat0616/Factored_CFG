import torch

def check_nan(input_dict):
    nans = 0
    for k, v in input_dict.items():
        k_nans = torch.isnan(v).sum()
        nans += k_nans
        if k_nans > 0:
            print(f'\nFound {k_nans} nan(s) in {k}, skipping step.')
    return nans


def update_stats(stats, elbo_fn):
    """Accumulate tracked summary statistics."""

    def _update(trace, dist='p'):
        for name, node in trace.nodes.items():
            if node['type'] == 'sample':
                k = 'log' + dist + '(' + name + ')'
                if k not in stats:
                    stats[k] = 0
                stats[k] += node['log_prob'].sum().item()
        return stats

    _update(elbo_fn.trace_storage['model'], dist='p')
    _update(elbo_fn.trace_storage['guide'], dist='q')
    return stats


def calculate_loss(pred_batch, target_batch, loss_norm="l1", soft_loss="BCElogits"):
    "Calculate the losses for pred_bacth"
    loss=0
    for k in pred_batch.keys():
        assert pred_batch[k].size()==target_batch[k].size(), f"{k} size does not match, pred_batch size {pred_batch[k].size()}; target batch size {target_batch[k].size()}"
        if k in ["age"]:
            if loss_norm=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
            elif loss_norm=="l2":
                loss+=torch.nn.MSELoss()(pred_batch[k], target_batch[k]) 
        elif k in ["sex", "finding", "density", 'circle','Smiling', 'Male', 'Young', 'Eyeglasses', 'Mouth_Slightly_Open']:
            if soft_loss=="BCElogits":
                loss+=torch.nn.BCEWithLogitsLoss()(pred_batch[k], target_batch[k])
            elif soft_loss=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
            # if k=="circle":
            #     print(f"circle loss {torch.nn.L1Loss()(pred_batch[k], target_batch[k])}")
        elif k in ["race", "scanner"]:
            if soft_loss=="BCElogits":
                loss+=torch.nn.CrossEntropyLoss()(pred_batch[k], target_batch[k])
            elif soft_loss=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
        else:
            raise ValueError(f"Key '{k}' not recognized in calculate_loss. Please update loss logic accordingly.")
    return loss
