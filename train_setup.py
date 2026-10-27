import sys
import os
import logging
import torch
from data_handling.mimic import MimicDataModule
from data_handling.embed import EmbedDataModule
from data_handling.celeba import CelebADataModule
from project_paths import EVAL_INDEX_ROOT
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from torch.utils.data import DataLoader,  Subset
from data_handling.sampler import SamplerFactory

def format_cfg_w(cfg_w):
    s = f"{cfg_w:.8f}".rstrip("0")
    if s.endswith("."):
        s = s[:-1]
    decimals = max(1, len(s.split(".")[1]) if "." in s else 0)
    return f"{cfg_w:.{decimals}f}"

def setup_directories(args, ckpt_dir="checkpoints"):
    parents_folder = "_".join([k for k in args.parents])
    save_dir = os.path.join(ckpt_dir, parents_folder, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"save_dir: {save_dir}")
    return save_dir

def setup_eval_directories(args, eval_dir="sampling_results"):
    parents_folder = "_".join([k for k in args.parents])
    cfg_mode = getattr(args, "cfg_mode", "global")

    # Determine cfg_weights folder name
    if cfg_mode == "none" or "no_cfg" in args.exp_name:
        cfg_mode_folder = "none"
        cfg_weights_folder = "no_cfg"
    elif cfg_mode == "global":
        cfg_mode_folder = "global"
        cfg_weights_folder = f"cfg_w_{format_cfg_w(args.cfg_w)}"
    elif cfg_mode == "per_attr":
        cfg_mode_folder = "per_attr"
        mode = getattr(args, "per_attr_mode", "null_this")
        weights = getattr(args, "cfg_attr_weights", {})
        weights_str = "_".join(f"{k}_{format_cfg_w(v)}" for k, v in weights.items())  # use underscore not hyphen
        cfg_weights_folder = f"{mode}_{weights_str}"
    elif cfg_mode == "interv_group":
        cfg_mode_folder = "interv_group"
        mode = getattr(args, "per_attr_mode", "null_this")
        w1 = getattr(args, "cfg_w_intervened", 1.0)
        w2 = getattr(args, "cfg_w_unintervened", 1.0)
        cfg_weights_folder = f"{mode}_int_{format_cfg_w(w1)}_unint_{format_cfg_w(w2)}"
    else:
        cfg_mode_folder = "unknown_mode"
        cfg_weights_folder = "cfg_unknown"

    # Combine full evaluation path
    save_dir = os.path.join(
        eval_dir,
        parents_folder,
        args.exp_name,
        args.timestep_respacing,
        cfg_mode_folder,
        cfg_weights_folder,
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"save evaluation dir: {save_dir}")
    return save_dir

def setup_results_directories_baselines(args, results_dir="sampling_results"):
    parents_folder = "_".join(args.parents)
    cfg_mode = getattr(args, "cfg_mode", "global")
    cfg_choice = getattr(args, "cfg_choice", "eq1")

    # ----------------------------
    # Build cfg_mode_folder + cfg_weights_folder
    # ----------------------------
    if cfg_mode == "none" or "no_cfg" in args.exp_name:
        cfg_mode_folder = "none"
        cfg_weights_folder = "no_cfg"

    elif cfg_mode == "global":
        cfg_mode_folder = "global"
        cfg_weights_folder = f"cfg_w_{format_cfg_w(args.cfg_w)}"

    elif cfg_mode == "per_attr":
        weights = getattr(args, "cfg_attr_weights", {}) or {}
        print(f"weights: {weights}, type: {type(weights)}")

        # Example: {"Smiling": 0.3, "Male": 0.2}
        weights_str = "_".join(
            f"{k}_{format_cfg_w(float(v))}"
            for k, v in sorted(weights.items())
        )

        cfg_mode_folder = "per_attr"
        cfg_weights_folder = f"per_attr_{weights_str}"

    elif cfg_mode == "interv_group":
        w1 = getattr(args, "cfg_w_intervened", 1.0)
        w2 = getattr(args, "cfg_w_unintervened", 1.0)

        cfg_mode_folder = "interv_group"
        cfg_weights_folder = f"int_{format_cfg_w(w1)}_unint_{format_cfg_w(w2)}"

    else:
        cfg_mode_folder = "unknown_mode"
        cfg_weights_folder = "cfg_unknown"

    # ----------------------------
    # Engine-based suffixes (MUST MATCH BASH)
    # ----------------------------
    engine = os.getenv("DIFF_ENGINE", "cfg").lower()

    if cfg_mode != "none":
        if engine == "cfgpp":
            cfg_mode_folder += "_cfgpp"

        elif engine == "apg":
            cfg_mode_folder += "_apg"

            apg_mom = getattr(args, "apg_momentum", 0.0)
            apg_eta = getattr(args, "apg_eta", 1.0)
            apg_norm = getattr(args, "apg_norm_threshold", 0.0)

            # Must match bash: _apg_mom_${mom}_eta_${eta}
            apg_suffix = f"_apg_mom_{apg_mom}_eta_{apg_eta}_apg_norm_threshold_{apg_norm}"

            # If norm_threshold should be included only when nonzero:
            # if abs(apg_norm) > 1e-12:
            #     apg_suffix += f"_rn_{apg_norm}"

            cfg_weights_folder += apg_suffix

    # ----------------------------
    # Build final save directory
    # (MUST match sbatch path in the bash script)
    # ----------------------------
    save_dir = os.path.join(
        results_dir,
        parents_folder,
        args.exp_name,
        args.timestep_respacing,
        cfg_choice,
        cfg_mode_folder,
        cfg_weights_folder,
    )

    os.makedirs(save_dir, exist_ok=True)
    print(f"save_dir: {save_dir}")
    return save_dir





def setup_results_directories(args, results_dir="sampling_results"):
    parents_folder = "_".join(args.parents)
    cfg_mode = getattr(args, "cfg_mode", "global")
    cfg_choice = getattr(args, "cfg_choice", "eq1")
    null_semb = getattr(args, "null_semb", False)
    use_cfgpp = getattr(args, "use_cfgpp", False)
    use_cfg_reverse = getattr(args, "use_cfg_reverse", False)
    # ----------------------------
    # Build cfg_mode_folder
    # ----------------------------
    if cfg_mode == "none" or "no_cfg" in args.exp_name:
        cfg_mode_folder = "none"
        cfg_weights_folder = "no_cfg"

    elif cfg_mode == "global":
        cfg_mode_folder = "global"
        cfg_weights_folder = f"cfg_w_{format_cfg_w(args.cfg_w)}"

    elif cfg_mode == "per_attr":
        weights = getattr(args, "cfg_attr_weights", {})
        print(f"weights: {weights}, type: {type(weights)}")

        # stable ordering of dict
        weights_str = "_".join(f"{k}_{format_cfg_w(float(v))}" for k, v in weights.items())
        cfg_mode_folder = "per_attr"
        cfg_weights_folder = f"per_attr_{weights_str}"

    elif cfg_mode == "interv_group":
        w1 = getattr(args, "cfg_w_intervened", 1.0)
        w2 = getattr(args, "cfg_w_unintervened", 1.0)
        cfg_mode_folder = "interv_group"
        cfg_weights_folder = f"int_{format_cfg_w(w1)}_unint_{format_cfg_w(w2)}"

    else:
        cfg_mode_folder = "unknown_mode"
        cfg_weights_folder = "cfg_unknown"

    # ----------------------------
    # Add null_semb if needed
    # ----------------------------
    if null_semb:
        cfg_mode_folder += "_null_semb"

    # ----------------------------
    # Add CFG++ suffix if active
    # ----------------------------
    # Does NOTHING if use_cfgpp = False → backward compatible
    if use_cfgpp and cfg_mode != "none":
        cfg_mode_folder += "_cfgpp"
    if use_cfg_reverse and cfg_mode != "none":
        cfg_mode_folder += "_cfg_in_reverse"
    # ----------------------------
    # Construct final save_dir
    # ----------------------------
    save_dir = os.path.join(
        results_dir,
        parents_folder,
        args.exp_name,
        args.timestep_respacing,
        cfg_choice,
        cfg_mode_folder,
        cfg_weights_folder,
    )

    os.makedirs(save_dir, exist_ok=True)
    print(f"save_dir: {save_dir}")
    return save_dir

    
def setup_tensorboard(args, model):
    """Setup metric summary writer."""
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(args.save_dir)

    hparams = {}
    for k, v in vars(args).items():
        if isinstance(v, list) or isinstance(v, torch.device):
            hparams[k] = str(v)
        elif isinstance(v, torch.Tensor):
            hparams[k] = v.item()
        else:
            hparams[k] = v

    writer.add_hparams(hparams, {"hparams": 0}, run_name=os.path.abspath(args.save_dir))
    return writer

def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f

def setup_optimizer(args, *modules):
    params = []
    for module in modules:
        if module is not None:
            params += list(module.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr, 
        weight_decay=args.wd, 
        betas=tuple(args.adam_betas),
        eps=1e-8,
    )
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_warmup(args.lr_warmup_steps)),  # Warm-up phase
            lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000000 - args.lr_warmup_steps, eta_min=args.lr*0.01)  # Cosine Decay
        ],
        milestones=[args.lr_warmup_steps]  # Switch from warm-up to cosine at `warmup_iters`
    )
    return optimizer, scheduler

def setup_logging(args, log_file="trainlog.txt"):
    # reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]
    # info logger for saving command line outputs during training
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, log_file)),
            logging.StreamHandler(),
        ],
        # filemode='a',  # append to file, 'w' for overwrite
        format="%(asctime)s, %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(args.exp_name)  # name the logger
    return logger


def setup_eval_logging(args, log_file="eval.txt"):
    # Reset root logger
    [logging.root.removeHandler(h) for h in logging.root.handlers[:]]

    # Configure logger without date
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, log_file)),
            logging.StreamHandler(),
        ],
        format="%(message)s",  # Remove timestamp
        level=logging.INFO,
    )

    # Create named logger
    logger = logging.getLogger(args.exp_name)
    return logger


def get_balanced_loader(
    dataset,
    batch_size,
    num_workers,
    pin_memory,
    class_arrays,
    alpha=0.75,
    kind="random",
):
    """
    Create a balanced DataLoader for a dataset, supporting one or more class-defining attributes.

    Args:
        dataset: A PyTorch Dataset object.
        batch_size: Int, total batch size.
        num_workers: Int, number of workers for DataLoader.
        pin_memory: Bool, for DataLoader.
        class_arrays: A single NumPy array (for one attribute) or a list of arrays (for multi-attribute).
                      e.g., dataset.scanner or [dataset.scanner, dataset.density]
        alpha: Dirichlet concentration parameter for SamplerFactory.
        kind: Sampler kind, e.g., 'random'.

    Returns:
        DataLoader with balanced sampling.
    """
    # Handle single or multiple class arrays
    if isinstance(class_arrays, np.ndarray):
        class_arrays = [class_arrays]

    # Stack the arrays to get tuples per sample: (attr1[i], attr2[i], ...)
    stacked = np.stack(class_arrays, axis=1)

    # Encode unique combinations into integer class labels
    _, composite_labels = np.unique(stacked, axis=0, return_inverse=True)

    # Group indices by composite label
    n_classes = composite_labels.max() + 1
    class_idx = [np.where(composite_labels == i)[0] for i in range(n_classes)]

    n_batches = len(dataset) // batch_size
    sampler = SamplerFactory().get(
        class_idx,
        batch_size,
        n_batches,
        alpha=alpha,
        kind=kind,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader

def get_balanced_test_loader(
    dataset,
    batch_size,
    num_workers,
    pin_memory,
    class_arrays,
    alpha=0.75,
    kind="random",
    args=None,
    n_eval_samples=3000,
):
    """
    Create a deterministic and reusable balanced test DataLoader.

    Saves fixed sample indices on first use, and reuses them later.

    Parameters
    ----------
    dataset : PyTorch dataset
        The dataset to sample from.

    batch_size : int
        Batch size.

    num_workers : int
        Dataloader workers.

    pin_memory : bool
        Dataloader pin memory flag.

    class_arrays : list of np.ndarray
        Each array gives one attribute per sample (e.g., scanner, density).

    alpha : float
        Dirichlet parameter for balancing.

    kind : str
        'random' or 'fixed' sampler type.

    args : argparse.Namespace
        Required to access args.hps to route the save path.
    """

    assert args is not None, "args must be provided to determine save path."

    # Determine the correct save path
    if "embed" in args.hps:
        save_root = EVAL_INDEX_ROOT / "embed"
    elif "celeba" in args.hps:
        save_root = EVAL_INDEX_ROOT / "celeba"
    elif "mimic" in args.hps:
        save_root = EVAL_INDEX_ROOT / "mimic"
    else:
        raise NotImplementedError(f"No fixed_eval_indices path defined for dataset: {args.hps}")

    os.makedirs(save_root, exist_ok=True)
    index_file = os.path.join(save_root, "fixed_test_indices.npy")

    if os.path.exists(index_file):
        print(f"Loading fixed evaluation indices from: {index_file}")
        indices = np.load(index_file)
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )
        return loader

    # If first run, generate a balanced sampler and record the selected indices
    print(f"Creating fixed balanced test loader and saving {n_eval_samples} samples...")

    if isinstance(class_arrays, np.ndarray):
        class_arrays = [class_arrays]
    stacked = np.stack(class_arrays, axis=1)
    _, composite_labels = np.unique(stacked, axis=0, return_inverse=True)
    n_classes = composite_labels.max() + 1
    class_idx = [np.where(composite_labels == i)[0] for i in range(n_classes)]

    n_batches = n_eval_samples // batch_size # Only use 1000 for now.
    sampler = SamplerFactory().get(
        class_idx,
        batch_size,
        n_batches,
        alpha=alpha,
        kind=kind,
    )

    # Extract fixed indices from the sampler
    flat_indices = []
    for batch in sampler:
        flat_indices.extend(batch)

    indices = np.array(flat_indices)
    np.save(index_file, indices)
    print(f"Saved fixed test indices to: {index_file}")

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return loader


def setup_dataloaders(args, balance_test=False, balance_train=True):
    if "mimic" in args.hps:
        data_module = MimicDataModule(args=args)
        batch_size = args.batch_size
        
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # ---------- Balanced Val Loader (by race, finding, sex) ----------
        if balance_test:
            test_loader = get_balanced_test_loader(
                dataset=data_module.dataset_test,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                class_arrays=[
                    data_module.dataset_test.race,
                    data_module.dataset_test.finding,
                    data_module.dataset_test.sex,
                ],
                alpha=0.9,
                kind="random",
                args=args,
            )
        else: 
            test_loader = data_module.test_dataloader()
        
        dataloaders = {
            "train": train_loader,
            "valid": val_loader,
            "test": test_loader,
        }

    elif "embed" in args.hps:
        data_module = EmbedDataModule(args=args)
        batch_size = args.batch_size

        # ---------- Balanced Train Loader (by circle) ----------
        if balance_train:
            train_loader = get_balanced_loader(
                dataset=data_module.dataset_train,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                class_arrays=[
                    data_module.dataset_train.circle,
                ],
                alpha=0.75,
                kind="random",
            )
        else:
            train_loader = data_module.train_dataloader()

        # ---------- Balanced Val Loader (by circle) ----------
        if balance_train:
            val_loader = get_balanced_loader(
                dataset=data_module.dataset_val,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                class_arrays=[
                    data_module.dataset_val.circle,
                ],
                alpha=0.75,
                kind="random",
            )
        else:
            val_loader = data_module.val_dataloader()

        # ---------- Balanced test Loader (by circle) ----------
        if balance_test:
            test_loader = get_balanced_test_loader(
                dataset=data_module.dataset_test,
                batch_size=batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                class_arrays=[
                    data_module.dataset_test.circle,
                    # data_module.dataset_test.density,
                ],
                alpha=0.9,
                kind="random",
                args=args,
            )
        else: 
            test_loader = data_module.test_dataloader()
        dataloaders = {
            "train": train_loader,
            "valid": val_loader,
            "test": test_loader,
        }
    
    elif "celeba" in args.hps:
        data_module = CelebADataModule(args=args)

        # Get dataset instance and attribute labels for balancing
        train_dataset = data_module.dataset_train
        val_dataset = data_module.dataset_val
        test_dataset = data_module.dataset_test  # You might not balance this

        # Compute composite class arrays for balanced sampling
        train_class_array = train_dataset.get_class_array_for_sampling(["Eyeglasses", "Young"])
        val_class_array = val_dataset.get_class_array_for_sampling(["Eyeglasses", "Young"])

        # Create balanced loaders
        if balance_train:
            train_loader = get_balanced_loader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                class_arrays=train_class_array,
                alpha=0.75,
                kind="random",
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                shuffle=True,
            )
        if balance_train:
            val_loader = get_balanced_loader(
                dataset=val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                class_arrays=val_class_array,
                alpha=0.75,
                kind="random",
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                shuffle=True,
            )


        if balance_test:
            test_class_array = test_dataset.get_class_array_for_sampling(["Eyeglasses", "Young"])
            test_loader = get_balanced_test_loader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                class_arrays=test_class_array,
                alpha=0.75,
                kind="random",
                args=args,
            )
        else:
            # Default test loader (no balancing)
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                shuffle=False,
            )

        dataloaders = {
            "train": train_loader,
            "valid": val_loader,
            "test": test_loader,
        }

    else:
        raise NotImplementedError(f"Unsupported dataset in hps: {args.hps}")

    return dataloaders
