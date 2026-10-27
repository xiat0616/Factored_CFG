import argparse
import ast
import sys

# Dictionary to store different hyperparameter configurations
HPARAMS_REGISTRY = {}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def value_for_none(v):
    """
    Convert 'None' string values or empty values to actual None.
    Otherwise, convert to float.
    """
    if not v or v == "None":
        return None
    return float(v)

class Hparams:
    """
    Class to store and manage hyperparameters.
    """
    def update(self, params_dict):
        """
        Update hyperparameters using a dictionary.
        """
        for k, v in params_dict.items():
            setattr(self, k, v)

# ==========================
# Mimic model
# ==========================
mimic = Hparams()
mimic.image_size = 256
mimic.in_channels = 1
mimic.out_channels = 1
mimic.model_channels = 32
mimic.num_res_blocks = 2
mimic.num_heads = 4
mimic.num_heads_upsample = -1
mimic.num_head_channels = -1
mimic.attention_resolutions = "16"
mimic.conv_resample = True
mimic.channel_mult = "1,2,3,4"
mimic.dropout = 0.0
mimic.dims = 2
mimic.use_checkpoint = False
mimic.resblock_updown = False
mimic.use_new_attention_order = False
mimic.embedding_mode = "concat_all"  # default for UNet usage
mimic.cfg_dropout_style = "cemb_only"  # Options: "together", "independent", "cemb_only"
mimic.ema_rate = 0.9999
mimic.time_embed_dim = 128
# Condition embedder
mimic.global_cond = False
mimic.cond_embed_dim = 128
# Semantic encoder, for diffae choice
mimic.use_semantic_encoder = False
mimic.sem_embed_dim = 128
mimic.use_vae_encoder = False
mimic.enc_channel_mult = "1,2,3,4"
mimic.kl_weight = 1e-4
mimic.kl_free_bits = 1e-2
mimic.lambda_decor = 0.0  # Weight for cosine similarity decorrelation loss between semb and cemb
# Diffusion
mimic.steps = 1000
mimic.sigma_small = False
mimic.sampler_name = "uniform"
mimic.noise_schedule = "linear"
mimic.use_kl = False
mimic.predict_xstart = False
mimic.rescale_timesteps = False
mimic.rescale_learned_sigmas = False
mimic.timestep_respacing = "ddim25"
# Data
mimic.parents = ["age", "race", "sex", "finding"]
mimic.shuffle = False
mimic.num_workers = 8
mimic.pin_memory = True
# Train
mimic.batch_size = 48
mimic.resume = ""
mimic.lr = 1e-4
mimic.wd = 1e-3
mimic.adam_betas = [0.9, 0.9999]
mimic.epochs = 5000
mimic.eval_freq = 1
mimic.start_epoch = 0
mimic.accu_steps = 1
mimic.grad_skip = 2500
mimic.grad_clip = 350
mimic.save_dir = None
mimic.plt_freq = 1000
mimic.lr_warmup_steps = 100
mimic.seed = 7
mimic.deterministic = False
# Classifier-free guidance
mimic.cfg_train = True  # True for CFG train and False for non CFG train.
mimic.cfg_w = 3.0       # CFG weight for inference
mimic.cfg_dropout = 0.5 # CFG dropout, 0 for fully conditional and 1 for fully unconditional.
HPARAMS_REGISTRY["mimic"] = mimic

# ==========================
# EMBED model
# ==========================
embed = Hparams()
embed.image_size = 192
embed.in_channels = 1
embed.out_channels = 1
embed.model_channels = 32
embed.num_res_blocks = 2
embed.num_heads = 4
embed.num_heads_upsample = -1
embed.num_head_channels = -1
embed.attention_resolutions = "16"
embed.conv_resample = True
embed.channel_mult = "1,2,3,4,5,6"
embed.dropout = 0.0
embed.dims = 2
embed.use_checkpoint = False
embed.resblock_updown = False
embed.use_new_attention_order = False
embed.embedding_mode = "concat_all"
embed.cfg_dropout_style = "cemb_only"
embed.ema_rate = 0.9999
embed.time_embed_dim = 128
# Condition embedder
embed.global_cond = False
embed.cond_embed_dim = 128
# Semantic encoder
embed.use_semantic_encoder = False
embed.sem_embed_dim = 128
embed.use_vae_encoder = False
embed.enc_channel_mult = "1,2,3,4,5,6,8"
embed.kl_weight = 1e-4
embed.kl_free_bits = 1e-2
embed.lambda_decor = 0.0
# Diffusion
embed.steps = 1000
embed.sigma_small = False
embed.sampler_name = "uniform"
embed.noise_schedule = "linear"
embed.use_kl = False
embed.predict_xstart = False
embed.rescale_timesteps = False
embed.rescale_learned_sigmas = False
embed.timestep_respacing = "ddim25"
# Data
embed.parents = ["sex", "scanner", "density", "BI-RADS"]
embed.shuffle = False
embed.num_workers = 8
embed.pin_memory = True
# Train
embed.batch_size = 48
embed.resume = ""
embed.lr = 1e-4
embed.wd = 1e-3
embed.adam_betas = [0.9, 0.9999]
embed.epochs = 5000
embed.eval_freq = 1
embed.start_epoch = 0
embed.accu_steps = 1
embed.grad_skip = 2500
embed.grad_clip = 350
embed.save_dir = None
embed.plt_freq = 1000
embed.lr_warmup_steps = 100
embed.seed = 7
embed.deterministic = False
embed.cfg_train = True
embed.cfg_w = 3.0
embed.cfg_dropout = 0.5
HPARAMS_REGISTRY["embed"] = embed

# ==========================
# CelebA model
# ==========================
celeba = Hparams()
celeba.image_size = 256
celeba.in_channels = 3
celeba.out_channels = 3
celeba.model_channels = 64
celeba.num_res_blocks = 2
celeba.num_heads = 4
celeba.num_heads_upsample = -1
celeba.num_head_channels = -1
celeba.attention_resolutions = "16"
celeba.conv_resample = True
celeba.channel_mult = "1,2,2,2"
celeba.dropout = 0.0
celeba.dims = 2
celeba.use_checkpoint = False
celeba.resblock_updown = False
celeba.use_new_attention_order = False
celeba.embedding_mode = "concat_all"
celeba.cfg_dropout_style = "cemb_only"
celeba.ema_rate = 0.9999
celeba.time_embed_dim = 128
celeba.global_cond = False
celeba.cond_embed_dim = 128
celeba.use_semantic_encoder = False
celeba.sem_embed_dim = 128
celeba.use_vae_encoder = False
celeba.enc_channel_mult = "1,2,2,2"
celeba.kl_weight = 1e-4
celeba.kl_free_bits = 1e-2
celeba.lambda_decor = 0.0
celeba.steps = 1000
celeba.sigma_small = False
celeba.sampler_name = "uniform"
celeba.noise_schedule = "linear"
celeba.use_kl = False
celeba.predict_xstart = False
celeba.rescale_timesteps = False
celeba.rescale_learned_sigmas = False
celeba.timestep_respacing = "ddim25"
celeba.parents = ["Eyeglasses", "Young", "Male", "Smiling"]
celeba.shuffle = False
celeba.num_workers = 8
celeba.pin_memory = True
celeba.batch_size = 48
celeba.resume = ""
celeba.lr = 1e-4
celeba.wd = 1e-3
celeba.adam_betas = [0.9, 0.9999]
celeba.epochs = 5000
celeba.eval_freq = 1
celeba.start_epoch = 0
celeba.accu_steps = 1
celeba.grad_skip = 2500
celeba.grad_clip = 350
celeba.save_dir = None
celeba.plt_freq = 1000
celeba.lr_warmup_steps = 100
celeba.seed = 7
celeba.deterministic = False
celeba.cfg_train = True
celeba.cfg_w = 3.0
celeba.cfg_dropout = 0.5
HPARAMS_REGISTRY["celeba"] = celeba


def setup_hparams(parser):
    """
    Set up hyperparameters using argument parser and the predefined registry.
    """
    hparams = Hparams()
    if 'ipykernel' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    
    if args.hps not in HPARAMS_REGISTRY:
        raise ValueError(f"Hyperparameter set '{args.hps}' not found in registry.")

    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__

    valid_args = set(args.__dict__.keys())
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"Unknown hyperparameter '{k}' not in default arguments.")

    parser.set_defaults(**hparams_dict)

    hparams.update(parser.parse_known_args()[0].__dict__)
    hparams.channel_mult = tuple(int(ch_mult) for ch_mult in hparams.channel_mult.split(","))
    hparams.enc_channel_mult = tuple(int(ch_mult) for ch_mult in hparams.enc_channel_mult.split(","))
    return hparams

    
def add_arguments(parser):
    """
    Add command-line arguments for configuring the model.
    """
    # UNet Model
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=3, help="Number of output channels")
    parser.add_argument("--model_channels", type=int, default=128, help="Base number of model channels")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks per level")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num_heads_upsample", type=int, default=-1, help="Number of heads for upsampling")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="Fixed attention head width")
    parser.add_argument("--attention_resolutions", type=str, default="16,8", help="Resolutions where attention is applied")
    parser.add_argument("--channel_mult", type=str, default="1,2,3,4,5,6", help="Channel multiplier per level")
    parser.add_argument("--conv_resample", type=str2bool, default=True, help="Use conv_resample.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--dims", type=int, default=2, help="Model dimensionality (1D, 2D, or 3D)")
    parser.add_argument("--use_checkpoint", type=str2bool, default=False, help="Use gradient checkpointing")
    parser.add_argument("--resblock_updown", type=str2bool, default=False, help="Use residual blocks for up/downsampling")
    parser.add_argument("--use_new_attention_order", type=str2bool, default=False, help="Alternative attention order")
    parser.add_argument(
        "--embedding_mode",
        type=str,
        default="concat_cemb_semb",
        choices=["concat_cemb_semb", "semb_shift_scale", "add_all", "concat_all", "none"],
        help="Embedding mode to combine temb, cemb, semb in ResBlock",
    )
    parser.add_argument(
        "--cfg_dropout_style",
        type=str,
        default="cemb_only",
        choices=["together", "independent", "cemb_only"],
        help="Which embedding(s) to drop during CFG dropout.",
    )

    parser.add_argument("--plt_freq", type=int, default=1000, help="Plot frequency")
    parser.add_argument("--ema_rate", type=float, default=0.9999,
                        help="Decay rate for the EMA of model parameters.")
    parser.add_argument("--time_embed_dim", type=int, default=512, help="Time embedding dim")
    parser.add_argument(
        "--cond_embed_dim",
        type=int,
        default=96,
        help="Total dimensionality of the conditional embedding (divisible by num parents)",
    )
    # Condition embedder
    parser.add_argument(
        "--global_cond",
        type=str2bool,
        default=False,
        help="Use global MLP for conditional embedding instead of per-parent",
    )
    parser.add_argument(
        "--parent_dims",
        type=str,
        default="{'sex': 1, 'age':1, 'race': 3, 'finding':1}",
        help="Dict of parent label dimensions.",
    )
    # Semantic encoder
    parser.add_argument("--use_semantic_encoder", type=str2bool, default=False,
                        help="Whether we use the semantic encoding or not.")
    parser.add_argument("--sem_embed_dim", type=int, default=0, help="Semantic embedding dim")
    parser.add_argument("--use_vae_encoder", type=str2bool, default=False,
                        help="Whether to use VAE encoder for semantic encoding or not.")
    parser.add_argument("--enc_channel_mult", type=str, default="", help="Channel multiplier per level for enc")
    parser.add_argument("--kl_weight", type=float, default=1e-4, help="The weight for latent KL loss.")
    parser.add_argument("--kl_free_bits", type=float, default=1e-2,
                        help="Minimum KL per dim to prevent collapse.")
    parser.add_argument("--lambda_decor", type=float, default=0.0,
                        help="Weight for cosine similarity-based decorrelation loss between semantic and conditional embeddings.")

    # Diffusion
    parser.add_argument("--steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--sigma_small", type=str2bool, default=False, help="Use small sigma")
    parser.add_argument("--noise_schedule", type=str, default="linear", help="Noise schedule type")
    parser.add_argument("--use_kl", type=str2bool, default=False, help="Use KL divergence for loss")
    parser.add_argument("--sampler_name", type=str, default="uniform",
                        help="What type of diffusion sample to use.")
    parser.add_argument("--predict_xstart", type=str2bool, default=False,
                        help="Predict xstart in diffusion")
    parser.add_argument("--rescale_timesteps", type=str2bool, default=False,
                        help="Rescale timesteps")
    parser.add_argument("--rescale_learned_sigmas", type=str2bool, default=False,
                        help="Rescale learned sigmas")
    parser.add_argument("--timestep_respacing", type=str, default="ddim25",
                        help="Timestep respacing strategy")
    # Data
    parser.add_argument(
        "--parents",
        help="Parents of x to condition on.",
        nargs="+",
        default=["race", "age", "sex", "finding"],
    )
    parser.add_argument("--shuffle", type=str2bool, default=True, help="Shuffle data.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader.")
    parser.add_argument("--pin_memory", type=str2bool, default=True,
                        help="Pin memory option for dataloader.")
    parser.add_argument("--use_manual", type=str2bool, default=False,
                        help="Whether to use manual labelled data for EMBED or not.")
    # Train
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--wd", type=float, default=1e-3, help="Weight decay rate.")
    parser.add_argument("--lr_warmup_steps", type=int, default=100, help="lr warmup steps.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Starting training epoch.")
    parser.add_argument("--epochs", type=int, default=5000, help="Total training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--adam_betas",
        help="Adam beta parameters.",
        nargs="+",
        type=float,
        default=[0.9, 0.9999],
    )
    parser.add_argument("--accu_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--grad_skip", type=float, default=500,
                        help="Skip update grad norm threshold.")
    parser.add_argument("--grad_clip", type=float, default=350,
                        help="Gradient clipping value.")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name.")
    parser.add_argument("--hps", type=str, default="mimic", help="Hyperparameter set.")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--eval_freq", type=int, default=5,
                        help="Train epochs per validation.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save model checkpoints")
    parser.add_argument("--deterministic", type=str2bool, default=True,
                        help="Enable deterministic execution.")
    # CFG params
    parser.add_argument("--cfg_train", type=str2bool, default=True,
                        help="Whether we do CFG training or not.")
    parser.add_argument("--cfg_w", type=float, default=3.0,
                        help="CFG weight, only for inference.")
    parser.add_argument("--cfg_dropout", type=float, default=0.5,
                        help="CFG dropout, only for training.")
    return parser


def add_sample_arguments(parser):
    """
    Add command-line arguments for sampling (CFG / CFG++ / APG).
    """
    # ------------------------------
    # Checkpoint + diffusion config
    # ------------------------------
    parser.add_argument("--ckpt_path", type=str, default="", help="Path to checkpoint.")
    parser.add_argument("--timestep_respacing", type=str, default="ddim250",
                        help="Timestep respacing strategy")

    # ------------------------------
    # Sample types
    # ------------------------------
    parser.add_argument(
        "--sample_types",
        nargs="+",
        default=["rs", "rec", "cf"],
        help="Types of sampling: rs (random), rec (reconstruction), cf (counterfactual)",
    )

    parser.add_argument(
        "--sample_do_keys",
        nargs="+",
        default=["sex", "race", "finding", "age"],
        help=(
            "Specify intervention keys. "
            "For multiple interventions use underscore '_': "
            "Example: --sample_do_keys Smiling Smiling_Male Young_Male_Eyeglasses"
        ),
    )

    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of sampled images total")
    parser.add_argument("--results_dir", type=str, default="sampling_results",
                        help="Directory for saving sampling outputs")

    # ------------------------------
    # Dataloader options
    # ------------------------------
    parser.add_argument("--shuffle", type=str2bool, default=False)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--use_manual", type=str2bool, default=False,
                        help="Whether to use manually-labelled EMBED data")

    # ======================================================
    # CFG / CFG++ / APG inference options
    #   - CFG / CFG++ use cfg_mode + cfg_choice + cfg_w
    #   - APG uses cfg_w as guidance scale as well; APG-specific
    #     params only affect the *shape* of the guidance, not its scale.
    # ======================================================
    parser.add_argument(
        "--cfg_mode",
        type=str,
        default="global",
        choices=["none", "global", "per_attr", "interv_group"],
        help="CFG mode used by GaussianDiffusion / CFG++ / APG models",
    )

    parser.add_argument(
        "--cfg_choice",
        type=str,
        default="eq1",
        choices=["eq1", "eq2", "eq3"],
        help="CFG interpolation formula (for CFG / CFG++)",
    )

    # -- Global CFG
    parser.add_argument(
        "--cfg_w",
        type=float,
        default=1.0,
        help="Global CFG weight (also used as guidance scale w for APG).",
    )

    # -- Per-attribute CFG
    parser.add_argument(
        "--cfg_attr_weights",
        type=str,
        default="",
        help="Dict of per-attribute CFG weights. Example: \"{'sex':1.2,'age':0.8}\"",
    )

    # -- Intervention-group CFG
    parser.add_argument("--cfg_w_intervened", type=float, default=1.0)
    parser.add_argument("--cfg_w_unintervened", type=float, default=1.0)

    # ======================================================
    # APG-specific guidance settings
    #   (No separate scale; APG uses cfg_w as w)
    # ======================================================
    parser.add_argument(
        "--apg_eta",
        type=float,
        default=1.0,
        help="APG η (controls balance of parallel vs orthogonal components).",
    )
    parser.add_argument(
        "--apg_norm_threshold",
        type=float,
        default=0.0,
        help="Optional norm threshold r for APG clipping (0 disables clipping).",
    )
    parser.add_argument(
        "--apg_momentum",
        type=float,
        default=0.0,
        help="Reverse momentum strength β for APG (0 = no temporal momentum).",
    )

    return parser
