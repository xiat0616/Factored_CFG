"""Portable paths for data and the original research module layout."""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("FCFG_DATA_ROOT", REPO_ROOT / "data")).expanduser()


def data_path(variable, default):
    return Path(os.environ.get(variable, default)).expanduser()


CELEBA_ROOT = data_path("CELEBA_ROOT", DATA_ROOT / "celeba" / "CelebA-HQ-img")
CELEBA_ATTR_PATH = data_path("CELEBA_ATTR_PATH", DATA_ROOT / "celeba" / "CelebAMask-HQ-attribute-anno.txt")
CELEBA_SPLIT_DIR = data_path("CELEBA_SPLIT_DIR", DATA_ROOT / "celeba_meta")
EMBED_ROOT = data_path("EMBED_ROOT", DATA_ROOT / "embed")
EMBED_META = data_path("EMBED_META", DATA_ROOT / "embed_meta")
MIMIC_ROOT = data_path("MIMIC_ROOT", DATA_ROOT / "mimic")
MIMIC_META = data_path("MIMIC_META", MIMIC_ROOT / "meta")
EVAL_INDEX_ROOT = data_path("FCFG_EVAL_INDEX_DIR", DATA_ROOT / "eval_indices")


def setup_import_paths():
    """Keep historical module names usable when loading research checkpoints."""
    for path in (REPO_ROOT, REPO_ROOT / "guided_diffusion", REPO_ROOT / "causal_model" / "pgm"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
