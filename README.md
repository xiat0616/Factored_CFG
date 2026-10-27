# Factored Classifier-Free Guidance

Official implementation of **Factored Classifier-Free Guidance (FCFG)**, ICML 2026.

**Tian Xia, Fabio De Sousa Ribeiro, Rajat R Rasal, Avinash Kori, Raghav Mehta, and Ben Glocker**

Imperial College London

[Paper](https://proceedings.mlr.press/v306/xia26b.html) · [arXiv](https://arxiv.org/abs/2506.14399) · [Data preparation](docs/DATA.md)

FCFG gives separate guidance strengths to semantic attributes or attribute groups during diffusion sampling. For counterfactual generation, this lets us strengthen an intervention while controlling changes to other attributes. The repository includes conditional diffusion models, DDIM inversion, counterfactual generation, and FCFG extensions to CFG++ and APG.

## Code overview

This repository provides training, counterfactual generation, and evaluation code for FCFG on CelebA-HQ, MIMIC-CXR, and EMBED.

| Component | Location |
| --- | --- |
| Attribute-split conditioning | `guided_diffusion/cond_embed.py` |
| CFG and FCFG | `guided_diffusion/gaussian_diffusion.py` |
| CFG++ / FCFG++ | `guided_diffusion/gaussian_diffusion_cfgpp.py` |
| APG / factored APG | `guided_diffusion/gaussian_diffusion_apg.py` |
| DDIM inversion, counterfactuals, and reversal | `guided_diffusion/sampling_utils.py` |
| Diffusion training | `main.py`, `guided_diffusion/trainer.py` |
| Data loaders | `data_handling/` |
| Attribute predictors / causal models | `causal_model/pgm/` |
| Counterfactual evaluation | `eval_cf.py`, `eval_utils.py` |
| Experiment recipes | `scripts/` |

**Datasets and pretrained checkpoints are not included.** See the [data preparation guide](docs/DATA.md) for setup instructions. Sampling requires a trained diffusion checkpoint; predictor-based evaluation requires a matching attribute predictor.

## Installation

Use Python 3.10 or newer. A CUDA-enabled PyTorch installation is recommended for training and image generation.

```bash
git clone https://github.com/xiat0616/Factored_CFG.git
cd Factored_CFG
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a matching torch/torchvision build for your CUDA version if needed.
python -m pip install -r requirements.txt
```

Run commands from the repository root. Check the installation and run the CPU smoke tests with:

```bash
python -m unittest discover -s tests -v
python main.py --help
python sample.py --help
python sample_baselines.py --help
python -m causal_model.pgm.train_pgm --help
python eval_cf.py --help
```

The smoke tests use synthetic inputs and require no datasets, checkpoints, or downloaded model weights.

## Data configuration

Follow [docs/DATA.md](docs/DATA.md) for dataset layouts, CSV columns, label encodings, and split behavior. Start with:

```bash
cp .env.example .env
# Edit .env to point to your local data and metadata.
set -a
source .env
set +a
```

| Dataset | Main experiment attributes | Image size in provided recipe |
| --- | --- | --- |
| CelebA-HQ | `Smiling Male Young` | 64 × 64, RGB |
| MIMIC-CXR | `age race sex finding` | 192 × 192, grayscale |
| EMBED | `density circle` | 192 × 192, grayscale |

The loaders construct train, validation, and test datasets together, so all required split files must be available even when sampling. Existing checkpoints determine architecture and attribute ordering; environment variables determine local data paths.

## Train a diffusion model

Launch training for each dataset with the following scripts. Each script uses the active Python environment and accepts command-line overrides.

```bash
bash scripts/train_celeba.sh
bash scripts/train_mimic.sh
bash scripts/train_embed.sh
```

Additional CelebA-HQ recipes:

```bash
bash scripts/train_celeba_smiling_eyeglasses.sh
bash scripts/train_celeba_four_attributes.sh
```

The latter conditions on `Smiling Male Young Mouth_Slightly_Open`. To override settings, append arguments, for example:

```bash
bash scripts/train_celeba.sh --batch_size 32 --epochs 100 --exp_name celeba_example
```

Checkpoints are saved to `checkpoints/<parents joined by underscores>/<exp_name>/`, including `last.pt` and the best-validation `checkpoint.pt`. Resume with `--resume /path/to/last.pt`. Use `PYTHON=/path/to/python bash scripts/train_celeba.sh` to select a particular interpreter.

## Generate counterfactuals

Set `CKPT` to a checkpoint trained with `Smiling Male Young`, such as the output of the main CelebA recipe:

```bash
CKPT=/path/to/last.pt
```

### Standard global CFG

```bash
python sample.py \
  --ckpt_path "$CKPT" \
  --sample_types cf \
  --sample_do_keys Smiling \
  --timestep_respacing ddim250 \
  --cfg_mode global --cfg_w 3.0 \
  --batch_size 20 --num_samples 100 \
  --results_dir sampling_results
```

### FCFG with intervened and invariant groups

```bash
python sample.py \
  --ckpt_path "$CKPT" \
  --sample_types cf \
  --sample_do_keys Smiling \
  --timestep_respacing ddim250 \
  --cfg_mode interv_group \
  --cfg_w_intervened 3.0 --cfg_w_unintervened 1.0 \
  --batch_size 20 --num_samples 100 \
  --results_dir sampling_results
```

These weights are illustrative settings. Choose them for your experiment. The standard implementation combines predictions as:

```text
eps_fcfg = eps_uncond
         + w_intervened   * (eps_intervened   - eps_uncond)
         + w_unintervened * (eps_unintervened - eps_uncond)
```

`interv_group` groups the keys passed in `--sample_do_keys` and their complement. For a causal graph requiring descendant updates, compute the counterfactual parent values and appropriate grouping before calling the sampler. The sampler flips binary attributes, cycles race labels, or resamples age. Parent-PGM propagation must be applied separately when required by the causal graph.

### Per-attribute weights and multiple interventions

```bash
python sample.py \
  --ckpt_path "$CKPT" \
  --sample_types cf \
  --sample_do_keys Smiling Smiling_Male \
  --cfg_mode per_attr \
  --cfg_attr_weights "{'Smiling': 3.0, 'Male': 1.0, 'Young': 1.0}" \
  --timestep_respacing ddim250 \
  --batch_size 20 --num_samples 100
```

An underscore joins attributes for a joint intervention; separate arguments request separate runs. Attribute names containing underscores, such as `Mouth_Slightly_Open`, are matched against checkpoint parent names. In `per_attr` mode, an omitted attribute contributes zero guidance. Factored modes require the attribute-split `CondEmbedder` (`global_cond=False`); a global MLP embedding does not provide meaningful per-attribute chunks.

Other sample types are `rs` for random samples and `rec` for reconstruction. Counterfactual runs also generate reversed counterfactuals. Use a fresh results directory for each new comparison: the sampling scripts append metadata when a CSV already exists.

### CFG++ and APG

Use `sample_baselines.py` to select the extended guidance engines. For example:

```bash
DIFF_ENGINE=cfgpp python sample_baselines.py \
  --ckpt_path "$CKPT" --sample_types cf --sample_do_keys Smiling \
  --cfg_mode interv_group \
  --cfg_w_intervened 0.8 --cfg_w_unintervened 0.2 \
  --timestep_respacing ddim250 --batch_size 20 --num_samples 100

DIFF_ENGINE=apg python sample_baselines.py \
  --ckpt_path "$CKPT" --sample_types cf --sample_do_keys Smiling \
  --cfg_mode interv_group \
  --cfg_w_intervened 3.0 --cfg_w_unintervened 1.0 \
  --apg_eta 0.0 --apg_momentum 0.0 --apg_norm_threshold 0.0 \
  --timestep_respacing ddim250 --batch_size 20 --num_samples 100
```

`DIFF_ENGINE` accepts `cfg`, `cfgpp`, or `apg` and is read when the process starts. CFG++ and APG outputs use engine suffixes in their result folder names. Use the default `--cfg_choice eq1` for result-folder naming; this argument does not select the guidance formula.

The APG examples use zero momentum. Temporal momentum requires buffers passed through the Python API; setting `--apg_momentum` alone does not activate it.

## Evaluate counterfactuals

Train a matching attribute predictor first:

```bash
bash scripts/train_predictor_celeba.sh
# Other datasets: train_predictor_mimic.sh or train_predictor_embed.sh
# CelebA variants: train_predictor_celeba_smiling_eyeglasses.sh
#                  train_predictor_celeba_four_attributes.sh
```

Predictor checkpoints are saved under `pgm_checkpoints/<parents>/<exp_name>/` as `checkpoint.pt` (best validation) and `checkpoint_current.pt` (latest). CelebA and EMBED predictor recipes use `sup_aux`; MIMIC and the four-attribute CelebA recipe use `sup_determ`. Use `--pgm_setup sup_pgm` to fit parent distributions.

```bash
python eval_cf.py \
  --pred_path /path/to/predictor/checkpoint.pt \
  --exp_name <diffusion_experiment_name> \
  --sample_do_keys Smiling Smiling_Male \
  --timestep_respacing ddim250 \
  --results_dir sampling_results \
  --skip_fid
```

Replace `<diffusion_experiment_name>` with the name stored in the diffusion checkpoint. The predictor must use the same dataset, image size, and ordered parent list as the diffusion experiment. The evaluator reads sampling CSVs and saves `all_cf_eval_results.csv` and comparison plots. Metrics include attribute accuracy/AUROC, F1, amplification measures, pixel MAE, and LPIPS where applicable; omit `--skip_fid` to additionally calculate input-versus-generated FID.

LPIPS and FID may download their evaluation network weights on first use. For comparisons with the paper, use matching trained models, preprocessing, data splits, evaluation indices, and guidance settings.

## Citation

```bibtex
@InProceedings{pmlr-v306-xia26b,
  title     = {Factored Classifier-Free Guidance},
  author    = {Xia, Tian and De Sousa Ribeiro, Fabio and Rasal, Rajat R and Kori, Avinash and Mehta, Raghav and Glocker, Ben},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages     = {138087--138140},
  year      = {2026},
  volume    = {306},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v306/xia26b.html}
}
```

## Acknowledgements and license

This implementation builds on [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion), [causal-gen](https://github.com/biomedia-mira/causal-gen), [ema-pytorch](https://github.com/lucidrains/ema-pytorch), [CLIP](https://github.com/openai/CLIP), and [torchvision](https://github.com/pytorch/vision). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [licenses/](licenses/) for their license notices.

Original project contributions use the [MIT license](LICENSE). Dataset and external model terms are separate.
