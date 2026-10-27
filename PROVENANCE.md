# Source provenance and release changes

This release was selected from the working tree of the research repository `diffusion_counterfactual`, at commit `013eeb82426aebfaf16771df3001e58e643b4267`. The source working tree was clean when inspected on 27 October 2026. The destination repository's original MIT license is retained.

## Preserved implementation

The guidance formulas, diffusion engines, U-Net architecture, conditioning embedder, optional semantic encoder, neural-network utilities, training loop, and causal/predictor model definitions were copied without algorithmic changes. Historical bare module names are retained through a repository-relative import-path helper for compatibility with research checkpoints.

Byte-for-byte unchanged Python files from the source snapshot:

- `causal_model/__init__.py`
- `causal_model/pgm/__init__.py`
- `causal_model/pgm/celeba_pgm.py`
- `causal_model/pgm/celeba_pgm_2.py`
- `causal_model/pgm/celeba_pgm_s_e.py`
- `causal_model/pgm/embed_pgm.py`
- `causal_model/pgm/mimic_pgm.py`
- `causal_model/pgm/pgm_layers.py`
- `causal_model/pgm/pgm_resnet.py`
- `causal_model/pgm/utils_pgm.py`
- `data_handling/__init__.py`
- `data_handling/base.py`
- `data_handling/sampler.py`
- `guided_diffusion/__init__.py`
- `guided_diffusion/cond_embed.py`
- `guided_diffusion/gaussian_diffusion.py`
- `guided_diffusion/gaussian_diffusion_apg.py`
- `guided_diffusion/gaussian_diffusion_cfgpp.py`
- `guided_diffusion/losses.py`
- `guided_diffusion/nn_utils.py`
- `guided_diffusion/plot_util.py`
- `guided_diffusion/resample.py`
- `guided_diffusion/respace.py`
- `guided_diffusion/respace_baselines.py`
- `guided_diffusion/semantic_encoder.py`
- `guided_diffusion/trainer.py`
- `guided_diffusion/unet.py`
- `guided_diffusion/utils.py`
- `guided_diffusion/utils_baselines.py`
- `hps.py`

## Portability and runtime fixes

- Replace institutional code and data paths with repository-relative imports and configurable environment variables (`project_paths.py`, `.env.example`). Create split directories only when needed, and store balanced evaluation indices under a configurable local directory.
- Guard CUDA device selection to allow small CPU runs. Import the optional TensorBoard writer only when requested.
- Replace the removed pandas `delim_whitespace` argument with its whitespace-separator equivalent. Restore a missing NumPy import and handle an absent marker-annotation path in the EMBED split helper.
- Remove a MIMIC diagnostic block that assumed every test loader was a `Subset`, which prevented ordinary training loaders from being used.
- Make reconstruction export support RGB images and datasets without `short_path`. Make CelebA counterfactual display titles use the checkpoint attributes. Respect the requested sample count and actual size of a partial final batch.
- Resolve intervention names against checkpoint parent names, including `Mouth_Slightly_Open`, and reject factored guidance for globally embedded checkpoints.
- Allow evaluation to select its results root and intervention folders, preserve the predictor's manual-label selection, load the appropriate CelebA predictor variant, and avoid modifying the supplied parent list when plotting. Keep amplification tensors one-dimensional for singleton batches.

Modified copied Python files:

- `causal_model/pgm/train_pgm.py`
- `data_handling/celeba.py`
- `data_handling/embed.py`
- `data_handling/mimic.py`
- `eval_cf.py`
- `eval_utils.py`
- `guided_diffusion/sampling_utils.py`
- `main.py`
- `sample.py`
- `sample_baselines.py`
- `train_setup.py`

## Experiment recipes

Portable launchers preserve the research settings from `train_celeba_cfg_64.sh`, `train_celeba_cfg_64_s_g.sh`, `train_celeba_cfg_64_2.sh`, `train_mimic_cfg_192.sh`, `train_embed_cfg_192.sh`, and the corresponding PGM launchers. Scheduler directives and named environment activation are removed; scripts use the active Python (or `PYTHON`) and accept trailing CLI overrides. The two-attribute CelebA predictor launcher defaults to `sup_aux` instead of its source launcher's `sup_pgm`, so that it trains an image-based evaluation predictor.

## Excluded artifacts and scope

No original dataset files, per-image metadata, annotations, fixed test indices, checkpoints, images, notebooks, logs, scheduler output, copied baseline repositories, legacy code, or backup implementations are included. Third-party notices are supplied separately.

The retained sampler directly modifies parent values and does not automatically propagate interventions through a fitted parent PGM. The APG CLI retains the original zero-momentum behavior unless momentum buffers are supplied programmatically. The historical `cfg_choice` label does not switch formulas in these engines. These limits are documented in the README rather than changing the research algorithms during packaging.

Synthetic CPU tests check guidance weighting, extended-engine sampling, a U-Net backward pass, dataset readers, and RGB/counterfactual output. This is a source release; exact paper-result reproduction additionally requires the original experimental artifacts and settings.
