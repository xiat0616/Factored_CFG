# Release validation

Validated on 27 October 2026 with Python 3.12.13 and PyTorch 2.10.0+cu128. All computation used CPU with CUDA hidden and one compute thread. Missing validation dependencies were placed in a temporary directory; existing environments were left unchanged.

## Passed checks

- All five synthetic tests in `tests/test_smoke.py`: known-prediction CFG/FCFG weight checks; finite CFG++ and APG DDIM outputs; conditional U-Net forward/backward; CelebA, MIMIC, and EMBED image/label readers; RGB reconstruction and counterfactual/reversal export with exact sample counts, including a partial final batch.
- One tiny diffusion training epoch using synthetic CelebA-format images and split CSVs, followed by checkpoint save/reload, single and joint interventions, and reversed counterfactual generation. Each requested output CSV contained exactly three samples.
- Help/import checks for `main.py`, `sample.py`, `sample_baselines.py`, `causal_model.pgm.train_pgm`, and `eval_cf.py`.
- Evaluation dataset loading with RGB images and the two-attribute CelebA variant.
- Shell syntax and argument capture for all ten launch recipes, including verification of conditioning dimensions and parent keys in the diffusion recipes.
- Python syntax, local documentation links, basic credential/private-path pattern scanning, destination file hashes, and `git diff --check` for tracked changes.
- Byte-for-byte comparison of the core guidance engines, U-Net, conditioning embedder, and training loop against the source snapshot. The original research checkout remained clean.

The synthetic integration data, generated images, and test checkpoints stayed in temporary directories and are not included in the release. No original dataset or trained paper checkpoint was used. Full training, GPU behavior, pretrained perceptual/FID evaluation, and reproduction of the paper's numerical results have not been validated by these checks. The dependency ranges in `requirements.txt` are not a lockfile for the historical experiments.
