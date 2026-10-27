# Dataset preparation

Datasets, per-image annotations, research split CSVs, fixed evaluation indices, and trained checkpoints are not distributed with this repository. Obtain the data from its official source and follow its access and use terms. The medical loaders expect preprocessed images and local metadata, rather than downloading or converting the original datasets automatically.

Configure paths through `.env.example`. Relative paths are resolved from the working directory; run commands from the repository root. These variables are read when the Python process starts. The environment overrides are also used when loading an existing research checkpoint.

## CelebA-HQ

Use the images and attribute annotation file from [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ):

```text
data/celeba/
  CelebA-HQ-img/0.jpg, 1.jpg, ...
  CelebAMask-HQ-attribute-anno.txt
```

Set `CELEBA_ROOT` to the image directory and `CELEBA_ATTR_PATH` to the annotation file. The loader reads the standard attribute file with its count line, attribute header, and -1/+1 values. It maps labels to 0/1 and images to [-1, 1]. The row order must correspond to the numbered HQ image filenames.

If `CELEBA_SPLIT_DIR` does not contain `celeba_train.csv`, `celeba_val.csv`, and `celeba_test.csv`, the original splitting procedure creates an approximately 80/10/10 split with random state 42. Each saved CSV has filenames as its first/index column and the attribute columns. Supply the original split files locally when comparing against an existing experiment.

The provided recipes use `Smiling Male Young`, `Smiling Eyeglasses`, or `Smiling Male Young Mouth_Slightly_Open`. Attribute order must match the checkpoint. The original loader balancing settings are retained, including balancing CelebA batches by `Eyeglasses` and `Young`.

## MIMIC-CXR

Obtain [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/) separately. The loader expects this local layout:

```text
data/mimic/
  data/<relative image paths>
  meta/mimic.sample.train.csv
  meta/mimic.sample.val.csv
  meta/mimic.sample.test.csv
```

Set `MIMIC_ROOT` to the parent directory and optionally set `MIMIC_META` to a different metadata directory. Prepare each CSV locally with these columns:

| Column | Meaning expected by the loader |
| --- | --- |
| `path_preproc` | Image path relative to `MIMIC_ROOT/data`, or an absolute local image path |
| `disease` | `No Finding`, `Pleural Effusion`, or `Other` (the last is excluded) |
| `sex`, `race` | Labels used in dataset distribution reporting |
| `sex_label` | Binary sex encoding used for the experiment |
| `race_label` | Integer race encoding in {0, 1, 2} |
| `age` | Age in years; divided by 100 by the loader |

Keep label encodings consistent with any predictor and diffusion checkpoints. Use subject-disjoint train/validation/test splits. The existing loader converts images to grayscale, normalizes each by its maximum, resizes, center crops, and scales to [-1, 1]. The raw DICOM conversion and construction of these research CSVs are not included.

## EMBED

Obtain [EMBED](https://registry.opendata.aws/emory-breast-imaging-dataset-embed/) separately. Set `EMBED_META` to your local split directory. The default loader uses `embed_train.csv`, `embed_val.csv`, and `embed_test.csv`; `--use_manual True` selects files with the suffix `_manual.csv` instead. The provided diffusion recipe uses the manual-labelled split files.

Each split CSV needs:

| Column | Meaning expected by the loader |
| --- | --- |
| `image_path` | Accessible local path to a preprocessed grayscale image |
| `shortimgpath` | Original short image identifier |
| `SimpleModelLabel` | Scanner/model code |
| `age_at_study` | Age in years |
| `tissueden` | Integer density category: A=0, B=1, C=2, D=3 |
| `circle`, `triangle` | Binary marker annotations |

The released experiment recipe conditions on `density circle`. The loader converts density to 0 for A/B and 1 for C/D. It keeps the largest foreground component, resizes the image, and scales it to [-1, 1].

`data_handling.embed.split_and_save_csvs` retains the original filtering and patient-level splitting utility for a prejoined metadata table plus marker annotations. `EMBED_ROOT` is used by that utility when constructing image paths under `images/png/1024x768`. The original joined metadata, manual marker annotations, and model-predicted marker annotations are not provided, so exact reconstruction of the research splits requires those local inputs.

## Fixed evaluation samples

Balanced test sampling saves `fixed_test_indices.npy` beneath `FCFG_EVAL_INDEX_DIR/<dataset>/` on its first run and reuses it afterwards. The original procedure draws 3,000 evaluation samples rounded down to a multiple of the batch size; sampling is with replacement. Keep the same split ordering, batch size, and fixed indices for comparisons. Move an old index file aside when changing datasets or splits; indices alone do not identify which CSV they came from.
