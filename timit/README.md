# TIMIT Data Layout

This directory separates original data, generated features, split metadata, and model outputs.

```text
timit/
  raw/        Original TIMIT TRAIN/TEST tree.
  features/   Generated normalized MFCC + delta `.npy` files.
  metadata/   Raw tree manifests used by the preprocessing notebook.
  splits/     CSV files consumed by training and evaluation.
  notebooks/  Preprocessing notebook.
  outputs/    Decoded transcript outputs.
```

The training code reads `splits/core_train.csv`, `splits/core_val.csv`, and `splits/core_test.csv`.
Each split row stores both `audio_path` and `feature_path`; the dataset loader uses `feature_path`
for model input.

Run `notebooks/preprocess.ipynb` to regenerate metadata, splits, and feature files.
