# RNN Transducer for TIMIT Grapheme ASR

This project implements an RNN Transducer (RNN-T) speech-to-text system in PyTorch for the TIMIT dataset. It follows the encoder, prediction network, and joint network design from the original RNN-T paper, but trains directly on graphemes instead of phonemes. The target vocabulary is built from transcript characters, so the model predicts written text characters end to end from acoustic features.

![RNN-T architecture](https://user-images.githubusercontent.com/61272193/156832630-ad0c7d31-b262-470e-9b88-77088adf90ff.png)

Architecture image from [msalhab96/RNN-Transducer](https://github.com/msalhab96/RNN-Transducer).

## What It Does

- Loads preprocessed TIMIT examples from CSV splits under `timit/splits/`.
- Reads frame-level acoustic features from `.npy` files referenced by each row's `feature_path`.
- Builds a character-level tokenizer from the training transcripts.
- Trains an RNN-T model with:
  - bidirectional LSTM encoder,
  - embedding-fed LSTM prediction network,
  - learnable joint network over encoder and prediction representations,
  - RNN-T loss from `warprnnt_pytorch`.
- Evaluates recognition quality with character error rate (CER).
- Saves checkpoints locally and logs metrics to Weights & Biases.

## Repository Layout

- `rnnt/model.py` defines the transducer, bidirectional LSTM encoder, prediction network, joint network, loss, and greedy/beam recognizers.
- `rnnt/data.py` defines the PyTorch dataset, dataloaders, and batch padding.
- `rnnt/tokenizer.py` builds the grapheme vocabulary.
- `rnnt/train.py` runs training, validation, checkpointing, and logging.
- `rnnt/search.py` writes decoded train, validation, and test transcripts.
- `config/config.yaml` contains model, data, training, and optimizer settings.
- `timit/raw/` contains the original TIMIT directory tree.
- `timit/features/` contains generated acoustic feature `.npy` files, mirroring the raw TIMIT tree.
- `timit/metadata/` contains source manifests used to scan the raw train/test trees.
- `timit/splits/` contains the train, validation, and test CSVs used by training.
- `timit/notebooks/preprocess.ipynb` regenerates metadata, splits, and features.
- `timit/outputs/` contains decoded transcript outputs.

## Data

The expected CSV format is:

```csv
audio_path,feature_path,transcript,duration
timit/raw/TRAIN/DR4/MMDM0/SI681.WAV,timit/features/TRAIN/DR4/MMDM0/SI681.npy,would such an act of refusal be useful,39936
```

For every `audio_path`, the loader reads acoustic features from `feature_path`, for example:

```text
timit/features/TRAIN/DR4/MMDM0/SI681.npy
```

The default configuration uses the core TIMIT splits:

- `timit/splits/core_train.csv`
- `timit/splits/core_val.csv`
- `timit/splits/core_test.csv`

The preprocessing notebook is at:

```text
timit/notebooks/preprocess.ipynb
```

## Setup

Create an environment and install the Python dependencies used by the code:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `warprnnt-pytorch` fails to build on the cluster, install it separately using the cluster's CUDA/PyTorch-specific instructions after installing the rest of the requirements.
The TIMIT files in this repo are NIST SPHERE files. The preprocessing notebook loads them with `soundfile` and uses `torchaudio` only for Kaldi-compatible MFCC extraction, so it does not require TorchCodec or FFmpeg.

## Training

From the repository root, run:

```bash
python rnnt/train.py
```

The default config trains for 100 epochs with batch size 1, SGD with momentum, gradient clipping, validation every epoch, and checkpoint saving every 5 epochs.

Training outputs are written to `info/` by default:

- `info/logger` for logs,
- `info/*.epoch` for checkpoints.

When `training.export_transcriptions_after_training` is enabled, training also
decodes the final restored model with each method in
`training.export_decode_methods` and writes tab-separated prediction/target
files to `timit/outputs/`:

- `timit/outputs/transcriptions_train_greedy.tsv`
- `timit/outputs/transcriptions_val_greedy.tsv`
- `timit/outputs/transcriptions_test_greedy.tsv`
- `timit/outputs/transcriptions_train_beam.tsv`
- `timit/outputs/transcriptions_val_beam.tsv`
- `timit/outputs/transcriptions_test_beam.tsv`

Metrics are logged to Weights & Biases when `wandb.enabled` is true in `config/config.yaml`.
Authenticate once before training:

```bash
wandb login
```

## Inference

To decode all train, validation, and test splits with the current model
initialization/checkpoint settings:

```bash
python rnnt/search.py
```

The decoded and reference transcripts are written as TSV files with
`sample_id`, `prediction`, and `target` columns:

```text
timit/outputs/transcriptions_train_greedy.tsv
timit/outputs/transcriptions_val_greedy.tsv
timit/outputs/transcriptions_test_greedy.tsv
timit/outputs/transcriptions_train_beam.tsv
timit/outputs/transcriptions_val_beam.tsv
timit/outputs/transcriptions_test_beam.tsv
```

## Notes

The original RNN-T paper trained the model on phoneme sequences for TIMIT. This project instead uses the text transcripts directly and optimizes over grapheme targets, making the output a character-level speech-to-text system rather than a phoneme recognizer.

The tokenizer treats the literal space character as a normal grapheme. It also adds `"<pad>"` for batching and `"<blank>"` for the RNN-T null transition. `"<blank>"` is prepended only to the prediction-network input history and is not stored as a transcript target.

Checkpoints produced by older versions of this repository are not compatible with the current model shapes.

Reference paper:

```bibtex
@article{DBLP:journals/corr/abs-1211-3711,
  author = {Alex Graves},
  title = {Sequence Transduction with Recurrent Neural Networks},
  journal = {CoRR},
  volume = {abs/1211.3711},
  year = {2012},
  url = {http://arxiv.org/abs/1211.3711}
}
```
