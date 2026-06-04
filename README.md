# RNN Transducer for TIMIT Grapheme ASR

This project implements an RNN Transducer (RNN-T) speech-to-text system in PyTorch for the TIMIT dataset. It follows the encoder, prediction network, and joint network design from the original RNN-T paper, but trains directly on graphemes instead of phonemes. The target vocabulary is built from transcript characters, so the model predicts written text characters end to end from acoustic features.

![RNN-T architecture](https://user-images.githubusercontent.com/61272193/156832630-ad0c7d31-b262-470e-9b88-77088adf90ff.png)

Architecture image from [msalhab96/RNN-Transducer](https://github.com/msalhab96/RNN-Transducer).

## What It Does

- Loads preprocessed TIMIT examples from CSV splits under `timit/`.
- Reads frame-level acoustic features from `.npy` files generated next to each `.WAV`.
- Builds a character-level tokenizer from the training transcripts.
- Trains an RNN-T model with:
  - bidirectional LSTM encoder,
  - LSTM prediction network,
  - additive joint network,
  - RNN-T loss from `warprnnt_pytorch`.
- Evaluates recognition quality with character error rate (CER).
- Saves checkpoints and TensorBoard logs under the configured experiment directory.

## Repository Layout

- `rnnt/model.py` defines the transducer, joint network, loss, and greedy recognizer.
- `rnnt/encoder.py` implements the bidirectional LSTM encoder.
- `rnnt/decoder.py` implements the LSTM prediction network over grapheme IDs.
- `rnnt/data.py` loads padded acoustic features and character targets.
- `rnnt/tokenizer.py` builds the grapheme vocabulary.
- `rnnt/train.py` runs training, validation, checkpointing, and logging.
- `rnnt/search.py` writes decoded validation transcripts.
- `config/config.yaml` contains model, data, training, and optimizer settings.
- `timit/` contains the TIMIT data, CSV splits, transcripts, and preprocessing notebook.

## Data

The expected CSV format is:

```csv
audio_path,transcript,duration
timit/data/TRAIN/DR4/MMDM0/SI681.WAV,would such an act of refusal be useful,39936
```

For every `audio_path`, the loader expects a matching `.npy` feature file beside the audio file, for example:

```text
timit/data/TRAIN/DR4/MMDM0/SI681.npy
```

The default configuration uses the core TIMIT splits:

- `timit/_core_train.csv`
- `timit/_core_val.csv`
- `timit/_core_test.csv`

## Setup

Create an environment and install the Python dependencies used by the code:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas pyyaml tensorboard editdistance warprnnt_pytorch
```

Install `warprnnt_pytorch` according to the platform-specific instructions for your PyTorch and CUDA setup. The model initialization succeeds without it, but training requires the RNN-T loss implementation.

## Training

From the repository root, run:

```bash
python rnnt/train.py
```

The default config trains for 100 epochs with batch size 1, SGD, gradient clipping, evaluation every 5 epochs, and checkpoint saving every 5 epochs.

Training outputs are written to `info/` by default:

- `info/logger` for logs,
- `info/visualizer/` for TensorBoard events,
- `info/*.epoch` for checkpoints.

To inspect TensorBoard logs:

```bash
tensorboard --logdir info/visualizer
```

## Inference

To decode the validation split with the current model initialization/checkpoint settings:

```bash
python rnnt/search.py
```

The decoded and reference transcripts are written to:

```text
timit/transcriptions_val.txt
```

## Notes

The original RNN-T paper trained the model on phoneme sequences for TIMIT. This project instead uses the text transcripts directly and optimizes over grapheme targets, making the output a character-level speech-to-text system rather than a phoneme recognizer.

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
