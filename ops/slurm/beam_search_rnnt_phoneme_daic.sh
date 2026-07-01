#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_rnnt_phoneme_beam_%j.out
#SBATCH --error=slurm_rnnt_phoneme_beam_%j.err
#SBATCH --job-name=rnnt_phoneme_beam
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

VENV_DIR="${VENV_DIR:-.venv-daic}"
BASE_CONFIG="${RNNT_CONFIG:-config/config_phoneme.yaml}"
RNNT_MODEL="${RNNT_MODEL:-info_phoneme/best.epoch}"
DECODE_OUTPUT_DIR="${DECODE_OUTPUT_DIR:-outputs_phoneme_beam}"
BEAM_WIDTH="${BEAM_WIDTH:-500}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONUNBUFFERED=1

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Virtualenv ${VENV_DIR} does not exist. Create it on a login node before sbatch."
    exit 1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Config ${BASE_CONFIG} does not exist."
    exit 1
fi

if [[ ! -f "${RNNT_MODEL}" ]]; then
    echo "Checkpoint ${RNNT_MODEL} does not exist."
    echo "Set RNNT_MODEL=/path/to/checkpoint when submitting this job."
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

mkdir -p "info_phoneme" "timit/${DECODE_OUTPUT_DIR}"

EVAL_CONFIG="$(mktemp "${TMPDIR:-/tmp}/rnnt_phoneme_beam_config.XXXXXX.yaml")"
export BASE_CONFIG
export RNNT_MODEL
export DECODE_OUTPUT_DIR
export BEAM_WIDTH
export RNNT_CONFIG="${EVAL_CONFIG}"

python - <<'PY'
import os
from pathlib import Path

import yaml

base_config = Path(os.environ["BASE_CONFIG"])
with base_config.open() as file:
    config = yaml.safe_load(file)

config["training"]["load_model"] = os.environ["RNNT_MODEL"]
config["training"]["export_transcriptions_after_training"] = False
config["training"]["export_decode_methods"] = ["beam"]
config["training"]["evaluate"] = False
config["wandb"]["enabled"] = False
config["data"]["transcriptions_dir"] = os.environ["DECODE_OUTPUT_DIR"]
config["model"]["decode"]["method"] = "beam"
config["model"]["decode"]["beam_width"] = int(os.environ["BEAM_WIDTH"])

eval_config = Path(os.environ["RNNT_CONFIG"])
with eval_config.open("w") as file:
    yaml.safe_dump(config, file, sort_keys=False)
PY

echo "Submit dir: ${SUBMIT_DIR}"
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-local}"
echo "Base config: ${BASE_CONFIG}"
echo "Eval config: ${RNNT_CONFIG}"
echo "Checkpoint: ${RNNT_MODEL}"
echo "Beam width: ${BEAM_WIDTH}"
echo "Output dir: timit/${DECODE_OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import torch

print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

# No-op when the split CSVs already contain complete `phonemes` targets.
python timit/add_phonemes_to_splits.py
python rnnt/search.py

python - <<'PY'
import csv
import os
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def edit_distance(left, right):
    prev = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        curr = [i]
        for j, right_token in enumerate(right, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (left_token != right_token),
                )
            )
        prev = curr
    return prev[-1]


output_dir = Path("timit") / Path(os.environ["DECODE_OUTPUT_DIR"])
summary_file = output_dir / "per_beam.tsv"
rows = []

for split in ("train", "val", "test"):
    transcription_file = output_dir / f"transcriptions_{split}_beam.tsv"
    total_edits = 0
    total_target_tokens = 0
    total_samples = 0

    with transcription_file.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in tqdm(reader, desc=f"{split} PER", unit="utt", dynamic_ncols=True):
            prediction = row["prediction"].split()
            target = row["target"].split()
            total_edits += edit_distance(prediction, target)
            total_target_tokens += len(target)
            total_samples += 1

    per = 100.0 * total_edits / total_target_tokens
    rows.append((split, total_samples, total_edits, total_target_tokens, per))

    per_file = output_dir / f"per_{split}_beam.txt"
    per_file.write_text(
        f"split\t{split}\n"
        f"samples\t{total_samples}\n"
        f"edits\t{total_edits}\n"
        f"target_tokens\t{total_target_tokens}\n"
        f"per\t{per:.6f}\n"
    )
    print(f"{split} beam PER: {per:.6f}%")

with summary_file.open("w", newline="") as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerow(["split", "samples", "edits", "target_tokens", "per"])
    writer.writerows(rows)

print(f"Wrote PER summary to {summary_file}")
PY

echo "Beam transcriptions:"
echo "  timit/${DECODE_OUTPUT_DIR}/transcriptions_train_beam.tsv"
echo "  timit/${DECODE_OUTPUT_DIR}/transcriptions_val_beam.tsv"
echo "  timit/${DECODE_OUTPUT_DIR}/transcriptions_test_beam.tsv"
echo "PER files:"
echo "  timit/${DECODE_OUTPUT_DIR}/per_train_beam.txt"
echo "  timit/${DECODE_OUTPUT_DIR}/per_val_beam.txt"
echo "  timit/${DECODE_OUTPUT_DIR}/per_test_beam.txt"
echo "  timit/${DECODE_OUTPUT_DIR}/per_beam.tsv"
