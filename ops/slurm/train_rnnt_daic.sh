#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_rnnt_train_%j.out
#SBATCH --error=slurm_rnnt_train_%j.err
#SBATCH --job-name=rnnt_train
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

if [[ -z "${VENV_DIR:-}" ]]; then
    if [[ -d ".venv-daic" ]]; then
        VENV_DIR=".venv-daic"
    elif [[ -d ".venv" ]]; then
        VENV_DIR=".venv"
    else
        VENV_DIR=".venv-daic"
    fi
fi
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-rnn-transducer}"
WANDB_NAME="${WANDB_NAME:-rnnt-daic-${SLURM_JOB_ID:-local}}"

export WANDB_MODE
export WANDB_PROJECT
export WANDB_NAME
export WANDB_DIR="${WANDB_DIR:-${SUBMIT_DIR}/wandb}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export PYTHONUNBUFFERED=1

mkdir -p info "${WANDB_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Virtualenv ${VENV_DIR} does not exist. Create it on a login node before sbatch."
    echo "From the repository root, run:"
    echo "  python -m venv ${VENV_DIR}"
    echo "  source ${VENV_DIR}/bin/activate"
    echo "  pip install --upgrade pip"
    echo "  pip install -r requirements.txt"
    echo "Then submit again with:"
    echo "  sbatch ops/slurm/train_rnnt_daic.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "Activated virtualenv: ${VENV_DIR}"

echo "Submit dir: ${SUBMIT_DIR}"
echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-local}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_NAME: ${WANDB_NAME}"
echo "WANDB_MODE: ${WANDB_MODE}"

python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
print(f"cuda version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
PY

python rnnt/train.py
