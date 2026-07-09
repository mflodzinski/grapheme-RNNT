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

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
    VENV_DIR="$(dirname "$(dirname "${VENV_ACTIVATE}")")"
elif [[ -z "${VENV_DIR:-}" ]]; then
    if [[ -d ".venv-daic" ]]; then
        VENV_DIR=".venv-daic"
    elif [[ -d ".venv" ]]; then
        VENV_DIR=".venv"
    elif [[ -f "/home/nfs/mlodzinski/venvs/mode-connectivity/bin/activate" ]]; then
        VENV_ACTIVATE="/home/nfs/mlodzinski/venvs/mode-connectivity/bin/activate"
        VENV_DIR="$(dirname "$(dirname "${VENV_ACTIVATE}")")"
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
    echo "Virtualenv ${VENV_DIR} does not exist; creating it now."
    python -m venv "${VENV_DIR}"
fi

VENV_ACTIVATE="${VENV_ACTIVATE:-${VENV_DIR}/bin/activate}"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    echo "Virtualenv activation script ${VENV_ACTIVATE} does not exist."
    echo "Pass VENV_DIR=/path/to/env or VENV_ACTIVATE=/path/to/env/bin/activate."
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_ACTIVATE}"
echo "Activated virtualenv: ${VENV_DIR}"

if [[ -z "${AUTO_INSTALL_REQUIREMENTS:-}" ]]; then
    if [[ "${VENV_DIR}" = /* ]]; then
        AUTO_INSTALL_REQUIREMENTS=0
    else
        AUTO_INSTALL_REQUIREMENTS=1
    fi
fi

if [[ "${AUTO_INSTALL_REQUIREMENTS}" == "1" ]]; then
    REQUIREMENTS_STAMP="${VENV_DIR}/.requirements-installed"
    if [[ ! -f "${REQUIREMENTS_STAMP}" || requirements.txt -nt "${REQUIREMENTS_STAMP}" ]]; then
        python -m pip install --upgrade pip
        python -m pip install -r requirements.txt
        touch "${REQUIREMENTS_STAMP}"
    fi
fi

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
