#!/usr/bin/env bash
#SBATCH --job-name=oa_mbe_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR is not set}"

source "$(conda info --base)/etc/profile.d/conda.sh"

set +u
conda activate "${REPO_DIR}/.conda_env"
set -u

cd "${REPO_DIR}"

mkdir -p logs checkpoints

source "$(conda info --base)/etc/profile.d/conda.sh"
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

set +u
conda activate "${REPO_DIR}/.conda_env"
set -u
IMG_DIR="${1:-}"
MASK_DIR="${2:-}"

if [[ -z "${IMG_DIR}" || -z "${MASK_DIR}" ]]; then
    echo "Usage:"
    echo "  sbatch submit_train_slurm.bash /path/to/images /path/to/masks"
    exit 1
fi

PATCH_SIZE="${PATCH_SIZE:-96}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-120}"

CKPT="checkpoints/best_model_ps${PATCH_SIZE}.pt"

echo "====================================================="
echo "Training orientation_agnostic_mbe"
echo "====================================================="
echo "Images     : ${IMG_DIR}"
echo "Masks      : ${MASK_DIR}"
echo "Patch size : ${PATCH_SIZE}"
echo "Batch size : ${BATCH_SIZE}"
echo "Epochs     : ${EPOCHS}"
echo "Checkpoint : ${CKPT}"
echo "====================================================="

nvidia-smi || true

python train.py \
    --img_dir "${IMG_DIR}" \
    --mask_dir "${MASK_DIR}" \
    --model_ckpt_save_dir "${CKPT}" \
    --patch_size "${PATCH_SIZE}" \
    --batch_size "${BATCH_SIZE}" \
    --training_epochs "${EPOCHS}" \
    --device cuda \
    --save_last
