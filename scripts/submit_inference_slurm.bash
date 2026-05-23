#!/usr/bin/env bash
#SBATCH --job-name=oa_mbe_infer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/infer_%j.out
#SBATCH --error=logs/infer_%j.err

set -euo pipefail
REPO_DIR="${REPO_DIR:?REPO_DIR is not set}"

source "$(conda info --base)/etc/profile.d/conda.sh"

#set +u
#conda activate "${REPO_DIR}/.conda_env"
#set -u

cd "${REPO_DIR}"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
REPO_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"

set +u
conda activate "${REPO_DIR}/.conda_env"
set -u
MODEL="${1:-}"
INPUT_DIR="${2:-}"

if [[ -z "${MODEL}" || -z "${INPUT_DIR}" ]]; then
    echo "Usage:"
    echo "  sbatch submit_inference_slurm.bash model.pt /path/to/images"
    exit 1
fi

PATCH_SIZE="${PATCH_SIZE:-96}"
TTA="${TTA:-adaptive}"

OUTDIR="${INPUT_DIR}/predicted_masks_${TTA}_ps${PATCH_SIZE}"

echo "====================================================="
echo "Inference orientation_agnostic_mbe"
echo "====================================================="
echo "Model      : ${MODEL}"
echo "Input dir  : ${INPUT_DIR}"
echo "Patch size : ${PATCH_SIZE}"
echo "TTA        : ${TTA}"
echo "Output dir : ${OUTDIR}"
echo "====================================================="

nvidia-smi || true

python inference.py \
    --model_ckpt_save_dir "${MODEL}" \
    --inference_dir "${INPUT_DIR}" \
    --output_dir "${OUTDIR}" \
    --patch_size "${PATCH_SIZE}" \
    --tta "${TTA}" \
    --device cuda
