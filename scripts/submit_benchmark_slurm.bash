#!/usr/bin/env bash
#SBATCH --job-name=oa_mbe_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

set -euo pipefail

mkdir -p logs benchmarks

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oa_mbe

IMG_DIR="${1:-}"
MASK_DIR="${2:-}"
MODEL96="${3:-}"
MODEL128="${4:-}"

if [[ -z "${IMG_DIR}" || -z "${MASK_DIR}" ]]; then
    echo "Usage:"
    echo "  sbatch submit_benchmark_slurm.bash images masks model96.pt model128.pt"
    exit 1
fi

OUTCSV="benchmarks/benchmark_$(date +%Y%m%d_%H%M%S).csv"

python benchmark_configs.py \
    --images "${IMG_DIR}" \
    --masks "${MASK_DIR}" \
    --model96 "${MODEL96}" \
    --model128 "${MODEL128}" \
    --tta_modes fast adaptive full \
    --output_csv "${OUTCSV}" \
    --device cuda

echo
echo "Benchmark CSV:"
echo "${OUTCSV}"
