#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$(pwd)/.conda_env"

python train.py "$@"
