#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "$(pwd)/.conda_env"
set -u

python train.py "$@"
