#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=".conda_env"

echo "====================================================="
echo "Setting up orientation_agnostic_mbe environment"
echo "====================================================="

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found"
    exit 1
fi

echo
echo "Creating/updating conda environment..."
conda env create -f environment.yml || conda env update -f environment.yml

if [[ ! -d "${ENV_DIR}" ]]; then
    conda env create --prefix "${ENV_DIR}" -f environment.yml
else
    conda env update --prefix "${ENV_DIR}" -f environment.yml
fi
echo
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$(pwd)/${ENV_DIR}"
echo
echo "Python:"
python --version

echo
echo "Torch CUDA status:"
python - << 'EOF'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
EOF

echo
echo "MONAI status:"
python - << 'EOF'
import monai
print("monai:", monai.__version__)
EOF

echo
echo "====================================================="
echo "Setup complete"
echo "====================================================="
