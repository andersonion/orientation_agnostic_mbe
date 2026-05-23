#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(pwd)"

sbatch \
  --export=ALL,REPO_DIR="${REPO_DIR}" \
  scripts/submit_train_slurm.bash "$@"