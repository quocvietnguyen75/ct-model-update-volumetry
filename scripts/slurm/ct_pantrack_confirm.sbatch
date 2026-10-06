#!/usr/bin/env bash
#SBATCH --job-name=ct-pantrack-confirm
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_PANTRACK_REMOTE_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_UPDATE_MODEL_FREEZE_SHA256:?}"
: "${CT_UPDATE_PANTRACK_STAGE:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MONAI_DATA_DIRECTORY="$CT_UPDATE_REMOTE_ROOT/runtime/monai"
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_*.py' -v
args=("$CT_UPDATE_PANTRACK_STAGE"
  --policy config/preanalysis/ct_pantrack_confirmation_010_2026-09-18.json
  --registry config/preanalysis/ct_pantrack_checkpoints_010_2026-09-18.json)
if [[ "$CT_UPDATE_PANTRACK_STAGE" == evaluate ]]; then
  : "${CT_UPDATE_PANTRACK_SEED:?}"
  args+=(--seed "$CT_UPDATE_PANTRACK_SEED")
fi
"$CT_UPDATE_PYTHON" -m careport.ct_pantrack_run "${args[@]}"
