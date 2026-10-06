#!/usr/bin/env bash
#SBATCH --job-name=ct-update-distill
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_REMOTE_DATA_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_UPDATE_CACHE_ROOT:?}"
: "${CT_UPDATE_RUN_ROOT:?}"
: "${CT_UPDATE_EXECUTION_POLICY:?}"
: "${CT_UPDATE_EVALUATION_ROOT:?}"
: "${CT_UPDATE_DISTILLATION_ROOT:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MONAI_DATA_DIRECTORY="$CT_UPDATE_REMOTE_ROOT/runtime/monai"
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_update_*.py' -v
"$CT_UPDATE_PYTHON" -m careport.ct_update_distillation_run \
  --config config/preanalysis/ct_update_001_2026-09-15.json \
  --distillation-policy config/preanalysis/ct_update_distillation_001_2026-09-16.json \
  --execution-policy "$CT_UPDATE_EXECUTION_POLICY" \
  --cache "$CT_UPDATE_CACHE_ROOT" --training "$CT_UPDATE_RUN_ROOT" \
  --evaluation "$CT_UPDATE_EVALUATION_ROOT" --output "$CT_UPDATE_DISTILLATION_ROOT"
