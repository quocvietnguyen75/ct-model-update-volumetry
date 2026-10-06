#!/usr/bin/env bash
#SBATCH --job-name=ct-update-evaluate
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_REMOTE_DATA_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_UPDATE_CACHE_ROOT:?}"
: "${CT_UPDATE_RUN_ROOT:?}"
: "${CT_UPDATE_EVALUATION_ROOT:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_update_*.py' -v
arguments=()
if [[ -n "${CT_UPDATE_EXECUTION_POLICY:-}" ]]; then
  arguments+=(--execution-policy "$CT_UPDATE_EXECUTION_POLICY")
fi
"$CT_UPDATE_PYTHON" -m careport.ct_update_evaluate \
  --config config/preanalysis/ct_update_001_2026-09-15.json \
  --cache "$CT_UPDATE_CACHE_ROOT" --training "$CT_UPDATE_RUN_ROOT" \
  --output "$CT_UPDATE_EVALUATION_ROOT" "${arguments[@]}"
