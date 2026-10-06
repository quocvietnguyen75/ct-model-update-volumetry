#!/usr/bin/env bash
#SBATCH --job-name=ct-update-calibrate
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_UPDATE_CACHE_ROOT:?}"
: "${CT_UPDATE_EXECUTION_POLICY:?}"
: "${CT_UPDATE_EVALUATION_ROOT:?}"
: "${CT_UPDATE_CALIBRATION_ROOT:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_update_*.py' -v
"$CT_UPDATE_PYTHON" -m careport.ct_update_calibration_run \
  --config config/preanalysis/ct_update_001_2026-09-15.json \
  --analysis-policy config/preanalysis/ct_update_calibration_001_2026-09-15.json \
  --execution-policy "$CT_UPDATE_EXECUTION_POLICY" \
  --cache "$CT_UPDATE_CACHE_ROOT" --evaluation "$CT_UPDATE_EVALUATION_ROOT" \
  --output "$CT_UPDATE_CALIBRATION_ROOT"
