#!/usr/bin/env bash
#SBATCH --job-name=ct-diagnostic-013
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=0
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_UPDATE_DIAGNOSTIC_POLICY_SHA256:?}"
: "${CT_UPDATE_DIAGNOSTIC_SOURCE_SHA256:?}"
: "${CT_UPDATE_MODEL_FREEZE_SHA256:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p test_ct_update_diagnostics.py -v
"$CT_UPDATE_PYTHON" -m careport.ct_update_diagnostic_run \
  --policy config/preanalysis/ct_diagnostics_013_2026-09-24.json \
  --confirmation-policy config/preanalysis/ct_pantrack_confirmation_010_2026-09-18.json \
  --registry config/preanalysis/ct_pantrack_checkpoints_010_2026-09-18.json
