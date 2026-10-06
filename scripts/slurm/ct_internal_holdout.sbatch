#!/usr/bin/env bash
#SBATCH --job-name=ct-internal-014
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_REMOTE_DATA_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_INTERNAL_POLICY_SHA256:?}"
: "${CT_INTERNAL_SOURCE_SHA256:?}"
: "${CT_INTERNAL_STAGE:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MONAI_DATA_DIRECTORY="$CT_UPDATE_REMOTE_ROOT/runtime/monai"
"$CT_UPDATE_PYTHON" -c 'import json,shutil,tempfile; print(json.dumps(dict(temporary_free_bytes=shutil.disk_usage(tempfile.gettempdir()).free)),flush=True)'
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_*.py' -v
args=("$CT_INTERNAL_STAGE" --policy config/preanalysis/ct_internal_holdout_014_2026-09-24.json)
if [[ "$CT_INTERNAL_STAGE" != prepare ]]; then
  : "${CT_INTERNAL_COHORT_FREEZE_SHA256:?}"
fi
if [[ "$CT_INTERNAL_STAGE" == evaluate ]]; then
  : "${CT_INTERNAL_SEED:?}"
  args+=(--seed "$CT_INTERNAL_SEED")
fi
"$CT_UPDATE_PYTHON" -m careport.ct_internal_run "${args[@]}"
