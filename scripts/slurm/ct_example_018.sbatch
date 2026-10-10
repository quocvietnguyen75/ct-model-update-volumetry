#!/usr/bin/env bash
#SBATCH --job-name=ct-example-018
#SBATCH --partition=mig
#SBATCH --time=0
# CT-UPDATE-018: one real lesion pair as an illustration. One GPU job; resources are set at submission.
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_REMOTE_DATA_ROOT:?}"
: "${CT_UPDATE_PANTRACK_REMOTE_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_STRONG_NNUNET_ROOT:?}"
: "${CT_UPDATE_MODEL_FREEZE_SHA256:?}"
: "${CT_STRONG_POLICY_SHA256:?}"
: "${CT_STRONG_SOURCE_SHA256:?}"
: "${CT_STRONG_FIX_SHA256:?}"
: "${CT_EXAMPLE_POLICY_SHA256:?}"
: "${CT_EXAMPLE_SOURCE_SHA256:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
test "$(sha256sum EXECUTION_SHA256SUMS | cut -d' ' -f1)" = "$CT_EXAMPLE_SOURCE_SHA256"
test "$CT_STRONG_SOURCE_SHA256" = "$CT_EXAMPLE_SOURCE_SHA256"
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_STRONG_NNUNET_ROOT:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
export nnUNet_compile=false
export nnUNet_raw="$CT_UPDATE_REMOTE_ROOT/runs/strong_017/nnUNet_raw"
export nnUNet_preprocessed="$CT_UPDATE_REMOTE_ROOT/runs/strong_017/nnUNet_preprocessed"
export MONAI_DATA_DIRECTORY="$CT_UPDATE_REMOTE_ROOT/runtime/monai"
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_*.py'
"$CT_UPDATE_PYTHON" -m careport.ct_example_run --policy config/preanalysis/ct_example_018_2026-10-09.json
