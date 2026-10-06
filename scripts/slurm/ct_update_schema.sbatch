#!/usr/bin/env bash
#SBATCH --job-name=ct-schema
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_REMOTE_DATA_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
: "${CT_UPDATE_SCHEMA_OUTPUT:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_update_*.py' -v
"$CT_UPDATE_PYTHON" -m careport.ct_update_data \
  --config config/preanalysis/ct_update_001_2026-09-15.json \
  --metadata-gate "$CT_UPDATE_REMOTE_ROOT/control/metadata_gate_001.json" \
  --output "$CT_UPDATE_SCHEMA_OUTPUT" --workers 4
