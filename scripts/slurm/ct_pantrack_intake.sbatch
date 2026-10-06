#!/usr/bin/env bash
#SBATCH --job-name=ct-pantrack-intake
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
set -euo pipefail
umask 077
: "${CT_UPDATE_CODE_ROOT:?}"
: "${CT_UPDATE_REMOTE_ROOT:?}"
: "${CT_UPDATE_PANTRACK_REMOTE_ROOT:?}"
: "${CT_UPDATE_PYTHON:?}"
: "${CT_UPDATE_PACKAGE_ROOT:?}"
cd "$CT_UPDATE_CODE_ROOT"
sha256sum --check --status EXECUTION_SHA256SUMS
export PYTHONPATH="$CT_UPDATE_CODE_ROOT/src:$CT_UPDATE_PACKAGE_ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$CT_UPDATE_PYTHON" -m unittest discover -s tests -p 'test_ct_pantrack_intake.py' -v
"$CT_UPDATE_PYTHON" -m careport.ct_pantrack_intake \
  --policy config/preanalysis/ct_update_pantrack_intake_002_2026-09-18.json \
  --transfer-receipt "$CT_UPDATE_REMOTE_ROOT/control/pantrack_transfer_verified_2026-09-18.json" \
  --output "$CT_UPDATE_REMOTE_ROOT/control/pantrack_intake_002.json" \
  --private-output "$CT_UPDATE_REMOTE_ROOT/control/pantrack_candidates_002_private.json" \
  --check-headers
