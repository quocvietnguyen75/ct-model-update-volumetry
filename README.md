# Longitudinal tumour volumetry after CT segmentation model updates

Analysis code, frozen analysis configurations, dated protocol amendments and verified aggregate results for the manuscript:

> Quoc Viet Nguyen, Phuc Tai Duong, Duc Hoai Anh Nguyen, Dzung Van Dinh and Zengchang Qin. *Longitudinal tumour volumetry after CT segmentation model updates: a controlled evaluation across training runs and cohorts.* Submitted to *Physics in Medicine & Biology*, 2026.

The study updates a CT tumour segmentation model with more training data. It then measures how retaining baseline measurements made by the old model, instead of reprocessing the baseline scan, changes the error of measured tumour-volume change. It compares a three-dimensional residual U-Net (three original runs and a prespecified 20-run replication) with nnU-Net (a ten-run replication), on an internal Longitudinal-CT holdout and the external PanTrack cohort.

This repository is a snapshot of the authors' working repository at the time of submission, restricted to this study. It contains no imaging data, no patient-level values and no trained weights.

## Data

Both datasets are public and distributed under CC BY-NC 4.0. Download them from their providers:
- **Longitudinal-CT** (version 2): [doi:10.57754/FDAT.75kj1-64747](https://doi.org/10.57754/FDAT.75kj1-64747);
- **PanTrack**: [huggingface.co/datasets/mrokuss/PanTrack](https://huggingface.co/datasets/mrokuss/PanTrack).

The code never hard-codes a data location. Dataset, cache and run roots come from environment variables; see "Running the experiments".

## Layout

| Path | Content |
|---|---|
| `src/careport/ct_*.py` | The analysis package. `careport` is the working repository's historical package name; only the `ct_*` modules belong to this study. |
| `tests/test_ct_*.py` | Synthetic unit tests. They need no data. |
| `scripts/ct_*.py` | Regenerate the manuscript's tables, figures and supplementary CSVs from the verified results, and check a manuscript's numbers against them |
| `scripts/slurm/ct_*.sbatch` | The Slurm jobs that ran each stage on an NVIDIA H100 |
| `config/preanalysis/ct_*.json` | Frozen analysis configurations. Each stage verifies their SHA-256 before it runs. |
| `newpaper/plan/CT_*AMENDMENT_*.md` | The dated protocol amendments. Each was written before the outcomes it governs were inspected. Amendment 001 creates the protocol. |
| `newpaper/research_selection_2026-09-13/CT_*` | The verified aggregate results reported in the paper (JSON and CSV) |
| `newpaper/ct_manuscript/tables/` | The supplementary data files `s1`–`s7`, and four aggregate tables the scripts read |

Paths keep the working repository's layout, because the configurations and amendments bind files by path and hash. Links in the amendments to internal planning documents point to files that are not included.

## Environment

The experiments ran with Python 3.10.12, PyTorch 2.6.0 (CUDA 12.4), MONAI 1.5.1, NiBabel 5.3.2, NumPy 2.2.6 and SciPy 1.15.3. The stronger-model replication also used nnU-Net v2.8.1. The figures use matplotlib. See `requirements.txt`.

## Quick checks

From the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

The suite runs 247 tests. Tests that need PyTorch, MONAI or nnU-Net are skipped when those packages are absent.

To regenerate the manuscript's figures, table fragments and supplementary CSVs from the verified results:

```bash
PYTHONPATH=src python scripts/ct_manuscript_figures.py   # figures into newpaper/ct_manuscript/figures/print/
mkdir -p newpaper/ct_manuscript/pmb_submission
PYTHONPATH=src python scripts/ct_manuscript_latex.py     # LaTeX table fragments and s6/s7 CSVs
```

The regenerated figures and table fragments are identical to those in the submitted manuscript, except Figure 4. That figure shows one PanTrack lesion pair, chosen by the rule in amendment 018. It is drawn from CT image crops, which this repository does not hold, so the figure script skips it. To draw it, run the 018 stage, which needs the downloaded PanTrack data and the nnU-Net models trained in the 017 stage, then set `CT_EXAMPLE_PRIVATE_NPZ` to the crop file it writes.

## Running the experiments

Each stage is a module run by one Slurm script. A stage refuses to run unless the SHA-256 of its frozen configuration, amendment and inputs match the values it is given. Those values come from environment variables ending in `_SHA256`. Roots come from `CT_UPDATE_*_ROOT` variables, for example `CT_UPDATE_REMOTE_DATA_ROOT`, `CT_UPDATE_PANTRACK_REMOTE_ROOT`, `CT_UPDATE_CACHE_ROOT` and `CT_UPDATE_RUN_ROOT`. Each Slurm script lists the variables it needs.

| Stage | Amendment | Slurm script | Module |
|---|---|---|---|
| Schema and access gate | 001 | `ct_update_schema` | `ct_update_data` |
| Patch caching, with the reference-volume recount | 001, 002 | `ct_update_preprocess` | `ct_update_preprocess` |
| Old model, update and continuation training | 001, 003, 004 | `ct_update_train` | `ct_update_train` |
| Development evaluation | 001–004 | `ct_update_evaluate` | `ct_update_evaluate` |
| Scratch and weight-interpolation controls | 005 | `ct_update_controls` | `ct_update_control_run` |
| Distillation controls | 006 | `ct_update_distill` | `ct_update_distillation_run` |
| Scalar correction maps | 004 | `ct_update_calibrate` | `ct_update_calibration_run` |
| PanTrack intake | 007, 009 | `ct_pantrack_intake` | `ct_pantrack_intake` |
| PanTrack evaluation | 008, 010–012 | `ct_pantrack_confirm` | `ct_pantrack_run` |
| Post hoc PanTrack diagnostics | 013 | `ct_update_diagnostic` | `ct_update_diagnostic_run` |
| Internal holdout evaluation | 014 | `ct_internal_holdout` | `ct_internal_run` |
| 20-run replication | 015 | `ct_replication_seed`, `ct_replication_analysis` | `ct_replication_run` |
| Post hoc descriptions | 016 | `ct_posthoc_016` | `ct_posthoc_run` |
| nnU-Net replication | 017, 017A | `ct_strong_017` | `ct_strong_run` |
| Example lesion pair (Figure 4) | 018 | `ct_example_018` | `ct_example_run` |

The amendment numbers above are a guide; the amendments themselves are authoritative. The analyses ran from hash-recorded code bundles of earlier versions of this code. Those execution records are available from the corresponding author.

## AI assistance

Claude Opus 5.5 (Anthropic) wrote and tested much of this code under the authors' direction, as the paper's AI statement describes. The authors take responsibility for it.

## Licence

The code is released under the MIT licence (see `LICENSE`). The licence does not cover the datasets, which keep their own terms.

## Contact

Quoc Viet Nguyen, College of Engineering and Computer Science, VinUniversity: viet.nq@vinuni.edu.vn.
