# CT-UPDATE-015: prespecified training-run replication

25 September 2026. Question: IMG-ALT-01. Specified before any new training run,
prediction or outcome exists. Machine-readable decisions:
`config/preanalysis/ct_replication_015_2026-09-25.json`.

## Why this extension exists

The completed study conditions every interval on three trained models. Its
patient bootstrap does not describe what a new training run of the same update
procedure would produce (roadmap of 19 September; internal assessment of 25
September). The PanTrack mixed-version increase of +2.20828 mL is dominated by
one run (per-run +0.93825, -0.65216 and +6.33874 mL), and the internal cohort
did not reproduce the average increase. The roadmap and the 20 September
literature review both permit one fixed seed-replication extension, provided
its estimand, precision target, budget and stopping rule are fixed first.

**Hypothesis provenance, disclosed.** On 25 September the completed per-run
tables were inspected. The baseline-retention cost was positive in five of six
run-by-cohort estimates (internal +0.70160, +0.62879, -0.00290; PanTrack
+0.72609, +0.68070, +2.31326 mL), while the ordinary-update benefit changed sign
across runs. This extension asks whether that pattern belongs to the update
procedure or to three particular runs. Because the hypothesis came from
inspected results, the original three runs are excluded from every primary
estimate below. They are displayed alongside, never pooled.

## What stays fixed

- Frozen configuration `ct_update_001`, architecture, patch policy, loss,
  optimizer, sampler, augmentation and the unchanged `careport.ct_update_train`.
- Budgets: old model 20,000 steps on the fixed 113-patient old subset; ordinary
  update 10,000 further steps on all 225 eligible training patients; continuation
  10,000 further steps on the old subset. Both descendants load the exact final
  old checkpoint. Final checkpoints only; no selection on any evaluation result.
- Development cache `derived/patches_001` and its receipts. No validation-set
  evaluation, calibration fitting, scratch, interpolation or distillation run is
  part of this extension; their three-run results remain as reported.
- Evaluation cohorts, reused through their existing hash-bound gates: internal
  holdout (cache `derived/internal_holdout_014`, 29 patients, 277 pairs) and
  PanTrack (cache `derived/pantrack_010`, 45 patients, 181 pairs). No new
  eligibility, preprocessing, reference access or cohort is introduced.
- Measurement: the unchanged `native_metrics` (native-grid volume, Dice, HD95),
  BF16 autocast inference exactly as before. Scoring may run in parallel worker
  processes; parallelism must not change any value, and a synthetic test must
  show equality with direct sequential scoring.
- Completed CT-UPDATE-010/013/014 results, their six-contrast family and every
  reported number are unchanged and are not reanalysed.

## Runs

Twenty new training seeds, 20261001 through 20261020, executed in that order
(execution IDs CT-REPL-015-01 to -20). None overlaps the original seeds. Each run
trains old, ordinary update and continuation, then evaluates those three final
checkpoints on both cohorts. The number of runs is fixed now and will not be
increased after any result is seen.

A run is complete when all three phases finish and both cohorts have exact
prediction coverage. Empty or poor predictions are outcomes and stay in the
analysis. A run that stops for a technical reason (scheduler, node, I/O or
nonfinite-loss failure) keeps its failed artifacts under a renamed directory and
may be resubmitted once with the same seed and inputs. A second failure is
reported and that seed is excluded. The primary analysis requires at least 15
completed runs; with fewer, results are reported descriptively only.

## Estimands and primary analysis

Weighting is unchanged: pair losses are averaged within patient, patients are
weighted equally, and here runs are weighted equally. For run r and patient i:

- Baseline-retention cost R = MAE(old/new) - MAE(new/new); positive favours
  reprocessing the baseline with the updated model.
- Ordinary-update benefit U = MAE(old/old) - MAE(old/new); positive favours the
  update when the historical baseline is retained.
- Contour benefit C = Dice(new) - Dice(old) at follow-up.

Each estimand is the mean over the 20 new runs and all cohort patients: the
expected contrast of the update procedure over training randomness. The primary
family is these three contrasts in each of the two cohorts (six contrasts).

Primary intervals use a two-way percentile bootstrap that resamples runs and
patients independently with replacement: 10,000 draws, generator seed 20261021,
run indices drawn first (shared by both cohorts), then internal patient indices,
then PanTrack patient indices (each shared across that cohort's contrasts).
Report 95% intervals and Bonferroni family intervals at quantiles 0.05/12 and
1 - 0.05/12. A contrast is called established in a cohort only if its family
interval excludes zero; otherwise it is not established, which is not
equivalence. Cohorts are not pooled and no cohort interaction is tested.

## Secondary, descriptive only

- Per-run patient-mean contrasts, their standard deviation and the number of
  runs with each sign; patient-only intervals conditional on the 20 runs;
  run-only intervals conditional on the cohort's patients.
- Signed baseline version shift, the patient-weighted mean of new-minus-old
  baseline volume (identically, signed old/new minus signed new/new error).
- Same-version change contrast MAE(old/old) - MAE(new/new), and continuation
  contrasts separating added data from added steps.
- Mean MAE and signed error for every strategy, follow-up Dice per model, empty
  predictions and nonempty HD95 per run.

No other contrast, subgroup, lesion-size stratum or pooled estimate is added
after results. The three original runs appear only as displayed points.

## Precision planning

Planning values come from the three original runs. They are rough, because a
variance estimated from three runs is itself very uncertain. Decomposing the
per-run and run-averaged patient-bootstrap widths suggests these approximate
family-interval half-widths with 20 runs:

| Contrast | Internal | PanTrack |
|---|---:|---:|
| Retention cost R (mL) | 0.47 | 1.91 |
| Update benefit U (mL) | 0.68 | 2.53 |
| Contour benefit C (Dice) | 0.02 | 0.03 |

Internal precision is limited mostly by run-by-patient variation, which more
runs reduce. PanTrack retention-cost precision is limited mainly by its 45
patients, so an interval that includes zero there would say little about
absence of an effect. These are planning approximations, not promised results.

## Execution, gates and resources

- Each run is one array task on the MIG profile
  `gpu:nvidia_h100_80gb_hbm3_3g.40gb:1`, eight CPUs, 48 GiB, unlimited wall
  time and at most one allocation at a time. Training peaked at 0.56 GB of
  allocated GPU memory in the original runs.
- The code bundle's `EXECUTION_SHA256SUMS`, this amendment, the replication
  configuration and every reused policy, freeze, registry, cache index, patch
  and native mask are verified by hash before use. The CT unit tests run in
  every job. Output directories must be new; artifacts are never overwritten.
- Ceiling: 80 cumulative allocation-hours for the whole extension. If the
  remaining runs would exceed it, submission pauses and the user decides,
  without inspecting any outcome. Storage: 5 GiB planned inside the existing
  250 GiB study cap. No download, dependency or dataset change is needed.
- The analysis job runs only after every submitted run has finished. A separate
  scalar verification recomputes all primary point estimates and intervals from
  the private predictions before the results are interpreted.
- Private predictions, checkpoints and patient-level values remain on the
  cluster. Only aggregate results and hashes enter the research record.

## Interpretation boundary

This extension estimates run-to-run behaviour of one fixed update procedure on
two fixed, already-inspected evaluation cohorts. It cannot make PanTrack or the
internal holdout untouched again, does not generalize to other architectures,
update schemes or clinical response assessment, and supplies no clinical margin.
Whatever the result, it is reported in full as a later, outcome-informed
extension of the completed study.
