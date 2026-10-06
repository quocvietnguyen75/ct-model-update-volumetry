# CT amendment 004: seed replication and simple volume correction

Date: 15 September 2026. Written after pilot 002 assessment, before the runs
and correction results below. This is adaptive development, not confirmatory
preregistration. The author instructed continuation of the authorized experiments.

## Material Passport

- Question: IMG-ALT-01, unchanged from amendment 001.
- Existing evidence: pilots 001 and 002, their assessments and preserved receipts.
- Human reading: not_attested. Review: inline; no independent reviewer.
- Protected test partition: sealed. No new dependencies, data or pretrained weights.

## Fixed replication batch

Repeat pilot 002's complete three-phase training with seeds 20260917 (pilot 003)
and 20260919 (pilot 004). Keep 20,000 old-model steps and 10,000 steps for each
descendant. Both descendants start from that seed's exact old checkpoint with
fresh optimizer states. The old-data membership salt remains unchanged; seeds
change initialization, sampling and augmentation, not patients or eligibility.
Use the same final-checkpoint selection, cache, architecture, optimizer,
localization, threshold, native-mask endpoint and CSV-reference sensitivity.

Report each new seed separately, including failures. Pilot 002's seed 20260915
is an already observed development reference. Display all three seeds without
choosing the best one; distinguish the two new runs from the earlier run. A
descriptive mean and range across seeds may supplement individual estimates.
Three seeds do not create 81 independent patients or establish seed-distribution
precision. Existing per-seed patient intervals retain their original definition.

Allocate at most two concurrent H100s for these two runs. Each training job has
a three-hour timeout and one-hour dependent validation job. Reuse the 11.8 GB
cache, preserve existing outputs, and check storage before launching. Expected
additional checkpoints are below 0.2 GB; the existing ceilings remain unchanged.
Do not repeatedly poll. Log progress and use scheduler dependencies/timeouts.

## Fixed simple-remedy analysis

For each seed, use its existing validation baseline predictions to fit scalar
affine mappings between old-model and descendant-model volumes. This is a
label-free mapping between measurement systems, not a reference-supervised
volume estimator. Reference volumes and follow-up values are used for scoring
only; they must never enter fitting. Models and segmentation masks stay fixed.

Assign validation patients to three balanced folds by sorting the SHA-256 of
`ct-update-004:calibration:` plus patient ID, breaking ties by ID, then assigning
rank modulo three. All lesions and both timepoints of a patient stay together.
For each held-out fold, fit on baseline predictions of the other two folds.
The 27 eligible patients give nine scoring patients and 18 fitting patients per
fold. This reuses development patients through cross-fitting; it is not an
independent validation of an unadapted research design.

Fit four mappings: old to new, new to old, old to continued, and continued to
old. Use patient-balanced weighted least squares: each patient's lesions have
total weight one. Fit y = intercept + slope*x with nonnegative slope; set the
slope to max(0, covariance/variance). If the weighted x variance is at most
1e-12 times max(1, weighted mean of x squared), use slope one and the weighted
mean difference as intercept. Keep this degenerate-fit flag. Clip mapped
volumes below zero to zero. No hyperparameter search, outlier exclusion or
post-result selection between directions is allowed.

Evaluate all four fixed remedies alongside the original five strategies:

- Map the stored old baseline onto the new scale, then use the new follow-up.
- Keep the old baseline and map the new follow-up onto the old scale.
- Apply the same two directions for the same-data continuation control.

The first direction approximates baseline reprocessing. The second asks whether
retaining the old measurement scale helps, while potentially retaining its
bias. Neither mapping is assumed to be correct or novel. Both original scans
remain available; actual new/new and continued/continued reprocessing are the
mandatory comparators. Corrected scalar volumes do not imply improved masks,
Dice or HD95. Follow-up contour scores remain those of the unchanged models.

Keep patient-balanced absolute and signed native-reference change errors.
Report each correction's benefit against its uncorrected mixed-version strategy,
and the residual cost of mapped-baseline correction against actual reprocessing.
Run the same scoring with CSV reference volumes as sensitivity; never refit the
mapping using either reference convention.

For the correction contrasts, use 2,000 paired patient-bootstrap draws, seed
20260920, stratified by the three fixed folds. Resample nine patients with
replacement within each fold, preserving patient multiplicities for fitting
and scoring. Refit the four mappings within every draw using only the other
folds. Report exploratory percentile intervals, conditional on the fixed folds
and trained models. This includes calibration-sample variation but not new
training-seed uncertainty or the adaptive choice of this analysis. No p-values,
clinical equivalence margin or claim of adequate power is introduced.

## Execution and interpretation gates

Use only completed validation prediction files whose hashes match their result
receipts and whose patient/lesion/timepoint keys exactly match the verified
validation cache. Reject missing, duplicate, extra, training or test records
before fitting. Verify the original cache/configuration receipts and each seed's
execution policy. Output only aggregates, fold counts, coefficient summaries
and hashes publicly; patient keys and predictions stay private on the cluster.

Write failing synthetic tests for affine fitting, patient weighting, held-out
patient isolation, reference/follow-up isolation, bootstrap refitting and input
gates before behavioral implementation. Run the full repository suite and the
cluster CT suite before accepting correction results.

The new seeds can train using the unchanged, already-tested pilot implementation
while the correction implementation is built and tested. Submit correction on
completed pilot 002 and as CPU work after successful evaluations of the new
seeds. No additional image inference is needed for scalar correction.

Assess the full fixed batch together. Do not stop when an interval first
excludes zero, discard an inconvenient seed, or call an imprecise null result
compatibility. Before a manuscript claim, matched-budget training from the
common initialization, relevant compatibility/longitudinal baselines, patient
overlap checks and a justified final analysis remain outstanding. No new method
or final test evaluation is authorized by this amendment.
