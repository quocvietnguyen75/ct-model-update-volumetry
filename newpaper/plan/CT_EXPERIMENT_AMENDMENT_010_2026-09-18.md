# CT-UPDATE-010: fixed PanTrack evaluation of existing update strategies

Date: 18 September 2026. Question: IMG-ALT-01. Author instruction: continue
the core study while the LongiSeg checkpoint comparison is deferred. This
amendment is written before PanTrack voxel/reference-volume inspection or
prediction. It implements one external evaluation; the internal test stays sealed.

## Material Passport

Academic-research-suite, experiment preparation/execution. Reuses the frozen
development protocol, all completed standard-method controls, the verified
PanTrack intake, and development-only precision planning. Human reading:
not_attested. Planning and implementation review: inline, same agent. This
record is not an independent review or a claim of publication readiness.

## Frozen scope and access sequence

1. Bind the 330-file transfer receipt, metadata/header receipt and private
   181-pair candidate manifest by SHA-256. Preserve all 45 candidate patients.
   Each pair is identified by patient, baseline scan, follow-up scan and label;
   repeated visits with the same label must not collide or overwrite each other.
2. Freeze hashes of all existing checkpoints and their lineage/configuration
   receipts. Fit the four previously defined nonnegative-slope affine mappings
   per seed using **only the existing 27 development patients' baseline model
   predictions**. Use equal patient weights, then equal lesions within patient,
   as in amendment 004. Fit on all development patients for external application;
   never use external volumes, follow-up values or reference masks in fitting.
   Freeze the resulting coefficient artifact before external arrays are read.
3. Close the reference-schema gate before image caching or prediction: verify
   file hashes, grid units/affines, orthogonal physical axes, bounded nonnegative
   integer instance masks, nonempty target labels and supplied XYZ centroid
   agreement within the existing 1.05-voxel tolerance. Save counts, complete
   native lesion bounds and volumes privately. Unexpected failure stops the
   batch; do not drop cases or change coordinates to maximize performance.
   Public reports contain only counts, status and hashes at this stage.
4. After all candidate references pass, extract each pair/timepoint image patch
   using the unchanged 96-cubed, 1.5-mm geometry, linear CT interpolation,
   HU clip [-1000,1000]/1000 and float16 cache. Reference prompts remain
   privileged localization. Native-mask volumes and IDs never become predictors.
   Retain truncated lesions and empty resampled targets. Full native reference
   support remains the denominator, including beyond the patch.
5. Run fixed checkpoints once on the identical complete cohort. No external
   training, tuning, checkpoint selection, early stopping, model selection or
   correction fitting. A missing/nonfinite/duplicate output fails completeness;
   no silent omissions or successful-case subset. Legitimately empty model masks
   have volume zero and Dice zero; their HD95 is undefined and counted separately.

## Fixed model and strategy set

Seeds: 20260915, 20260917, 20260919 (pilot executions 002/003/004). Models:
old, new, same-data continued, scratch, BCWI old-weight 0.25/0.50/0.75, and
distillation lambda 1/10. Reconstruct BCWI from the exact old/new states and
require the saved development state fingerprints to match. No new model training.

Report old/old and, for every non-old candidate, both old/candidate and
candidate/candidate volume-change error. Include the four fixed scalar
correction strategies for new and continued. Retain all seeds and strengths.
No LongiSeg reproduction or superiority claim is made. No CSV-reference
sensitivity exists for PanTrack. Full native-mask evaluation, logit threshold
zero, BF16 inference and existing empty-mask conventions remain unchanged.

## Estimands and analysis fixed before outcomes

Primary endpoint remains absolute error in predicted follow-up minus baseline
volume versus full native-mask reference change, in mL. For each seed, average
lesion/visit-pair losses within patient, then patients equally. For the main
conditional-on-trained-models estimate, average each patient's loss over the
three seeds before averaging patients or resampling. Do not average volumes
across seeds before taking absolute errors. Do not count repeated visits or
three seeds as additional independent patients.

Three primary paired contrasts, with their signs explicitly retained:

- Ordinary-update benefit: old/old MAE minus old/new MAE (positive favors update).
- Retaining-baseline cost: old/new MAE minus new/new MAE (positive favors reprocessing).
- Contour benefit: new minus old follow-up Dice (positive favors update).

Use 10,000 whole-patient bootstrap draws, NumPy default_rng seed **20260924**,
with common patient selections for all strategies/contrasts/seeds. Report
ordinary 95% percentile intervals and conservative Bonferroni-adjusted percentile
intervals for the primary family, reserving **six** tests for these three contrasts
across external and any later internal confirmation. The three unused internal
slots are not reassigned. Finite-sample bootstrap coverage is approximate.
Individual-seed intervals and all nonprimary control contrasts are descriptive;
their nominal intervals do not establish family-controlled discoveries. The
main intervals condition on the fixed trained checkpoints and fixed development
calibration coefficients, not the full training/calibration-sampling process.

Report MAE, signed error, follow-up Dice, empty-prediction counts, and HD95 over
nonempty predictions with its patient denominator. Secondary contrasts include
each control's mixed-version benefit versus ordinary update, its Dice difference,
and cost versus new/new reprocessing; corrections retain the previous four
benefit contrasts. Publish complete tables, not a selected best seed/strength.

## Precision and interpretation

Freeze **5 mL** as an operational target for interval half-width on the two
primary volume contrasts; it is a reporting-resolution target, not a clinically
validated margin, noninferiority threshold, or assertion that 5 mL matters
clinically. Report whether ordinary and adjusted intervals meet it regardless
of effect direction. The development-only conservative scenario predicted about
9.87 mL ordinary half-width at 45 patients, so target attainment is not assumed.
There is no justified clinical preservation/equivalence margin. An interval
crossing zero cannot demonstrate equivalence or preserved accuracy.

Treat this as external-dataset transfer, without attributing effects solely
to scanner/hospital shift. Distinct compressed image hashes do not prove patient
independence across pseudonyms; preserve that provenance limitation. Reference
annotations and supplied centroids are imperfect and privileged respectively.
Assess all results once after completeness/lineage checks. This batch authorizes
no adaptive strength search, new cohort acquisition, or automatic paper claim.

## Resources, reproducibility and stop rules

Use the approved imaging runtime; no new dependency, model or dataset download.
Allocate at most **5 GiB additional** storage within the unchanged **250 GiB
study / 900 GiB project caps**, including a **2 GiB cache cap**. The 362-patch
upper-bound projection is below 1 GiB plus small metadata; verify before writes.
CPU schema/cache job: 4 CPUs, 24 GiB RAM, 45 minutes. Each seed inference job:
one H100, 8 CPUs, 48 GiB RAM, 4 hours; at most two simultaneously. Aggregation
is CPU-only and depends on all three successful seed jobs. Failed artifacts
are preserved; a repair requires a recorded diagnosis before any rerun.

Failing-first synthetic tests must cover repeated visits, full native-volume
denominators, XYZ/shear/missing-label failures, exact prediction coverage,
patient/seed weighting, paired bootstrap and primary multiplicity. Freeze a
source bundle, checkpoint registry, coefficients, policy and their hashes before
submission. Run the complete local suite and required cluster synthetic tests.
No changes to the original development code/configuration or internal-test access.
