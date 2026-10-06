# CT-UPDATE-014: frozen internal patient holdout evaluation

24 September 2026, before access to internal test CSV bodies, native reference
arrays, CT arrays or model predictions. PanTrack outcomes and amendment 013
diagnostics are already known. This evaluation is prospectively specified for
the still-unopened internal holdout, within a study informed by external results.

This amendment implements the [internal specification](CT_INTERNAL_HOLDOUT_SPECIFICATION_2026-09-24.md).
It opens a separate, gated evaluation entry point. The original development
entry points and policies must continue rejecting test data. It changes no
primary endpoint, trained checkpoint, development calibration or storage cap.
The user's continuing study authorization covers execution; no dependency or
dataset acquisition is proposed.

## Population and reference

Use exactly the original split manifest's 30 nominal test patients, disjoint from
its 240 train and 30 validation patients. Determine eligible patients only after
the metadata gate passes. Apply the unchanged `classify_rows` logic: unique
positive lesion IDs; topology UNCHANGED; explicitly clear linking; no merger
destinations; positive finite CSV volumes; valid BL/FU image indices and native
XYZ centres. Retain all image indices. Do not filter by target/challenge flags,
lesion size, model quality, crop coverage or successful predictions.

BL and FU are the provided roles, not validated elapsed acquisition times. No
chronological interval or growth-rate inference is permitted. Model inputs are
CT patches and reference-derived centres only. Reference volumes, target flags,
patient identifiers and other CSV fields are not model predictors.

The primary reference is the complete native instance mask, with mL from the
affine determinant. Validate matching image/mask geometry, explicit millimetres,
orthogonal nondegenerate axes, positive target support and centre agreement
within 1.05 voxels. A missing target or geometry/centroid mismatch stops the
cohort before model inference; it is not a reason for silent exclusion.

Amendment 002 established how redundant CSV summaries are checked. Carry its
procedure forward explicitly: retain the original failed schema report if its
only issue is CSV/native disagreement above one voxel or 0.5%. Recount every
flagged label using direct equality/count, compare whole-mask and bounded-box
counts, and compare affine- and header-spacing-derived physical volumes.
Only unanimous agreement with the original native count allows a separate
resolution receipt. Preserve CSV warnings, original files, all eligible cases,
and both receipt hashes. Any other issue remains fatal. Do not raise tolerance
or assume the discrepancy is explained by rounding. This is the same native-
mask endpoint, not a test-informed choice between references.

## Models, preprocessing and controls

Use seeds 20260915, 20260917 and 20260919 and all nine settings: old, new,
continued, scratch, bcwi_025, bcwi_050, bcwi_075, kd_001 and kd_010. Bind the
existing CT-UPDATE-010 checkpoint registry, completed seed state fingerprints
and development-only four-map freeze. Load the identical stored checkpoints;
recompute the same interpolation weights. No new training, checkpoint selection,
strength selection, calibration fitting or external comparator acquisition.

Reuse 96x96x96 patches at 1.5 mm, reference-centred native XYZ geometry,
HU clipping [-1000,1000] and division by 1000, float16 cache, BF16 inference,
native-grid logit interpolation, threshold zero and the existing native metrics.
Retain truncated lesions and empty resampled targets. Empty model predictions
have volume/Dice zero and undefined HD95, remaining in MAE and Dice analyses.
Report HD95 on nonempty predictions with its patient denominator.

## Estimands and complete-batch analysis

For each pair, reference change is FU minus BL native volume. Compute signed
prediction-minus-reference change error and its absolute value before averaging.
Average pairs within patient, the three fixed seeds within patient, then patients
equally. Reuse the established paired-patient bootstrap with 10,000 draws,
bootstrap seed 20260924 and family size six (three PanTrack contrasts plus these
three internal contrasts). The three primary contrasts are:

1. Ordinary update benefit: MAE(old/old) minus MAE(old/new).
2. Historical-baseline retention cost: MAE(old/new) minus MAE(new/new).
3. Contour benefit: follow-up Dice(new) minus Dice(old).

Report ordinary 95% and the existing approximate Bonferroni percentile intervals
for the fixed-seed average. Keep each seed's results and all standard secondary
comparisons. Intervals condition on these models and maps; they do not integrate
future training or calibration-fit uncertainty. No cohort pooling, clinical
equivalence margin, response classification or additional primary hypothesis.
The 5 mL precision target remains reporting resolution only.

Report all 21 strategies (17 unmapped and four previously fitted scalar maps),
MAE, signed errors, nine follow-up Dice summaries, HD95 and empty counts. The
map direction and zero clipping stay as in amendment 010. For the amendment 002
CSV-reference sensitivity, replace both reference volumes by CSV mm3/1000 for
all eligible pairs and compute all strategy point estimates and primary-contrast
point differences, by seed and fixed-seed average. No new significance testing
or preferential reference selection. Dice and HD95 remain native-mask based.

Do not inspect partial model outcome summaries. Require all three seeds, all
models and complete cohort coverage before analysis. Preserve failures and
contrary results. PanTrack influence/size diagnostics do not select this cohort
or become additional primary endpoints. Recheck point-estimate arithmetic after
the complete results exist and before interpretation.

## Ordered access and integrity gates

1. Freeze this amendment, machine policy and tested source manifest. Bind raw
   inventory/split/file-hash manifests, transfer receipt, old source/execution
   policies, checkpoint registry, state receipts and maps. Check original input
   roles, BL/FU role conventions, CSV header allowlist, split grouping, and
   storage before extraction. Hash the selected test files without interpreting
   their contents; this integrity check is not test outcome inspection.
2. After metadata passes, read only test CSV bodies and the native references
   needed for the frozen eligibility/schema audit. Save a private exact reference
   manifest and public exclusion/warning counts. Apply the conditional recount
   above. Lock membership before any predictions.
3. After reference schema/resolution passes, project cache space, then extract
   the same patches. Verify image/mask file hashes and geometry again. Record
   every patch hash, coverage and failure count. Preserve original native values.
4. Freeze successful metadata, reference, preprocessing and exact cohort hashes
   in a separate execution receipt before submitting any GPU evaluation.
5. Require that receipt, unchanged source/policy, patch hashes, native mask hashes,
   checkpoint/tensor lineage and synthetic inference success at every seed.
   Generate completion receipts binding outputs to all these inputs.
6. Verify every seed receipt and complete prediction payload before parsing a
   complete batch for outcome analysis. Write new exclusive result artifacts.

The complete source bundle and exact run bindings, including a metadata-only
capacity check, must be recorded before step 2. Every unexpected data or runtime
failure stops the dependent stage. Corrections preserve original artifacts and
require documentation before renewed outcome access; no overwrite/retry loop
that silently changes membership or the scientific choices above is allowed.

## Resources, privacy and verification

Existing study/project caps remain 268,435,456,000 and 966,367,641,600 bytes.
Reserve at most 16 GiB additional space, including an 8 GiB patch-cache limit
and 512 MiB for non-cache outputs, private tables, logs and code. Actual usage
and free space are checked before each stage. No raw data are copied or deleted.
If the complete eligible cohort does not fit, stop before extraction and revise
the storage plan without inspecting model outcomes.

Use one H100 at a time; the three evaluations run sequentially with success
dependencies. All jobs have unlimited wall time. CPU preparation and analysis
request no GPU. Use the approved isolated runtime and existing NumPy/SciPy/
PyTorch/NiBabel/MONAI installations. No downloads or dependency additions.

Use synthetic failing-first tests for the new access boundary, linkage/multi-index
identity, schema/recount handling, cache integrity, empty predictions, complete
seed coverage, reference sensitivity and patient/seed averaging. Run the full
local suite and the applicable cluster tests before clinical inputs. Review the
new entry points and record actual verification coverage; do not label a same-
agent review independent. Patient-level outputs remain private on the cluster.
Repository artifacts contain aggregate statistics, metadata and hashes only.

Material Passport: existing licensed study files and frozen model artifacts;
internal holdout specified after external results, before internal outcomes;
human_reading=not_attested. No new-method, clinical-utility or publication claim.
