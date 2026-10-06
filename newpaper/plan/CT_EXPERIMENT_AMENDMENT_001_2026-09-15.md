# CT model-update experiment amendment 001

Date: 15 September 2026. Question: IMG-ALT-01.
Status: execution authorized by the author; written before lesion-value or
model-outcome inspection. This creates the CT development protocol. It does
not amend the ICU endpoints or reopen the VeriFact holdout.

## Material Passport

- Artifact: CT development experiment protocol, version 001.
- Author instruction: "Now proceed with the experiments."
- Evidence inherited: CT contribution brief and its September 14 provenance;
  September 15 local inventory and cluster staging manifest.
- Human reading: not_attested. Independent review: not performed.
- Results: none inspected when this amendment was written.

## Question and first decision

Can adding training patients improve a fixed CT tumour segmenter's contours
while preserving reference-based longitudinal volume-change accuracy when the
earlier measurement came from the old model?

The first experiment tests the basic effect and its explanations. It does not
introduce a new architecture or establish that a paper will have positive
results. Both CT timepoints remain available. Preserve old/old, old/new and
new/new measurements and a same-data continued-training control.

## Access and schema sequence

1. Verify the received 2,309 files against the private source SHA-256 manifest.
   The received relative-path/size fingerprint and split-manifest hash must
   also match the local snapshot. This is transfer integrity only.
2. Close the metadata gate using manifest grouping, filenames, CSV headers,
   declared input roles, timepoint interpretation and storage projection.
   inputsTr includes train, validation and test patients; folder names never
   define training membership. No CSV body is read at this gate.
3. Audit training and validation data: image headers and affine geometry,
   physical units, image/mask alignment, CSV linkage and label conventions.
   Read reference values only for schema validation and declared eligibility,
   before model selection. Keep per-patient findings in private cluster files.
   Do not decode test images, test masks or test CSV bodies at this stage.
4. Only after that audit passes, extract bounded development patches, perform
   a synthetic/runtime check, then train and evaluate the development pilot.
   A technical/schema failure stops dependent work. Record the failure and
   fix it with synthetic reproduction before resubmission.

BL/FU are the supplied two study timepoints. Acquisition dates and elapsed
follow-up intervals are unavailable in the inspected manifest. No growth-per-
day or chronological recruitment claim is allowed. Image index 01/02 denotes
the supplied image mapping, not automatically an additional follow-up visit.

## Development cohort and protected inputs

Use the published patient split: 240 train, 30 validation, 30 held-out test.
Rank training IDs by SHA-256 of `ct-update-001:20260915:` followed by the ID.
The first 120 form the old-data subset; the updated pool is all 240. This
simulates additional training data, not a temporal or hospital shift.

The initial target is matched, already localized lesions. Require topology
UNCHANGED, explicit linking_unclear=False, valid integer image mappings at
both timepoints and a verified one-to-one positive label identity. Missing
linking_unclear is unknown and excluded from the primary cohort, with aggregate
counts retained. Exclude new, disappearing, merging or otherwise ambiguous
correspondence; UNCHANGED does not mean unchanged volume. Never filter on
target_lesion, use_for_challenge, model error, or response magnitude.

Images and the supplied reference-derived centroids are the only model inputs.
Centroids provide privileged localization, not simulated clinical click error.
Reference masks are supervision/evaluation targets. Reference volumes, topology,
linkage flags, lesion type and target/challenge flags are never predictors.
The baseline reference contour is not substituted for an old-model prediction.

Derive reference volumes from the complete native-grid lesion masks in mm3/1000,
and compare with the CSV mm3 convention as a schema check. Resolve coordinate
axis order and label identity from primary documentation and training metadata;
do not silently guess or choose the convention by validation performance.

The official [data description](https://fdat.uni-tuebingen.de/records/75kj1-64747)
defines voxel centroids, image IDs, mm3 reference volumes and topology fields.
Its inconsistent 300/333 count remains a provenance caveat; the supplied copy
has 300 patient IDs. No original archive checksum has been verified.

## Pilot specification

Use a fixed MONAI 1.5.1 3D U-Net with channels (16,32,64,128,256), four factor-2
downsamplings, two residual units, instance normalization, one image channel
and one foreground logit. Fixed centred patches contain 96 cubed voxels at
1.5 mm spacing (144 mm per side). Crop size never uses lesion size. Clip CT
values to [-1000,1000] HU and divide by 1000. Use linear image interpolation
and nearest-neighbour target interpolation. Verify native-grid reconstruction
on synthetic geometry before measuring volumes. Record crop truncation; retain
truncated lesions in the primary estimate with full native reference volumes.

Patient-uniform, then lesion-uniform and timepoint-uniform training sampling;
random axis reflections applied identically to image and target. Fixed Dice
plus binary cross-entropy loss; AdamW, learning rate 0.0003, weight decay 0.00001,
batch size 4. All runs share preprocessing and model definition.

The initial feasibility seed is 20260915. Train the old model for 4,000 steps.
From that exact checkpoint and with fresh optimizer states, run 4,000 more
steps on (a) all 240 training patients and (b) the same old-data 120 patients.
The latter separates added-data effects from additional optimization. Use the
final checkpoint for this pilot; no cherry-picking by validation volume error.
One seed is developmental evidence only. A matched-budget larger-data model
trained from the common initialization and repeated seeds remain required
before attributing a stable effect to data updates in a manuscript.

Evaluate all eligible validation patients at both timepoints. Threshold logits
at zero, use a fixed mapping to the native reference grid, and do not apply
postprocessing selected on results. Audit target-label recovery, finite values
and image/mask geometry before any outcome is accepted.

## Analysis and continuation

For each lesion, absolute change error is
abs(predicted_FU_mL - predicted_BL_mL - reference_FU_mL + reference_BL_mL).
Average lesions within patients, then patients equally. Report old/old,
old/new, new/new and the analogous continued-training controls, paired patient
contrasts, signed change error and follow-up Dice separately. Repeated seeds
do not increase the patient sample size.

The prespecified contrasts are L_oo-L_on (ordinary-update benefit), L_on-L_nn
(cost of retaining the earlier measurement), and L_oo-L_nn (refresh benefit).
Use a paired patient bootstrap with 2,000 draws and seed 20260916 for exploratory
95% percentile intervals. No confirmatory p-value, equivalence, non-inferiority,
clinical-benefit or adequately-powered-test claim is authorized by this pilot.

A subsequent dated amendment must fix the calibration split, affine correction,
BCWI/distillation settings, repeated-run budget and final uncertainty analysis
before those results are inspected. The 30 test patients stay sealed until
the complete analysis and comparator choices are frozen. The existing
contribution brief's stronger longitudinal comparator and overlap audit remain
requirements for claims beyond this controlled mechanism study.

Do not invent a magnitude threshold or change the primary population after
seeing results. Report whether the intended update actually improves contours;
if it does not, that run cannot support the premise of an accuracy-improving
update. A null or unstable effect does not establish compatibility.

## Execution and storage

Use cluster CPU jobs for integrity, schema and preprocessing; H100 for training
and inference. Dataset, runtime and output roots come only from environment
variables. Public artifacts contain aggregates, metadata and hashes only.
Raw images, patient IDs, patches, predictions and detailed logs remain private.

Retain the existing 250 GiB new-study and 900 GiB project ceilings. Provision
at most 64 GiB for development patch caches, 16 GiB for model/optimizer files,
4 GiB for runtime additions and 4 GiB for audit/log overhead. Verify current
project usage plus this 88 GiB allowance before extraction; the earlier raw
copy remains counted in study usage. No deletion or cap increase is implied.

The [runtime proposal](CT_RUNTIME_PROPOSAL_2026-09-15.md) requests three pinned
packages; installation waits for the dependency answer required by AGENTS.md.
Existing model weights and the approximately 700 GB synthetic corpus are not
needed for this pilot. The author authorized experiments, so the earlier
no-experiment staging scope is superseded for these declared development steps.
