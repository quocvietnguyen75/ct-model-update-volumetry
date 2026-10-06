# CT-UPDATE-007: PanTrack integrity and metadata intake

Date: 18 September 2026. Question: IMG-ALT-01. Written before patient metadata
inspection in this continuation. The author acquired and transferred PanTrack
and instructed us to continue. This amendment permits intake validation; it
does not authorize opening external segmentation/measurement outcomes.

## Material Passport

Academic-research-suite, experiment preparation. Source: the prior comparator
audit, dataset documentation and completed CT development record. Human reading:
not_attested. Review is inline, same agent. Metadata eligibility is provisional
until geometry and label-instance validation are separately authorized.

## Scope and fixed rules

1. Verify every transferred core file against the author's local copy by SHA-256.
   Exclude `.cache` and optional `totalseg`. Expected core: 330 files and
   57,820,749,445 bytes. Preserve private per-file manifests and publish only
   their hashes and aggregate counts. Raw files remain unchanged.
2. Read `patients.json`, `tracking.json`, and `recommended_split.json` as intake
   metadata after successful transfer verification. Inspect the dataset card
   and file/header metadata as needed. No image or mask arrays, lesion volumes,
   model predictions, or external model scores may be inspected at this stage.
3. Validate patient-to-scan ownership, unique scan IDs and strictly increasing
   valid calendar dates. Every scan must have exactly one image and label file.
   Every tracking step must refer to the same patient's consecutive scans.
   A filename date establishes ordering only, not biological follow-up duration.
4. Candidate eligibility mirrors the existing persistent one-to-one lesion scope:
   finite baseline and follow-up points, a single unchanged positive label ID,
   and a unique target within the step. Exclude missing follow-up points, missing
   baseline points, multiple/relabelled targets and target collisions. Report
   exclusion reasons separately. A missing annotation does not prove biological
   disappearance. Registration-propagated points do not control eligibility.
5. Read the recommended split solely to quantify patient leakage across its
   scan groups. Our proposed external evaluation uses every eligible external
   patient, with no PanTrack training, calibration, tuning or method selection.
   Do not adopt the author's suggested scan split automatically.
6. All lesions and consecutive visit pairs belonging to a patient remain one
   inference cluster. Retain all eligible pairs; average pair losses within
   patient, then patients equally, consistent with the existing estimand.
7. Only CT intensity and supplied reference-centroid localization are permitted
   model inputs for the existing segmenters. Patient/date IDs, split membership,
   organ descriptions, masks, volumes and topology labels are not predictors.
   Reference centroids remain privileged localization and are explicitly disclosed.
8. Preserve the existing 250 GiB study and 900 GiB project caps. Intake metadata,
   manifests and reports receive at most 1 GiB additional storage; no image cache
   is allocated. The observed post-transfer study footprint is approximately
   121.27 GB, below the study cap. Verify current free space before intake.

Any unexpected schema or structural ambiguity fails the intake rather than
silently dropping a case or repairing a field. A structural refinement must be
documented before rerunning. Only aggregate diagnostics leave private storage.
The strict parser must pass failing-first synthetic tests for chronology,
cross-patient pairs, split overlap, duplicate IDs, unsafe paths and missing data.

## Limits and next gate

Integrity and metadata checks establish data readiness, not valid volume outcomes,
patient independence across differently pseudonymized sources, or adequate
statistical precision. Checkpoint training provenance for LongiSeg remains open.
Before external inference, freeze a separate confirmation specification, model
and preprocessing hashes, full geometry/label checks, and the exact contrasts.
The original internal test remains sealed. This amendment changes neither the
primary volume-change endpoint nor any existing result or final model selection.
