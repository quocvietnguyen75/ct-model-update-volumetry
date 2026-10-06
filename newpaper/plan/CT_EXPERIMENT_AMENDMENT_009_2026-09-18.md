# CT-UPDATE-009: recognize PanTrack's documented absent-target sentinel

Date: 18 September 2026. Question: IMG-ALT-01. Structural clarification of
[amendment 007](CT_EXPERIMENT_AMENDMENT_007_2026-09-18.md), written after the
metadata-only failure of job 52974 and before its corrected rerun. No external
image/mask arrays, reference volumes or model outputs have been inspected.

## Evidence and failure

Job 52974 failed with `Invalid follow-up target labels` before producing an
intake receipt or candidate manifest. The frozen first code bundle remains
preserved. A hash-verified metadata diagnostic found 217 tracking entries with
single-integer target lists; 36 use `[0]`, all with missing follow-up points.
There were no other nonpositive or noninteger target labels.

The pinned author guide explicitly documents `[0]` as an absent follow-up
lesion. The parser incorrectly required every listed target to be positive,
so this is an implementation omission, not an undocumented dataset defect.
[Author tracking format](https://github.com/MIC-DKFZ/LongiSeg/blob/5f5987be17e100889c507fe52645e7ddfbb036cf/documentation/lesion_tracking.md).
Local source SHA-256:
`fb615e2cb987a3c420c4ba03b06b70a18d1b6366e485c89e4e34506839fbfdf2`.

## Fixed interpretation before rerun

- Accept exactly the integer list `[0]` as the documented absent-target
  sentinel. It never identifies a foreground mask, a candidate pair, or a
  zero-volume measurement. Do not infer biological disappearance independently
  from the annotation convention.
- Reject negative, noninteger, boolean, duplicated, and mixed zero/positive
  target lists. Empty lists retain their existing non-correspondence handling.
- Count collisions using positive target labels only. Two absent annotations
  do not share a lesion target merely because both contain zero.
- Keep the fixed candidate scope: finite baseline/follow-up points, one
  unchanged positive label, and a unique target. No previously ineligible
  lesion becomes eligible under this clarification.
- Use mutually exclusive exclusion precedence: missing baseline point;
  documented absent follow-up target; missing follow-up point; positive-target
  collision; non-one-to-one or relabelled correspondence. This makes the known
  sentinel visible rather than hiding it among generic missing-point counts.

Write failing synthetic coverage for sentinel exclusion with missing and
present prompts, multiple absent entries, and malformed target lists before
the implementation change. Run the complete local test suite and the bounded
cluster intake tests. Deploy a new `pantrack_intake_002` bundle and retain the
failed `001` bundle, job log and receipt. The dataset remains unchanged.

## Unchanged boundaries

All access, storage, patient grouping, image/header, input-privilege and
confirmation gates in amendment 007 remain in force. This refinement changes
neither an endpoint nor a scientific hypothesis. The external cohort and
internal test remain closed to outcome inspection. No download, dependency,
GPU job, new model or additional dataset is authorized by this amendment.
