# CT-UPDATE-016: post hoc descriptive analyses of the replication predictions

2 October 2026. Question: IMG-ALT-01. Specified before any of the summaries below was computed, except for the review-computed values disclosed under "Provenance". Machine-readable decisions: `config/preanalysis/ct_posthoc_016_2026-10-02.json`.

## Why this extension exists

The ARS pre-submission review of 2 October found that the paper's magnitudes are reported only as absolute, patient-averaged millilitres ([review](../ct_manuscript/review_2026-10-02/editorial_decision.md), items R5, R8, R9, R10, S7 and S15). The review also found that three interpretive claims needed evidence:
- whether overlap gains reveal the version shift;
- whether the cost requires an average upward shift;
- whether the cost depends on lesion size.

The user approved this package on 2 October ([revision plan](CT_REVISION_PLAN_2026-10-02.md)). Every analysis here is post hoc and descriptive. None tests a hypothesis, none joins the primary family, and the replication's six primary contrasts and their family-wise decisions are unchanged.

**Provenance, disclosed.** On 2 October the review computed three run-level quantities from the published per-run table (`s6_replication_runs.csv`):
- Spearman correlation between per-run contour benefit and baseline shift: +0.74 in PanTrack, +0.23 internally;
- Spearman correlation between per-run baseline shift and retention cost: +0.27 in PanTrack, +0.35 internally;
- retention cost positive in 8 of the 9 runs, pooled over both cohorts, whose mean baseline shift was negative.

Section 4 recomputes them from the stored predictions and reports them as review-motivated descriptions. Nothing else below has been computed; no lesion-level, stratified or relative value from the replication has been inspected.

## What stays fixed

- The 20 completed CT-UPDATE-015 runs (seeds 20261001–20261020). Their stored private predictions are read through the original gates: each seed receipt must bind the 015 policy (SHA-256 3f629e05…) and the 015 source manifest (SHA-256 74feac6e…), and each prediction file must match the hash in its receipt.
- The 015 aggregate result file (SHA-256 2c53cd69…), which also fixes the seed list.
- The frozen evaluation cohorts through their hash-bound caches: internal holdout, 29 patients and 277 pairs; PanTrack, 45 patients and 181 pairs.
- The change-error definition: for strategy (a, b), e = V̂_FU^b − V̂_BL^a − (V_FU^ref − V_BL^ref), with old/old, old/new and new/new as in 015.
- No new training, inference, cohort, eligibility rule or reference access. Outputs contain aggregates only.

## Analyses

All are computed for each cohort separately; cohorts are never pooled. "Two-way interval" means the 015 percentile bootstrap over runs and patients, with run draws shared by both cohorts and drawn first. These intervals are descriptive, with no multiplicity adjustment.

1. **Lesion-size strata (R5, R10).**
   - Pairs are grouped by reference baseline volume: below 1 mL, 1 to below 10 mL, and 10 mL or more. These are the cut-points of the 24 September diagnostics (CT-UPDATE-013).
   - Within a stratum, a patient's pair losses are averaged over that patient's pairs in the stratum. Patients without such pairs are excluded. Patients and runs are weighted equally.
   - Reported per stratum: the patients and pairs it contains; mean change MAE of old/old, old/new and new/new; the retention cost and update benefit with two-way 95% intervals (10,000 draws, generator seed 20261101); and the runs with each sign.
   - A stratum with fewer than five patients gets point estimates and run signs only, without intervals.
2. **Relative scale (R5).**
   - Each pair's absolute change error is expressed as a percentage of its reference baseline volume and averaged over the 20 runs.
   - Reported over pairs: the median and interquartile range for old/old, old/new and new/new; the same summaries of the pair-level relative retention cost, old/new minus new/new; and the share of pairs whose run-averaged relative retention cost is positive.
3. **Lesion weighting (R5).** The retention cost recomputed with pairs instead of patients weighted equally within each run. Runs are still weighted equally.
4. **Run-level associations (R8, R9).**
   - Spearman rank correlations across the 20 runs, with ties given average ranks:
     - per-run contour benefit against per-run signed baseline shift;
     - per-run signed baseline shift against per-run retention cost.
   - The number of runs with a negative mean baseline shift, and how many of those had a positive retention cost.
   - With 20 runs these are descriptions; no p-value is computed.
5. **Lesion-level version disagreement (R9).** For every run and pair, the signed baseline difference d = V̂_BL^new − V̂_BL^old. Reported per cohort, pooled over runs and pairs:
   - the shares of d below, at and above zero;
   - the 5th, 25th, 50th, 75th and 95th percentiles of d;
   - the mean of |d|;
   - the median and interquartile range of d as a percentage of the reference baseline volume;
   - the range across runs of the share below zero.
6. **Interval sensitivity (S7).**
   - The six primary family-wise intervals are recomputed with 100,000 two-way draws (generator seed 20261102) and are otherwise unchanged. They are reported next to the published 10,000-draw intervals.
   - For each primary contrast and cohort, the exact one-sided binomial probability of at least the observed number of positive runs among the nonzero runs, under a probability of one half. This describes sign consistency; it is not an additional test of the primary hypothesis.
7. **Exploratory response-category illustration (S15).** Each pair's measured relative volume change, 100 × (V_FU − V_BL)/V_BL, is classified as:
   - "decrease" at or below the volume change of a sphere whose diameter falls by 30%, that is 100 × (0.7³ − 1) = −65.7%;
   - "increase" at or above the volume change of a sphere whose diameter grows by 20%, that is 100 × (1.2³ − 1) = 72.8%;
   - "stable" otherwise;
   - "undefined" when the predicted baseline volume is zero.

   The thresholds are the RECIST 1.1 diameter thresholds converted to volume. RECIST applies to the sum of target-lesion diameters per patient and adds a 5 mm absolute rule, so this lesion-level classification is an illustration only.

   Reported per run and then as the mean and range over runs:
   - the share of pairs whose category differs between old/new and new/new;
   - each strategy's agreement with the reference category;
   - the share of pairs where new/new matches the reference but old/new does not, and the reverse.

## Execution

- **Jobs.** One CPU analysis job, followed (afterok) by one CPU verification job, on partition `mig` without a GPU: four CPUs, 24 GiB, a four-hour limit.
- **Code.** A new code bundle whose `EXECUTION_SHA256SUMS` is verified before use, with the CT unit tests run first in each job.
- **Output.** Written exclusively to `runs/posthoc_016/`.
- **Storage.** Under 10 MiB, inside the existing 250 GiB study cap.
- **Verification.** It recomputes every point estimate, count, quantile and correlation through a separate scalar path. Intervals are reproduced with the same resampling code: a reproducibility check, not an independent derivation.
- **Records.** Only the aggregate result and verification files are copied into the research record.

## Interpretation boundary

These are post hoc descriptions of one fixed update procedure on two fixed, already-inspected cohorts. They do not change any primary contrast, decision or interval.
- Stratum and relative summaries describe where the retention cost arises. They are not tests of a size interaction, and the strata are confounded with patients and, across cohorts, with cancer type and anatomy.
- The correlations come from 20 runs.
- The response-category illustration is not a clinical-response assessment.

Every result is reported, whatever its direction.
