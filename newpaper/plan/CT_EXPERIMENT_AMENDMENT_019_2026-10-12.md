# CT-UPDATE-019: post hoc compatibility and patient-level response analyses of the stored replication predictions

12 October 2026. Question IMG-ALT-01.
- **Specified before:** any of the quantities below was computed. "Provenance" lists what was already known.
- **Machine-readable decisions:** `config/preanalysis/ct_compat_019_2026-10-12.json`.

## Why this extension exists

Two reviews of the 10 October manuscript asked for it:
- **Prof. Zengchang Qin's co-author review:**
  - point 2: focus on the impact on clinical response assessment;
  - point 3: define a quantitative compatibility metric that generalizes beyond the studied data and models.
- **The 12 October ARS review:**
  - R2: a reference scale for the millilitre results;
  - R3: the precision of the studied segmenters;
  - R6: the a priori expectation;
  - Journal-Fit W1, the only Major finding: practical significance.

On 12 October the user approved this package (decision D2 in [the revision plan](CT_REVISION_PLAN_2026-10-12.md)).

The definitions follow the metrology literature reviewed in [the literature update](../ct_manuscript/literature_2026-10-12/REVIEW.md):
- the VIM's repeatability and reproducibility conditions;
- QIBA's statistics;
- Bland and Altman's limits of agreement on the log scale.

**Status of these analyses.**
- Every analysis here is post hoc and descriptive.
- None tests a hypothesis, and none joins the primary family.
- No primary contrast, interval or decision of CT-UPDATE-015 or CT-UPDATE-017 changes.

**Provenance, disclosed.** Known before this amendment:
- every published aggregate of CT-UPDATE-015, -016 and -017, including:
  - the lesion-level relative quantiles of the between-version baseline difference;
  - the pair-level relative retention costs;
  - the lesion-level category shares;
- four ratios computed in the revision plan from published means. Mixed-version change MAE exceeded same-version change MAE by about 14% and 25% for the U-Net (internal, PanTrack) and 5% and 14% for nnU-Net.

Not yet computed from the stored predictions: any log-scale quantity, any second moment, any patient-level category, and any interval for the quantities below.

## What stays fixed

**Stored predictions.** They are read through the gates that bound their analyses:
- **CT-UPDATE-015** (U-Net, 20 runs, seeds 20261001–20261020): through the verified result file (SHA-256 2c53cd69…), the seed-receipt hashes it lists, and the prediction-file hash in each receipt.
- **CT-UPDATE-017** (nnU-Net ResEnc-M, 10 runs, seeds 20261201–20261210, evaluated after correction 017A): through its verified result file (SHA-256 e8bddf9c…), the seed-receipt hashes it lists, and the prediction-file hash in each receipt.

**Cohorts.** The frozen evaluation cohorts, through their hash-bound caches:
- internal holdout: 29 patients, 277 pairs;
- PanTrack: 45 patients, 181 pairs.

Every run's predictions must cover them exactly.

**Change-error definitions.** As in 015:
- e = V̂_FU^b − V̂_BL^a − (V_FU^ref − V_BL^ref) for strategy (a, b);
- old/old, old/new and new/new strategies only. The continuation model is not used.

**No new data or models.** No new training, inference, cohort, eligibility rule or reference access.

**Outputs.** Aggregates only.

## Analyses

**General rules.**
- Each analysis runs for each segmenter and each cohort separately. Segmenters and cohorts are never pooled.
- **Two-way interval:**
  - the percentile bootstrap of 015 and 017, resampling runs and patients independently;
  - 10,000 draws, with generator seed 20261301 for the U-Net and 20261302 for nnU-Net;
  - run draws are shared by the two cohorts of a segmenter and drawn first.
  - These intervals are descriptive, with no multiplicity adjustment.
- **Weighting:**
  - within a run, values are averaged over a patient's pairs, then over patients;
  - runs are weighted equally;
  - a ratio is computed from the two-way means of its numerator and denominator, in the point estimate and in every draw.

### 1. Relative compatibility cost (millilitre scale)

**Definition.**
- A_old,new and A_new,new are the patient-balanced, run-averaged change MAEs in mL, as defined in 015.
- The prespecified retention cost is A_old,new − A_new,new. The manuscript renames it the compatibility cost.

**Reported:**
- the relative compatibility cost, 100 × (A_old,new/A_new,new − 1), the percentage by which mixing versions raised change MAE over reprocessing;
- its value in each run;
- the number of runs above zero;
- its two-way interval.

### 2. Version agreement of baseline volume (log scale)

**Valid pair-runs.** A pair-run is valid when V̂_BL^old > 0, V̂_BL^new > 0 and V̂_FU^new > 0. Reference volumes are positive by eligibility. The same valid set is used in analyses 2 and 3.

**Definition.** For each valid pair-run, d = ln(V̂_BL^new/V̂_BL^old).

**Averaging.**
- Within each run, d and d² are averaged over a patient's valid pairs, then over the patients with at least one valid pair.
- The run means are then averaged.

**Summaries.**
- bias b, the mean of d;
- second moment m₂, the mean of d²;
- standard deviation s = √(m₂ − b²);
- 95% limits of agreement b ± 1.96 s.

**Reported:**
- on the ratio scale, as percentages: 100 × (exp(b) − 1) and 100 × (exp(b ± 1.96 s) − 1), with two-way intervals for the bias, s and both limits;
- the numbers of valid and excluded pair-runs, per segmenter and cohort.

**What it describes.** Between-version differences on the same images: the measurement's reproducibility when only the model version changes. Repeat-scan variability is not part of it.

### 3. Change-error decomposition and compatibility index (log scale)

**Definitions.** For each valid pair-run:
- e = ln(V̂_FU^new/V̂_BL^new) − ln(V_FU^ref/V_BL^ref);
- e_mix = ln(V̂_FU^new/V̂_BL^old) − ln(V_FU^ref/V_BL^ref) = e + d.

**Patient-balanced means**, averaged as in analysis 2:
- M_same, the mean of e²;
- M_mix, the mean of e_mix²;
- D, the mean of d²;
- X, twice the mean of e·d.

**Reported:**
- the four means, with the identity M_mix = M_same + D + X checked to numerical precision;
- the compatibility index κ = √(D/M_same);
- the predicted inflation of root-mean-square change error when the cross term is zero, √(1 + κ²) − 1;
- the observed inflation, √(M_mix/M_same) − 1;
- the cross-term ratio X/M_same;
- a two-way interval for each.

### 4. Patient-level volumetric response categories

**Assessments.**
- **Internal cohort:** one per patient. Each patient has one baseline and one follow-up examination, and all of the patient's eligible pairs are used, across image indices.
- **PanTrack:** one per patient and consecutive scan pair (baseline scan, follow-up scan), using that scan pair's eligible lesion pairs.

**Burden.** The burden at each timepoint is the summed volume of the assessment's lesions, for the reference and for each model.

**Categories.**
- Each burden change, 100 × (FU − BL)/BL, is classified for old/old, old/new, new/new and the reference.
- The categories are those of CT-UPDATE-016:
  - decrease: at most −65.7%;
  - increase: at least +72.8%;
  - stable: otherwise;
  - undefined: when the measured baseline burden is zero.

**Conditions.** Within each run, a patient's value is the share of their assessments meeting a condition; patients and runs are weighted equally. The conditions are:
- (a) the old/new category differs from new/new;
- (b) old/new and new/new differ on increase versus no increase (the progression split);
- (c) mixing loses the reference category: new/new matches the reference and old/new does not;
- (d) mixing gains the reference category: the reverse of (c);
- (e) spurious increase: old/new is increase and new/new is not;
- (f) masked increase: new/new is increase and old/new is not;
- (g) spurious decrease and (h) masked decrease, defined in the same way;
- (i) agreement with the reference category, for each of old/old, old/new and new/new.

**Reported:**
- the mean, range over runs and two-way interval for (a)–(d) and (i);
- the mean and range for (e)–(h);
- the number of assessments and the reference category counts.

**Limits.**
- The result is not a RECIST assessment:
  - no new or non-target lesions;
  - no target-lesion selection;
  - no nadir;
  - no 5 mm rule;
  - volumes, not diameters.
- PanTrack assessments compare consecutive scans, not each follow-up with the baseline.
- The thresholds are the RECIST 1.1 diameter thresholds converted to sphere volumes. That is a convention, not an empirical equivalence.

## Execution

- **Jobs.** One CPU analysis job, followed (afterok) by one CPU verification job, with no GPU:
  - partition `mig,main` (scheduling only);
  - four CPUs, 24 GiB, a four-hour limit.
- **Code.** A new bundle whose `EXECUTION_SHA256SUMS` is checked before use. The CT unit tests run first in each job.
- **Output.**
  - Written only to `runs/compat_019/`.
  - Storage is under 10 MiB, inside the existing study cap.
- **Verification.**
  - It recomputes every point estimate, ratio, count and share through a separate scalar path.
  - It reproduces the intervals with the same resampling code. That is a reproducibility check, not an independent derivation.
- **Records.** Only the aggregate result and verification files are copied into the research record.

## Interpretation and reporting boundary

**Scope of the results.**
- These are post hoc descriptions of two fixed update procedures on two fixed, already-inspected cohorts.
- The log-scale analyses condition on non-zero measurements.
- κ and the inflation values describe the two studied segmenters. They do not predict another update.
- The patient-level categories are not a clinical response assessment.

**Reporting.**
- Every output of analyses 1–4 is reported, in the main text or the supplement, whatever its direction.
- No output is selected for the abstract by its value. The abstract reports, for both segmenters and both cohorts:
  - the relative compatibility cost;
  - condition (a) of analysis 4.
