# CT-UPDATE-008: development-only confirmation precision planning

Date: 18 September 2026. Written before calculating the new variance summaries.
Question IMG-ALT-01; primary endpoint unchanged. No external outcome access.

Use only the previously verified validation predictions from CT-PILOT-002,
003 and 004 (27 shared patients, 198 lesion pairs). Verify source hashes and
exact patient/pair membership. For each seed, calculate the patient-level cost
of retaining the old baseline: mean absolute change error for old/new minus
that for new/new. This is the existing registered contrast, not a new endpoint.

Average each patient's contrast across the three fixed seeds before estimating
between-patient dispersion. Also report each seed's dispersion. Do not treat
the three runs as 81 patients. Retain all patients and lesions, without trimming,
subgroup selection, transformations or choosing the most favorable seed.

Report normal-approximation interval half-widths at 27, 30, 45, 100 and 200
patients and required counts for half-widths of 1, 2 and 5 mL. Show both marginal
95% intervals and Bonferroni coverage for six contrasts (three core contrasts
in each of two cohorts). The widths 1/2/5 mL are an explicit planning grid,
not clinically justified preservation margins or a hidden method-selection rule.

Use as the planning dispersion the maximum of the averaged-seed patient SD and
the three individual-seed patient SDs. Show 1.0, 1.5 and 2.0 multipliers on this
dispersion to make sensitivity to heavier-tailed external errors visible.
These are deliberately conservative relative to the observed point estimates
of SD; they are not upper confidence bounds on population variance. The formula
uses normal quantiles, not a power calculation, a guarantee of bootstrap
coverage, or an assumption that the new disease cohort has the same variance.

This study-planning output can demonstrate that available counts are unlikely to
give a narrow estimate under comparable variability. It cannot establish any
PanTrack result, clinical effect threshold, superiority, equivalence, or paper
readiness. Publish aggregate SDs, counts, formula settings and hashes only.
All patient-level inputs and intermediate contrasts remain private on cluster.
No external/internal test model outputs, new training or package installation
are permitted by this amendment.
