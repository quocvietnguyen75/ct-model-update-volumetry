# CT-UPDATE-013: bounded post-hoc explanation of PanTrack results

Recorded 24 September 2026, before opening patient-level predictions for these
new diagnostics. The complete CT-UPDATE-010 aggregate results have already been
inspected. This is a post-hoc analysis, not a new confirmatory test. It implements
step 1 of the [completion roadmap](CT_PAPER_COMPLETION_ROADMAP_2026-09-19.md)
and the [post-result literature review](CT_POSTRESULT_LITERATURE_REVIEW_2026-09-20.md).

## Question and scope

How much of the observed mixed-version error is associated with a change in the
measurement scale, same-version longitudinal error, and concentration among
patients? Is the explanation consistent across the three fixed training seeds?

Use every saved prediction for all 45 PanTrack patients, 181 eligible pairs,
three seeds (20260915, 20260917, 20260919), and nine registered models. No new
cohort, exclusions, model selection, fitting, training or inference. The internal
test remains sealed. No raw images, masks or new payloads need to be downloaded.
The original endpoints, reference definitions, family of six primary contrasts,
and published aggregate results remain unchanged.

## Frozen diagnostic definitions

For one pair, let Rb and Rf be reference volumes, and Bm and Fm be predicted
baseline and follow-up volumes for model m. The following signed quantities are
in millilitres:

    Eoo = (Fold - Bold) - (Rf - Rb)
    Emix = (Fm - Bold) - (Rf - Rb)
    Esame = (Fm - Bm) - (Rf - Rb)
    baseline_shift = Bm - Bold
    followup_shift = Fm - Fold
    Emix - Esame = baseline_shift
    Emix - Eoo = followup_shift

Check these identities numerically. For absolute error, report separately:

    total_difference = |Emix| - |Eoo|
    retention_difference = |Emix| - |Esame|
    same_version_difference = |Esame| - |Eoo|
    total_difference = retention_difference + same_version_difference

These are accounting identities, not causal attribution, an original theorem,
or proof that mixed-version measurement must be worse. No correlation between
algebraically coupled quantities will be offered as mechanism validation.

1. Calculate pair losses first, average within patient, then average the three
   fixed seeds within patient, then give each patient equal weight. Never take
   absolute values after averaging signed errors or seeds. Report each seed and
   the fixed-seed average, including contrary signs.
2. For every model report signed and absolute volume error at each timepoint,
   same-image old-to-model volume shifts, Dice, the three change errors and their
   three differences. Preserve all standard controls; do not select a winner.
3. Show reference baseline-size strata **<1 mL, 1 to <10 mL, and >=10 mL**.
   Within each stratum average pairs within patient before averaging patients.
   A patient can occur in more than one stratum. Display denominators; stratum
   means cannot be combined using pair counts to reproduce the primary mean.
   These descriptive bins are not clinical thresholds or treatment criteria.
4. For the new model's three error differences, report all-patient positive,
   negative and zero counts, median, central 80% empirical range, top-one and
   top-five shares of absolute contribution, and the range of leave-one-patient-
   out means. These are influence diagnostics, not alternative main results,
   confidence intervals or grounds to remove patients. Retain complete patient
   contribution tables in private cluster storage only.
5. Repeated appearances of the same patient/scan/lesion will be deduplicated for
   the separate same-image summaries. Confirm reference and prediction metrics
   agree before deduplicating. Retain all pairs for the longitudinal analysis.

## Execution and integrity gates

The machine-readable policy binds existing aggregate results, preprocessing,
model freeze, reference/cache indices and seed completion receipts by SHA-256.
Check these and the prediction payload hashes before parsing predictions.
Reuse the original membership, timestamps, reference and prediction validation.
Reproduce every unmapped strategy's MAE and signed error and every model's
follow-up Dice, for all seeds and the seed average, within 1e-9 absolute/relative
tolerance. Scalar corrections remain in the original complete results; they are
not model outputs and have no voxel-level diagnostic in this addendum.

Write synthetic failing tests before behavioral code. Run the relevant tests
and the full local unit suite, then cluster tests against the hashed source
bundle. No dependency installation. Execute in a CPU Slurm allocation with
unlimited wall time, in accordance with the user's instruction. Do not modify
the completed jobs, original artifacts or sealed test data.

Reserve at most 64 MiB additional space, including code, logs and private/public
diagnostic outputs, under the existing 250 GiB study / 900 GiB project caps.
Check actual project usage and free space before reading prediction values.
Use an exclusive new output directory. Public outputs contain only aggregate
statistics, counts, hashes and provenance; no patient identifiers or values.

## Interpretation and next action

No new p-values, bootstrap intervals, superiority claims or equivalence margins
are introduced. The old patient-bootstrap intervals condition on the existing
three trained model sets and do not quantify variability of future seeds.
An all-patient result may still be concentrated among a few influential cases.
Same-version error and historical-baseline retention can contribute with
different signs; neither can be assumed to explain the entire result.

After all diagnostics pass, document what they support and what they weaken.
Prepare the separate held-out internal evaluation specification and gates next;
this addendum itself does not open the internal test. If a technical discrepancy
appears, preserve artifacts and document a correction before further outcomes.
The objective remains a defensible empirical paper; publication readiness and
acceptance are not established by this analysis.

Material Passport: existing authorized PanTrack predictions and frozen metadata;
post-hoc same-agent analysis; human_reading=not_attested. No independent review
or novelty clearance is claimed.
