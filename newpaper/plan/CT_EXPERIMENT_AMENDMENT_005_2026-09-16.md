# CT amendment 005: initialization and weight-interpolation controls

Date: 16 September 2026. Written after the complete amendment 004 batch and
before these control outcomes. This remains adaptive development under the
author's instruction to continue experiments. No frozen endpoint is changed.

## Material Passport

- Question IMG-ALT-01; amendments 001–004 remain active.
- Evidence: [three-seed assessment](CT_REPLICATION_BATCH_ASSESSMENT_2026-09-16.md).
- Human reading: not_attested. Review: inline, no independent reviewer.
- No new data, model weights, packages, storage increase or test access.

## Matched-budget model from the common initialization

For each of seeds 20260915, 20260917 and 20260919, train a separate model using
all 225 eligible training patients from the same seeded random initialization
recipe used for its old model. Train 20,000 steps, save the intermediate weights,
then load those weights for a further 10,000 steps with a fresh optimizer and
the same reset seed/sampler procedure as the ordinary update. This matches the
30,000 total optimization steps and the 20,000-step optimizer restart of the
old-plus-update path. Its first phase uses all patients instead of the old subset.
It is called scratch_base followed by scratch; it is not initialized from old.pt.

Keep architecture, loss, batch size, learning rate, augmentation, fixed final
checkpoint and native reconstruction unchanged. Record hashes of both checkpoints,
the shared initialization recipe and their parent relationship. Use all patients
in both scratch stages; do not change old-training membership in the cached index.
This tests the training-data exposure path, not proof that either path converged.

## Fixed existing compatibility baseline

Apply the basic BCWI rule from Schumann et al. (EACL 2024), section 4, equation 6:
theta_alpha = alpha*theta_old + (1-alpha)*theta_new. Alpha is the old-model
weight. Evaluate exactly 0.25, 0.50 and 0.75 for each seed, retaining the existing
alpha=0 new-model and alpha=1 old-model results as endpoints. Use the fine-tuned
new checkpoint descending from that seed's old model, not the scratch checkpoint.
Match every tensor name, shape and dtype; nonfloating state must be identical.
Do not change parent weights in place or interpolate optimizer state.

Source: [BCWI paper](https://aclanthology.org/2024.eacl-long.174.pdf), equation 6
verified on 16 September. This is an adaptation of an existing classification
method to segmentation/volume measurement, not a new algorithm or a reproduction
of its text-classification experiments. No Fisher/Soup extension, tuning sweep
or selected best alpha is introduced. Report all fixed points as a tradeoff.

## Evaluation and interpretation

Evaluate scratch and the three interpolated models at both timepoints for all
27 eligible validation patients and 198 pairs. Reuse the gated native-volume,
Dice and HD95 measurement code. Cache/reference masks, thresholds, known-lesion
localization, full references, empty-mask handling and CSV sensitivity stay fixed.
Bind existing predictions to their completed hashes and exact validation keys.

Report old/old, old/new, new/new and same-data continuation controls alongside
old/candidate and candidate/candidate for each of the four candidates. Report
signed and absolute patient-balanced change error, follow-up Dice and conditional
HD95 with empty-mask counts. For every candidate report: (i) old/new minus
old/candidate error, (ii) old/candidate minus candidate/candidate error, (iii)
new/new minus candidate/candidate error, and (iv) candidate minus new follow-up
Dice. Use 2,000 paired patient bootstrap draws with seed 20260916, the existing
original-metric bootstrap. Report all intervals as exploratory and conditional
on each seed; do not treat seeds as new patients or select alpha by interval sign.

Compatibility at a point with worse Dice may simply trade away update accuracy.
Showing a useful compromise requires reporting both outcomes, not choosing one
metric after inspection. Failure of this small fixed grid would not establish
failure of all compatibility methods. Distillation and the stronger longitudinal
imaging comparator/overlap audit remain outstanding. Test data remain sealed.

## Execution

Use at most two concurrent H100s. Each seed's one-H100 job trains the scratch
control and then evaluates its four candidates, with a three-hour timeout and
48 GiB host-memory limit. Start the third seed after the first two jobs finish
successfully; retain every earlier run. No automatic best-seed retries.
Additional saved checkpoints are below 0.2 GB; reuse the verified patch cache,
check the existing storage projection and preserve all raw/derived artifacts.

Before execution, add failing synthetic tests for scratch phase/patient routing,
checkpoint lineage, interpolation identity/midpoint/shape and state preservation,
patient-balanced metrics, empty boundaries and exact validation membership.
Run the full repository suite and cluster CT tests. Write aggregate results and
hashes publicly; keep patient predictions, keys and detailed logs private.
