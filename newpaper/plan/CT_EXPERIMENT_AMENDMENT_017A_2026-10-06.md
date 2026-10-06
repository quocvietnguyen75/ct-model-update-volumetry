# CT-UPDATE-017 correction A: evaluate the trained nnU-Net weights

6 October 2026. Question IMG-ALT-01. Written after the invalid evaluation was found and before any corrected evaluation. Machine-readable binding: `config/preanalysis/ct_strong_017_fix_a_2026-10-06.json`. [Amendment 017](CT_EXPERIMENT_AMENDMENT_017_2026-10-02.md) stays in force except where this correction says otherwise.

## What went wrong

All ten runs of attempt 002 trained and were evaluated, and the analysis (job 56849) and its verification (job 56850) passed on 5 October. On 6 October, reading the verified aggregates showed that the evaluation was invalid:
- nnU-Net follow-up Dice was 0.001 to 0.012 for every model in both cohorts, against 0.54 to 0.61 for the much weaker U-Net;
- measured volumes moved by up to about 2,800 mL between models, close to the volume of the whole 96³ patch (about 2,986 mL);
- some models returned empty masks for most lesions.

**Cause.** The evaluation adapter `ct_nnunet.make_predictor` built a freshly initialized network and passed the trained weights to nnU-Net's `manual_initialization` only as a parameter list. nnU-Net 2.8.1 loads that list only inside `predict_logits_from_preprocessed_data`. The adapter called `predict_sliding_window_return_logits` directly, which uses the network as given. Every evaluation therefore used an untrained, randomly initialized network.

The synthetic end-to-end check had tested the weight transfer between training phases. For prediction it checked only the output shape and the threshold rule, which an untrained network also passes.

## What remains valid

- **Training.** Each run's three final checkpoints are bound by hash in its `training_complete.json`, whose hashes are registered in the binding file.
  - nnU-Net's own training-time pseudo-Dice, measured on its training batches, ended each phase at 0.86 to 0.93 in the three runs inspected (seeds 20261201, 20261205 and 20261210).
  - The update and continuation phases began with pseudo-Dice of 0.58 to 0.92, so they did start from the old model's weights.
  - Training is not repeated.
- **Unchanged inputs.** The preparation receipt (SHA-256 64292af8…), the schedule decision (long: 500, 250 and 250 epochs; SHA-256 85fe8936…), the policy (SHA-256 74706675…) and the code under which training ran (source manifest 71da4506…).

## The correction

1. **Weights.** `make_predictor` loads the trained weights into the network before handing it to nnU-Net. It then checks that every weight of the predictor's network equals the checkpoint's, and any mismatch stops the job.
2. **Synthetic check.** It also reports this weight equality, and its test fails on the old code.
3. **Sanity gate, before any evaluation cohort is read.**
   - One GPU job predicts the first 20 development patches with a nonempty target from the old-model training set. These are training data, so no evaluation cohort is touched.
   - It uses each of run 20261201's three models through the corrected path.
   - It requires a mean Dice of at least 0.5 for each model; otherwise nothing is evaluated.
   - The threshold is fixed here, before the job runs. It is far below the training-time pseudo-Dice and is meant only to catch a gross failure such as this one.
4. **Re-evaluation.** Each of the ten runs is evaluated again from its registered final checkpoints. The inference settings, cohorts, gates, measurement, analysis, bootstrap seed, contrast family and interpretation rules are unchanged.
   - Each run's invalid `evaluation` folder is renamed `evaluation_attempt002_untrained`, and the invalid `analysis` folder `analysis_attempt002_untrained`. Both are kept.
   - The new receipts record both the code that trained the models and the code that evaluated them.
5. **Resubmission limit.** This uses the single resubmission that amendment 017 allows for a technical failure, for the evaluation stage only.

## What the invalid attempt can and cannot inform

The invalid aggregates were read before the defect was found. They describe untrained networks and carry no information about how the trained models behave, so the correction draws on no outcome knowledge. They stay in the record and will be disclosed as a failed first evaluation:
- [results](../research_selection_2026-09-13/CT_STRONG_RESULTS_INVALID_ATTEMPT002_2026-10-05.json), SHA-256 78247f46…;
- [verification](../research_selection_2026-09-13/CT_STRONG_VERIFICATION_INVALID_ATTEMPT002_2026-10-05.json), SHA-256 0492a32c….

No design, analysis or interpretation choice changes.

## Interpretation boundary

As in amendment 017. Every result of the corrected evaluation is reported, whatever its direction.
