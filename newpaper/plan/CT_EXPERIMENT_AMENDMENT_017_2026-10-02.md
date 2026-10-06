# CT-UPDATE-017: stronger-model replication with nnU-Net

2 October 2026. Question: IMG-ALT-01. Specified before any nnU-Net training, prediction or outcome exists. Machine-readable decisions: `config/preanalysis/ct_strong_017_2026-10-02.json`.

## Why this extension exists

The 2 October review asked whether the retention cost depends on the weak, compact U-Net the study used, which reaches follow-up Dice of only 0.55–0.64 ([review](../ct_manuscript/review_2026-10-02/editorial_decision.md), item R6). Both methodology and domain reviewers suggested a small replication with a strong configured segmenter. The user approved this package on 2 October ([revision plan](CT_REVISION_PLAN_2026-10-02.md)). On the design, the user left the choice to "whatever works best for the paper", asked for several MIG slices, and left the run count to the agent.

**The question:** with a strong configured segmenter, does retaining the old model's baseline measurement still add volume-change error after an update, and how large is the cost compared with the U-Net?

## Design: change only the segmenter

Everything except the segmenter and its training recipe stays as in CT-UPDATE-015.

- **Model:** nnU-Net v2.8.1 with its residual-encoder "M" configuration (planner `nnUNetPlannerResEncM`, 3D full resolution, fold `all`), the field reference recommended by Isensee et al. (2024). It is installed from 43 verified wheels into a separate package folder, and the existing runtime is unchanged.
- **Data:** the same lesion-centred development patches (`derived/patches_001`, 96³ voxels at 1.5 mm, index SHA-256 ffd054e0…). Membership comes from the same flags the U-Net used:
  - old dataset: the 113 old-subset patients, 1952 patches;
  - expanded dataset: all 225 training patients, 4048 patches;
  - the 27 validation patients and every evaluation patient are excluded.

  The patches are already clipped and scaled, so nnU-Net runs with no normalization and no resampling, and every model receives the inputs the U-Net received. Plans are made on the old dataset and transferred unchanged to the expanded one, so all three models share one architecture.
- **Update protocol:**
  - old model: trained on the old dataset;
  - update: starts from the old model's final weights (all layers, the segmentation head included) and trains on the expanded dataset;
  - continuation: starts from the same weights and trains on the old dataset.

  The update and continuation each get half the old model's epochs, as in the U-Net design. Each phase reseeds and restarts nnU-Net's optimizer and learning-rate schedule. Final checkpoints are used; nothing is selected on any result.
- **Training recipe:** nnU-Net's own data augmentation, loss, deep supervision and SGD schedule, with 250 iterations per training epoch and 50 validation iterations. The epoch budget is fixed by the timing rule below.
- **Evaluation:** both frozen cohorts (internal holdout, 29 patients and 277 pairs; PanTrack, 45 patients and 181 pairs) through their original hash-bound caches, gates and the unchanged native-grid measurement. nnU-Net's two-class output is converted to the lesion-minus-background logit and thresholded at zero, exactly where its softmax exceeds one half. Inference uses nnU-Net's sliding window with its default mirroring.

## Runs and timing rule

- Ten runs, seeds 20261201–20261210, each training the three phases and evaluating them. The run count is fixed now and will not be increased after any result.
- A technical failure keeps its artifacts and may be resubmitted once; at least eight completed runs are required for the primary analysis.
- Execution: a Slurm array on the `mig` partition (one 3g.40gb H100 slice, eight CPUs and 48 GiB per task, unlimited wall time), at most five tasks at once.
- **Timing rule, fixed now and based on speed only.** Before the array is submitted, a timing check trains the old phase for three epochs on the real old dataset and evaluates nothing. Let s be its mean seconds per epoch over epochs 2 and 3. Then:
  - if 1000 × s plus a 1.5-hour evaluation allowance is at most 30 hours, the schedule is 500 / 250 / 250 epochs (nnU-Net's 500-epoch schedule for the old model);
  - otherwise it is 250 / 125 / 125 (nnU-Net's 250-epoch schedule).

  The chosen schedule is recorded before the array starts. No other budget change is allowed without a new dated amendment.

## Analysis

The weighting and contrasts are those of CT-UPDATE-015, applied to the ten nnU-Net runs.

- **Primary:** the retention cost, update benefit and contour benefit in each cohort. They form a six-contrast family, with a two-way percentile bootstrap over runs and patients (10,000 draws, generator seed 20261211, run draws shared by both cohorts) and Bonferroni family-wise intervals. "Established" again means the family-wise interval excludes zero.
- **Secondary (descriptive):** the CT-UPDATE-015 secondary contrasts, which include:
  - the baseline version shift;
  - the continuation retention cost;
  - mean MAE and signed error for every strategy;
  - follow-up Dice per model (the strength check against the U-Net);
  - empty predictions and HD95.
- **Also descriptive:** the CT-UPDATE-016 summaries for the nnU-Net runs, with the same settings and seeds as 016: lesion-size strata, relative scale, lesion weighting, run-level associations, lesion-level version disagreement and the response-category illustration.
- **Comparison with the U-Net (descriptive only):** the two studies are reported side by side. No pooled estimate or model-by-update interaction test is made.

## Gates, outputs and storage

- **Verification before use:** the code bundle's `EXECUTION_SHA256SUMS`, this amendment, the policy, the 015 policy and cohort gates, the patch cache index and every patch hash. The CT unit tests run in every job.
- **Outputs:** directories are new and never overwritten. Private datasets, checkpoints and predictions stay on the cluster; only aggregate results and hashes enter the record.
- **Verification after the run:** a separate scalar verification recomputes every primary point estimate and interval before interpretation.
- **Storage:** about 64 GiB planned for nnU-Net datasets, checkpoints and predictions, inside the 250 GiB study cap (117 GB in use on 2 October).

## Interpretation boundary

- This tests one strong configured segmenter under the study's privileged, reference-centred localization and one simulated added-data update, on two already-inspected cohorts.
- It does not test whole-scan nnU-Net with its own lesion detection, other update types or clinical response.
- The nnU-Net runs are new training realizations and are not pooled with the U-Net runs.

Every result is reported, whatever its direction.
