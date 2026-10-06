# CT amendment 006: ordinary same-image distillation

Date: 16 September 2026. Written after all amendment 005 outcomes and before
any distillation training or outcomes. This is adaptive development within
IMG-ALT-01 under the author's instruction to continue experiments.

## Material Passport

- Evidence: [complete controls assessment](CT_CONTROLS_BATCH_ASSESSMENT_2026-09-16.md)
  and the September 14 contribution brief's required ordinary-distillation control.
- Human reading: not_attested. Implementation/review: inline.
- Frozen data, endpoint, patient splits and test seal remain unchanged. No new
  dataset, checkpoint, dependency, or storage-cap increase is required.

## Fixed training comparison

For each seed 20260915, 20260917 and 20260919, initialize two students from that
seed's existing 20,000-step old checkpoint. Each has a frozen, evaluation-mode
copy of the same old network as teacher. The teacher and student receive the
identical augmented single-timepoint image tensor. Never constrain baseline
and follow-up predictions to each other: real lesion change is not a training
error. Neither the teacher nor optimizer carries forward from another strength.

Train each student for 10,000 steps on all 225 eligible training patients,
using the existing sampler, architecture, batch size, optimizer restart,
learning rate, augmentation and mixed-precision recipe. Use the final checkpoint.
The ordinary update is the existing zero-distillation endpoint.

Let p = sigmoid(old_logit / T) and q = sigmoid(student_logit / T), with **T=2**.
Use the original supervised DiceCE plus **lambda * T^2 * mean_voxel KL(p || q)**,
where binary KL includes both foreground and background probabilities. The mean
is over the batch and spatial voxels. Fix **lambda=1 and lambda=10**, named
kd_001 and kd_010. These two decade-separated strengths are a bounded sensitivity
comparison chosen now, not optimized values from this dataset or a replication
of another paper's hyperparameters. Compute the KL stably in float32 with a
detached teacher; verify its direction and gradients against synthetic examples.

This is a conventional soft-target distillation adaptation for binary
segmentation, based on [Hinton et al.](https://arxiv.org/html/1503.02531v1).
The BCWI paper also evaluates old-teacher distillation, but its baseline adds
negative-flip weighting. We do not reproduce or rename that stronger variant
here. [BCWI, section 6](https://aclanthology.org/2024.eacl-long.174.pdf)
Uniform voxel weighting can emphasize background; supervised Dice remains
present, and evaluation uses lesion Dice and physical volume error rather than
voxel agreement. Failure of these two settings cannot reject all distillation
or positive-congruent methods.

## Evaluation

Evaluate both candidates at both timepoints for the same 27 validation patients
and 198 pairs, preserving full native references, patch geometry, logit threshold,
empty-mask rules and CSV sensitivity. Verify old/new/continued source predictions
and all checkpoint hashes before evaluation. Retain the prior scratch/BCWI
results for the complete comparison; do not rerun or silently replace them.

Report old/candidate and candidate/candidate MAE and signed error, follow-up
Dice, conditional HD95 and empty counts. Use the same four candidate contrasts
as amendment 005: mixed-error benefit vs old/new, retaining-old-baseline cost,
same-version benefit vs new/new, and Dice difference vs new. Retain the existing
2,000 paired-patient bootstrap draws with seed 20260916. These are exploratory,
unadjusted, per-training-seed intervals. Report both strengths and all seeds;
do not select the best setting or call numerical similarity non-inferiority.

## Execution and gates

One job per seed trains both students sequentially and evaluates both. Use at
most two H100s concurrently; the third depends on the first two succeeding.
Each job requests one GPU, 8 CPUs, 48 GiB memory and a three-hour timeout.
Preserve earlier checkpoints; the six new checkpoints add about 0.12 GB.
Validate the unchanged storage policy and existing cache/index/source receipts.

Before execution, write failing synthetic tests for binary KL direction,
temperature scaling, teacher detachment, same-image inputs, teacher immutability,
finite extreme logits, fixed policy/seed membership, candidate cohort completeness,
and generalized candidate aggregation. Verify the new loss on synthetic H100
data before patient training; require the teacher state hash to stay unchanged.
Run the full repository suite and cluster CT tests. Record package/code hashes,
teacher/parent lineage, loss components, policy hash and final checkpoint hashes.
Keep private predictions on the cluster and export aggregate results only.
