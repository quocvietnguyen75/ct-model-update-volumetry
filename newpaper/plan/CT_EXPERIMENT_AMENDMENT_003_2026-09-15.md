# CT model-update amendment 003: establish a usable old model

Date: 15 September 2026. Written after pilot 001 validation, before pilot 002
execution. This is a transparent development adaptation, not a preregistration
made before all outcome inspection. No test data have been inspected.

Pilot 001's old model returned 190/198 empty follow-up masks. Same-data extra
training almost matched the larger-data descendant. See the
[assessment](CT_PILOT_001_ASSESSMENT_2026-09-15.md). The first update comparison
cannot establish preservation of a useful historical measurement system.

For pilot 002, train a fresh old model from the same seeded initialization for
20,000 optimizer steps on the same 120 assigned IDs (113 eligible patients).
Then create two descendants from its exact final checkpoint, with fresh
optimizers: 10,000 steps on all training IDs (225 eligible patients) and
10,000 steps on the same old-data patients. Run seed 20260915. The longer budget
is an engineering choice motivated by the failed baseline; it is not proven
sufficient for convergence. All other hyperparameters and final-checkpoint
selection remain fixed. Do not compare these runs as independent random seeds.

Keep the patch cache, patient split, eligible pairs, complete native-mask
endpoint, CSV sensitivity, threshold, architecture and optimizer settings.
No raw/derived data deletion, endpoint change, package addition or download is
needed. Reuse the verified 11.8 GB development cache. Preserve pilot 001's
checkpoints, logs and results.

The separate execution policy binds the original configuration SHA-256 and may
override only training step counts and the model random seed. Validate cache
receipts against the original data configuration; bind checkpoints and evaluation
to the resulting execution configuration and policy. Unexpected keys or a stale
source hash must fail. This avoids relabelling old preprocessing as a new audit.

Use the existing patient-balanced volume-change contrasts and follow-up contour
metrics on all eligible validation pairs after the three phases finish.
First assess whether the old model segments lesions meaningfully and whether
extra same-data training still accounts for most gains. Do not invent a
post-hoc Dice threshold that is presented as clinical adequacy. Persistent
empty predictions, a still rapidly falling old-training loss, or dominant
same-data improvement means the update claim remains premature.

Pilot 002 remains a developmental baseline check. Repeated seeds, matched-budget
from-initialization training, simple correction, BCWI/distillation and the
appropriate longitudinal comparator remain outstanding. Test evaluation waits
for those design decisions and a justified final analysis. No dedicated new
method is selected by this amendment.

Allocate one H100 for the sequential three-phase run, with a three-hour limit;
submit validation after successful completion. Report job IDs and receipts.
Avoid continuous polling while long jobs run; retain file progress logs and
Slurm timeouts so the author can monitor them directly.
