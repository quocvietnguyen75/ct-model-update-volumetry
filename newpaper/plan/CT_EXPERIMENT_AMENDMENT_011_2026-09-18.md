# CT-UPDATE-011: recover external inference after scheduler timeouts

Date: 18 September 2026. Question: IMG-ALT-01. Administrative execution
amendment to CT-UPDATE-010, recorded before any external outcome summary.

## Diagnosis and preserved evidence

Preparation job 52978 completed with exit 0:0 in 31:47. Native-instance checks
passed for all 45 patients, 181 pairs and 161 scans. All 362 patches were cached
in 961,007,192 bytes; no reference bounding box was truncated and no resampled
target was empty. Reuse this successful preparation and its original hashes.

Inference jobs 52979 and 52980 reached the four-hour scheduler limit. Their
terminal states are TIMEOUT, despite the allocation-level exit-code field
showing 0:0; batch steps were cancelled with 0:15. Their flushed private output
files contain 181 and 96 lines respectively, with no seed-completion receipts.
Both had passed the 88 CT tests and synthetic H100 inference check. Third-seed
job 52981 and analysis job 52982 cannot satisfy their successful-job dependencies.

The established failure is an underestimated wall-time allowance, not a
completed scientific result. Full native-grid scoring is included in each
prediction record. Its CPU work is a plausible performance cost, but no timing
profile establishes the dominant bottleneck. At the observed whole-job rates,
362 records would take approximately 8 and 15.1 hours respectively; these are
rough extrapolations, not promised completion times. No partial metric values
or model rankings have been inspected for this decision.

## Bounded recovery

1. Record scheduler states, counts, log hashes and partial-output hashes. Cancel
   only the two pending jobs with failed dependencies, 52981 and 52982.
2. Preserve the two incomplete run directories under names containing their
   original job IDs. Verify containment, absence of each destination, and
   unchanged file hashes after the rename. Keep all original logs, source
   bundles, model states, policies and preparation artifacts.
3. Re-run all 362 timepoints for each of the same three seeds using the exact
   source bundle and frozen checkpoint/calibration hashes from amendment 010.
   Do not merge or resume incomplete predictions. Use new scheduler IDs and
   logs. Each seed receives one H100, 8 CPUs, 48 GiB RAM and **24 hours**; at
   most two run concurrently. The third depends on both earlier successful
   seeds. The CPU analysis depends on all three successes and retains its
   original resources and complete-cohort gates.
4. Verify the original source manifest, preparation receipts, checkpoint freeze
   and storage gate before submission. Each job again runs the unchanged
   88-test CT suite and the inference runtime check. No application code change
   or new dependency is required for this scheduling repair.

The 5-GiB additional allowance, 2-GiB cache cap, 250-GiB study cap and 900-GiB
project cap remain in force, including preserved partial outputs. There is no
new training, data/model download, cohort change, endpoint change, tuning,
calibration refit or internal-test access. The scientific policy remains
CT-UPDATE-010 with its original hash. If another run fails, preserve its evidence
and diagnose again before any further retry; no automatic unbounded retries.

The recovery receipt and current experiment state carry the actual replacement
job IDs. Analyze only the complete replacement batch under amendment 010.

Material Passport: academic-research-suite execution recovery; inline review
by the same agent, human reading not_attested. This is an execution amendment,
not independent validation or evidence of research success.
