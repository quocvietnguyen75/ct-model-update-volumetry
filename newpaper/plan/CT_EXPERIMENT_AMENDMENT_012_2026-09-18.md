# CT-UPDATE-012: remove scheduler time limits at the author's request

Date: 18 September 2026. Question: IMG-ALT-01. Administrative amendment to
the resource settings of amendments 010 and 011.

The author explicitly requested unlimited runtime and a joint progress/bottleneck
check in eight hours. The cluster's main partition reports MaxTime=UNLIMITED.
Remove the wall-time limits from current inference jobs 53003, 53004 and 53005,
and dependent analysis job 53006. Prefer updating jobs in place to preserve
their running processes and scheduler dependencies; verify the scheduler reports
TimeLimit=UNLIMITED for each job. The execution receipt records what succeeded.

This supersedes amendment 011's 24-hour inference allowance and amendment 010's
15-minute analysis allowance for this batch. Keep at most two concurrent H100s,
all memory/CPU allocations, storage caps, data/model hashes, endpoint definitions
and complete-cohort analysis gates. No scientific source or frozen bundle is
edited, and no results are inspected to choose this administrative change.

Record an eight-hour checkpoint from the actual update time in UTC and Bangkok
time. At that check, inspect completed-record counts and their rate, process
health, CPU/GPU utilization, memory and I/O before attributing a bottleneck or
changing implementation. The checkpoint is a planned joint review; this document
does not establish an automatic monitoring or notification task.

## Execution correction

The attempted in-place update used `scontrol update JobId=53003 TimeLimit=0`.
That was an agent error: this command set a literal zero limit and job 53003
timed out after 11:35; its dependent jobs 53005/53006 were cancelled automatically.
An immediate attempt to set UNLIMITED found that job already ended. The same
UNLIMITED request for running job 53004 was denied by scheduler permissions.

Use replacement submissions instead. The
[official sbatch reference](https://slurm.schedmd.com/sbatch.html#OPT_time)
defines `sbatch --time=0` as unlimited; the
[scontrol reference](https://slurm.schedmd.com/scontrol.html#OPT_TimeLimit)
restricts increases to a running job's time limit to privileged users. The zero
submission value must not be reused for a live `scontrol` update.

Submit the replacement jobs on hold, verify every scheduler record explicitly
reports TimeLimit=UNLIMITED, and only then stop the remaining finite-time job,
preserve both incomplete run directories and release the replacements. Keep
all prior receipts, logs and partial outputs. Record the new job IDs and the
eight-hour checkpoint from release time. No partial prediction values are
inspected or reused.
