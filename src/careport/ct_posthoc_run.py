"""Execute CT-UPDATE-016: read the stored CT-UPDATE-015 predictions through their gates, summarize and verify."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace

from careport.ct_pantrack_run import checked_path, read_bound_json, storage_gate, write_json
from careport.ct_posthoc import PROTOCOL_ID, summarize_posthoc, validate_posthoc_policy, verify_posthoc
from careport.ct_replication import COHORTS, file_sha256, validate_replication_policy
from careport.ct_update_gate import GateError

RUN_METADATA = ('completed_at_utc', 'policy_sha256', 'source_manifest_sha256', 'amendment_sha256',
                'replication_results_sha256', 'seed_receipt_sha256', 'storage', 'verification_pending')


def load_context(policy_path: Path, code: Path | None = None) -> SimpleNamespace:
    """Bind this bundle and policy, and the frozen 015 policy and source manifest that the receipts name."""
    from careport.ct_internal_holdout import validate_holdout_policy
    from careport.ct_pantrack_confirm import validate_confirmation_policy
    code = (code or Path.cwd()).resolve()
    policy = read_bound_json(policy_path, os.environ['CT_POSTHOC_POLICY_SHA256'])
    validate_posthoc_policy(policy)
    if file_sha256(code/policy['amendment_file']) != policy['amendment_sha256']:
        raise GateError('Post hoc amendment changed')
    source_sha = file_sha256(code/'EXECUTION_SHA256SUMS')
    if source_sha != os.environ['CT_POSTHOC_SOURCE_SHA256']:
        raise GateError('Post hoc source manifest differs from the submitted bundle')
    replication = read_bound_json(code/policy['replication_policy_file'], policy['replication_policy_sha256'])
    validate_replication_policy(replication)
    if file_sha256(code/replication['amendment_file']) != replication['amendment_sha256']:
        raise GateError('Replication amendment changed')
    if os.environ.get('CT_UPDATE_MODEL_FREEZE_SHA256') != replication['pantrack_model_freeze_sha256']:
        raise GateError('PanTrack model-freeze binding differs from the replication policy')
    if policy['cohorts'] != replication['cohorts'] or policy['runs'] != len(replication['seeds']):
        raise GateError('Post hoc cohorts or run count differ from the replication policy')
    base = read_bound_json(code/replication['source_config_file'], replication['source_config_sha256'])
    internal_policy = read_bound_json(code/replication['internal_policy_file'], replication['internal_policy_sha256'])
    pantrack_policy = read_bound_json(code/replication['pantrack_policy_file'], replication['pantrack_policy_sha256'])
    read_bound_json(code/replication['pantrack_registry_file'], replication['pantrack_registry_sha256'])
    validate_holdout_policy(internal_policy, base)
    validate_confirmation_policy(pantrack_policy)
    project = Path(os.environ[replication['project_root_env']]).resolve()
    roots = {c: Path(os.environ[replication[f'{c}_dataset_root_env']]).resolve() for c in COHORTS}
    if any(not root.is_relative_to(project) or root == project for root in roots.values()):
        raise GateError('Dataset root is outside the registered study storage')
    frozen = SimpleNamespace(policy=replication, policy_sha=policy['replication_policy_sha256'],
                             source_sha=policy['replication_source_manifest_sha256'], code=code, base=base,
                             internal_policy=internal_policy, pantrack_policy=pantrack_policy, project=project,
                             roots=roots, run_root=checked_path(project, replication['run_root_relative']))
    return SimpleNamespace(policy=policy, policy_sha=file_sha256(policy_path), source_sha=source_sha, code=code,
                           project=project, replication=frozen,
                           output=checked_path(project, policy['output_root_relative']))


def replication_results(ctx) -> dict:
    """The verified 015 aggregate result: exact bytes, complete status, and the 015 policy and source."""
    path = checked_path(ctx.project, ctx.policy['replication_results_relative'])
    result = read_bound_json(path, ctx.policy['replication_results_sha256'])
    if (result.get('status') != 'REPLICATION_ANALYSIS_COMPLETE' or result.get('missing_seeds')
            or len(result.get('seeds', [])) != ctx.policy['runs']
            or result.get('policy_sha256') != ctx.policy['replication_policy_sha256']
            or result.get('source_manifest_sha256') != ctx.policy['replication_source_manifest_sha256']):
        raise GateError('Replication result is incomplete or belongs to another policy or source')
    return result


def core_result(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in RUN_METADATA}


def _frozen_runs(ctx) -> tuple[dict, dict, dict]:
    from careport.ct_replication_run import completed_runs, load_cohorts, seed_folder
    replication = replication_results(ctx)
    cohorts = load_cohorts(ctx.replication)
    runs, receipts = completed_runs(ctx.replication, set(replication['seeds']))
    for receipt in receipts:
        receipt_path = seed_folder(ctx.replication, receipt['seed'])/'evaluation'/'seed_complete.json'
        if file_sha256(receipt_path) != replication['seed_receipt_sha256'][str(receipt['seed'])]:
            raise GateError('A seed receipt differs from the one the replication analysis bound')
    return replication, runs, {c: cohorts[c][1] for c in COHORTS}


def analyze(ctx) -> None:
    replication, runs, expected = _frozen_runs(ctx)
    storage = storage_gate(ctx.project, ctx.policy)
    result = summarize_posthoc(runs, expected, ctx.policy)
    result.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), policy_sha256=ctx.policy_sha,
                  source_manifest_sha256=ctx.source_sha, amendment_sha256=ctx.policy['amendment_sha256'],
                  replication_results_sha256=ctx.policy['replication_results_sha256'],
                  seed_receipt_sha256=replication['seed_receipt_sha256'], storage=storage, verification_pending=True)
    output = ctx.output/'analysis'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'posthoc_results.json', result)
    print(json.dumps(dict(status=result['status'], runs=result['runs'],
                          result_sha256=file_sha256(output/'posthoc_results.json'))), flush=True)


def verify(ctx) -> None:
    path = ctx.output/'analysis'/'posthoc_results.json'
    result_sha = file_sha256(path)
    result = json.loads(path.read_bytes())
    if (result.get('policy_sha256') != ctx.policy_sha or result.get('source_manifest_sha256') != ctx.source_sha
            or result.get('status') != 'POSTHOC_ANALYSIS_COMPLETE'):
        raise GateError('Post hoc result belongs to another policy or source')
    replication, runs, expected = _frozen_runs(ctx)
    if result.get('seed_receipt_sha256') != replication['seed_receipt_sha256']:
        raise GateError('Post hoc result was computed from other seed receipts')
    report = verify_posthoc(core_result(result), runs, expected, ctx.policy)
    if file_sha256(path) != result_sha:
        raise GateError('Post hoc result changed during verification')
    report.update(checked_at_utc=datetime.now(timezone.utc).isoformat(), result_sha256=result_sha,
                  policy_sha256=ctx.policy_sha, source_manifest_sha256=ctx.source_sha, inference_rerun=False,
                  independent_agent=False, human_reading='not_attested')
    write_json(ctx.output/'analysis'/'posthoc_verification.json', report)
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['analyze', 'verify'])
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    ctx = load_context(args.policy)
    analyze(ctx) if args.stage == 'analyze' else verify(ctx)


if __name__ == '__main__':
    main()
