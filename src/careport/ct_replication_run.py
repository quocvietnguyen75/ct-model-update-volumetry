"""Execute CT-UPDATE-015: fixed seed lookup, trainer policy, evaluation, analysis and verification."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

from careport.ct_pantrack_run import checked_path, read_bound_json, storage_gate, write_json, write_records
from careport.ct_replication import (
    COHORTS, MODELS, PROTOCOL_ID, canonical_sha256, evaluate_cohort, execution_policy, file_sha256,
    seed_for_index, summarize_replication, validate_replication_coverage, validate_replication_policy,
    verify_replication,
)
from careport.ct_update_gate import GateError

SHA256 = re.compile('[0-9a-f]{64}')


def write_execution_policy(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise GateError('Existing execution policy differs from the fixed replication policy')
        return
    with path.open('xb') as stream:
        stream.write(payload)


def load_training(project: Path, policy: dict, base: dict, index: int) -> tuple[int, dict, dict, dict]:
    from careport.ct_update_control_run import load_checkpoint
    from careport.ct_update_train import execution_config
    execution = execution_policy(policy, base, index)
    seed = execution['seed']
    config = execution_config(base, execution)
    config_sha = canonical_sha256(config)
    folder = checked_path(project, f"{policy['run_root_relative']}/seed_{seed}/training")
    completion = folder/'training_complete.json'
    if (not completion.is_file()
            or json.loads(completion.read_text()) != {'status': 'THREE_PHASES_COMPLETE', 'test_used': False}):
        raise GateError('Replication training is incomplete')
    if json.loads((folder/'execution_config.json').read_text()) != config:
        raise GateError('Stored training configuration differs from the fixed replication policy')
    states, receipts = {}, {}
    states['old'], receipts['old'] = load_checkpoint(folder, 'old', config_sha, None, seed,
                                                     policy['training']['old_steps'])
    for phase in ('new', 'continued'):
        states[phase], receipts[phase] = load_checkpoint(folder, phase, config_sha, receipts['old']['checkpoint_sha256'],
                                                         seed, policy['training']['update_steps'])
    for phase, receipt in receipts.items():
        if (receipt.get('execution_id') != execution['execution_id']
                or receipt.get('eligible_training_patients') != policy['training']['eligible_training_patients'][phase]):
            raise GateError('Replication checkpoint belongs to another execution or training cache')
    return seed, config, {m: states[m] for m in MODELS}, receipts


def validate_seed_receipt(receipt: dict, policy: dict, index: int, policy_sha: str, source_sha: str) -> None:
    seed = seed_for_index(policy, index)
    required = dict(status='REPLICATION_SEED_COMPLETE', protocol_id=PROTOCOL_ID, index=index, seed=seed,
                    execution_id=f"{policy['execution_id_prefix']}{index:02d}", models=list(MODELS),
                    policy_sha256=policy_sha, source_manifest_sha256=source_sha,
                    amendment_sha256=policy['amendment_sha256'])
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in required.items()):
        raise GateError('Replication seed receipt does not bind the fixed run, policy and source')
    cohorts = receipt.get('cohorts')
    if not isinstance(cohorts, dict) or set(cohorts) != set(COHORTS):
        raise GateError('Replication seed receipt lacks a cohort')
    for cohort in COHORTS:
        entry, frozen = cohorts[cohort], policy['cohorts'][cohort]
        if (entry.get('patients') != frozen['patients'] or entry.get('pairs') != frozen['pairs']
                or entry.get('timepoints') != 2*frozen['pairs']
                or not isinstance(entry.get('predictions_sha256'), str)
                or not SHA256.fullmatch(entry['predictions_sha256'])):
            raise GateError('Replication cohort receipt is incomplete')


def load_context(policy_path: Path) -> SimpleNamespace:
    from careport.ct_internal_holdout import validate_holdout_policy
    from careport.ct_pantrack_confirm import validate_confirmation_policy
    policy = read_bound_json(policy_path, os.environ['CT_REPLICATION_POLICY_SHA256'])
    validate_replication_policy(policy)
    code = Path.cwd().resolve()
    if file_sha256(code/policy['amendment_file']) != policy['amendment_sha256']:
        raise GateError('Replication amendment changed')
    source_sha = file_sha256(code/'EXECUTION_SHA256SUMS')
    if source_sha != os.environ['CT_REPLICATION_SOURCE_SHA256']:
        raise GateError('Replication source manifest differs from the submitted bundle')
    base = read_bound_json(code/policy['source_config_file'], policy['source_config_sha256'])
    internal_policy = read_bound_json(code/policy['internal_policy_file'], policy['internal_policy_sha256'])
    pantrack_policy = read_bound_json(code/policy['pantrack_policy_file'], policy['pantrack_policy_sha256'])
    read_bound_json(code/policy['pantrack_registry_file'], policy['pantrack_registry_sha256'])
    validate_holdout_policy(internal_policy, base)
    validate_confirmation_policy(pantrack_policy)
    if os.environ.get('CT_UPDATE_MODEL_FREEZE_SHA256') != policy['pantrack_model_freeze_sha256']:
        raise GateError('PanTrack model-freeze binding differs from the replication policy')
    project = Path(os.environ[policy['project_root_env']]).resolve()
    roots = {c: Path(os.environ[policy[f'{c}_dataset_root_env']]).resolve() for c in COHORTS}
    if any(not root.is_relative_to(project) or root == project for root in roots.values()):
        raise GateError('Replication dataset root is outside the registered study storage')
    return SimpleNamespace(policy=policy, policy_sha=file_sha256(policy_path), source_sha=source_sha, code=code,
                           base=base, internal_policy=internal_policy, pantrack_policy=pantrack_policy,
                           project=project, roots=roots,
                           run_root=checked_path(project, policy['run_root_relative']))


def load_cohorts(ctx) -> dict[str, tuple[Path, list[dict]]]:
    """Frozen evaluation caches through their original, hash-bound gates."""
    from careport.ct_internal_run import load_verified_cache
    from careport.ct_pantrack_run import load_cache
    freeze = read_bound_json(checked_path(ctx.project, ctx.internal_policy['freeze_relative']),
                             ctx.policy['internal_freeze_sha256'])
    if freeze.get('policy_sha256') != ctx.policy['internal_policy_sha256']:
        raise GateError('Internal cohort freeze belongs to another policy')
    split = read_bound_json(ctx.roots['internal']/'data_split.json', ctx.base['expected_split_sha256'])
    internal = load_verified_cache(ctx.project, ctx.internal_policy, freeze, set(split['test']))
    bindings = SimpleNamespace(policy=ctx.code/ctx.policy['pantrack_policy_file'],
                               registry=ctx.code/ctx.policy['pantrack_registry_file'])
    pantrack_cache, pantrack_records, _ = load_cache(bindings, ctx.pantrack_policy, ctx.project)
    cohorts = {'internal': internal, 'pantrack': (pantrack_cache, pantrack_records)}
    for cohort, (_, records) in cohorts.items():
        frozen = ctx.policy['cohorts'][cohort]
        if len(records) != 2*frozen['pairs'] or len({r['patient'] for r in records}) != frozen['patients']:
            raise GateError('Frozen evaluation cohort size differs from the replication policy')
    return cohorts


def mask_locator(ctx, cohort: str):
    if cohort == 'internal':
        return lambda r: (checked_path(ctx.roots['internal'], r['mask_relative']), r['native_mask_sha256'])
    return lambda r: (checked_path(ctx.roots['pantrack'], f"labels/{r['scan']}.nii.gz"), r['native_mask_sha256'])


def seed_folder(ctx, seed: int) -> Path:
    return checked_path(ctx.project, f"{ctx.policy['run_root_relative']}/seed_{seed}")


def evaluate(ctx, index: int) -> None:
    import torch
    from careport.ct_update_control_run import state_fingerprint
    from careport.ct_update_train import make_model, seed_runtime
    started = time.monotonic()
    seed, config, states, receipts = load_training(ctx.project, ctx.policy, ctx.base, index)
    cohorts = load_cohorts(ctx)
    storage_gate(ctx.project, ctx.policy)
    output = seed_folder(ctx, seed)/'evaluation'
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise GateError('Replication inference requires an allocated H100 profile')
    seed_runtime(seed)
    fingerprints = {m: state_fingerprint(states[m]) for m in MODELS}
    models = {}
    for name in MODELS:
        model = make_model(config)
        model.load_state_dict(states[name], strict=True)
        models[name] = model.cuda().eval().requires_grad_(False)
    del states
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        probe = torch.zeros((1, 1, *config['patch']['shape']), device='cuda')
        for model in models.values():
            value = model(probe)
            if value.shape != probe.shape or not torch.isfinite(value).all():
                raise GateError('Replication synthetic inference check failed')
    device = torch.cuda.get_device_name(0)
    print(json.dumps(dict(status='REPLICATION_GPU_RUNTIME_PASS', seed=seed, device=device)), flush=True)
    torch.cuda.reset_peak_memory_stats()

    def infer(image):
        # Identical to the frozen evaluation loops: float32 patch, BF16 autocast, float32 logits.
        tensor = torch.from_numpy(image[None, None]).cuda()
        logits = {}
        for name, model in models.items():
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits[name] = model(tensor)[0, 0].float().cpu().numpy()
        return logits

    workers = ctx.policy['execution']['scoring_workers']
    cohort_receipts = {}
    for cohort in COHORTS:
        cache, records = cohorts[cohort]
        rows = evaluate_cohort(records, cache, mask_locator(ctx, cohort), infer, workers)
        validate_replication_coverage(rows, records)
        path = output/f'{cohort}_predictions_private.jsonl'
        write_records(path, rows)
        cohort_receipts[cohort] = dict(patients=len({r['patient'] for r in rows}), pairs=len(rows)//2,
                                       timepoints=len(rows), predictions_sha256=file_sha256(path))
        print(json.dumps(dict(stage='cohort_complete', seed=seed, cohort=cohort, timepoints=len(rows))), flush=True)
    receipt = dict(status='REPLICATION_SEED_COMPLETE', protocol_id=PROTOCOL_ID, index=index, seed=seed,
                   execution_id=f"{ctx.policy['execution_id_prefix']}{index:02d}", models=list(MODELS),
                   policy_sha256=ctx.policy_sha, source_manifest_sha256=ctx.source_sha,
                   amendment_sha256=ctx.policy['amendment_sha256'],
                   checkpoint_sha256={m: receipts[m]['checkpoint_sha256'] for m in MODELS},
                   model_state_sha256=fingerprints,
                   training_elapsed_seconds=sum(r['elapsed_seconds'] for r in receipts.values()),
                   evaluation_elapsed_seconds=time.monotonic() - started,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), device=device,
                   scoring_workers=workers, cohorts=cohort_receipts,
                   completed_at_utc=datetime.now(timezone.utc).isoformat())
    validate_seed_receipt(receipt, ctx.policy, index, ctx.policy_sha, ctx.source_sha)
    write_json(output/'seed_complete.json', receipt)
    print(json.dumps(dict(status=receipt['status'], seed=seed, index=index)), flush=True)


def completed_runs(ctx, seeds=None) -> tuple[dict, list[dict]]:
    runs, receipts = {c: [] for c in COHORTS}, []
    for index, seed in enumerate(ctx.policy['seeds'], start=1):
        if seeds is not None and seed not in seeds:
            continue
        folder = seed_folder(ctx, seed)/'evaluation'
        if not (folder/'seed_complete.json').is_file():
            if seeds is not None:
                raise GateError('A reported replication run is no longer complete')
            continue
        receipt = json.loads((folder/'seed_complete.json').read_bytes())
        validate_seed_receipt(receipt, ctx.policy, index, ctx.policy_sha, ctx.source_sha)
        for cohort in COHORTS:
            path = folder/f'{cohort}_predictions_private.jsonl'
            if file_sha256(path) != receipt['cohorts'][cohort]['predictions_sha256']:
                raise GateError('Completed replication predictions changed')
            runs[cohort].append((seed, [json.loads(line) for line in path.read_bytes().splitlines()]))
        receipts.append(receipt)
    return runs, receipts


def analyze(ctx) -> None:
    cohorts = load_cohorts(ctx)
    runs, receipts = completed_runs(ctx)
    result = summarize_replication(runs, {c: cohorts[c][1] for c in COHORTS}, ctx.policy)
    result.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), policy_sha256=ctx.policy_sha,
                  source_manifest_sha256=ctx.source_sha, amendment_sha256=ctx.policy['amendment_sha256'],
                  seed_receipt_sha256={str(r['seed']): file_sha256(seed_folder(ctx, r['seed'])/'evaluation'/'seed_complete.json')
                                       for r in receipts},
                  training_elapsed_seconds=sum(r['training_elapsed_seconds'] for r in receipts),
                  evaluation_elapsed_seconds=sum(r['evaluation_elapsed_seconds'] for r in receipts),
                  arithmetic_verification_pending=True)
    output = ctx.run_root/'analysis'
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'replication_results.json', result)
    print(json.dumps(dict(status=result['status'], runs=result['runs'], missing=result['missing_seeds'],
                          result_sha256=file_sha256(output/'replication_results.json'))), flush=True)


def verify(ctx) -> None:
    cohorts = load_cohorts(ctx)
    path = ctx.run_root/'analysis'/'replication_results.json'
    result_sha = file_sha256(path)
    result = json.loads(path.read_bytes())
    if (result.get('policy_sha256') != ctx.policy_sha or result.get('source_manifest_sha256') != ctx.source_sha
            or result.get('status') != 'REPLICATION_ANALYSIS_COMPLETE'):
        raise GateError('Replication result belongs to another policy or source')
    runs, receipts = completed_runs(ctx, set(result['seeds']))
    for receipt in receipts:
        folder = seed_folder(ctx, receipt['seed'])/'evaluation'
        if file_sha256(folder/'seed_complete.json') != result['seed_receipt_sha256'][str(receipt['seed'])]:
            raise GateError('Seed receipt changed after analysis')
    verified = verify_replication(result, runs, {c: cohorts[c][1] for c in COHORTS}, ctx.policy)
    if file_sha256(path) != result_sha:
        raise GateError('Replication result changed during verification')
    report = dict(verified, protocol_id=PROTOCOL_ID, checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  result_sha256=result_sha, policy_sha256=ctx.policy_sha, source_manifest_sha256=ctx.source_sha,
                  inference_rerun=False, independent_agent=False, human_reading='not_attested')
    write_json(ctx.run_root/'analysis'/'replication_verification.json', report)
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['seed', 'execution-policy', 'evaluate', 'analyze', 'verify'])
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    os.umask(0o077)
    ctx = load_context(args.policy)
    if args.stage == 'seed':
        print(seed_for_index(ctx.policy, args.index))
    elif args.stage == 'execution-policy':
        seed = seed_for_index(ctx.policy, args.index)
        folder = seed_folder(ctx, seed)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'execution_policy.json'
        write_execution_policy(path, execution_policy(ctx.policy, ctx.base, args.index))
        print(path)
    elif args.stage == 'evaluate':
        evaluate(ctx, args.index)
    elif args.stage == 'analyze':
        analyze(ctx)
    else:
        verify(ctx)


if __name__ == '__main__':
    main()
