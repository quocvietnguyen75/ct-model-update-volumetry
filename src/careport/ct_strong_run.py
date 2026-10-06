"""Execute CT-UPDATE-017: prepare nnU-Net datasets, fix the schedule by timing, train and evaluate runs, analyze.

Stages: prepare (CPU), smoke (GPU, timing only), run (GPU, one array task per seed), analyze and verify (CPU).
Correction A (amendment 017A) adds sanity (GPU, training patches only) and re-evaluates registered training.
nnU-Net reads its folders when first imported, so each stage sets nnUNet_results before anything imports nnunetv2.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace

from careport.ct_pantrack_run import checked_path, read_bound_json, storage_gate, write_json, write_records
from careport.ct_replication import COHORTS, MODELS, file_sha256, validate_replication_policy
from careport.ct_strong import (PHASES, PROTOCOL_ID, choose_schedule, summarize_strong, validate_strong_policy,
                                verify_strong)
from careport.ct_update_gate import GateError

SHA256 = re.compile('[0-9a-f]{64}')
RUN_METADATA = ('completed_at_utc', 'policy_sha256', 'source_manifest_sha256', 'amendment_sha256',
                'schedule_sha256', 'seed_receipt_sha256', 'storage', 'verification_pending',
                'training_source_manifest_sha256', 'correction_sha256', 'sanity_receipt_sha256')
# Correction A (amendment 017A): re-evaluate the registered checkpoints with their trained weights loaded.
FIX_FILE = 'config/preanalysis/ct_strong_017_fix_a_2026-10-06.json'


def validate_fix(fix: dict, policy: dict, policy_sha: str) -> None:
    """The correction must bind this policy, every registered run's training and a fixed sanity rule."""
    sanity, invalid = fix.get('sanity') or {}, fix.get('invalid_attempt') or {}
    training = fix.get('training_complete_sha256') or {}
    hashes = [fix.get(k) for k in ('amendment_sha256', 'base_policy_sha256', 'training_source_manifest_sha256',
                                   'prepare_receipt_sha256', 'schedule_sha256')]
    hashes += [*training.values(), invalid.get('results_sha256'), invalid.get('verification_sha256')]
    threshold = sanity.get('minimum_mean_dice')
    if (fix.get('protocol_id') != PROTOCOL_ID or fix.get('correction') != 'A' or fix.get('retrain') is not False
            or fix.get('base_policy_sha256') != policy_sha or set(training) != {str(s) for s in policy['seeds']}
            or not all(isinstance(h, str) and SHA256.fullmatch(h) for h in hashes)
            or sanity.get('seed') not in policy['seeds'] or sanity.get('development_subset') != 'old'
            or type(sanity.get('patches')) is not int or sanity['patches'] < 1
            or type(threshold) not in (int, float) or not 0 < threshold <= 1
            or invalid.get('evaluation_folder_renamed_to') in (None, '', 'evaluation')
            or invalid.get('analysis_folder_renamed_to') in (None, '', 'analysis')
            or fix.get('patient_level_values_in_outputs') is not False):
        raise GateError('Correction binding is incomplete or does not match the policy')


def correction_ready(ctx, seed: int, folder: Path) -> None:
    """Correction A evaluates registered training only, once the invalid evaluation is preserved under its new name."""
    marker = folder/'training_complete.json'
    if not marker.is_file() or file_sha256(marker) != ctx.fix['training_complete_sha256'][str(seed)]:
        raise GateError('Correction A re-evaluates registered training only; nothing is retrained')
    if not (folder/ctx.fix['invalid_attempt']['evaluation_folder_renamed_to']).is_dir() or (folder/'evaluation').exists():
        raise GateError('The invalid evaluation must be preserved under its new name before re-evaluation')


def load_context(policy_path: Path, code: Path | None = None) -> SimpleNamespace:
    from careport.ct_internal_holdout import validate_holdout_policy
    from careport.ct_pantrack_confirm import validate_confirmation_policy
    code = (code or Path.cwd()).resolve()
    policy = read_bound_json(policy_path, os.environ['CT_STRONG_POLICY_SHA256'])
    validate_strong_policy(policy)
    if file_sha256(code/policy['amendment_file']) != policy['amendment_sha256']:
        raise GateError('Strong-model amendment changed')
    source_sha = file_sha256(code/'EXECUTION_SHA256SUMS')
    if source_sha != os.environ['CT_STRONG_SOURCE_SHA256']:
        raise GateError('Strong-model source manifest differs from the submitted bundle')
    replication = read_bound_json(code/policy['replication_policy_file'], policy['replication_policy_sha256'])
    validate_replication_policy(replication)
    if file_sha256(code/replication['amendment_file']) != replication['amendment_sha256']:
        raise GateError('Replication amendment changed')
    if os.environ.get('CT_UPDATE_MODEL_FREEZE_SHA256') != replication['pantrack_model_freeze_sha256']:
        raise GateError('PanTrack model-freeze binding differs from the replication policy')
    if policy['cohorts'] != replication['cohorts']:
        raise GateError('Strong-model cohorts differ from the frozen replication cohorts')
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
    root = checked_path(project, policy['nnunet']['root_relative'])
    folders = {'nnUNet_raw': root/'nnUNet_raw', 'nnUNet_preprocessed': root/'nnUNet_preprocessed'}
    if any(Path(os.environ.get(k, '')).resolve() != v.resolve() for k, v in folders.items()):
        raise GateError('nnU-Net folders must be the registered ones under the strong-model run root')
    frozen = SimpleNamespace(policy=replication, policy_sha=policy['replication_policy_sha256'], code=code, base=base,
                             internal_policy=internal_policy, pantrack_policy=pantrack_policy, project=project,
                             roots=roots, run_root=checked_path(project, replication['run_root_relative']))
    policy_sha = file_sha256(policy_path)
    fix, fix_sha, training_source = None, None, source_sha
    if os.environ.get('CT_STRONG_FIX_SHA256'):
        fix_sha = os.environ['CT_STRONG_FIX_SHA256']
        fix = read_bound_json(code/FIX_FILE, fix_sha)
        validate_fix(fix, policy, policy_sha)
        if file_sha256(code/fix['amendment_file']) != fix['amendment_sha256']:
            raise GateError('Correction amendment changed')
        training_source = fix['training_source_manifest_sha256']
    return SimpleNamespace(policy=policy, policy_sha=policy_sha, source_sha=source_sha, code=code,
                           project=project, replication=frozen, root=root, raw=folders['nnUNet_raw'],
                           preprocessed=folders['nnUNet_preprocessed'], fix=fix, fix_sha=fix_sha,
                           training_source_sha=training_source)


def validate_schedule(schedule: dict, policy: dict, policy_sha: str, source_sha: str) -> dict:
    """The recorded schedule must be the fixed rule applied to its own timing, for this policy and bundle."""
    if (schedule.get('status') != 'STRONG_SCHEDULE_FIXED' or schedule.get('protocol_id') != PROTOCOL_ID
            or schedule.get('policy_sha256') != policy_sha or schedule.get('source_manifest_sha256') != source_sha
            or schedule.get('outcomes_inspected') is not False):
        raise GateError('Schedule record belongs to another policy or source, or followed outcome access')
    decision = choose_schedule(schedule.get('epoch_seconds', []), policy)
    if any(schedule.get(k) != decision[k] for k in ('schedule', 'epochs', 'seconds_per_epoch',
                                                    'projected_long_run_hours')):
        raise GateError('Schedule record does not follow from its timing under the fixed rule')
    return decision


def validate_strong_receipt(receipt: dict, policy: dict, index: int, policy_sha: str, source_sha: str,
                            schedule_sha: str) -> None:
    if not 1 <= index <= len(policy['seeds']):
        raise GateError('Run index outside the registered seeds')
    required = dict(status='STRONG_SEED_COMPLETE', protocol_id=PROTOCOL_ID, index=index, seed=policy['seeds'][index - 1],
                    models=list(MODELS), policy_sha256=policy_sha, source_manifest_sha256=source_sha,
                    schedule_sha256=schedule_sha)
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in required.items()):
        raise GateError('Run receipt does not bind the fixed run, policy, source and schedule')
    cohorts = receipt.get('cohorts')
    if not isinstance(cohorts, dict) or set(cohorts) != set(COHORTS):
        raise GateError('Run receipt lacks a cohort')
    for cohort in COHORTS:
        entry, frozen = cohorts[cohort], policy['cohorts'][cohort]
        if (entry.get('patients') != frozen['patients'] or entry.get('pairs') != frozen['pairs']
                or entry.get('timepoints') != 2 * frozen['pairs']
                or not SHA256.fullmatch(str(entry.get('predictions_sha256', '')))):
            raise GateError('Run cohort receipt is incomplete')


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_h100():
    import torch
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise GateError('This stage requires an allocated H100 profile')
    return torch.device('cuda'), torch.cuda.get_device_name(0)


def prepare(ctx) -> None:
    import importlib.metadata as metadata
    from careport.ct_nnunet import (CONFIGURATION, DATASETS, PLANNER, PLANS, dataset_name, folder_digest,
                                    transfer_plans, write_dataset)
    marker = ctx.root/'prepare_complete.json'
    if marker.exists() or ctx.raw.exists() or ctx.preprocessed.exists():
        raise GateError('Strong-model preparation already started; inspect it before any retry')
    storage = storage_gate(ctx.project, ctx.policy)
    cache = checked_path(ctx.project, ctx.policy['development_cache_relative'])
    index = (cache/'cache_index_private.jsonl').read_bytes()
    if hashlib.sha256(index).hexdigest() != ctx.policy['development_cache_index_sha256']:
        raise GateError('Development cache index differs from the registered one')
    records = [json.loads(line) for line in index.splitlines()]
    manifests = {subset: write_dataset(cache, records, subset, ctx.raw) for subset in DATASETS}
    for subset, manifest in manifests.items():
        planned = ctx.policy['datasets'][subset]
        if manifest['cases'] != planned['cases'] or manifest['patients'] != planned['patients']:
            raise GateError('nnU-Net dataset size differs from the registered membership')
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    ids = [DATASETS['old'][0], DATASETS['expanded'][0]]
    extract_fingerprints(ids, num_processes=4, check_dataset_integrity=True, verbose=False, show_progress_bar=False)
    plan_experiments([ids[0]], experiment_planner_class_name=PLANNER)
    transfer_plans(ctx.raw, ctx.preprocessed)
    preprocess(ids, plans_identifier=PLANS, configurations=(CONFIGURATION,), num_processes=(4,), verbose=False,
               show_progress_bar=False)
    plans = {s: ctx.preprocessed/dataset_name(s)/f'{PLANS}.json' for s in DATASETS}
    configurations = {s: json.loads(p.read_text())['configurations'][CONFIGURATION] for s, p in plans.items()}
    if configurations['old'] != configurations['expanded']:
        raise GateError('The two datasets do not share one nnU-Net configuration')
    configuration = configurations['old']
    receipt = dict(status='STRONG_PREPARATION_COMPLETE', protocol_id=PROTOCOL_ID, policy_sha256=ctx.policy_sha,
                   source_manifest_sha256=ctx.source_sha, nnunet_version=metadata.version('nnunetv2'),
                   datasets=manifests, plans_sha256={s: file_sha256(p) for s, p in plans.items()},
                   patch_size=configuration['patch_size'], batch_size=configuration['batch_size'],
                   normalization=configuration['normalization_schemes'],
                   architecture=configuration['architecture']['network_class_name'],
                   n_stages=configuration['architecture']['arch_kwargs']['n_stages'],
                   features_per_stage=configuration['architecture']['arch_kwargs']['features_per_stage'],
                   preprocessed_digest={s: folder_digest(ctx.preprocessed/dataset_name(s)) for s in DATASETS},
                   storage=storage, evaluation_cohorts_read=False, completed_at_utc=_now())
    write_json(marker, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('datasets', 'storage')}), flush=True)


def _training_source(ctx) -> str:
    """The bundle that prepared, timed and trained: the current one, or the one correction A registered."""
    return getattr(ctx, 'training_source_sha', None) or ctx.source_sha


def _prepared(ctx) -> tuple[dict, str]:
    path = ctx.root/'prepare_complete.json'
    receipt = json.loads(path.read_bytes())
    if (receipt.get('status') != 'STRONG_PREPARATION_COMPLETE' or receipt.get('policy_sha256') != ctx.policy_sha
            or receipt.get('source_manifest_sha256') != _training_source(ctx)):
        raise GateError('Preparation receipt belongs to another policy or source')
    sha = file_sha256(path)
    if getattr(ctx, 'fix', None) is not None and sha != ctx.fix['prepare_receipt_sha256']:
        raise GateError('Preparation receipt differs from the one correction A registered')
    return receipt, sha


def smoke(ctx) -> None:
    _, prepare_sha = _prepared(ctx)
    path = ctx.root/'schedule.json'
    if path.exists():
        raise GateError('The schedule is already fixed')
    os.environ['nnUNet_results'] = str(ctx.root/'smoke'/'results')
    device, name = _require_h100()
    from careport.ct_nnunet import train_phase
    _, receipt = train_phase('old', ctx.policy['schedule_rule']['smoke_epochs'], ctx.preprocessed, device,
                             seed=ctx.policy['seeds'][0])
    decision = choose_schedule(receipt['epoch_seconds'], ctx.policy)
    record = dict(decision, status='STRONG_SCHEDULE_FIXED', protocol_id=PROTOCOL_ID, policy_sha256=ctx.policy_sha,
                  source_manifest_sha256=ctx.source_sha, prepare_receipt_sha256=prepare_sha, device=name,
                  smoke_checkpoint_sha256=receipt['checkpoint_sha256'], outcomes_inspected=False, decided_at_utc=_now())
    write_json(path, record)
    print(json.dumps(record), flush=True)


def _schedule(ctx) -> tuple[dict, str]:
    path = ctx.root/'schedule.json'
    record = json.loads(path.read_bytes())
    sha = file_sha256(path)
    if getattr(ctx, 'fix', None) is not None and sha != ctx.fix['schedule_sha256']:
        raise GateError('Schedule differs from the one correction A registered')
    return validate_schedule(record, ctx.policy, ctx.policy_sha, _training_source(ctx)), sha


def _sanity_receipt(ctx) -> str | None:
    """Under correction A, evaluation and analysis need a passed sanity check for this binding; returns its hash."""
    if getattr(ctx, 'fix', None) is None:
        return None
    path = ctx.root/ctx.fix['sanity']['receipt_relative']
    if not path.is_file():
        raise GateError('Correction A requires its sanity check before any evaluation')
    receipt = json.loads(path.read_bytes())
    if receipt.get('status') != 'STRONG_SANITY_PASS' or receipt.get('correction_sha256') != ctx.fix_sha:
        raise GateError('Correction A requires a passed sanity check for this binding')
    return file_sha256(path)


def _trained_checkpoints(folder: Path, seed: int, schedule_sha: str) -> tuple[dict, dict]:
    """The run's completed training and its three final checkpoints, each still matching its recorded hash."""
    training = json.loads((folder/'training_complete.json').read_bytes())
    if training.get('seed') != seed or training.get('schedule_sha256') != schedule_sha:
        raise GateError('Completed training belongs to another seed or schedule')
    checkpoints = {p: folder/training['checkpoints'][p] for p in PHASES}
    if any(file_sha256(checkpoints[p]) != training['checkpoint_sha256'][p] for p in PHASES):
        raise GateError('A trained checkpoint changed after training')
    return training, checkpoints


def sanity(ctx) -> None:
    """Correction A: before any evaluation cohort is read, the corrected predictor must segment training patches."""
    if getattr(ctx, 'fix', None) is None:
        raise GateError('The sanity stage belongs to correction A')
    plan = ctx.fix['sanity']
    path = ctx.root/plan['receipt_relative']
    if path.exists():
        raise GateError('The sanity check has already run; inspect it before any retry')
    _, schedule_sha = _schedule(ctx)
    seed = plan['seed']
    folder = ctx.root/f'run_{seed}'
    marker = folder/'training_complete.json'
    if not marker.is_file() or file_sha256(marker) != ctx.fix['training_complete_sha256'][str(seed)]:
        raise GateError('The sanity check uses registered training only')
    _, checkpoints = _trained_checkpoints(folder, seed, schedule_sha)
    device, name = _require_h100()
    from careport.ct_nnunet import binary_dice, foreground_logit, make_predictor, sanity_patches
    cache = checked_path(ctx.project, ctx.policy['development_cache_relative'])
    index = (cache/'cache_index_private.jsonl').read_bytes()
    if hashlib.sha256(index).hexdigest() != ctx.policy['development_cache_index_sha256']:
        raise GateError('Development cache index differs from the registered one')
    patches = sanity_patches(cache, [json.loads(line) for line in index.splitlines()], plan['patches'])
    mean_dice = {}
    for model in MODELS:
        predictor = make_predictor(checkpoints[model], device)
        scores = [binary_dice(foreground_logit(predictor, image) > 0, target) for _, image, target in patches]
        mean_dice[model] = sum(scores) / len(scores)
    passed = all(value >= plan['minimum_mean_dice'] for value in mean_dice.values())
    receipt = dict(status='STRONG_SANITY_PASS' if passed else 'STRONG_SANITY_FAIL', protocol_id=PROTOCOL_ID,
                   correction='A', seed=seed, patches=len(patches), data='old-model training patches',
                   minimum_mean_dice=plan['minimum_mean_dice'], mean_dice=mean_dice, device=name,
                   correction_sha256=ctx.fix_sha, source_manifest_sha256=ctx.source_sha, schedule_sha256=schedule_sha,
                   evaluation_cohorts_read=False, completed_at_utc=_now())
    write_json(path, receipt)
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise GateError('The corrected predictor fails the training-data sanity check; nothing is evaluated')


def run(ctx, index: int) -> None:
    from careport.ct_replication import evaluate_cohort, validate_replication_coverage
    from careport.ct_replication_run import load_cohorts, mask_locator
    if not 1 <= index <= len(ctx.policy['seeds']):
        raise GateError('Run index outside the registered seeds')
    decision, schedule_sha = _schedule(ctx)
    seed = ctx.policy['seeds'][index - 1]
    folder = ctx.root/f'run_{seed}'
    if getattr(ctx, 'fix', None) is not None:
        correction_ready(ctx, seed, folder)
        _sanity_receipt(ctx)
    os.environ['nnUNet_results'] = str(folder/'results')
    device, name = _require_h100()
    from careport.ct_nnunet import foreground_logit, make_predictor, train_phase
    started = time.monotonic()
    marker = folder/'training_complete.json'
    if not marker.exists():
        folder.mkdir(parents=True, exist_ok=False)  # a failed attempt is renamed and kept before a resubmission
        checkpoints, receipts = {}, {}
        for phase in PHASES:
            parent = None if phase == 'old' else checkpoints['old']
            checkpoints[phase], receipts[phase] = train_phase(phase, decision['epochs'][phase], ctx.preprocessed,
                                                              device, seed=seed, parent=parent)
            write_json(folder/f'{phase}_receipt.json', receipts[phase])
        write_json(marker, dict(status='THREE_PHASES_COMPLETE', seed=seed, schedule_sha256=schedule_sha,
                                checkpoints={p: str(c.relative_to(folder)) for p, c in checkpoints.items()},
                                checkpoint_sha256={p: r['checkpoint_sha256'] for p, r in receipts.items()},
                                evaluation_cohorts_read=False))
    training, checkpoints = _trained_checkpoints(folder, seed, schedule_sha)
    output = folder/'evaluation'
    output.mkdir(exist_ok=False)
    cohorts = load_cohorts(ctx.replication)
    storage_gate(ctx.project, ctx.policy)
    predictors = {m: make_predictor(checkpoints[m], device) for m in MODELS}

    def infer(image):
        return {m: foreground_logit(predictors[m], image) for m in MODELS}

    workers = ctx.policy['execution']['scoring_workers']
    cohort_receipts = {}
    for cohort in COHORTS:
        cache, records = cohorts[cohort]
        rows = evaluate_cohort(records, cache, mask_locator(ctx.replication, cohort), infer, workers)
        validate_replication_coverage(rows, records)
        path = output/f'{cohort}_predictions_private.jsonl'
        write_records(path, rows)
        cohort_receipts[cohort] = dict(patients=len({r['patient'] for r in rows}), pairs=len(rows) // 2,
                                       timepoints=len(rows), predictions_sha256=file_sha256(path))
    receipt = dict(status='STRONG_SEED_COMPLETE', protocol_id=PROTOCOL_ID, index=index, seed=seed, models=list(MODELS),
                   policy_sha256=ctx.policy_sha, source_manifest_sha256=ctx.source_sha, schedule_sha256=schedule_sha,
                   schedule=decision['schedule'], epochs=decision['epochs'],
                   checkpoint_sha256=training['checkpoint_sha256'], device=name, cohorts=cohort_receipts,
                   training_source_manifest_sha256=_training_source(ctx),
                   correction_sha256=getattr(ctx, 'fix_sha', None),
                   stage_elapsed_seconds=time.monotonic() - started, completed_at_utc=_now())
    validate_strong_receipt(receipt, ctx.policy, index, ctx.policy_sha, ctx.source_sha, schedule_sha)
    write_json(output/'seed_complete.json', receipt)
    print(json.dumps(dict(status=receipt['status'], seed=seed, index=index)), flush=True)


def _completed_runs(ctx, schedule_sha: str, seeds=None) -> tuple[dict, dict]:
    runs, receipts = {c: [] for c in COHORTS}, {}
    for index, seed in enumerate(ctx.policy['seeds'], start=1):
        if seeds is not None and seed not in seeds:
            continue
        path = ctx.root/f'run_{seed}'/'evaluation'/'seed_complete.json'
        if not path.is_file():
            if seeds is not None:
                raise GateError('A reported run is no longer complete')
            continue
        receipt = json.loads(path.read_bytes())
        validate_strong_receipt(receipt, ctx.policy, index, ctx.policy_sha, ctx.source_sha, schedule_sha)
        for cohort in COHORTS:
            predictions = path.parent/f'{cohort}_predictions_private.jsonl'
            if file_sha256(predictions) != receipt['cohorts'][cohort]['predictions_sha256']:
                raise GateError('Completed predictions changed')
            runs[cohort].append((seed, [json.loads(line) for line in predictions.read_bytes().splitlines()]))
        receipts[str(seed)] = file_sha256(path)
    return runs, receipts


def analyze(ctx) -> None:
    from careport.ct_replication_run import load_cohorts
    _, schedule_sha = _schedule(ctx)
    sanity_sha = _sanity_receipt(ctx)
    fix = getattr(ctx, 'fix', None)
    if fix is not None and not (ctx.root/fix['invalid_attempt']['analysis_folder_renamed_to']).is_dir():
        raise GateError('The invalid analysis must be preserved under its new name before re-analysis')
    runs, receipts = _completed_runs(ctx, schedule_sha)
    cohorts = load_cohorts(ctx.replication)
    result = summarize_strong(runs, {c: cohorts[c][1] for c in COHORTS}, ctx.policy)
    result.update(completed_at_utc=_now(), policy_sha256=ctx.policy_sha, source_manifest_sha256=ctx.source_sha,
                  amendment_sha256=ctx.policy['amendment_sha256'], schedule_sha256=schedule_sha,
                  seed_receipt_sha256=receipts, verification_pending=True,
                  training_source_manifest_sha256=_training_source(ctx),
                  correction_sha256=getattr(ctx, 'fix_sha', None), sanity_receipt_sha256=sanity_sha)
    output = ctx.root/'analysis'
    output.mkdir(exist_ok=False)
    write_json(output/'strong_results.json', result)
    print(json.dumps(dict(status=result['status'], runs=result['runs'], missing=result['missing_seeds'],
                          result_sha256=file_sha256(output/'strong_results.json'))), flush=True)


def verify(ctx) -> None:
    from careport.ct_replication_run import load_cohorts
    path = ctx.root/'analysis'/'strong_results.json'
    result_sha = file_sha256(path)
    result = json.loads(path.read_bytes())
    _, schedule_sha = _schedule(ctx)
    if (result.get('policy_sha256') != ctx.policy_sha or result.get('source_manifest_sha256') != ctx.source_sha
            or result.get('schedule_sha256') != schedule_sha
            or result.get('status') != 'STRONG_REPLICATION_ANALYSIS_COMPLETE'):
        raise GateError('Strong-model result belongs to another policy, source or schedule')
    runs, receipts = _completed_runs(ctx, schedule_sha, set(result['seeds']))
    if receipts != result['seed_receipt_sha256']:
        raise GateError('Seed receipts changed after analysis')
    cohorts = load_cohorts(ctx.replication)
    report = verify_strong({k: v for k, v in result.items() if k not in RUN_METADATA}, runs,
                           {c: cohorts[c][1] for c in COHORTS}, ctx.policy)
    if file_sha256(path) != result_sha:
        raise GateError('Strong-model result changed during verification')
    report.update(checked_at_utc=_now(), result_sha256=result_sha, policy_sha256=ctx.policy_sha,
                  source_manifest_sha256=ctx.source_sha, independent_agent=False, human_reading='not_attested')
    write_json(ctx.root/'analysis'/'strong_verification.json', report)
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'sanity', 'run', 'analyze', 'verify'])
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    os.umask(0o077)
    ctx = load_context(args.policy)
    if args.stage == 'prepare':
        prepare(ctx)
    elif args.stage == 'smoke':
        smoke(ctx)
    elif args.stage == 'sanity':
        sanity(ctx)
    elif args.stage == 'run':
        if args.index is None:
            raise GateError('The run stage needs --index')
        run(ctx, args.index)
    elif args.stage == 'analyze':
        analyze(ctx)
    else:
        verify(ctx)


if __name__ == '__main__':
    main()
