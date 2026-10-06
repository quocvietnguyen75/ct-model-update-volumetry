"""Execute frozen PanTrack preparation, inference and complete-batch analysis."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from careport.ct_pantrack_confirm import (
    MODELS, candidate_records, inspect_instances, summarize_external,
    validate_confirmation_policy, validate_prediction_coverage,
)
from careport.ct_update_calibration import fit_affine
from careport.ct_update_gate import GateError, inventory_files, metadata_fingerprint, validate_storage


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def checked_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or not (root/path).resolve().is_relative_to(root.resolve()):
        raise GateError('Artifact path escapes its registered root')
    return root/path


def read_bound_json(path: Path, expected: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise GateError('Bound artifact content changed')
    return json.loads(raw)


def write_json(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_records(path: Path, records: list[dict]) -> None:
    with path.open('x', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False)+'\n')


def fit_development_maps(rows: list[dict]) -> dict:
    counts = Counter(row['patient'] for row in rows)
    weights = [1/counts[row['patient']] for row in rows]
    result = {}
    for source, target in [('old', 'new'), ('new', 'old'), ('old', 'continued'), ('continued', 'old')]:
        fitted = fit_affine([r[source+'_bl'] for r in rows], [r[target+'_bl'] for r in rows], weights)
        result[source+'_to_'+target] = dict(slope=fitted.slope, intercept_ml=fitted.intercept,
                                          degenerate=fitted.degenerate)
    return result


def storage_gate(project: Path, policy: dict) -> dict:
    usage = sum(p.stat().st_size for p in project.rglob('*') if p.is_file() and not p.is_symlink())
    return validate_storage(policy['storage'], usage, shutil.disk_usage(project).free)


def load_states(project: Path, code: Path, registry: dict, run: dict) -> tuple[dict, dict]:
    from careport.ct_update_control_run import load_checkpoint, state_fingerprint
    from careport.ct_update_controls import interpolate_state
    from careport.ct_update_train import execution_config
    base = read_bound_json(code/registry['source_config_file'], registry['source_config_sha256'])
    execution = json.loads((code/run['execution_policy_file']).read_text())
    config = execution_config(base, execution)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_sha != run['config_sha256'] or config['seed'] != run['seed']:
        raise GateError('External model configuration differs from registered development model')
    states = {}
    for name, artifact in run['checkpoints'].items():
        path = checked_path(project, artifact['path'])
        receipt = artifact['receipt']
        if path.name != name+'.pt':
            raise GateError('Checkpoint filename and registered phase disagree')
        state, actual = load_checkpoint(path.parent, name, config_sha, receipt['parent_sha256'],
                                        run['seed'], receipt['steps'])
        if actual != receipt:
            raise GateError('Checkpoint receipt differs from frozen registry')
        states[name] = state
    states.pop('scratch_base')
    for alpha in (.25, .50, .75):
        states[f'bcwi_{int(alpha*100):03d}'] = interpolate_state(states['old'], states['new'], alpha)
    if set(states) != set(MODELS):
        raise GateError('Incomplete frozen model set')
    for name, expected in run['state_sha256'].items():
        if state_fingerprint(states[name]) != expected:
            raise GateError('External model tensor state differs from development state')
    return states, config


def freeze_models(args, policy, registry, project, code):
    from careport.ct_update_calibration_run import load_validation
    output = checked_path(project, 'control/pantrack_models_010.json')
    if output.exists():
        raise GateError('External model/calibration freeze already exists')
    base = read_bound_json(code/registry['source_config_file'], registry['source_config_sha256'])
    frozen = []
    for run in registry['runs']:
        states, _ = load_states(project, code, registry, run)
        del states
        evaluation = checked_path(project, run['development_evaluation_relative'])
        if (digest(evaluation/'development_results.json') != run['development_results_sha256']
                or digest(evaluation/'predictions_private.jsonl') != run['development_predictions_sha256']):
            raise GateError('Calibration source differs from completed development evidence')
        execution = json.loads((code/run['execution_policy_file']).read_text())
        rows, _ = load_validation(base, execution, project/'derived/patches_001', evaluation)
        if len({r['patient'] for r in rows}) != 27 or len(rows) != 198:
            raise GateError('Unexpected development calibration membership')
        frozen.append(dict(seed=run['seed'], maps=fit_development_maps(rows),
                           development_predictions_sha256=run['development_predictions_sha256'],
                           checkpoint_hashes={name: artifact['receipt']['checkpoint_sha256']
                                              for name, artifact in run['checkpoints'].items()}))
    result = dict(status='EXTERNAL_MODELS_AND_CALIBRATION_FROZEN', protocol_id=policy['protocol_id'],
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(), policy_sha256=digest(args.policy),
                  checkpoint_registry_sha256=digest(args.registry), runs=frozen,
                  external_arrays_read=False, internal_test_values_read=False,
                  calibration_fitting_patients=27, calibration_fitting_pairs=198)
    write_json(output, result)
    print(json.dumps(dict(status=result['status'], runs=len(frozen), sha256=digest(output))), flush=True)


def load_freeze(args, project):
    path = project/'control/pantrack_models_010.json'
    submitted_hash = os.environ.get('CT_UPDATE_MODEL_FREEZE_SHA256')
    if not submitted_hash:
        raise GateError('External coefficient freeze is not bound to the submitted job')
    value = read_bound_json(path, submitted_hash)
    if (value.get('status') != 'EXTERNAL_MODELS_AND_CALIBRATION_FROZEN'
            or value.get('policy_sha256') != digest(args.policy)
            or value.get('checkpoint_registry_sha256') != digest(args.registry)
            or value.get('external_arrays_read') is not False
            or [r['seed'] for r in value['runs']] != [20260915, 20260917, 20260919]):
        raise GateError('External model/calibration freeze is stale or incomplete')
    return value, digest(path)


def prepare(args, policy, project, dataset):
    import nibabel as nib
    from careport.ct_update_preprocess import make_patch_pair, projected_cache_bytes
    _, freeze_sha = load_freeze(args, project)
    intake = read_bound_json(project/'control/pantrack_intake_002.json', policy['intake_sha256'])
    candidates = read_bound_json(project/'control/pantrack_candidates_002_private.json', policy['candidate_sha256'])
    manifest = read_bound_json(project/'control/pantrack_remote_manifest_2026-09-18.json', policy['file_manifest_sha256'])
    if intake['status'] != 'METADATA_GATE_PASS' or intake['metadata_sha256'] != policy['metadata_sha256']:
        raise GateError('External intake does not match frozen policy')
    if metadata_fingerprint(inventory_files(dataset)) != policy['metadata_sha256']:
        raise GateError('External file inventory changed')
    records = candidate_records(candidates['pairs'])
    if (len(records) != 2*policy['expected_pairs']
            or len({r['patient'] for r in records}) != policy['expected_patients']):
        raise GateError('Candidate cohort count changed')
    projection = projected_cache_bytes(len(records), policy['patch']['shape'], policy['storage']['cache_cap_bytes'])
    storage = storage_gate(project, policy)
    output = checked_path(project, 'derived/pantrack_010')
    if output.exists():
        raise GateError('External preparation output already exists')
    file_hashes = {f['path']: f['sha256'] for f in manifest['files']}
    for index, entry in enumerate(manifest['files']):
        if digest(checked_path(dataset, entry['path'])) != entry['sha256']:
            raise GateError('External source file changed since transfer verification')
        if (index+1) % 50 == 0:
            print(json.dumps(dict(stage='integrity', verified=index+1, total=len(manifest['files']))), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    grouped = defaultdict(list)
    for record in records:
        grouped[record['scan']].append(record)
    references = []
    for number, (scan, entries) in enumerate(sorted(grouped.items())):
        image = nib.load(dataset/'images'/f'{scan}_0000.nii.gz')
        label = nib.load(dataset/'labels'/f'{scan}.nii.gz')
        if (len(image.shape) != 3 or image.shape != label.shape
                or image.header.get_xyzt_units()[0] != 'mm' or label.header.get_xyzt_units()[0] != 'mm'
                or not np.allclose(image.affine, label.affine, rtol=0, atol=1e-4)):
            raise GateError('External native geometry changed or does not agree')
        rows = inspect_instances(np.asanyarray(label.dataobj), image.affine, entries,
                                 policy['centroid_tolerance_voxels'])
        for row in rows:
            row['native_mask_sha256'] = file_hashes[f'labels/{scan}.nii.gz']
        references.extend(rows)
        if (number+1) % 20 == 0:
            print(json.dumps(dict(stage='reference_schema', scans_verified=number+1, total=len(grouped))), flush=True)
    write_records(output/'reference_records_private.jsonl', references)
    write_json(output/'reference_schema.json', dict(status='EXTERNAL_REFERENCE_SCHEMA_PASS',
               patients=45, pairs=181, timepoints=362, scans_checked=len(grouped),
               policy_sha256=digest(args.policy), model_freeze_sha256=freeze_sha,
               reference_index_sha256=digest(output/'reference_records_private.jsonl'),
               image_voxels_read=False, label_voxels_read=True, model_outputs_read=False,
               internal_test_values_read=False))
    # Every candidate reference has passed before any CT patch is extracted.
    (output/'patches').mkdir()
    by_scan = defaultdict(list)
    for record in references:
        by_scan[record['scan']].append(record)
    cached, counts = [], Counter()
    for number, (scan, entries) in enumerate(sorted(by_scan.items())):
        image = np.asanyarray(nib.load(dataset/'images'/f'{scan}_0000.nii.gz').dataobj)
        labels = np.asanyarray(nib.load(dataset/'labels'/f'{scan}.nii.gz').dataobj)
        for record in entries:
            patch, target, affine, truncated = make_patch_pair(image, labels, record, policy['patch'])
            relative = 'patches/'+record['pair_id']+'_'+record['timepoint']+'.npz'
            path = checked_path(output, relative)
            with path.open('xb') as stream:
                np.savez(stream, image=patch, target=target)
            counts['patches'] += 1
            counts['cache_bytes'] += path.stat().st_size
            counts['truncated_reference_boxes'] += truncated
            counts['empty_resampled_targets'] += not bool(target.any())
            if counts['cache_bytes'] > projection:
                raise GateError('External cache exceeded its fixed projection')
            cached.append({**record, 'patch_file': relative, 'patch_sha256': digest(path),
                           'patch_affine': affine.tolist(), 'reference_box_truncated': truncated})
        if (number+1) % 20 == 0:
            print(json.dumps(dict(stage='patches', patches_written=counts['patches'], total=len(records))), flush=True)
    write_records(output/'cache_index_private.jsonl', cached)
    result = dict(status='EXTERNAL_PREPROCESSING_PASS', protocol_id=policy['protocol_id'],
                  patients=45, pairs=181, counts=dict(counts), projected_cache_bytes=projection, storage=storage,
                  policy_sha256=digest(args.policy), model_freeze_sha256=freeze_sha,
                  intake_sha256=policy['intake_sha256'], candidate_sha256=policy['candidate_sha256'],
                  index_sha256=digest(output/'cache_index_private.jsonl'),
                  reference_schema_sha256=digest(output/'reference_schema.json'),
                  internal_test_values_read=False, model_outputs_read=False)
    write_json(output/'preprocessing_report.json', result)
    print(json.dumps(result), flush=True)


def load_cache(args, policy, project):
    _, freeze_sha = load_freeze(args, project)
    cache = project/'derived/pantrack_010'
    report = json.loads((cache/'preprocessing_report.json').read_text())
    if (report.get('status') != 'EXTERNAL_PREPROCESSING_PASS'
            or report.get('policy_sha256') != digest(args.policy)
            or report.get('model_freeze_sha256') != freeze_sha
            or report.get('internal_test_values_read') is not False
            or report.get('intake_sha256') != policy['intake_sha256']
            or report.get('candidate_sha256') != policy['candidate_sha256']
            or report.get('index_sha256') != digest(cache/'cache_index_private.jsonl')):
        raise GateError('External preprocessing receipt is stale or incomplete')
    schema = read_bound_json(cache/'reference_schema.json', report['reference_schema_sha256'])
    if (schema['status'] != 'EXTERNAL_REFERENCE_SCHEMA_PASS'
            or schema['reference_index_sha256'] != digest(cache/'reference_records_private.jsonl')):
        raise GateError('External instance-label schema gate did not pass')
    records = [json.loads(line) for line in (cache/'cache_index_private.jsonl').read_bytes().splitlines()]
    candidates = read_bound_json(project/'control/pantrack_candidates_002_private.json', policy['candidate_sha256'])
    expected = candidate_records(candidates['pairs'])
    key = lambda r: (r['pair_id'], r['timepoint'])
    actual = {key(r): r for r in records}
    if len(actual) != len(records) or set(actual) != {key(r) for r in expected}:
        raise GateError('External cache membership differs from frozen candidates')
    for row in expected:
        if any(actual[key(row)][name] != value for name, value in row.items()):
            raise GateError('External cache identity or prompt differs from frozen candidates')
    return cache, records, report


def evaluate(args, policy, registry, project, dataset, code):
    import nibabel as nib
    import torch
    from careport.ct_update_evaluate import native_metrics
    from careport.ct_update_train import make_model, seed_runtime
    from careport.ct_update_control_run import state_fingerprint
    cache, records, report = load_cache(args, policy, project)
    storage_gate(project, policy)
    runs = [r for r in registry['runs'] if r['seed'] == args.seed]
    if len(runs) != 1:
        raise GateError('Unregistered external training seed')
    states, config = load_states(project, code, registry, runs[0])
    fingerprints = {name: state_fingerprint(state) for name, state in states.items()}
    output = checked_path(project, f'runs/pantrack_010_seed_{args.seed}')
    output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise GateError('External inference requires an allocated H100')
    seed_runtime(args.seed)
    models = {}
    for name in MODELS:
        model = make_model(config)
        model.load_state_dict(states[name], strict=True)
        models[name] = model.cuda().eval().requires_grad_(False)
    del states
    # Inference-only runtime check; no optimizer or clinical inputs.
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        image = torch.zeros((1, 1, *policy['patch']['shape']), device='cuda')
        for model in models.values():
            value = model(image)
            if value.shape != image.shape or not torch.isfinite(value).all():
                raise GateError('External synthetic inference check failed')
    print(json.dumps(dict(status='EXTERNAL_GPU_RUNTIME_PASS', models=len(models), seed=args.seed)), flush=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    predictions = []
    previous = None
    with (output/'predictions_private.jsonl').open('x', encoding='utf-8') as stream:
        for number, record in enumerate(records):
            patch_path = checked_path(cache, record['patch_file'])
            if digest(patch_path) != record['patch_sha256']:
                raise GateError('External image patch changed after preprocessing')
            with np.load(patch_path, allow_pickle=False) as payload:
                image = torch.from_numpy(payload['image'].astype(np.float32)[None, None]).cuda()
            if record['scan'] != previous:
                path = dataset/'labels'/f"{record['scan']}.nii.gz"
                if digest(path) != record['native_mask_sha256']:
                    raise GateError('External native mask changed after schema audit')
                volume = nib.load(path)
                labels, affine = np.asanyarray(volume.dataobj), volume.affine
                previous = record['scan']
            if not np.allclose(affine, record['native_affine'], rtol=0, atol=1e-4):
                raise GateError('External inference native geometry differs')
            measured = {}
            for name, model in models.items():
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(image)[0, 0].float().cpu().numpy()
                measured[name] = native_metrics(logits, record['patch_affine'], affine, labels,
                                               record['lesion_id'], record['reference_voxels'])
            result = {k: record[k] for k in ('patient', 'pair_id', 'scan', 'timepoint', 'lesion_id', 'reference_volume_ml')}
            result['predictions'] = measured
            predictions.append(result)
            stream.write(json.dumps(result, allow_nan=False)+'\n')
            stream.flush()
            if (number+1) % 10 == 0:
                print(json.dumps(dict(stage='inference', seed=args.seed, completed=number+1, total=len(records))), flush=True)
    validate_prediction_coverage(predictions, records)
    result = dict(status='EXTERNAL_SEED_COMPLETE', protocol_id=policy['protocol_id'], seed=args.seed,
                  patients=45, pairs=181, timepoints=len(records), models=list(MODELS),
                  policy_sha256=digest(args.policy), preprocessing_report_sha256=digest(cache/'preprocessing_report.json'),
                  index_sha256=report['index_sha256'], model_freeze_sha256=report['model_freeze_sha256'],
                  checkpoint_registry_sha256=digest(args.registry), model_state_sha256=fingerprints,
                  private_predictions_sha256=digest(output/'predictions_private.jsonl'),
                  elapsed_seconds=time.monotonic()-started,
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), internal_test_values_read=False)
    write_json(output/'seed_complete.json', result)
    print(json.dumps(result), flush=True)


def analyze(args, policy, project):
    cache, expected, preprocessing = load_cache(args, policy, project)
    frozen, freeze_sha = load_freeze(args, project)
    seed_records, receipts = [], []
    for seed in policy['seeds']:
        folder = project/f'runs/pantrack_010_seed_{seed}'
        receipt = json.loads((folder/'seed_complete.json').read_text())
        if (receipt.get('status') != 'EXTERNAL_SEED_COMPLETE' or receipt.get('seed') != seed
                or receipt.get('policy_sha256') != digest(args.policy)
                or receipt.get('checkpoint_registry_sha256') != digest(args.registry)
                or receipt.get('index_sha256') != preprocessing['index_sha256']
                or receipt.get('model_freeze_sha256') != freeze_sha
                or receipt.get('internal_test_values_read') is not False
                or receipt.get('private_predictions_sha256') != digest(folder/'predictions_private.jsonl')):
            raise GateError('External seed completion/lineage gate failed')
        seed_records.append([json.loads(line) for line in (folder/'predictions_private.jsonl').read_bytes().splitlines()])
        receipts.append(receipt)
    specification = policy['analysis']
    result = summarize_external(seed_records, expected, [r['maps'] for r in frozen['runs']],
                                specification['bootstrap_draws'], specification['bootstrap_seed'],
                                specification['primary_family_size'], specification['precision_target_halfwidth_ml'])
    for seed, row in zip(policy['seeds'], result['per_seed']):
        row['training_seed'] = seed
    result.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), policy_sha256=digest(args.policy),
                  checkpoint_registry_sha256=digest(args.registry), model_freeze_sha256=freeze_sha,
                  preprocessing_report_sha256=digest(cache/'preprocessing_report.json'), seed_receipts=receipts)
    output = checked_path(project, 'runs/pantrack_010_analysis')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'external_results.json', result)
    print(json.dumps(dict(status=result['status'], patients=result['patients'], pairs=result['pairs'],
                         results_sha256=digest(output/'external_results.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'prepare', 'evaluate', 'analyze'])
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    os.umask(0o077)
    policy = json.loads(args.policy.read_text())
    validate_confirmation_policy(policy)
    registry = read_bound_json(args.registry, policy['checkpoint_registry_sha256'])
    if (registry['protocol_id'] != policy['protocol_id']
            or [r['seed'] for r in registry['runs']] != policy['seeds']):
        raise GateError('External checkpoint registry has an unexpected scope')
    project = Path(os.environ[policy['project_root_env']]).resolve()
    dataset = Path(os.environ[policy['dataset_root_env']]).resolve()
    if not dataset.is_relative_to(project):
        raise GateError('Dataset root is outside the registered study storage')
    code = Path.cwd()
    if args.stage == 'freeze': freeze_models(args, policy, registry, project, code)
    elif args.stage == 'prepare': prepare(args, policy, project, dataset)
    elif args.stage == 'evaluate': evaluate(args, policy, registry, project, dataset, code)
    else: analyze(args, policy, project)


if __name__ == '__main__':
    main()
