"""Train the matched-budget control and evaluate a fixed BCWI grid."""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from careport.ct_update_calibration_run import load_validation
from careport.ct_update_controls import interpolate_state, summarize_controls, validate_control_records
from careport.ct_update_data import volume_paths
from careport.ct_update_evaluate import native_metrics
from careport.ct_update_gate import GateError, validate_input_policy, validate_storage
from careport.ct_update_train import (
    execution_config, make_model, runtime_smoke, seed_runtime, train_phase,
    validate_cache_receipt,
)


def validate_control_policy(policy: dict, base: dict, execution: dict) -> None:
    expected = {
        'analysis_id': 'CT-CONTROLS-001', 'amendment': 'CT-UPDATE-005',
        'source_config_sha256': hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
        'execution_ids': ['CT-PILOT-002', 'CT-PILOT-003', 'CT-PILOT-004'],
        'training_seeds': [20260915, 20260917, 20260919], 'scratch_stage_steps': [20000, 10000],
        'scratch_stage_data': 'all_training_patients_both_stages', 'optimizer_restart_at_step': 20000,
        'scratch_checkpoint_selection': 'final', 'bcwi_old_weight_alphas': [.25, .5, .75],
        'bcwi_source': 'https://aclanthology.org/2024.eacl-long.174.pdf',
        'bootstrap_draws': 2000, 'bootstrap_seed': 20260916,
        'test_values_permitted': False, 'confirmatory_inference': False,
    }
    if json.dumps(policy, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise GateError('Unsupported or stale control policy')
    merged = execution_config(base, execution)
    pairs = dict(zip(expected['execution_ids'], expected['training_seeds']))
    if (pairs.get(execution['execution_id']) != merged['seed']
            or [merged['training'][k] for k in ('old_steps', 'update_steps')] != [20000, 10000]):
        raise GateError('Control execution seed or budget differs from the fixed batch')


def load_checkpoint(directory: Path, phase: str, config_sha: str, parent_sha: str | None,
                    seed: int, steps: int) -> tuple[dict, dict]:
    path = directory / (phase + '.pt')
    receipt = json.loads((directory / (phase + '_receipt.json')).read_text())
    if (receipt.get('status') != 'TRAINING_COMPLETE'
            or hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get('checkpoint_sha256')):
        raise GateError('Control source checkpoint differs from its completed receipt')
    stored = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(phase=phase, config_sha256=config_sha, parent_sha256=parent_sha, seed=seed, steps=steps)
    if any(stored.get(k) != v or receipt.get(k) != v for k, v in expected.items()):
        raise GateError('Control source checkpoint lineage/configuration is inconsistent')
    return stored['state_dict'], receipt


def state_fingerprint(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def infer_controls(models: dict, records: list[dict], originals: list[dict], cache: Path,
                   root: Path, output: Path) -> list[dict]:
    import nibabel as nib

    old_records = {(r['patient'], r['lesion_id'], r['timepoint']): r for r in originals}

    @lru_cache(maxsize=1)
    def mask_for(patient, timepoint, index):
        _, path = volume_paths(root, patient, timepoint, index)
        volume = nib.load(path)
        return np.asanyarray(volume.dataobj), volume.affine

    results = []
    with (output / 'predictions_private.jsonl').open('x', encoding='utf-8') as stream:
        for i, record in enumerate(records):
            path = cache / record['patch_file']
            if not path.resolve().is_relative_to(cache.resolve()):
                raise GateError('Control patch path escapes the cache')
            with np.load(path, allow_pickle=False) as payload:
                image = torch.from_numpy(payload['image'].astype(np.float32)[None, None]).cuda()
            labels, geometry = mask_for(record['patient'], record['timepoint'], record['image_index'])
            if not np.allclose(geometry, record['native_affine'], rtol=0, atol=1e-4):
                raise GateError('Control reference geometry differs from audited cache')
            original = old_records[(record['patient'], record['lesion_id'], record['timepoint'])]
            result = {**original, 'predictions': dict(original['predictions'])}
            for name, model in models.items():
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(image)[0, 0].float().cpu().numpy()
                result['predictions'][name] = native_metrics(logits, record['patch_affine'], geometry,
                                                            labels, record['lesion_id'], record['reference_voxels'])
            results.append(result)
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            if (i + 1) % 20 == 0:
                print(json.dumps({'control_timepoints_evaluated': i + 1, 'total': len(records)}), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'control-policy', 'execution-policy', 'cache', 'training', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.config.read_text())
    policy = json.loads(args.control_policy.read_text())
    execution = json.loads(args.execution_policy.read_text())
    validate_input_policy(base)
    validate_control_policy(policy, base, execution)
    project = Path(os.environ[base['project_root_env']])
    usage = sum(p.stat().st_size for p in project.rglob('*') if p.is_file() and not p.is_symlink())
    validate_storage(base['storage'], usage, shutil.disk_usage(project).free)
    _, source = load_validation(base, execution, args.cache, args.evaluation)
    index_bytes = (args.cache / 'cache_index_private.jsonl').read_bytes()
    validate_cache_receipt(json.loads((args.cache / 'preprocessing_report.json').read_text()), index_bytes, base)
    all_records = [json.loads(line) for line in index_bytes.splitlines()]
    records = sorted((r for r in all_records if r['split'] == 'val'),
                     key=lambda r: (r['patient'], r['timepoint'], r['image_index'], r['lesion_id']))
    source_bytes = (args.evaluation / 'predictions_private.jsonl').read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != source['private_predictions_sha256']:
        raise GateError('Source predictions changed after validation gate')
    originals = [json.loads(line) for line in source_bytes.splitlines()]
    config = execution_config(base, execution)
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    old, old_receipt = load_checkpoint(args.training, 'old', config_sha, None, config['seed'], 20000)
    new, new_receipt = load_checkpoint(args.training, 'new', config_sha,
                                      old_receipt['checkpoint_sha256'], config['seed'], 10000)
    for name, receipt in [('old', old_receipt), ('new', new_receipt)]:
        if receipt != source['checkpoint_receipts'][name]:
            raise GateError('Checkpoint receipt differs from original evaluated source')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'control_policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    (args.output / 'execution_config.json').write_text(json.dumps(config, indent=2) + '\n')
    started = time.monotonic()
    runtime_smoke(config, args.output)
    seed_runtime(config['seed'])
    initialization = state_fingerprint(make_model(config).state_dict())
    scratch_base = train_phase('scratch_base', config, args.cache, all_records, args.output, None)
    train_phase('scratch', config, args.cache, all_records, args.output, scratch_base)
    scratch_base_hash = hashlib.sha256(scratch_base.read_bytes()).hexdigest()
    _, scratch_base_receipt = load_checkpoint(args.output, 'scratch_base', config_sha, None, config['seed'], 20000)
    scratch, scratch_receipt = load_checkpoint(args.output, 'scratch', config_sha, scratch_base_hash,
                                              config['seed'], 10000)
    states = {'scratch': scratch}
    for alpha in policy['bcwi_old_weight_alphas']:
        states[f'bcwi_{int(alpha * 100):03d}'] = interpolate_state(old, new, alpha)
    models = {}
    for name, state in states.items():
        model = make_model(config)
        model.load_state_dict(state, strict=True)
        models[name] = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    predictions = infer_controls(models, records, originals, args.cache,
                                 Path(os.environ[base['dataset_root_env']]), args.output)
    validate_control_records(predictions, all_records)
    result = summarize_controls(predictions, policy['bootstrap_draws'], policy['bootstrap_seed'])
    for reference, original in [('native', source), ('csv_reference_sensitivity', source['csv_reference_sensitivity'])]:
        for name, metrics in original['strategies'].items():
            for metric in ('mae_ml', 'signed_error_ml'):
                if not math.isclose(result[reference]['strategies'][name][metric], metrics[metric], abs_tol=1e-9):
                    raise GateError('Original control comparators did not reproduce')
    result.update(analysis_id=policy['analysis_id'], amendment=policy['amendment'],
                  execution_id=execution['execution_id'], training_seed=config['seed'], config_sha256=config_sha,
                  control_policy_sha256=hashlib.sha256(args.control_policy.read_bytes()).hexdigest(),
                  source_results_sha256=hashlib.sha256((args.evaluation / 'development_results.json').read_bytes()).hexdigest(),
                  source_predictions_sha256=source['private_predictions_sha256'],
                  private_predictions_sha256=hashlib.sha256((args.output / 'predictions_private.jsonl').read_bytes()).hexdigest(),
                  initialization_state_sha256=initialization,
                  candidate_state_sha256={name: state_fingerprint(state) for name, state in states.items()},
                  scratch_checkpoint_receipts={'scratch_base': scratch_base_receipt, 'scratch': scratch_receipt},
                  original_metrics_reproduced=True, elapsed_seconds=time.monotonic() - started,
                  peak_evaluation_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output / 'control_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
