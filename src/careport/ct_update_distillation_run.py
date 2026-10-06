"""Execute the fixed distillation batch after data, provenance and runtime gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from careport.ct_update_calibration_run import load_validation
from careport.ct_update_control_run import infer_controls, load_checkpoint, state_fingerprint
from careport.ct_update_controls import summarize_controls, validate_control_records
from careport.ct_update_distillation import (
    distillation_runtime_smoke, train_distilled, validate_distillation_policy,
)
from careport.ct_update_gate import GateError, validate_input_policy, validate_storage
from careport.ct_update_train import execution_config, make_model, runtime_smoke, validate_cache_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'distillation-policy', 'execution-policy', 'cache', 'training', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.config.read_text())
    policy_bytes = args.distillation_policy.read_bytes()
    policy = json.loads(policy_bytes)
    policy_sha = hashlib.sha256(policy_bytes).hexdigest()
    execution = json.loads(args.execution_policy.read_text())
    validate_input_policy(base)
    validate_distillation_policy(policy, base, execution)
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
    if old_receipt != source['checkpoint_receipts']['old']:
        raise GateError('Teacher differs from the evaluated old-model checkpoint')
    old_state_sha = state_fingerprint(old)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'distillation_policy.json').write_bytes(policy_bytes)
    (args.output / 'execution_config.json').write_text(json.dumps(config, indent=2) + '\n')
    started = time.monotonic()
    runtime_smoke(config, args.output)
    distillation_runtime_smoke(config, args.output)
    states, receipts = {}, {}
    for name, coefficient in policy['candidates'].items():
        states[name], receipts[name] = train_distilled(name, coefficient, policy, policy_sha, config,
                                                      old, old_receipt['checkpoint_sha256'],
                                                      args.cache, all_records, args.output)
        if receipts[name]['teacher_state_sha256'] != old_state_sha:
            raise GateError('Training teacher state differs from the source old model')
    models = {}
    for name, state in states.items():
        model = make_model(config)
        model.load_state_dict(state, strict=True)
        models[name] = model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    predictions = infer_controls(models, records, originals, args.cache,
                                 Path(os.environ[base['dataset_root_env']]), args.output)
    candidates = tuple(policy['candidates'])
    validate_control_records(predictions, all_records, candidates=candidates)
    result = summarize_controls(predictions, policy['bootstrap_draws'], policy['bootstrap_seed'],
                                candidates=candidates)
    for reference, original in [('native', source), ('csv_reference_sensitivity', source['csv_reference_sensitivity'])]:
        for name, metrics in original['strategies'].items():
            for metric in ('mae_ml', 'signed_error_ml'):
                if not math.isclose(result[reference]['strategies'][name][metric], metrics[metric],
                                    rel_tol=0, abs_tol=1e-9):
                    raise GateError('Original distillation comparators did not reproduce')
    result.update(analysis_id=policy['analysis_id'], amendment=policy['amendment'],
                  execution_id=execution['execution_id'], training_seed=config['seed'],
                  config_sha256=config_sha, distillation_policy_sha256=policy_sha,
                  source_results_sha256=hashlib.sha256((args.evaluation / 'development_results.json').read_bytes()).hexdigest(),
                  source_predictions_sha256=source['private_predictions_sha256'],
                  private_predictions_sha256=hashlib.sha256((args.output / 'predictions_private.jsonl').read_bytes()).hexdigest(),
                  source_old_checkpoint_receipt=old_receipt, candidate_checkpoint_receipts=receipts,
                  candidate_state_sha256={name: state_fingerprint(state) for name, state in states.items()},
                  original_metrics_reproduced=True, elapsed_seconds=time.monotonic() - started,
                  peak_evaluation_gpu_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.output / 'distillation_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
