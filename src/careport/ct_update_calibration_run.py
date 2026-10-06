"""Receipt-gated CPU analysis of completed CT validation predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from careport.ct_update_calibration import assign_folds, calibration_analysis
from careport.ct_update_gate import GateError, validate_input_policy, validate_storage


def validate_prediction_records(records: list[dict], index: list[dict]) -> list[dict]:
    from careport.ct_update_evaluate import paired_prediction_rows

    if any(r['split'] not in {'train', 'val'} for r in index):
        raise GateError('Unexpected/test partition in calibration cache')
    key = lambda r: (r['patient'], r['lesion_id'], r['timepoint'])
    validation = [r for r in index if r['split'] == 'val']
    expected = {key(r): r for r in validation}
    received = {key(r): r for r in records}
    if (not expected or len(expected) != len(validation) or len(received) != len(records)
            or set(expected) != set(received)):
        raise GateError('Predictions do not exactly match the validation cache')
    for identity, prediction in received.items():
        for field in ('reference_volume_ml', 'csv_volume_mm3'):
            if not math.isclose(prediction[field], expected[identity][field], rel_tol=1e-12, abs_tol=1e-12):
                raise GateError('Prediction references differ from verified cache')
        if set(prediction['predictions']) != {'old', 'new', 'continued'}:
            raise GateError('Unexpected calibration model set')
    native = paired_prediction_rows(records, False)
    csv = paired_prediction_rows(records, True)
    for row, sensitivity in zip(native, csv):
        row['csv_reference_bl'] = sensitivity['reference_bl']
        row['csv_reference_fu'] = sensitivity['reference_fu']
    return native


def validate_analysis_policy(policy: dict, config: dict) -> None:
    expected = {
        'analysis_id': 'CT-CALIBRATION-001', 'amendment': 'CT-UPDATE-004',
        'source_config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'folds': 3, 'fold_salt': 'ct-update-004:calibration:', 'fit_timepoint': 'bl',
        'fit_target': 'paired_model_predictions_only', 'weighting': 'patient_then_lesion_equal',
        'slope_constraint': 'nonnegative', 'degenerate_variance_relative_tolerance': 1e-12,
        'degenerate_slope': 1.0, 'mapped_volume_floor_ml': 0.0,
        'directions': [['old', 'new'], ['new', 'old'], ['old', 'continued'], ['continued', 'old']],
        'bootstrap_draws': 2000, 'bootstrap_seed': 20260920,
        'bootstrap': 'patient_within_fixed_fold_with_refitting', 'reference_sensitivity': 'csv',
        'test_values_permitted': False, 'confirmatory_inference': False,
    }
    if json.dumps(policy, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise GateError('Unsupported or stale calibration policy')


def load_validation(config: dict, policy: dict, cache: Path, evaluation: Path) -> tuple[list[dict], dict]:
    from careport.ct_update_train import execution_config, validate_cache_receipt

    index_bytes = (cache/'cache_index_private.jsonl').read_bytes()
    validate_cache_receipt(json.loads((cache/'preprocessing_report.json').read_text()), index_bytes, config)
    merged = execution_config(config, policy)
    config_sha = hashlib.sha256(json.dumps(merged, sort_keys=True).encode()).hexdigest()
    summary = json.loads((evaluation/'development_results.json').read_text())
    if (summary.get('status') != 'DEVELOPMENT_ONLY' or summary.get('test_evaluated') is not False
            or summary.get('confirmatory_inference') is not False or summary.get('config_sha256') != config_sha
            or summary.get('execution_id') != policy['execution_id']):
        raise GateError('Source evaluation is incomplete, unsafe or from another run')
    receipts = summary['checkpoint_receipts']
    for phase in ('old', 'new', 'continued'):
        receipt = receipts[phase]
        parent = None if phase == 'old' else receipts['old']['checkpoint_sha256']
        if (receipt['status'] != 'TRAINING_COMPLETE' or receipt['phase'] != phase
                or receipt['parent_sha256'] != parent or receipt['config_sha256'] != config_sha
                or receipt['execution_id'] != policy['execution_id'] or receipt['seed'] != policy['seed']
                or receipt['steps'] != policy['old_steps' if phase == 'old' else 'update_steps']):
            raise GateError('Source evaluation checkpoint lineage is inconsistent')
    predictions = (evaluation/'predictions_private.jsonl').read_bytes()
    if hashlib.sha256(predictions).hexdigest() != summary['private_predictions_sha256']:
        raise GateError('Source prediction hash differs from completed evaluation')
    records = [json.loads(line) for line in predictions.splitlines()]
    index = [json.loads(line) for line in index_bytes.splitlines()]
    rows = validate_prediction_records(records, index)
    if summary['patients'] != len({r['patient'] for r in rows}) or summary['lesions'] != len(rows):
        raise GateError('Source evaluation counts differ from validated predictions')
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'analysis-policy', 'execution-policy', 'cache', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    policy = json.loads(args.analysis_policy.read_text())
    execution = json.loads(args.execution_policy.read_text())
    validate_input_policy(config)
    validate_analysis_policy(policy, config)
    project = Path(os.environ[config['project_root_env']])
    usage = sum(p.stat().st_size for p in project.rglob('*') if p.is_file() and not p.is_symlink())
    validate_storage(config['storage'], usage, shutil.disk_usage(project).free)
    rows, source = load_validation(config, execution, args.cache, args.evaluation)
    folds = assign_folds(sorted({r['patient'] for r in rows}), policy['folds'], policy['fold_salt'])
    args.output.mkdir(parents=True, exist_ok=False)
    folds_bytes = (json.dumps(folds, sort_keys=True) + '\n').encode()
    (args.output/'folds_private.json').write_bytes(folds_bytes)
    started = time.monotonic()
    result = calibration_analysis(rows, folds, policy['bootstrap_draws'], policy['bootstrap_seed'])
    for reference, original in [('native', source), ('csv', source['csv_reference_sensitivity'])]:
        for strategy, values in original['strategies'].items():
            for metric in ('mae_ml', 'signed_error_ml'):
                if not math.isclose(result[reference]['strategies'][strategy][metric], values[metric], abs_tol=1e-9):
                    raise GateError('Uncorrected metrics do not reproduce the completed evaluation')
    result.update(analysis_id=policy['analysis_id'], amendment=policy['amendment'],
                  execution_id=execution['execution_id'], training_seed=execution['seed'],
                  config_sha256=source['config_sha256'],
                  analysis_policy_sha256=hashlib.sha256(args.analysis_policy.read_bytes()).hexdigest(),
                  source_results_sha256=hashlib.sha256((args.evaluation/'development_results.json').read_bytes()).hexdigest(),
                  source_predictions_sha256=source['private_predictions_sha256'],
                  folds_private_sha256=hashlib.sha256(folds_bytes).hexdigest(),
                  unchanged_followup_dice=source['patient_mean_followup_dice'],
                  original_metrics_reproduced=True, elapsed_seconds=time.monotonic() - started)
    (args.output/'calibration_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
