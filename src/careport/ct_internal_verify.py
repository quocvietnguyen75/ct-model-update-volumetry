"""Separate scalar arithmetic verification of the completed CT-UPDATE-014 batch.

This does not call the production summary or its patient-array implementation.
Input gates reuse the frozen execution code; no inference or fitting is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

from careport.ct_pantrack_confirm import MODELS, PRIMARY, validate_prediction_coverage


def recompute_patient_values(records: list[dict], maps: dict) -> tuple[list[str], dict, dict]:
    """Compute pair errors first, then equal patient means using scalar arithmetic."""
    pairs = defaultdict(dict)
    for row in records:
        key, tp = (row['patient'], row['pair_id']), row['timepoint']
        if tp not in ('bl', 'fu') or tp in pairs[key]:
            raise ValueError('Duplicate or invalid paired role')
        pairs[key][tp] = row
    if not pairs or any(set(pair) != {'bl', 'fu'} for pair in pairs.values()):
        raise ValueError('Incomplete paired predictions')
    collected = defaultdict(lambda: defaultdict(list))
    hd = {model: defaultdict(list) for model in MODELS}
    empty = dict.fromkeys(MODELS, 0)
    for (patient, _), pair in pairs.items():
        bl, fu = pair['bl'], pair['fu']
        target = fu['reference_volume_ml'] - bl['reference_volume_ml']
        b = {m: bl['predictions'][m]['volume_ml'] for m in MODELS}
        f = {m: fu['predictions'][m]['volume_ml'] for m in MODELS}
        changes = {'old_old': f['old'] - b['old']}
        for model in MODELS[1:]:
            changes['old_' + model] = f[model] - b['old']
            changes[model + '_' + model] = f[model] - b[model]
        for model in ('new', 'continued'):
            forward, reverse = maps['old_to_' + model], maps[model + '_to_old']
            changes['mapped_old_' + model] = f[model] - max(0, forward['slope'] * b['old'] + forward['intercept_ml'])
            changes['old_mapped_' + model] = max(0, reverse['slope'] * f[model] + reverse['intercept_ml']) - b['old']
        for strategy, change in changes.items():
            collected[patient]['mae:' + strategy].append(abs(change - target))
            collected[patient]['signed:' + strategy].append(change - target)
        for model in MODELS:
            measured = fu['predictions'][model]
            collected[patient]['dice:' + model].append(measured['dice'])
            empty[model] += int(measured['empty_prediction'])
            if measured['hd95_mm'] is not None:
                hd[model][patient].append(measured['hd95_mm'])
    patients = sorted(collected)
    values = {metric: [statistics.fmean(collected[p][metric]) for p in patients]
              for metric in collected[patients[0]]}
    boundary = {m: dict(empty_followup_predictions=empty[m],
                        patients_with_defined_hd95=len(hd[m]),
                        patient_mean_hd95_mm_nonempty_only=statistics.fmean(
                            statistics.fmean(v) for v in hd[m].values()) if hd[m] else None)
                for m in MODELS}
    return patients, values, boundary


def _contrasts(values: dict) -> dict:
    differences = {
        'ordinary_update_benefit': ('mae:old_old', 'mae:old_new'),
        'retaining_baseline_cost': ('mae:old_new', 'mae:new_new'),
        'contour_benefit': ('dice:new', 'dice:old'),
    }
    return {name: np.subtract(values[a], values[b]) for name, (a, b) in differences.items()}


def verify_result(result: dict, seeds: list[list[dict]], expected: list[dict],
                  maps: list[dict], policy: dict) -> dict:
    """Reproduce frozen point estimates, primary intervals and reference sensitivity."""
    if (len(seeds) != 3 or len(maps) != 3 or result['seeds'] != 3
            or result['status'] != 'INTERNAL_EVALUATION_COMPLETE'
            or result['patients'] != len({r['patient'] for r in expected})
            or result['pairs'] * 2 != len(expected)
            or [r['training_seed'] for r in result['per_seed']] != policy['seeds']):
        raise ValueError('Incomplete result or cohort')
    for key in ('bootstrap_draws', 'bootstrap_seed', 'primary_family_size'):
        if result[key] != policy['analysis'][key]:
            raise ValueError('Analysis configuration changed')
    differences = []

    def close(actual, target):
        if target is None:
            if actual is not None:
                raise ValueError('Undefined boundary metric was replaced')
            return
        delta = abs(float(actual) - float(target))
        if not math.isfinite(delta) or delta > 1e-9:
            raise ValueError('Reported aggregate differs from scalar recalculation')
        differences.append(delta)

    count = result['patients']
    draws = np.random.default_rng(policy['analysis']['bootstrap_seed']).integers(
        0, count, size=(policy['analysis']['bootstrap_draws'], count))

    def points(summary, values, csv_only=False, adjusted=False):
        names = {k[4:] for k in values if k.startswith('mae:')}
        if set(summary['strategies']) != names or len(names) != 21:
            raise ValueError('Strategy coverage changed')
        for name in names:
            close(summary['strategies'][name]['mae_ml'], statistics.fmean(values['mae:' + name]))
            close(summary['strategies'][name]['signed_error_ml'], statistics.fmean(values['signed:' + name]))
        contrasts = _contrasts(values)
        for name, values_array in contrasts.items():
            if csv_only:
                close(summary['primary_contrast_points'][name], statistics.fmean(values_array))
                continue
            reported = summary['primary_contrasts'][name]
            close(reported['mean'], statistics.fmean(values_array))
            sampled = values_array[draws].mean(axis=1)
            for actual, target in zip(reported['percentile_95_interval'], np.quantile(sampled, [.025, .975])):
                close(actual, target)
            if adjusted:
                tail = .05 / (2 * policy['analysis']['primary_family_size'])
                for actual, target in zip(reported['bonferroni_family_interval'], np.quantile(sampled, [tail, 1-tail])):
                    close(actual, target)
        if not csv_only:
            if set(summary['patient_mean_followup_dice']) != set(MODELS):
                raise ValueError('Contour model coverage changed')
            for model in MODELS:
                close(summary['patient_mean_followup_dice'][model], statistics.fmean(values['dice:' + model]))

    native_arrays, csv_arrays, patient_order = [], [], None
    csv_reference = {(r['pair_id'], r['timepoint']): r['csv_volume_mm3'] / 1000 for r in expected}
    sensitivity = result['csv_reference_sensitivity']
    for slot, (rows, coefficients) in enumerate(zip(seeds, maps)):
        validate_prediction_coverage(rows, expected)
        patients, values, boundary = recompute_patient_values(rows, coefficients)
        if patient_order is not None and patients != patient_order:
            raise ValueError('Seed patient membership differs')
        patient_order = patients
        native_arrays.append(values)
        summary = result['per_seed'][slot]
        points(summary, values)
        for model, fields in boundary.items():
            for name, value in fields.items():
                close(summary['boundary_diagnostics'][model][name], value)
        changed = [dict(row, reference_volume_ml=csv_reference[(row['pair_id'], row['timepoint'])]) for row in rows]
        _, csv_values, _ = recompute_patient_values(changed, coefficients)
        csv_arrays.append(csv_values)
        csv_summary = sensitivity['per_seed'][slot]
        if csv_summary['training_seed'] != policy['seeds'][slot]:
            raise ValueError('CSV sensitivity seed order changed')
        points(csv_summary, csv_values, csv_only=True)
        for name in PRIMARY:
            close(csv_summary['primary_difference_from_native'][name],
                  statistics.fmean(_contrasts(csv_values)[name]) - statistics.fmean(_contrasts(values)[name]))

    def average(arrays):
        return {metric: [statistics.fmean(a[metric][i] for a in arrays) for i in range(count)]
                for metric in arrays[0]}

    native_average, csv_average = average(native_arrays), average(csv_arrays)
    points(result['seed_averaged'], native_average, adjusted=True)
    points(sensitivity['seed_averaged'], csv_average, csv_only=True)
    for name in PRIMARY:
        close(sensitivity['seed_averaged']['primary_difference_from_native'][name],
              statistics.fmean(_contrasts(csv_average)[name]) - statistics.fmean(_contrasts(native_average)[name]))
    maximum = max(abs(statistics.fmean(c[k]) - statistics.fmean(n[k]))
                  for n, c in zip(native_arrays, csv_arrays) for k in n if k.startswith('mae:'))
    close(sensitivity['maximum_absolute_per_seed_strategy_mae_difference_ml'], maximum)
    return dict(numeric_values_checked=len(differences), max_absolute_difference=max(differences),
                complete_prediction_membership_verified=True, private_reference_identity_verified=True,
                strategy_count=21, native_and_csv_point_estimates_reproduced=True,
                primary_intervals_reproduced=True, boundary_diagnostics_reproduced=True,
                descriptive_secondary_intervals_recomputed=False)


def main() -> None:
    # These are gate helpers only; no production summary implementation is called.
    from careport.ct_internal_holdout import validate_complete_seeds
    from careport.ct_internal_run import evaluation_inputs, load_context
    from careport.ct_pantrack_run import checked_path, digest, read_bound_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    ctx = load_context(args)
    for line in (ctx.code / 'EXECUTION_SHA256SUMS').read_text().splitlines():
        wanted, relative = line.split(maxsplit=1)
        if digest(checked_path(ctx.code, relative)) != wanted:
            raise ValueError('Frozen execution source changed')
    freeze, _, expected = evaluation_inputs(ctx)
    receipts, folders = [], []
    for seed in ctx.policy['seeds']:
        folder = ctx.project / f'runs/internal_holdout_014_seed_{seed}'
        folders.append(folder)
        receipts.append(json.loads((folder / 'seed_complete.json').read_bytes()))
    validate_complete_seeds(receipts, ctx.policy, freeze, ctx.states)
    for folder, receipt in zip(folders, receipts):
        if digest(folder / 'predictions_private.jsonl') != receipt['private_predictions_sha256']:
            raise ValueError('Completed prediction payload changed')
    for run in ctx.registry['runs']:
        for artifact in run['checkpoints'].values():
            if digest(checked_path(ctx.project, artifact['path'])) != artifact['receipt']['checkpoint_sha256']:
                raise ValueError('Registered checkpoint file changed')
    result_path = ctx.project / 'runs/internal_holdout_014_analysis/internal_results.json'
    result_sha = digest(result_path)
    result = read_bound_json(result_path, result_sha)
    for key, value in dict(policy_sha256=ctx.policy_sha, source_manifest_sha256=ctx.source_sha,
                           cohort_freeze_sha256=digest(ctx.project / ctx.policy['freeze_relative']),
                           model_freeze_sha256=ctx.policy['model_freeze_sha256'],
                           checkpoint_registry_sha256=ctx.policy['checkpoint_registry_sha256'],
                           seed_receipt_sha256={str(r['seed']): digest(f / 'seed_complete.json')
                                               for r, f in zip(receipts, folders)}).items():
        if result.get(key) != value:
            raise ValueError('Result lineage differs from the complete frozen batch')
    rows = [[json.loads(line) for line in (f / 'predictions_private.jsonl').read_bytes().splitlines()] for f in folders]
    verified = verify_result(result, rows, expected, [r['maps'] for r in ctx.maps['runs']], ctx.policy)
    if digest(result_path) != result_sha:
        raise ValueError('Analysis result changed during verification')
    report = dict(status='INTERNAL_ARITHMETIC_REPRODUCED', protocol_id='CT-UPDATE-014',
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), result_sha256=result_sha,
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  policy_sha256=ctx.policy_sha, source_manifest_sha256=ctx.source_sha,
                  cohort_freeze_sha256=digest(ctx.project / ctx.policy['freeze_relative']),
                  patients=result['patients'], pairs=result['pairs'], seeds=3, **verified,
                  inference_rerun=False, independent_agent=False, human_reading='not_attested',
                  scope='Separate scalar point arithmetic for all strategies, Dice, HD95 and CSV sensitivity; native primary bootstrap intervals',
                  original_result_preserved=True)
    destination = ctx.project / 'control/internal_holdout_014_arithmetic_verification_2026-09-25.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
