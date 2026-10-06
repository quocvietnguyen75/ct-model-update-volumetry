"""Fixed interpolation and patient-balanced control comparisons."""

from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
import torch

from careport.ct_update_calibration_run import validate_prediction_records
from careport.ct_update_gate import GateError


CANDIDATES = ('scratch', 'bcwi_025', 'bcwi_050', 'bcwi_075')
MODELS = ('old', 'new', 'continued', *CANDIDATES)


def candidate_models(candidates: tuple[str, ...]) -> tuple[str, ...]:
    if (not isinstance(candidates, tuple) or not candidates
            or any(not isinstance(name, str) or not name.isidentifier() for name in candidates)
            or len(set(candidates)) != len(candidates)
            or set(candidates) & {'old', 'new', 'continued'}):
        raise GateError('Invalid candidate model names')
    return ('old', 'new', 'continued', *candidates)


def interpolate_state(old: dict, new: dict, alpha: float) -> dict:
    if (not math.isfinite(alpha) or not 0 <= alpha <= 1 or not old
            or set(old) != set(new)):
        raise GateError('Invalid interpolation weights or model keys')
    result = {}
    for name, before in old.items():
        after = new[name]
        if (before.shape != after.shape or before.dtype != after.dtype
                or not torch.isfinite(before).all() or not torch.isfinite(after).all()):
            raise GateError('Incompatible or nonfinite model state')
        if not torch.is_floating_point(before):
            if not torch.equal(before, after):
                raise GateError('Nonfloating model state differs')
            result[name] = before.clone()
        elif alpha in (0, 1):
            result[name] = (before if alpha == 1 else after).clone()
        else:
            result[name] = before * alpha + after * (1 - alpha)
    return result


def validate_control_records(records: list[dict], index: list[dict], *,
                             candidates: tuple[str, ...] = CANDIDATES) -> None:
    models = candidate_models(candidates)
    if any(set(r['predictions']) != set(models) for r in records):
        raise GateError('Unexpected control model set')
    originals = [{**r, 'predictions': {m: r['predictions'][m] for m in ('old', 'new', 'continued')}}
                 for r in records]
    validate_prediction_records(originals, index)


def _pairs(records: list[dict], models: tuple[str, ...]) -> dict:
    groups = defaultdict(dict)
    for record in records:
        if set(record['predictions']) != set(models):
            raise GateError('Unexpected control model set')
        for model in models:
            value = record['predictions'][model]
            if (not math.isfinite(value['volume_ml']) or value['volume_ml'] < 0
                    or not math.isfinite(value['dice']) or not 0 <= value['dice'] <= 1
                    or (value['hd95_mm'] is not None and
                        (not math.isfinite(value['hd95_mm']) or value['hd95_mm'] < 0))):
                raise GateError('Invalid control metric')
            if value['empty_prediction']:
                if value['volume_ml'] != 0 or value['dice'] != 0 or value['hd95_mm'] is not None:
                    raise GateError('Inconsistent empty prediction metrics')
            elif value['volume_ml'] <= 0 or value['hd95_mm'] is None:
                raise GateError('Nonempty prediction has missing boundary or volume')
        for field in ('reference_volume_ml', 'csv_volume_mm3'):
            if not math.isfinite(record[field]) or record[field] <= 0:
                raise GateError('Invalid control reference volume')
        pair = groups[(record['patient'], record['lesion_id'])]
        if record['timepoint'] not in {'bl', 'fu'} or record['timepoint'] in pair:
            raise GateError('Duplicate or invalid control timepoint')
        pair[record['timepoint']] = record
    if not groups or any(set(pair) != {'bl', 'fu'} for pair in groups.values()):
        raise GateError('Incomplete control evaluation pairs')
    return groups


def summarize_controls(records: list[dict], bootstrap_draws: int, seed: int, *,
                       candidates: tuple[str, ...] = CANDIDATES) -> dict:
    models = candidate_models(candidates)
    groups = _pairs(records, models)
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise GateError('Invalid control bootstrap count')
    patients = sorted({key[0] for key in groups})
    by_patient = {p: [pair for key, pair in groups.items() if key[0] == p] for p in patients}
    samples = np.random.default_rng(seed).integers(0, len(patients), (bootstrap_draws, len(patients)))

    def estimate(values: np.ndarray, unit: str) -> dict:
        low, high = np.quantile(values[samples].mean(axis=1), [.025, .975])
        return {'mean': float(values.mean()), 'unit': unit,
                'exploratory_95_percentile_interval': [float(low), float(high)]}

    dice = {model: np.array([np.mean([pair['fu']['predictions'][model]['dice']
                                    for pair in by_patient[p]]) for p in patients]) for model in models}
    strategies = {'old_old': ('old', 'old'), 'old_new': ('old', 'new'), 'new_new': ('new', 'new'),
                  'old_continued': ('old', 'continued'), 'continued_continued': ('continued', 'continued')}
    for candidate in candidates:
        strategies['old_' + candidate] = ('old', candidate)
        strategies[candidate + '_' + candidate] = (candidate, candidate)
    result = {}
    for reference, field, divisor in [('native', 'reference_volume_ml', 1),
                                       ('csv_reference_sensitivity', 'csv_volume_mm3', 1000)]:
        absolute, signed = {}, {}
        for name, (baseline, followup) in strategies.items():
            errors = [[pair['fu']['predictions'][followup]['volume_ml']
                       - pair['bl']['predictions'][baseline]['volume_ml']
                       - (pair['fu'][field] - pair['bl'][field]) / divisor
                       for pair in by_patient[p]] for p in patients]
            absolute[name] = np.array([np.mean(np.abs(e)) for e in errors])
            signed[name] = np.array([np.mean(e) for e in errors])
        candidate_results = {}
        for candidate in candidates:
            mixed, refreshed = absolute['old_' + candidate], absolute[candidate + '_' + candidate]
            candidate_results[candidate] = {
                'mixed_benefit_vs_new': estimate(absolute['old_new'] - mixed, 'mL'),
                'retaining_baseline_cost': estimate(mixed - refreshed, 'mL'),
                'refresh_benefit_vs_new': estimate(absolute['new_new'] - refreshed, 'mL'),
                'followup_dice_difference_vs_new': estimate(dice[candidate] - dice['new'], 'Dice'),
            }
        result[reference] = {'strategies': {name: {'mae_ml': float(absolute[name].mean()),
                                                  'signed_error_ml': float(signed[name].mean())}
                                            for name in strategies}, 'candidates': candidate_results}
    boundaries = {}
    for model in models:
        distances = defaultdict(list)
        empty = 0
        for pair in groups.values():
            metrics = pair['fu']['predictions'][model]
            empty += metrics['empty_prediction']
            if metrics['hd95_mm'] is not None:
                distances[pair['fu']['patient']].append(metrics['hd95_mm'])
        boundaries[model] = {
            'patient_mean_hd95_mm_nonempty_only': float(np.mean([np.mean(v) for v in distances.values()])) if distances else None,
            'empty_followup_predictions': int(empty), 'patients_with_defined_hd95': len(distances)}
    result.update(status='DEVELOPMENT_ONLY', patients=len(patients), lesions=len(groups),
                  patient_mean_followup_dice={m: float(v.mean()) for m, v in dice.items()},
                  boundary_diagnostics=boundaries, bootstrap_draws=bootstrap_draws, bootstrap_seed=seed,
                  test_evaluated=False, confirmatory_inference=False)
    return result
