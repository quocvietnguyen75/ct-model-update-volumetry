"""Post-hoc, patient-weighted accounting of frozen CT model-update errors."""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import fmean

from careport.ct_pantrack_confirm import MODELS, validate_prediction_coverage
from careport.ct_update_gate import GateError

SIZE_BINS = ('lt1', '1to10', 'ge10')
DIFFERENCES = ('mixed_minus_old_ml', 'mixed_minus_same_ml', 'same_minus_old_ml')


def size_bin(volume: float) -> str:
    if not math.isfinite(volume) or volume <= 0:
        raise GateError('Reference lesion volume must be finite and positive')
    return 'lt1' if volume < 1 else '1to10' if volume < 10 else 'ge10'


def pair_metrics(bl: dict, fu: dict, model: str) -> dict[str, float]:
    reference = fu['reference_volume_ml'] - bl['reference_volume_ml']
    b, f = bl['predictions'][model]['volume_ml'], fu['predictions'][model]['volume_ml']
    bo, fo = bl['predictions']['old']['volume_ml'], fu['predictions']['old']['volume_ml']
    old, mixed, same = fo - bo - reference, f - bo - reference, f - b - reference
    baseline_error, followup_error = b - bl['reference_volume_ml'], f - fu['reference_volume_ml']
    result = dict(old_old_mae_ml=abs(old), mixed_mae_ml=abs(mixed), same_mae_ml=abs(same),
                  old_old_signed_ml=old, mixed_signed_ml=mixed, same_signed_ml=same,
                  mixed_minus_old_ml=abs(mixed)-abs(old),
                  mixed_minus_same_ml=abs(mixed)-abs(same),
                  same_minus_old_ml=abs(same)-abs(old),
                  baseline_shift_ml=b-bo, followup_shift_ml=f-fo,
                  baseline_signed_error_ml=baseline_error, followup_signed_error_ml=followup_error,
                  baseline_absolute_error_ml=abs(baseline_error),
                  followup_absolute_error_ml=abs(followup_error),
                  baseline_dice=bl['predictions'][model]['dice'],
                  followup_dice=fu['predictions'][model]['dice'])
    residuals = (mixed-same-(b-bo), mixed-old-(f-fo),
                 result['mixed_minus_old_ml']-result['mixed_minus_same_ml']-result['same_minus_old_ml'])
    result['identity_max_residual_ml'] = max(abs(x) for x in residuals)
    if result['identity_max_residual_ml'] > 1e-9:
        raise GateError('Diagnostic error identity does not hold within tolerance')
    return result


def _mean_metrics(rows: list[dict]) -> dict[str, float]:
    return {key: fmean(r[key] for r in rows) for key in rows[0]}


def _quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    position = (len(values)-1) * probability
    lower = int(position)
    upper = min(lower+1, len(values)-1)
    return values[lower] + (position-lower) * (values[upper]-values[lower])


def _influence(values: list[float]) -> dict:
    count, total = len(values), math.fsum(values)
    absolute = sorted((abs(x) for x in values), reverse=True)
    mass = math.fsum(absolute)
    leave_one_out = [(total-x)/(count-1) for x in values] if count > 1 else []
    return dict(patients=count, mean_ml=total/count, median_ml=_quantile(values, .5),
                central_80_percent_empirical_range_ml=[_quantile(values, .1), _quantile(values, .9)],
                positive_patients=sum(x > 0 for x in values),
                negative_patients=sum(x < 0 for x in values), zero_patients=sum(x == 0 for x in values),
                top1_absolute_contribution_share=absolute[0]/mass if mass else None,
                top5_absolute_contribution_share=math.fsum(absolute[:5])/mass if mass else None,
                leave_one_out_mean_range_ml=[min(leave_one_out), max(leave_one_out)] if leave_one_out else None)


def _patient_means(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['patient'], row['model'])].append(row['metrics'])
    return [dict(patient=p, model=m, metrics=_mean_metrics(values))
            for (p, m), values in sorted(grouped.items())]


def _model_means(patient_rows: list[dict]) -> dict:
    return {m: _mean_metrics([r['metrics'] for r in patient_rows if r['model'] == m]) for m in MODELS}


def _batch(patient_rows: list[dict], strata: dict, images: list[dict]) -> dict:
    newer = [r['metrics'] for r in patient_rows if r['model'] == 'new']
    return dict(models=_model_means(patient_rows), size_strata=strata,
                new_model_influence={k: _influence([r[k] for r in newer]) for k in DIFFERENCES},
                same_image_models=_model_means(images))


def _unique_images(records: list[dict]) -> list[dict]:
    seen = {}
    for row in records:
        key = (row['patient'], row['scan'], row['lesion_id'])
        if key in seen:
            previous = seen[key]
            if (previous['reference_volume_ml'] != row['reference_volume_ml']
                    or previous['predictions'] != row['predictions']):
                raise GateError('Repeated image/lesion has inconsistent reference or predictions')
        seen[key] = row
    return list(seen.values())


def diagnose(seed_records: dict[int, list[dict]], expected: list[dict]) -> tuple[dict, dict]:
    """Return public aggregate results and separate private contribution records."""
    if not seed_records:
        raise GateError('No complete seeds supplied')
    patients = {r['patient'] for r in expected}
    pair_keys = {(r['patient'], r['pair_id']) for r in expected}
    seed_patients, seed_images, seed_strata, per_seed, private_pairs = [], [], [], [], []
    private_patients = []
    max_residual = 0.
    unique_count = None
    for seed, records in sorted(seed_records.items()):
        validate_prediction_coverage(records, expected)
        grouped = defaultdict(dict)
        for row in records:
            size_bin(row['reference_volume_ml'])
            grouped[(row['patient'], row['pair_id'])][row['timepoint']] = row
        pair_rows = []
        for (patient, pair_id), pair in sorted(grouped.items()):
            if set(pair) != {'bl', 'fu'}:
                raise GateError('Diagnostic pair lacks one timepoint')
            for model in MODELS:
                metrics = pair_metrics(pair['bl'], pair['fu'], model)
                max_residual = max(max_residual, metrics['identity_max_residual_ml'])
                pair_rows.append(dict(seed=seed, patient=patient, pair_id=pair_id, model=model,
                                      size_bin=size_bin(pair['bl']['reference_volume_ml']), metrics=metrics))
        patient_rows = _patient_means(pair_rows)
        private_pairs.extend(pair_rows)
        private_patients.extend(dict(seed=seed, **r) for r in patient_rows)
        seed_patients.extend(patient_rows)
        strata, stratum_rows = {}, {}
        for label in SIZE_BINS:
            selected = [r for r in pair_rows if r['size_bin'] == label]
            stratum_rows[label] = _patient_means(selected)
            strata[label] = dict(patients=len({r['patient'] for r in selected}),
                                 pairs=len({(r['patient'], r['pair_id']) for r in selected}),
                                 models=_model_means(stratum_rows[label]) if selected else None)
        seed_strata.append(stratum_rows)
        images = _unique_images(records)
        if unique_count is not None and len(images) != unique_count:
            raise GateError('Unique image count differs across seeds')
        unique_count = len(images)
        image_rows = []
        for row in images:
            for model in MODELS:
                volume = row['predictions'][model]['volume_ml']
                error = volume-row['reference_volume_ml']
                metrics = dict(signed_error_ml=error, absolute_error_ml=abs(error),
                               shift_from_old_ml=volume-row['predictions']['old']['volume_ml'],
                               dice=row['predictions'][model]['dice'])
                image_rows.append(dict(patient=row['patient'], model=model, metrics=metrics))
        image_means = _patient_means(image_rows)
        seed_images.extend(image_means)
        per_seed.append(dict(training_seed=seed, **_batch(patient_rows, strata, image_means)))
    averaged_patients = _patient_means(seed_patients)
    averaged_strata = {}
    for label in SIZE_BINS:
        rows = [r for batch in seed_strata for r in batch[label]]
        averaged_strata[label] = dict(per_seed[0]['size_strata'][label],
                                     models=_model_means(_patient_means(rows)) if rows else None)
    result = dict(status='POSTHOC_DIAGNOSTICS_COMPLETE', protocol_id='CT-UPDATE-013',
                  patients=len(patients), pairs=len(pair_keys), seeds=list(sorted(seed_records)),
                  models=list(MODELS), unique_image_lesions=unique_count,
                  maximum_identity_residual_ml=max_residual, per_seed=per_seed,
                  seed_averaged=_batch(averaged_patients, averaged_strata, _patient_means(seed_images)),
                  inferential_tests_performed=False, internal_test_values_read=False)
    return result, dict(patient_contributions=private_patients,
                        seed_averaged_patient_contributions=averaged_patients,
                        pair_contributions=private_pairs)


def verify_reproduction(result: dict, original: dict) -> dict:
    """Recheck original statistics with separate scalar arithmetic, no bootstrap."""
    original_seeds = {r['training_seed']: r for r in original['per_seed']}
    if set(original_seeds) != set(result['seeds']):
        raise GateError('Frozen result seed set differs')
    batches = [(result['seed_averaged'], original['seed_averaged'])]
    batches.extend((r, original_seeds[r['training_seed']]) for r in result['per_seed'])
    errors = []
    for current, frozen in batches:
        for model in MODELS:
            values = current['models'][model]
            strategies = [('old_old', 'same')] if model == 'old' else [
                ('old_'+model, 'mixed'), (model+'_'+model, 'same')]
            comparisons = [(values['followup_dice'], frozen['patient_mean_followup_dice'][model])]
            for name, prefix in strategies:
                comparisons.extend([(values[prefix+'_mae_ml'], frozen['strategies'][name]['mae_ml']),
                                    (values[prefix+'_signed_ml'], frozen['strategies'][name]['signed_error_ml'])])
            for actual, wanted in comparisons:
                if not math.isclose(actual, wanted, rel_tol=1e-9, abs_tol=1e-9):
                    raise GateError('Diagnostic calculation does not reproduce the original analysis')
                errors.append(abs(actual-wanted))
    return dict(values_checked=len(errors), max_absolute_difference=max(errors))
