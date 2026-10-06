"""Frozen PanTrack reference checks and patient-clustered external analysis."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import numpy as np
from scipy import ndimage

from careport.ct_pantrack_intake import _point, _scan_owner, IntakeError
from careport.ct_update_data import patch_affine
from careport.ct_update_gate import GateError

MODELS = ('old', 'new', 'continued', 'scratch', 'bcwi_025', 'bcwi_050',
          'bcwi_075', 'kd_001', 'kd_010')
PRIMARY = ('ordinary_update_benefit', 'retaining_baseline_cost', 'contour_benefit')
MAPS = ('old_to_new', 'new_to_old', 'old_to_continued', 'continued_to_old')


def validate_confirmation_policy(policy: dict) -> None:
    fixed = dict(protocol_id='CT-UPDATE-010', cohort='PanTrack',
                 role='external_evaluation_only_no_tuning', models=list(MODELS),
                 seeds=[20260915, 20260917, 20260919], expected_patients=45, expected_pairs=181,
                 model_input_roles=['ct_image', 'reference_centroid'],
                 training_permitted=False, external_fitting_permitted=False,
                 internal_test_values_permitted=False, inspect_native_references_after_intake=True,
                 inspect_predictions_after_reference_gate=True, centroid_tolerance_voxels=1.05)
    if any(type(policy.get(k)) is not type(v) or policy[k] != v for k, v in fixed.items()):
        raise GateError('Unapproved external confirmation policy')
    analysis = policy['analysis']
    if (analysis != dict(bootstrap_draws=10000, bootstrap_seed=20260924, primary_family_size=6,
                         precision_target_halfwidth_ml=5.0, primary_contrasts=list(PRIMARY),
                         weighting='pair_losses_within_patient_then_seed_average_then_equal_patients',
                         clinical_margin_ml=None, csv_reference_sensitivity=False)
            or policy['patch'] != dict(shape=[96, 96, 96], spacing_mm=[1.5]*3,
                                       hu_clip=[-1000, 1000], image_dtype='float16', mask_dtype='uint8')
            or policy['storage'] != dict(study_cap_bytes=268435456000, project_cap_bytes=966367641600,
                                        planned_additional_bytes=5368709120, cache_cap_bytes=2147483648)):
        raise GateError('External analysis, preprocessing or storage policy changed')


def candidate_records(pairs: list[dict]) -> list[dict]:
    if not pairs:
        raise GateError('Empty external cohort')
    result, seen = [], set()
    for pair in pairs:
        try:
            patient, bl, fu, label = (pair[k] for k in ('patient', 'baseline', 'followup', 'label'))
            owner_bl, date_bl = _scan_owner(bl)
            owner_fu, date_fu = _scan_owner(fu)
            key = (patient, bl, fu, label)
            if (type(label) is not int or label <= 0 or owner_bl != patient or owner_fu != patient
                    or date_bl >= date_fu or key in seen):
                raise GateError('Invalid or duplicate external lesion/visit pair')
            seen.add(key)
            pair_id = hashlib.sha256(json.dumps(key, separators=(',', ':')).encode()).hexdigest()
            for timepoint, scan, name in [('bl', bl, 'baseline_point'), ('fu', fu, 'followup_point')]:
                center = _point(pair[name])
                if center is None:
                    raise GateError('Eligible external pair lacks a finite prompt')
                result.append(dict(patient=patient, pair_id=pair_id, scan=scan, lesion_id=label,
                                   timepoint=timepoint, centroid=list(center)))
        except (KeyError, TypeError, IntakeError) as error:
            raise GateError('Malformed external pair manifest') from error
    return sorted(result, key=lambda r: (r['scan'], r['lesion_id'], r['pair_id'], r['timepoint']))


def inspect_instances(labels, affine, entries: list[dict], tolerance: float) -> list[dict]:
    if (labels.ndim != 3 or labels.dtype.kind not in 'iu' or labels.size == 0
            or labels.min() < 0 or labels.max() > 100000):
        raise GateError('Reference mask is not bounded nonnegative integer instances')
    patch_affine(affine, (0, 0, 0), (3, 3, 3), (1, 1, 1))
    counts = np.bincount(labels.ravel(order='K'))
    boxes = ndimage.find_objects(labels)
    voxel_ml = abs(float(np.linalg.det(np.asarray(affine)[:3, :3]))) / 1000
    records = []
    for entry in entries:
        identity = entry['lesion_id']
        if identity <= 0 or identity >= len(counts) or not counts[identity]:
            raise GateError('External candidate instance is absent from its native mask')
        box = boxes[identity - 1]
        coordinates = np.nonzero(labels[box] == identity)
        actual = np.array([float(c.mean()) + s.start for c, s in zip(coordinates, box)])
        center = np.asarray(entry['centroid'], dtype=float)
        if (center.shape != (3,) or not np.isfinite(center).all()
                or np.any(center < 0) or np.any(center > np.asarray(labels.shape)-1)
                or np.max(np.abs(actual-center)) > tolerance):
            raise GateError('External prompt does not match its native XYZ instance centroid')
        records.append({**entry, 'reference_voxels': int(counts[identity]),
                        'reference_volume_ml': int(counts[identity]) * voxel_ml,
                        'native_shape': list(labels.shape), 'native_affine': np.asarray(affine).tolist(),
                        'native_lesion_box': [[s.start, s.stop] for s in box]})
    return records


def validate_prediction_coverage(records: list[dict], expected: list[dict]) -> None:
    key = lambda r: (r['pair_id'], r['timepoint'])
    wanted, got = {key(r): r for r in expected}, {key(r): r for r in records}
    if (not wanted or len(wanted) != len(expected) or len(got) != len(records) or set(wanted) != set(got)):
        raise GateError('External predictions do not exactly cover the frozen reference manifest')
    for identity, record in got.items():
        source = wanted[identity]
        if (any(record[k] != source[k] for k in ('patient', 'scan', 'lesion_id', 'timepoint'))
                or not math.isclose(record['reference_volume_ml'], source['reference_volume_ml'],
                                    rel_tol=1e-12, abs_tol=1e-12)
                or set(record['predictions']) != set(MODELS)):
            raise GateError('External prediction identity, reference or model set differs')
        for measured in record['predictions'].values():
            volume, dice, hd = (measured[k] for k in ('volume_ml', 'dice', 'hd95_mm'))
            if (type(volume) not in (int, float) or not math.isfinite(volume) or volume < 0
                    or type(dice) not in (int, float) or not math.isfinite(dice) or not 0 <= dice <= 1
                    or measured['empty_prediction'] is not (volume == 0)
                    or (volume == 0 and (dice != 0 or hd is not None))
                    or (volume > 0 and (hd is None or not math.isfinite(hd) or hd < 0))):
                raise GateError('Invalid external prediction metric or empty-mask convention')


def _patient_arrays(records: list[dict], coefficients: dict) -> tuple[list[str], dict, dict]:
    if set(coefficients) != set(MAPS):
        raise GateError('Incomplete frozen affine correction set')
    for fitted in coefficients.values():
        if (not math.isfinite(fitted['slope']) or fitted['slope'] < 0
                or not math.isfinite(fitted['intercept_ml'])):
            raise GateError('Invalid frozen affine coefficients')
    grouped = defaultdict(dict)
    for record in records:
        grouped[(record['patient'], record['pair_id'])][record['timepoint']] = record
    values = defaultdict(lambda: defaultdict(list))
    boundary = {m: defaultdict(list) for m in MODELS}
    empty = {m: 0 for m in MODELS}
    for (patient, _), pair in grouped.items():
        if set(pair) != {'bl', 'fu'}:
            raise GateError('External pair lacks one timepoint')
        bl, fu = pair['bl'], pair['fu']
        reference = fu['reference_volume_ml'] - bl['reference_volume_ml']
        b = {m: bl['predictions'][m]['volume_ml'] for m in MODELS}
        f = {m: fu['predictions'][m]['volume_ml'] for m in MODELS}
        changes = {'old_old': f['old']-b['old']}
        for model in MODELS[1:]:
            changes['old_'+model] = f[model]-b['old']
            changes[model+'_'+model] = f[model]-b[model]
        for model in ('new', 'continued'):
            forward, reverse = coefficients['old_to_'+model], coefficients[model+'_to_old']
            mapped_bl = max(0., forward['slope']*b['old'] + forward['intercept_ml'])
            mapped_fu = max(0., reverse['slope']*f[model] + reverse['intercept_ml'])
            changes['mapped_old_'+model] = f[model]-mapped_bl
            changes['old_mapped_'+model] = mapped_fu-b['old']
        for strategy, change in changes.items():
            values[patient]['mae:'+strategy].append(abs(change-reference))
            values[patient]['signed:'+strategy].append(change-reference)
        for model in MODELS:
            measured = fu['predictions'][model]
            values[patient]['dice:'+model].append(measured['dice'])
            empty[model] += measured['empty_prediction']
            if measured['hd95_mm'] is not None:
                boundary[model][patient].append(measured['hd95_mm'])
    patients = sorted(values)
    arrays = {name: np.array([np.mean(values[p][name]) for p in patients])
              for name in values[patients[0]]}
    diagnostics = {m: dict(empty_followup_predictions=empty[m],
                           patients_with_defined_hd95=len(boundary[m]),
                           patient_mean_hd95_mm_nonempty_only=float(np.mean([
                               np.mean(v) for v in boundary[m].values()])) if boundary[m] else None)
                   for m in MODELS}
    return patients, arrays, diagnostics


def summarize_external(seed_records: list[list[dict]], expected: list[dict], coefficients: list[dict],
                       draws: int, seed: int, family: int, target: float) -> dict:
    if len(seed_records) != 3 or len(coefficients) != 3 or draws < 1 or family != 6:
        raise GateError('External seed set or analysis parameters are incomplete')
    all_arrays, diagnostics = [], []
    patients = None
    for records, maps in zip(seed_records, coefficients):
        validate_prediction_coverage(records, expected)
        current, arrays, boundary = _patient_arrays(records, maps)
        if patients is not None and current != patients:
            raise GateError('External patient membership differs across seeds')
        patients = current
        all_arrays.append(arrays)
        diagnostics.append(boundary)
    samples = np.random.default_rng(seed).integers(0, len(patients), size=(draws, len(patients)))

    def interval(array, primary=False, volume=True):
        boot = array[samples].mean(axis=1)
        low, high = np.quantile(boot, [.025, .975]).tolist()
        out = dict(mean=float(array.mean()), percentile_95_interval=[low, high])
        if primary:
            adjusted = np.quantile(boot, [.05/(2*family), 1-.05/(2*family)]).tolist()
            out['bonferroni_family_interval'] = adjusted
            if volume:
                out['halfwidth_95_ml'] = (high-low)/2
                out['halfwidth_adjusted_ml'] = (adjusted[1]-adjusted[0])/2
                out['precision_target_95_met'] = (high-low)/2 <= target
                out['precision_target_adjusted_met'] = (adjusted[1]-adjusted[0])/2 <= target
        return out

    def summarize(arrays, adjusted):
        mae = {k[4:]: v for k, v in arrays.items() if k.startswith('mae:')}
        dice = {m: arrays['dice:'+m] for m in MODELS}
        primary = dict(ordinary_update_benefit=mae['old_old']-mae['old_new'],
                       retaining_baseline_cost=mae['old_new']-mae['new_new'],
                       contour_benefit=dice['new']-dice['old'])
        secondary = {}
        for m in MODELS[2:]:
            secondary[m+'_mixed_benefit'] = mae['old_new']-mae['old_'+m]
            secondary[m+'_dice_difference'] = dice[m]-dice['new']
            secondary[m+'_cost_vs_new_reprocessing'] = mae['old_'+m]-mae['new_new']
        for m in ('new', 'continued'):
            for corrected in ('mapped_old_'+m, 'old_mapped_'+m):
                secondary[corrected+'_benefit'] = mae['old_'+m]-mae[corrected]
        return dict(strategies={s: dict(mae_ml=float(v.mean()),
                                        signed_error_ml=float(arrays['signed:'+s].mean())) for s, v in mae.items()},
                    patient_mean_followup_dice={m: float(v.mean()) for m, v in dice.items()},
                    primary_contrasts={k: interval(v, adjusted, k != 'contour_benefit') for k, v in primary.items()},
                    descriptive_secondary_contrasts={k: interval(v) for k, v in secondary.items()})

    averaged = {k: np.mean([a[k] for a in all_arrays], axis=0) for k in all_arrays[0]}
    return dict(status='EXTERNAL_EVALUATION_COMPLETE', protocol_id='CT-UPDATE-010', cohort='PanTrack',
                patients=len(patients), pairs=len(expected)//2, seeds=3,
                seed_averaged=summarize(averaged, True),
                per_seed=[dict(**summarize(a, False), boundary_diagnostics=d)
                          for a, d in zip(all_arrays, diagnostics)],
                bootstrap_draws=draws, bootstrap_seed=seed, primary_family_size=family,
                precision_target_halfwidth_ml=target, clinical_equivalence_margin_ml=None,
                intervals_conditional_on_fixed_models_and_calibration=True,
                patient_resampling=True, internal_test_values_read=False,
                external_fitting_performed=False)
