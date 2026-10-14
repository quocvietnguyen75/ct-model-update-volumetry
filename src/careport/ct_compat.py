"""CT-UPDATE-019: compatibility and patient-level response analyses of the stored replication predictions.

Post hoc and descriptive (amendment CT-UPDATE-019). The inputs are the frozen CT-UPDATE-015 (U-Net) and
CT-UPDATE-017 (nnU-Net) predictions on the two frozen evaluation cohorts; the outputs are aggregates only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math

import numpy as np

from careport.ct_posthoc import response_category, volume_thresholds
from careport.ct_replication import COHORTS, resampling_weights, validate_replication_coverage
from careport.ct_update_gate import GateError

PROTOCOL_ID = 'CT-UPDATE-019'
SEGMENTERS = ('unet', 'nnunet')
STRATEGIES = {'old_old': ('old', 'old'), 'old_new': ('old', 'new'), 'new_new': ('new', 'new')}
ASSESSMENT_UNITS = {'internal': 'patient', 'pantrack': 'patient_and_consecutive_scan_pair'}
DIAMETER_THRESHOLDS = {'decrease': -0.3, 'increase': 0.2}
LOA_MULTIPLIER = 1.96
CATEGORIES = ('decrease', 'stable', 'increase', 'undefined')
INTERVAL_CONDITIONS = ('differs', 'progression_split_differs', 'mixing_loses_reference', 'mixing_gains_reference')
DIRECTION_CONDITIONS = ('spurious_increase', 'masked_increase', 'spurious_decrease', 'masked_decrease')
AGREEMENT = tuple(f'agrees:{name}' for name in STRATEGIES)
CONDITIONS = INTERVAL_CONDITIONS + DIRECTION_CONDITIONS + AGREEMENT
LOG_TERMS = ('d', 'd2', 'e2', 'emix2', 'ed')
TOLERANCE = 1e-9


def validate_compat_policy(policy: dict) -> None:
    try:
        segmenters = policy['segmenters']
        ok = (policy['protocol_id'] == PROTOCOL_ID
              and set(policy['cohorts']) == set(COHORTS)
              and all(type(policy['cohorts'][c][k]) is int and policy['cohorts'][c][k] > 1
                      for c in COHORTS for k in ('patients', 'pairs'))
              and set(segmenters) == set(SEGMENTERS)
              and all(type(s['runs']) is int and s['runs'] >= 2 and len(s['seeds']) == s['runs']
                      and len(set(s['seeds'])) == s['runs'] and type(s['bootstrap_seed']) is int
                      for s in segmenters.values())
              and segmenters['unet']['bootstrap_seed'] != segmenters['nnunet']['bootstrap_seed']
              and type(policy['bootstrap_draws']) is int and policy['bootstrap_draws'] >= 100
              and policy['limits_of_agreement_multiplier'] == LOA_MULTIPLIER
              and policy['response_diameter_thresholds'] == DIAMETER_THRESHOLDS
              and policy['assessment_unit'] == ASSESSMENT_UNITS)
    except (KeyError, TypeError):
        ok = False
    if not ok:
        raise GateError('Invalid CT-UPDATE-019 policy')


def join_pairs(records: list[dict]) -> list[dict]:
    """One entry per lesion pair, sorted by patient and pair, with scans, references and old/new volumes."""
    joined = defaultdict(dict)
    for row in records:
        key, role = (row['patient'], row['pair_id']), row['timepoint']
        if role not in ('bl', 'fu') or role in joined[key]:
            raise GateError('Duplicate or invalid paired role')
        joined[key][role] = row
    pairs = []
    for (patient, pair_id), roles in sorted(joined.items()):
        if set(roles) != {'bl', 'fu'}:
            raise GateError('Compatibility pair lacks one timepoint')
        bl, fu = roles['bl'], roles['fu']
        pairs.append(dict(patient=patient, pair_id=pair_id, bl_scan=bl['scan'], fu_scan=fu['scan'],
                          ref_bl=float(bl['reference_volume_ml']), ref_fu=float(fu['reference_volume_ml']),
                          bl={m: float(bl['predictions'][m]['volume_ml']) for m in ('old', 'new')},
                          fu={m: float(fu['predictions'][m]['volume_ml']) for m in ('old', 'new')}))
    return pairs


def log_terms(pair: dict) -> dict | None:
    """Log-scale version difference and change errors; None when a measured volume the terms need is zero."""
    if pair['ref_bl'] <= 0 or pair['ref_fu'] <= 0:
        raise GateError('Reference volumes must be positive')
    old_bl, new_bl, new_fu = pair['bl']['old'], pair['bl']['new'], pair['fu']['new']
    if min(old_bl, new_bl, new_fu) <= 0:
        return None
    reference = math.log(pair['ref_fu'] / pair['ref_bl'])
    return dict(d=math.log(new_bl / old_bl), e=math.log(new_fu / new_bl) - reference,
                e_mix=math.log(new_fu / old_bl) - reference)


def _patients(runs_pairs: list[list[dict]]) -> list[str]:
    keys = [(p['patient'], p['pair_id']) for p in runs_pairs[0]]
    if not keys or any([(p['patient'], p['pair_id']) for p in pairs] != keys for pairs in runs_pairs):
        raise GateError('Pair membership differs across runs')
    return sorted({p['patient'] for p in runs_pairs[0]})


def cohort_matrices(runs_pairs: list[list[dict]]) -> dict:
    """Runs x patients means: change MAE over all pairs, log-scale terms over valid pair-runs."""
    patients = _patients(runs_pairs)
    index = {p: k for k, p in enumerate(patients)}
    shape = (len(runs_pairs), len(patients))
    sums = {name: np.zeros(shape) for name in ('mae:old_new', 'mae:new_new') + LOG_TERMS}
    pair_counts, log_counts = np.zeros(shape), np.zeros(shape)
    valid = excluded = 0
    for r, pairs in enumerate(runs_pairs):
        for p in pairs:
            i = index[p['patient']]
            target = p['ref_fu'] - p['ref_bl']
            sums['mae:old_new'][r, i] += abs(p['fu']['new'] - p['bl']['old'] - target)
            sums['mae:new_new'][r, i] += abs(p['fu']['new'] - p['bl']['new'] - target)
            pair_counts[r, i] += 1
            terms = log_terms(p)
            if terms is None:
                excluded += 1
                continue
            valid += 1
            log_counts[r, i] += 1
            d, e = terms['d'], terms['e']
            for name, value in (('d', d), ('d2', d * d), ('e2', e * e), ('emix2', terms['e_mix'] ** 2),
                                ('ed', e * d)):
                sums[name][r, i] += value
    mask = log_counts > 0
    result = dict(patients=patients, valid_pair_runs=valid, excluded_pair_runs=excluded, log_mask=mask)
    for name in ('mae:old_new', 'mae:new_new'):
        result[name] = sums[name] / pair_counts
    for name in LOG_TERMS:
        result[name] = np.divide(sums[name], log_counts, out=np.zeros(shape), where=mask)
    return result


def two_way_means(values, mask, run_weights: np.ndarray, patient_weights: np.ndarray) -> np.ndarray:
    """Per resample: patients averaged within each run (valid cells only), then runs averaged.

    With every cell valid this equals the frozen CT-UPDATE-015 two-way bootstrap mean.
    """
    values, mask = np.asarray(values, dtype=float), np.asarray(mask, dtype=bool)
    runs, patients = values.shape
    if (mask.shape != values.shape or run_weights.shape[1] != runs or patient_weights.shape[1] != patients
            or run_weights.shape[0] != patient_weights.shape[0]):
        raise GateError('Compatibility matrix and resampling weights disagree')
    cells = np.where(mask, values, 0.0)
    if not np.isfinite(cells).all():
        raise GateError('Non-finite compatibility value')
    numerator = patient_weights @ cells.T
    denominator = patient_weights @ mask.T.astype(float)
    present = denominator > 0
    run_means = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=present)
    weights = run_weights * present
    total = weights.sum(axis=1)
    if (total <= 0).any():
        raise GateError('A resample contains no valid run')
    return (weights * run_means).sum(axis=1) / total


def point_mean(values, mask) -> float:
    runs, patients = np.asarray(values).shape
    return float(two_way_means(values, mask, np.ones((1, runs)), np.ones((1, patients)))[0])


def _interval(draws) -> list[float]:
    return [float(v) for v in np.quantile(np.asarray(draws, dtype=float), [0.025, 0.975])]


def _ones(matrix) -> tuple[np.ndarray, np.ndarray]:
    runs, patients = np.asarray(matrix).shape
    return np.ones((1, runs)), np.ones((1, patients))


def relative_cost(matrices: dict, run_weights: np.ndarray, patient_weights: np.ndarray) -> dict:
    """100 x (MAE old/new / MAE new/new - 1): the compatibility cost relative to the same-version error."""
    mixed, same = matrices['mae:old_new'], matrices['mae:new_new']
    full = np.ones_like(mixed, dtype=bool)

    def percent(rw, pw):
        return 100 * (two_way_means(mixed, full, rw, pw) / two_way_means(same, full, rw, pw) - 1)
    per_run = [float(100 * (mixed[r].mean() / same[r].mean() - 1)) for r in range(mixed.shape[0])]
    return dict(mean_percent=float(percent(*_ones(mixed))[0]),
                two_way_percentile_95=_interval(percent(run_weights, patient_weights)),
                per_run_percent=per_run, runs_positive=int(sum(v > 0 for v in per_run)),
                runs_negative=int(sum(v < 0 for v in per_run)))


def version_agreement(matrices: dict, run_weights: np.ndarray, patient_weights: np.ndarray,
                      multiplier: float) -> dict:
    """Bias and limits of agreement of ln(V_new / V_old) on the same baseline images, back-transformed."""
    mask = matrices['log_mask']

    def moments(rw, pw):
        bias = two_way_means(matrices['d'], mask, rw, pw)
        second = two_way_means(matrices['d2'], mask, rw, pw)
        return bias, np.sqrt(np.maximum(second - bias * bias, 0.0))
    bias, sd = (float(v[0]) for v in moments(*_ones(mask)))
    draws_bias, draws_sd = moments(run_weights, patient_weights)

    def percent(x):
        return 100 * (np.exp(x) - 1)
    return dict(bias_log=bias, sd_log=sd, bias_percent=float(percent(bias)),
                limits_percent=[float(percent(bias - multiplier * sd)), float(percent(bias + multiplier * sd))],
                bias_percent_two_way_95=_interval(percent(draws_bias)), sd_log_two_way_95=_interval(draws_sd),
                lower_limit_percent_two_way_95=_interval(percent(draws_bias - multiplier * draws_sd)),
                upper_limit_percent_two_way_95=_interval(percent(draws_bias + multiplier * draws_sd)),
                valid_pair_runs=int(matrices['valid_pair_runs']),
                excluded_pair_runs=int(matrices['excluded_pair_runs']))


def decomposition(matrices: dict, run_weights: np.ndarray, patient_weights: np.ndarray) -> dict:
    """M_mix = M_same + D + X on the log scale, the compatibility index and the inflation of RMS change error."""
    mask = matrices['log_mask']

    def terms(rw, pw):
        mean = lambda name: two_way_means(matrices[name], mask, rw, pw)
        return mean('e2'), mean('emix2'), mean('d2'), 2 * mean('ed')
    same, mixed, d2, cross = (float(v[0]) for v in terms(*_ones(mask)))
    if same <= 0:
        raise GateError('Same-version log change error has no variation')
    residual = mixed - (same + d2 + cross)
    if abs(residual) > TOLERANCE * max(1.0, mixed):
        raise GateError('The log-scale change-error identity does not hold')
    draws_same, draws_mixed, draws_d2, draws_cross = terms(run_weights, patient_weights)
    return dict(m_same=same, m_mix=mixed, d2=d2, cross=cross, identity_residual=residual,
                kappa=math.sqrt(d2 / same), predicted_inflation_percent=100 * (math.sqrt(1 + d2 / same) - 1),
                observed_inflation_percent=100 * (math.sqrt(mixed / same) - 1), cross_ratio=cross / same,
                kappa_two_way_95=_interval(np.sqrt(draws_d2 / draws_same)),
                predicted_inflation_percent_two_way_95=_interval(100 * (np.sqrt(1 + draws_d2 / draws_same) - 1)),
                observed_inflation_percent_two_way_95=_interval(100 * (np.sqrt(draws_mixed / draws_same) - 1)),
                cross_ratio_two_way_95=_interval(draws_cross / draws_same))


def assessment_key(pair: dict, cohort: str) -> tuple:
    if cohort == 'internal':
        return (pair['patient'],)
    if cohort == 'pantrack':
        return (pair['patient'], pair['bl_scan'], pair['fu_scan'])
    raise GateError('Unknown cohort for patient-level assessment')


def assessments(pairs: list[dict], cohort: str) -> list[dict]:
    """Summed lesion volumes per assessment: per patient (internal) or per consecutive scan pair (PanTrack)."""
    grouped = defaultdict(list)
    for p in pairs:
        grouped[assessment_key(p, cohort)].append(p)
    result = []
    for key, members in sorted(grouped.items()):
        result.append(dict(patient=key[0], lesions=len(members),
                           ref_bl=sum(p['ref_bl'] for p in members), ref_fu=sum(p['ref_fu'] for p in members),
                           bl={m: sum(p['bl'][m] for p in members) for m in ('old', 'new')},
                           fu={m: sum(p['fu'][m] for p in members) for m in ('old', 'new')}))
    return result


def assessment_conditions(assessment: dict, diameter_thresholds: dict) -> dict:
    thresholds = volume_thresholds(diameter_thresholds)
    categories = {'reference': response_category(assessment['ref_bl'], assessment['ref_fu'], thresholds)}
    for name, (baseline, followup) in STRATEGIES.items():
        categories[name] = response_category(assessment['bl'][baseline], assessment['fu'][followup], thresholds)
    mixed, same, reference = categories['old_new'], categories['new_new'], categories['reference']
    increase = lambda c: c == 'increase'
    decrease = lambda c: c == 'decrease'
    result = dict(differs=int(mixed != same), progression_split_differs=int(increase(mixed) != increase(same)),
                  mixing_loses_reference=int(same == reference != mixed),
                  mixing_gains_reference=int(mixed == reference != same),
                  spurious_increase=int(increase(mixed) and not increase(same)),
                  masked_increase=int(increase(same) and not increase(mixed)),
                  spurious_decrease=int(decrease(mixed) and not decrease(same)),
                  masked_decrease=int(decrease(same) and not decrease(mixed)))
    for name in STRATEGIES:
        result['agrees:' + name] = int(categories[name] == reference)
    result['categories'] = categories
    return result


def response_matrices(runs_pairs: list[list[dict]], cohort: str,
                      diameter_thresholds: dict) -> tuple[list[str], dict, dict]:
    """Runs x patients shares of each patient's assessments meeting each condition, with counts."""
    patients = _patients(runs_pairs)
    index = {p: k for k, p in enumerate(patients)}
    shape = (len(runs_pairs), len(patients))
    sums, counts = {c: np.zeros(shape) for c in CONDITIONS}, np.zeros(shape)
    reference_counts, undefined, total = None, Counter(), None
    for r, pairs in enumerate(runs_pairs):
        found = assessments(pairs, cohort)
        categories = Counter()
        for assessment in found:
            i = index[assessment['patient']]
            conditions = assessment_conditions(assessment, diameter_thresholds)
            for name in CONDITIONS:
                sums[name][r, i] += conditions[name]
            counts[r, i] += 1
            categories[conditions['categories']['reference']] += 1
            undefined.update(n for n in STRATEGIES if conditions['categories'][n] == 'undefined')
        observed = {c: categories.get(c, 0) for c in CATEGORIES}
        if reference_counts is None:
            reference_counts, total = observed, len(found)
        elif observed != reference_counts or len(found) != total:
            raise GateError('Reference assessments differ across runs')
    if (counts == 0).any():
        raise GateError('A patient has no assessment')
    matrices = {name: sums[name] / counts for name in CONDITIONS}
    return patients, matrices, dict(assessments=total, reference_category_counts=reference_counts,
                                    undefined_assessment_runs={n: undefined.get(n, 0) for n in STRATEGIES})


def response_summary(matrices: dict, counts: dict, run_weights: np.ndarray, patient_weights: np.ndarray) -> dict:
    result = dict(counts)
    for name in CONDITIONS:
        matrix = matrices[name]
        full = np.ones_like(matrix, dtype=bool)
        per_run = matrix.mean(axis=1)
        entry = dict(mean=point_mean(matrix, full), run_min=float(per_run.min()), run_max=float(per_run.max()))
        if name not in DIRECTION_CONDITIONS:
            entry['two_way_percentile_95'] = _interval(two_way_means(matrix, full, run_weights, patient_weights))
        result[name] = entry
    return result


def _run_seeds(runs: dict, registered: list[int]) -> list[int]:
    if set(runs) != set(COHORTS):
        raise GateError('Runs must cover exactly both cohorts')
    seeds = [seed for seed, _ in runs[COHORTS[0]]]
    if any([seed for seed, _ in runs[c]] != seeds for c in COHORTS) or seeds != list(registered):
        raise GateError('Runs differ from the registered seeds')
    return seeds


def summarize_compat(segmenter_runs: dict, expected: dict[str, list[dict]], policy: dict) -> dict:
    validate_compat_policy(policy)
    if set(segmenter_runs) != set(SEGMENTERS) or set(expected) != set(COHORTS):
        raise GateError('Compatibility inputs must cover both segmenters and both cohorts')
    counts = {c: policy['cohorts'][c]['patients'] for c in COHORTS}
    for cohort in COHORTS:
        frozen = policy['cohorts'][cohort]
        if (len(expected[cohort]) != 2 * frozen['pairs']
                or len({r['patient'] for r in expected[cohort]}) != frozen['patients']):
            raise GateError('Frozen evaluation cohort size differs from the policy')
    segmenters = {}
    for name in SEGMENTERS:
        spec, runs = policy['segmenters'][name], segmenter_runs[name]
        seeds = _run_seeds(runs, spec['seeds'])
        run_weights, patient_weights = resampling_weights(policy['bootstrap_draws'], spec['bootstrap_seed'],
                                                          len(seeds), counts)
        cohorts = {}
        for cohort in COHORTS:
            for _, records in runs[cohort]:
                validate_replication_coverage(records, expected[cohort])
            runs_pairs = [join_pairs(records) for _, records in runs[cohort]]
            matrices = cohort_matrices(runs_pairs)
            if len(matrices['patients']) != counts[cohort]:
                raise GateError('Patient count differs from the policy')
            _, response, response_counts = response_matrices(runs_pairs, cohort,
                                                             policy['response_diameter_thresholds'])
            weights = run_weights, patient_weights[cohort]
            cohorts[cohort] = dict(
                patients=len(matrices['patients']), pairs=len(runs_pairs[0]),
                relative_compatibility_cost=relative_cost(matrices, *weights),
                version_agreement=version_agreement(matrices, *weights, policy['limits_of_agreement_multiplier']),
                decomposition=decomposition(matrices, *weights),
                patient_response=response_summary(response, response_counts, *weights))
        segmenters[name] = dict(runs=len(seeds), seeds=seeds, bootstrap_seed=spec['bootstrap_seed'], cohorts=cohorts)
    return dict(status='COMPAT_ANALYSIS_COMPLETE', protocol_id=PROTOCOL_ID, descriptive_only=True,
                primary_family_changed=False, segmenters_pooled=False, cohorts_pooled=False,
                patient_level_values_included=False, bootstrap_draws=policy['bootstrap_draws'],
                weighting='pairs_within_patient_then_patients_within_run_then_runs_equal',
                log_scale_valid_pair_run='old and new baseline and new follow-up volumes positive',
                limits_of_agreement_multiplier=policy['limits_of_agreement_multiplier'],
                response_volume_thresholds_percent=volume_thresholds(policy['response_diameter_thresholds']),
                assessment_unit=policy['assessment_unit'], segmenters=segmenters)


# Scalar verification path: plain loops, no shared matrix code.

def _scalar_cells(runs_pairs: list[list[dict]], cohort: str, thresholds: dict) -> list[dict]:
    runs = []
    for pairs in runs_pairs:
        cells = defaultdict(lambda: defaultdict(list))
        for p in pairs:
            target = p['ref_fu'] - p['ref_bl']
            cell = cells[p['patient']]
            cell['mae:old_new'].append(abs(p['fu']['new'] - p['bl']['old'] - target))
            cell['mae:new_new'].append(abs(p['fu']['new'] - p['bl']['new'] - target))
            olds, news, fu = p['bl']['old'], p['bl']['new'], p['fu']['new']
            if olds > 0 and news > 0 and fu > 0:
                ref = math.log(p['ref_fu']) - math.log(p['ref_bl'])
                d = math.log(news) - math.log(olds)
                e = math.log(fu) - math.log(news) - ref
                mixed = math.log(fu) - math.log(olds) - ref
                for name, value in (('d', d), ('d2', d * d), ('e2', e * e), ('emix2', mixed * mixed),
                                    ('ed', e * d)):
                    cell[name].append(value)
        for a in assessments(pairs, cohort):
            conditions = assessment_conditions(a, thresholds)
            for name in CONDITIONS:
                cells[a['patient']]['response:' + name].append(conditions[name])
        runs.append(cells)
    return runs


def _scalar_mean(runs: list[dict], name: str) -> float:
    means = []
    for cells in runs:
        values = [sum(c[name]) / len(c[name]) for c in cells.values() if c.get(name)]
        means.append(sum(values) / len(values))
    return sum(means) / len(means)


def _scalar_expectations(runs_pairs: list[list[dict]], cohort: str, policy: dict) -> dict:
    runs = _scalar_cells(runs_pairs, cohort, policy['response_diameter_thresholds'])
    mean = lambda name: _scalar_mean(runs, name)
    k = policy['limits_of_agreement_multiplier']
    bias, second = mean('d'), mean('d2')
    sd = math.sqrt(max(second - bias * bias, 0.0))
    same, mixed, d2, cross = mean('e2'), mean('emix2'), mean('d2'), 2 * mean('ed')
    expected = {
        ('relative_compatibility_cost', 'mean_percent'): 100 * (mean('mae:old_new') / mean('mae:new_new') - 1),
        ('version_agreement', 'bias_log'): bias, ('version_agreement', 'sd_log'): sd,
        ('version_agreement', 'bias_percent'): 100 * (math.exp(bias) - 1),
        ('version_agreement', 'limits_percent', 0): 100 * (math.exp(bias - k * sd) - 1),
        ('version_agreement', 'limits_percent', 1): 100 * (math.exp(bias + k * sd) - 1),
        ('decomposition', 'm_same'): same, ('decomposition', 'm_mix'): mixed,
        ('decomposition', 'd2'): d2, ('decomposition', 'cross'): cross,
        ('decomposition', 'kappa'): math.sqrt(d2 / same),
        ('decomposition', 'predicted_inflation_percent'): 100 * (math.sqrt(1 + d2 / same) - 1),
        ('decomposition', 'observed_inflation_percent'): 100 * (math.sqrt(mixed / same) - 1),
        ('decomposition', 'cross_ratio'): cross / same,
    }
    for r, cells in enumerate(runs):
        mixed_run = sum(sum(c['mae:old_new']) / len(c['mae:old_new']) for c in cells.values()) / len(cells)
        same_run = sum(sum(c['mae:new_new']) / len(c['mae:new_new']) for c in cells.values()) / len(cells)
        expected[('relative_compatibility_cost', 'per_run_percent', r)] = 100 * (mixed_run / same_run - 1)
    for name in CONDITIONS:
        expected[('patient_response', name, 'mean')] = mean('response:' + name)
    valid = sum(1 for cells in runs for c in cells.values() for _ in c.get('d', []))
    total = sum(len(pairs) for pairs in runs_pairs)
    expected[('version_agreement', 'valid_pair_runs')] = valid
    expected[('version_agreement', 'excluded_pair_runs')] = total - valid
    expected[('patient_response', 'assessments')] = len(assessments(runs_pairs[0], cohort))
    return expected


def _lookup(container, path):
    for key in path:
        container = container[key]
    return container


def verify_compat(result: dict, segmenter_runs: dict, expected: dict[str, list[dict]], policy: dict) -> dict:
    """Recompute every point value through the scalar path; reproduce the intervals with the same code."""
    reproduced = summarize_compat(segmenter_runs, expected, policy)
    checked = 0
    for name in SEGMENTERS:
        for cohort in COHORTS:
            got = result['segmenters'][name]['cohorts'][cohort]
            runs_pairs = [join_pairs(records) for _, records in segmenter_runs[name][cohort]]
            for path, value in _scalar_expectations(runs_pairs, cohort, policy).items():
                stored = _lookup(got, path)
                if not math.isclose(stored, value, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
                    raise GateError(f'Compatibility value differs on the scalar path: {name} {cohort} {path}')
                checked += 1
    if result != reproduced:
        raise GateError('Compatibility result is not reproduced by the analysis code')
    return dict(status='COMPAT_VERIFIED', protocol_id=PROTOCOL_ID, values_checked=checked,
                intervals_reproduced=True, intervals_independent=False)
