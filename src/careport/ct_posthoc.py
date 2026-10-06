"""CT-UPDATE-016: post hoc descriptive analyses of the stored CT-UPDATE-015 replication predictions.

Every output is an aggregate (counts, means, quantiles, correlations); no patient or lesion value leaves this
module. Weighting follows the replication: pair losses within patient, then patients and runs equally.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import statistics

import numpy as np

from careport.ct_replication import (COHORTS, _counts, contrast_values, interval_summary, patient_values,
                                     resampling_weights, validate_replication_coverage)
from careport.ct_update_gate import GateError

PROTOCOL_ID = 'CT-UPDATE-016'
STRATEGIES = {'old_old': ('old', 'old'), 'old_new': ('old', 'new'), 'new_new': ('new', 'new')}
PRIMARY = ('retaining_baseline_cost', 'ordinary_update_benefit', 'contour_benefit')
STRATUM_CONTRASTS = ('retaining_baseline_cost', 'ordinary_update_benefit')
CATEGORIES = ('decrease', 'stable', 'increase', 'undefined')
INTERVAL_FIELDS = ('mean', 'two_way_percentile_95', 'per_run_means', 'runs_positive', 'runs_negative')
TOLERANCE = 1e-9


def validate_posthoc_policy(policy: dict) -> None:
    """Structure only; the exact values are bound by the policy file's hash in the run module."""
    try:
        ok = (policy['protocol_id'] == PROTOCOL_ID and type(policy['runs']) is int and policy['runs'] >= 2
              and set(policy['cohorts']) == set(COHORTS)
              and all(type(policy['cohorts'][c][k]) is int for c in COHORTS for k in ('patients', 'pairs'))
              and [s[0] for s in policy['size_strata_ml']] == ['lt1', '1to10', 'ge10']
              and type(policy['minimum_patients_for_interval']) is int
              and all(type(policy[b][k]) is int for b in ('stratum_bootstrap', 'sensitivity_bootstrap')
                      for k in ('draws', 'seed'))
              and type(policy['sensitivity_bootstrap']['family_size']) is int
              and all(0 < q < 1 for q in policy['shift_quantiles'])
              and set(policy['response_diameter_thresholds']) == {'decrease', 'increase'})
    except (KeyError, TypeError, IndexError):
        ok = False
    if not ok:
        raise GateError('Invalid CT-UPDATE-016 policy')


def pair_rows(records: list[dict]) -> list[dict]:
    """One entry per lesion pair, sorted by patient and pair, with reference and predicted volumes."""
    joined = defaultdict(dict)
    for row in records:
        key, role = (row['patient'], row['pair_id']), row['timepoint']
        if role not in ('bl', 'fu') or role in joined[key]:
            raise GateError('Duplicate or invalid paired role')
        joined[key][role] = row
    pairs = []
    for (patient, pair_id), roles in sorted(joined.items()):
        if set(roles) != {'bl', 'fu'}:
            raise GateError('Post hoc pair lacks one timepoint')
        bl, fu = roles['bl'], roles['fu']
        baseline = {m: float(bl['predictions'][m]['volume_ml']) for m in ('old', 'new')}
        pairs.append(dict(patient=patient, pair_id=pair_id, ref_bl=float(bl['reference_volume_ml']),
                          ref_fu=float(fu['reference_volume_ml']), bl=baseline,
                          fu={m: float(fu['predictions'][m]['volume_ml']) for m in ('old', 'new')},
                          baseline_shift=baseline['new'] - baseline['old']))
    return pairs


def strategy_errors(pair: dict) -> dict[str, float]:
    target = pair['ref_fu'] - pair['ref_bl']
    return {name: pair['fu'][b] - pair['bl'][a] - target for name, (a, b) in STRATEGIES.items()}


def stratum_of(volume: float, strata) -> str:
    for name, low, high in strata:
        if volume >= low and (high is None or volume < high):
            return name
    raise GateError('Reference baseline volume lies outside every stratum')


def stratum_patient_values(pairs: list[dict], stratum: str, strata) -> tuple[list[str], dict[str, np.ndarray]]:
    """Equal-weight patient means over the patient's pairs in one stratum; other patients are excluded."""
    collected = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        if stratum_of(pair['ref_bl'], strata) == stratum:
            for name, error in strategy_errors(pair).items():
                collected[pair['patient']]['mae:' + name].append(abs(error))
    patients = sorted(collected)
    if not patients:
        return [], {}
    values = {f'mae:{s}': np.array([np.mean(collected[p]['mae:' + s]) for p in patients]) for s in STRATEGIES}
    values['retaining_baseline_cost'] = values['mae:old_new'] - values['mae:new_new']
    values['ordinary_update_benefit'] = values['mae:old_old'] - values['mae:old_new']
    return patients, values


def _point_summary(matrix: np.ndarray) -> dict:
    per_run = matrix.mean(axis=1)
    return dict(mean=float(matrix.mean()), per_run_means=[float(v) for v in per_run],
                runs_positive=int(np.sum(per_run > 0)), runs_negative=int(np.sum(per_run < 0)))


def stratum_resampling(draws: int, seed: int, run_count: int, sizes: list[tuple[tuple[str, str], int]]):
    """Fixed draw order: runs (shared), then each sufficiently large cohort stratum's patients in policy order."""
    rng = np.random.default_rng(seed)
    runs = _counts(rng.integers(0, run_count, size=(draws, run_count)), run_count)
    patients = {}
    for key, size in sizes:
        patients[key] = _counts(rng.integers(0, size, size=(draws, size)), size)
    return runs, patients


def _quartiles(values) -> dict:
    q25, median, q75 = (float(v) for v in np.quantile(np.asarray(values, dtype=float), [.25, .5, .75]))
    return dict(median=median, q25=q25, q75=q75)


def _same_pairs(runs_pairs: list[list[dict]]) -> list[tuple]:
    keys = [(p['patient'], p['pair_id']) for p in runs_pairs[0]]
    if any([(p['patient'], p['pair_id']) for p in pairs] != keys for pairs in runs_pairs):
        raise GateError('Pair membership differs across runs')
    return keys


def relative_summary(runs_pairs: list[list[dict]]) -> dict:
    """Absolute change error as a percentage of the reference baseline volume, averaged over runs per pair."""
    keys = _same_pairs(runs_pairs)
    relative = {name: np.zeros(len(keys)) for name in STRATEGIES}
    cost = np.zeros(len(keys))
    for pairs in runs_pairs:
        for j, pair in enumerate(pairs):
            errors = strategy_errors(pair)
            for name in STRATEGIES:
                relative[name][j] += 100 * abs(errors[name]) / pair['ref_bl']
            cost[j] += 100 * (abs(errors['old_new']) - abs(errors['new_new'])) / pair['ref_bl']
    runs = len(runs_pairs)
    summary = {f'{name}_percent': _quartiles(values / runs) for name, values in relative.items()}
    summary['retention_cost_percent'] = dict(_quartiles(cost / runs), share_positive=float(np.mean(cost > 0)))
    summary['pairs'] = len(keys)
    return summary


def lesion_weighted_cost(runs_pairs: list[list[dict]]) -> dict:
    per_run = []
    for pairs in runs_pairs:
        differences = [abs(e['old_new']) - abs(e['new_new']) for e in map(strategy_errors, pairs)]
        per_run.append(float(np.mean(differences)))
    return _point_summary(np.asarray(per_run)[:, None])


def _average_ranks(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='mergesort')
    ranks, ordered = np.empty(len(values)), values[order]
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and ordered[end + 1] == ordered[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(x, y) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise GateError('Spearman correlation needs matched series of at least three values')
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    if denominator == 0:
        raise GateError('Spearman correlation is undefined for a constant series')
    return float((rx * ry).sum() / denominator)


def run_associations(per_run: dict[str, list[float]]) -> dict:
    shift, cost = per_run['baseline_version_shift_ml'], per_run['retaining_baseline_cost']
    negative = [k for k, value in enumerate(shift) if value < 0]
    return dict(runs=len(shift), spearman_contour_benefit_vs_baseline_shift=spearman(per_run['contour_benefit'], shift),
                spearman_baseline_shift_vs_retention_cost=spearman(shift, cost),
                runs_with_negative_mean_shift=len(negative),
                negative_shift_runs_with_positive_retention_cost=sum(cost[k] > 0 for k in negative))


def shift_distribution(runs_pairs: list[list[dict]], quantiles) -> dict:
    """Signed between-version baseline difference (new minus old) over all runs and pairs."""
    d = np.array([p['baseline_shift'] for pairs in runs_pairs for p in pairs])
    relative = np.array([100 * p['baseline_shift'] / p['ref_bl'] for pairs in runs_pairs for p in pairs])
    per_run = [float(np.mean([p['baseline_shift'] < 0 for p in pairs])) for pairs in runs_pairs]
    return dict(values=int(d.size), share_negative=float(np.mean(d < 0)), share_zero=float(np.mean(d == 0)),
                share_positive=float(np.mean(d > 0)),
                quantiles={str(q): float(np.quantile(d, q)) for q in quantiles},
                mean_absolute_ml=float(np.mean(np.abs(d))), relative_percent=_quartiles(relative),
                per_run_share_negative=dict(min=min(per_run), max=max(per_run)))


def volume_thresholds(diameter: dict) -> dict:
    """Percent volume change of a sphere whose diameter changes by the given fractions."""
    return {name: 100 * ((1 + fraction) ** 3 - 1) for name, fraction in diameter.items()}


def response_category(baseline: float, followup: float, thresholds: dict) -> str:
    if baseline <= 0:
        return 'undefined'
    change = 100 * (followup - baseline) / baseline
    if change <= thresholds['decrease']:
        return 'decrease'
    if change >= thresholds['increase']:
        return 'increase'
    return 'stable'


def _range(values: list[float]) -> dict:
    return dict(mean=float(np.mean(values)), min=float(min(values)), max=float(max(values)))


def response_illustration(runs_pairs: list[list[dict]], diameter: dict) -> dict:
    """Exploratory lesion-level categories; RECIST applies to patients' summed diameters, not single lesions."""
    thresholds = volume_thresholds(diameter)
    shares = defaultdict(list)
    reference_counts = None
    for pairs in runs_pairs:
        categories = {'reference': [response_category(p['ref_bl'], p['ref_fu'], thresholds) for p in pairs]}
        for name, (a, b) in STRATEGIES.items():
            categories[name] = [response_category(p['bl'][a], p['fu'][b], thresholds) for p in pairs]
        counts = {c: categories['reference'].count(c) for c in CATEGORIES}
        if reference_counts is None:
            reference_counts = counts
        elif counts != reference_counts:
            raise GateError('Reference categories differ across runs')
        n, reference = len(pairs), categories['reference']
        defined = [j for j in range(n) if 'undefined' not in (categories['old_new'][j], categories['new_new'][j])]
        if not defined:
            raise GateError('No pair has a defined category under both strategies')
        shares['share_old_new_differs_from_new_new'].append(
            sum(categories['old_new'][j] != categories['new_new'][j] for j in defined) / len(defined))
        shares['share_mixing_loses_reference_category'].append(
            sum(categories['new_new'][j] == reference[j] != categories['old_new'][j] for j in range(n)) / n)
        shares['share_mixing_gains_reference_category'].append(
            sum(categories['old_new'][j] == reference[j] != categories['new_new'][j] for j in range(n)) / n)
        for name in STRATEGIES:
            shares['agreement:' + name].append(sum(categories[name][j] == reference[j] for j in range(n)) / n)
            shares['undefined:' + name].append(categories[name].count('undefined'))
    result = {k: _range(v) for k, v in shares.items() if ':' not in k}
    result['agreement_with_reference'] = {s: _range(shares['agreement:' + s]) for s in STRATEGIES}
    result['undefined_pairs_per_run'] = {s: _range(shares['undefined:' + s]) for s in STRATEGIES}
    result['reference_category_counts'] = reference_counts
    result['volume_thresholds_percent'] = thresholds
    return result


def sign_probability(positive: int, negative: int) -> float:
    """One-sided binomial probability of at least `positive` positive runs among the nonzero runs, p = 1/2."""
    n = positive + negative
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(positive, n + 1)) / 2 ** n


def _run_seeds(runs: dict, policy: dict) -> list[int]:
    if set(runs) != set(COHORTS):
        raise GateError('Post hoc runs must cover exactly both cohorts')
    seeds = [seed for seed, _ in runs[COHORTS[0]]]
    if any([seed for seed, _ in runs[c]] != seeds for c in COHORTS) or len(set(seeds)) != len(seeds):
        raise GateError('Post hoc run sets differ between cohorts or repeat a seed')
    if len(seeds) != policy['runs']:
        raise GateError('Post hoc analysis requires every registered run')
    return seeds


def summarize_posthoc(runs: dict[str, list[tuple[int, list[dict]]]], expected: dict[str, list[dict]],
                      policy: dict) -> dict:
    validate_posthoc_policy(policy)
    seeds = _run_seeds(runs, policy)
    if set(expected) != set(COHORTS):
        raise GateError('Frozen cohorts are incomplete')
    strata = policy['size_strata_ml']
    pairs_by_cohort, primary = {}, {}
    for cohort in COHORTS:
        frozen = policy['cohorts'][cohort]
        if (len(expected[cohort]) != 2 * frozen['pairs']
                or len({r['patient'] for r in expected[cohort]}) != frozen['patients']):
            raise GateError('Frozen evaluation cohort size differs from the post hoc policy')
        runs_pairs, grids, reference = [], defaultdict(list), None
        for _, records in runs[cohort]:
            validate_replication_coverage(records, expected[cohort])
            patients, values = patient_values(records)
            if reference is None:
                reference = patients
            elif patients != reference:
                raise GateError('Patient membership differs across runs')
            for name, array in contrast_values(values).items():
                grids[name].append(array)
            runs_pairs.append(pair_rows(records))
        _same_pairs(runs_pairs)
        pairs_by_cohort[cohort] = runs_pairs
        primary[cohort] = {name: np.vstack(rows) for name, rows in grids.items()}
    membership, sizes = {}, []
    for cohort in COHORTS:
        for name, _, _ in strata:
            patients, _ = stratum_patient_values(pairs_by_cohort[cohort][0], name, strata)
            membership[(cohort, name)] = patients
            if len(patients) >= policy['minimum_patients_for_interval']:
                sizes.append(((cohort, name), len(patients)))
    plan = policy['stratum_bootstrap']
    run_weights, stratum_weights = stratum_resampling(plan['draws'], plan['seed'], len(seeds), sizes)
    sensitivity_plan = policy['sensitivity_bootstrap']
    sensitivity_runs, sensitivity_patients = resampling_weights(
        sensitivity_plan['draws'], sensitivity_plan['seed'], len(seeds),
        {c: primary[c]['contour_benefit'].shape[1] for c in COHORTS})
    cohorts = {}
    for cohort in COHORTS:
        runs_pairs = pairs_by_cohort[cohort]
        size_strata = []
        for name, low, high in strata:
            members = membership[(cohort, name)]
            entry = dict(stratum=name, lower_ml=low, upper_ml=high, patients=len(members),
                         pairs=sum(stratum_of(p['ref_bl'], strata) == name for p in runs_pairs[0]))
            if members:
                grids = defaultdict(list)
                for pairs in runs_pairs:
                    patients, values = stratum_patient_values(pairs, name, strata)
                    if patients != members:
                        raise GateError('Stratum membership differs across runs')
                    for key, value in values.items():
                        grids[key].append(value)
                matrices = {key: np.vstack(rows) for key, rows in grids.items()}
                entry['mae_ml'] = {s: float(matrices['mae:' + s].mean()) for s in STRATEGIES}
                for contrast in STRATUM_CONTRASTS:
                    if (cohort, name) in stratum_weights:
                        summary = interval_summary(matrices[contrast], run_weights, stratum_weights[(cohort, name)])
                        entry[contrast] = {k: summary[k] for k in INTERVAL_FIELDS}
                    else:
                        entry[contrast] = _point_summary(matrices[contrast])
            size_strata.append(entry)
        per_run = {name: [float(v) for v in primary[cohort][name].mean(axis=1)]
                   for name in ('contour_benefit', 'baseline_version_shift_ml', 'retaining_baseline_cost')}
        sensitivity = {}
        for name in PRIMARY:
            summary = interval_summary(primary[cohort][name], sensitivity_runs, sensitivity_patients[cohort],
                                       sensitivity_plan['family_size'])
            sensitivity[name] = dict({k: summary[k] for k in ('mean', 'two_way_percentile_95', 'two_way_family_interval',
                                                              'family_interval_excludes_zero', 'runs_positive',
                                                              'runs_negative')},
                                     sign_probability=sign_probability(summary['runs_positive'],
                                                                       summary['runs_negative']))
        cohorts[cohort] = dict(patients=policy['cohorts'][cohort]['patients'], pairs=policy['cohorts'][cohort]['pairs'],
                               size_strata=size_strata, relative_scale=relative_summary(runs_pairs),
                               lesion_weighted=lesion_weighted_cost(runs_pairs),
                               run_associations=run_associations(per_run),
                               baseline_shift_distribution=shift_distribution(runs_pairs, policy['shift_quantiles']),
                               interval_sensitivity=sensitivity,
                               response_illustration=response_illustration(runs_pairs,
                                                                           policy['response_diameter_thresholds']))
    return dict(status='POSTHOC_ANALYSIS_COMPLETE', protocol_id=PROTOCOL_ID, runs=len(seeds), seeds=seeds,
                descriptive_only=True, primary_family_changed=False, cohorts_pooled=False,
                patient_level_values_included=False, size_strata_ml=strata,
                stratum_bootstrap=dict(plan), sensitivity_bootstrap=dict(sensitivity_plan), cohorts=cohorts)


# Verification: an independent scalar path for every point estimate, plus exact reproduction of the summary.

def _scalar_pairs(records: list[dict]) -> list[tuple]:
    rows = {(r['patient'], r['pair_id'], r['timepoint']): r for r in records}
    out = []
    for (patient, pair_id, role), fu in sorted(rows.items()):
        if role != 'fu':
            continue
        bl = rows[(patient, pair_id, 'bl')]
        out.append((patient, float(bl['reference_volume_ml']), float(fu['reference_volume_ml']),
                    {m: float(bl['predictions'][m]['volume_ml']) for m in ('old', 'new')},
                    {m: float(fu['predictions'][m]['volume_ml']) for m in ('old', 'new')}))
    return out


def _scalar_errors(ref_bl, ref_fu, bl, fu) -> dict:
    return {name: fu[b] - bl[a] - (ref_fu - ref_bl) for name, (a, b) in STRATEGIES.items()}


def _scalar_quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _scalar_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    for i, v in enumerate(values):
        below = sum(1 for w in values if w < v)
        equal = sum(1 for w in values if w == v)
        ranks[i] = below + (equal + 1) / 2
    return ranks


def _scalar_spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _scalar_ranks(x), _scalar_ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


def verify_posthoc(result: dict, runs: dict, expected: dict, policy: dict) -> dict:
    again = summarize_posthoc(runs, expected, policy)
    if json.dumps(again, sort_keys=True) != json.dumps(result, sort_keys=True):
        raise GateError('Post hoc result is not reproduced exactly')
    strata = policy['size_strata_ml']
    checked = 0

    def check(actual, wanted):
        nonlocal checked
        if not math.isclose(actual, wanted, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise GateError(f'Scalar recomputation differs: {actual!r} != {wanted!r}')
        checked += 1

    thresholds = volume_thresholds(policy['response_diameter_thresholds'])
    for cohort in COHORTS:
        stored = result['cohorts'][cohort]
        runs_pairs = [_scalar_pairs(records) for _, records in runs[cohort]]
        # Size strata: patient means within stratum, then patients, then runs.
        for entry in stored['size_strata']:
            per_run_cost, per_run_update, levels = [], [], defaultdict(list)
            for pairs in runs_pairs:
                by_patient = defaultdict(lambda: defaultdict(list))
                for patient, ref_bl, ref_fu, bl, fu in pairs:
                    if stratum_of(ref_bl, strata) == entry['stratum']:
                        for name, error in _scalar_errors(ref_bl, ref_fu, bl, fu).items():
                            by_patient[patient][name].append(abs(error))
                if not by_patient:
                    continue
                means = {p: {n: statistics.fmean(v) for n, v in d.items()} for p, d in by_patient.items()}
                per_run_cost.append(statistics.fmean(m['old_new'] - m['new_new'] for m in means.values()))
                per_run_update.append(statistics.fmean(m['old_old'] - m['old_new'] for m in means.values()))
                for s in STRATEGIES:
                    levels[s].append(statistics.fmean(m[s] for m in means.values()))
            if per_run_cost:
                check(entry['retaining_baseline_cost']['mean'], statistics.fmean(per_run_cost))
                check(entry['ordinary_update_benefit']['mean'], statistics.fmean(per_run_update))
                for s in STRATEGIES:
                    check(entry['mae_ml'][s], statistics.fmean(levels[s]))
        # Relative scale and lesion weighting.
        pair_count = len(runs_pairs[0])
        relative = {s: [0.0] * pair_count for s in STRATEGIES}
        cost = [0.0] * pair_count
        lesion_runs = []
        for pairs in runs_pairs:
            differences = []
            for j, (_, ref_bl, ref_fu, bl, fu) in enumerate(pairs):
                errors = _scalar_errors(ref_bl, ref_fu, bl, fu)
                for s in STRATEGIES:
                    relative[s][j] += 100 * abs(errors[s]) / ref_bl / len(runs_pairs)
                cost[j] += 100 * (abs(errors['old_new']) - abs(errors['new_new'])) / ref_bl / len(runs_pairs)
                differences.append(abs(errors['old_new']) - abs(errors['new_new']))
            lesion_runs.append(statistics.fmean(differences))
        for s in STRATEGIES:
            for key, q in (('q25', .25), ('median', .5), ('q75', .75)):
                check(stored['relative_scale'][f'{s}_percent'][key], _scalar_quantile(relative[s], q))
        check(stored['relative_scale']['retention_cost_percent']['median'], _scalar_quantile(cost, .5))
        check(stored['relative_scale']['retention_cost_percent']['share_positive'],
              sum(c > 0 for c in cost) / pair_count)
        check(stored['lesion_weighted']['mean'], statistics.fmean(lesion_runs))
        # Run-level associations from scalar patient means.
        per_run = defaultdict(list)
        for _, records in runs[cohort]:
            by_patient = defaultdict(lambda: defaultdict(list))
            fu_rows = {(r['patient'], r['pair_id']): r for r in records if r['timepoint'] == 'fu'}
            for patient, ref_bl, ref_fu, bl, fu in _scalar_pairs(records):
                errors = _scalar_errors(ref_bl, ref_fu, bl, fu)
                by_patient[patient]['cost'].append(abs(errors['old_new']) - abs(errors['new_new']))
                by_patient[patient]['shift'].append(errors['old_new'] - errors['new_new'])
            for (patient, _), row in fu_rows.items():
                by_patient[patient]['contour'].append(row['predictions']['new']['dice'] - row['predictions']['old']['dice'])
            for key in ('cost', 'shift', 'contour'):
                per_run[key].append(statistics.fmean(statistics.fmean(d[key]) for d in by_patient.values()))
        associations = stored['run_associations']
        check(associations['spearman_contour_benefit_vs_baseline_shift'],
              _scalar_spearman(per_run['contour'], per_run['shift']))
        check(associations['spearman_baseline_shift_vs_retention_cost'],
              _scalar_spearman(per_run['shift'], per_run['cost']))
        negative = [k for k, v in enumerate(per_run['shift']) if v < 0]
        check(associations['runs_with_negative_mean_shift'], len(negative))
        check(associations['negative_shift_runs_with_positive_retention_cost'],
              sum(per_run['cost'][k] > 0 for k in negative))
        # Lesion-level baseline differences.
        shifts = [bl['new'] - bl['old'] for pairs in runs_pairs for _, _, _, bl, _ in pairs]
        distribution = stored['baseline_shift_distribution']
        check(distribution['share_negative'], sum(d < 0 for d in shifts) / len(shifts))
        check(distribution['mean_absolute_ml'], statistics.fmean(abs(d) for d in shifts))
        for q in policy['shift_quantiles']:
            check(distribution['quantiles'][str(q)], _scalar_quantile(shifts, q))
        # Response illustration.
        changed = []
        for pairs in runs_pairs:
            pairs_defined, differs = 0, 0
            for _, ref_bl, ref_fu, bl, fu in pairs:
                a = response_category(bl['old'], fu['new'], thresholds)
                b = response_category(bl['new'], fu['new'], thresholds)
                if 'undefined' not in (a, b):
                    pairs_defined += 1
                    differs += a != b
            changed.append(differs / pairs_defined)
        check(stored['response_illustration']['share_old_new_differs_from_new_new']['mean'], statistics.fmean(changed))
        # Sensitivity point estimates and sign probabilities.
        for name in PRIMARY:
            entry = stored['interval_sensitivity'][name]
            check(entry['sign_probability'], sign_probability(entry['runs_positive'], entry['runs_negative']))
        check(stored['interval_sensitivity']['retaining_baseline_cost']['mean'],
              statistics.fmean(per_run['cost']))
    return dict(status='POSTHOC_VERIFIED', protocol_id=PROTOCOL_ID, values_checked=checked,
                summary_reproduced_exactly=True, intervals_recomputed_with_same_code=True)
