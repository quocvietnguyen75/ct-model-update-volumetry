"""CT-UPDATE-017 analysis: the CT-UPDATE-015 statistics and the CT-UPDATE-016 descriptions for the nnU-Net runs.

Aggregates only. The primary family, weighting and bootstrap are those of CT-UPDATE-015; the descriptive summaries
reuse the CT-UPDATE-016 definitions. The epoch schedule follows a timing-only rule fixed in amendment 017.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import statistics

from careport.ct_posthoc import summarize_posthoc, verify_posthoc
from careport.ct_replication import (COHORTS, MODELS, PRIMARY, SECONDARY, STRATEGIES, _boundary, _scalar_contrasts,
                                     _scalar_patient_values, contrast_values, interval_summary, patient_values,
                                     resampling_weights, validate_replication_coverage)
from careport.ct_update_gate import GateError

PROTOCOL_ID = 'CT-UPDATE-017'
PHASES = ('old', 'new', 'continued')
TOLERANCE = 1e-9


def validate_strong_policy(policy: dict) -> None:
    try:
        seeds = policy['seeds']
        schedules = policy['epoch_schedules']
        rule = policy['schedule_rule']
        analysis = policy['analysis']
        ok = (policy['protocol_id'] == PROTOCOL_ID and all(type(s) is int for s in seeds)
              and len(set(seeds)) == len(seeds) == policy['runs']
              and 1 <= policy['minimum_completed_runs'] <= policy['runs'] and policy['models'] == list(MODELS)
              and set(policy['cohorts']) == set(COHORTS)
              and set(schedules) == {'long', 'short'}
              and all(set(s) == set(PHASES) and all(type(e) is int and e > 0 for e in s.values())
                      for s in schedules.values())
              and all(type(e) is int and e >= 1 for e in rule['timing_epochs'])
              and max(rule['timing_epochs']) <= rule['smoke_epochs']
              and rule['long_epoch_total'] == sum(schedules['long'].values())
              and analysis['primary_contrasts'] == list(PRIMARY)
              and all(type(analysis[k]) is int for k in ('primary_family_size', 'bootstrap_draws', 'bootstrap_seed'))
              and isinstance(policy['descriptive'], dict))
    except (KeyError, TypeError, ValueError):
        ok = False
    if not ok:
        raise GateError('Invalid CT-UPDATE-017 policy')


def choose_schedule(epoch_seconds: list[float], policy: dict) -> dict:
    """Timing-only rule: the long schedule if 1000 epochs plus the evaluation allowance fit the hour limit."""
    validate_strong_policy(policy)
    rule = policy['schedule_rule']
    if (len(epoch_seconds) < rule['smoke_epochs']
            or any(not (isinstance(s, (int, float)) and math.isfinite(s) and s > 0) for s in epoch_seconds)):
        raise GateError('Timing check returned too few or invalid epoch durations')
    seconds = statistics.fmean(epoch_seconds[e - 1] for e in rule['timing_epochs'])
    projected = rule['long_epoch_total'] * seconds / 3600 + rule['evaluation_allowance_hours']
    name = 'long' if projected <= rule['long_if_projected_run_hours_at_most'] else 'short'
    return dict(schedule=name, epochs=dict(policy['epoch_schedules'][name]), seconds_per_epoch=seconds,
                projected_long_run_hours=projected, epoch_seconds=list(epoch_seconds))


def run_seeds(runs: dict, policy: dict) -> list[int]:
    if set(runs) != set(COHORTS):
        raise GateError('Runs must cover exactly both cohorts')
    seeds = [seed for seed, _ in runs[COHORTS[0]]]
    if any([seed for seed, _ in runs[c]] != seeds for c in COHORTS):
        raise GateError('Cohort run sets differ')
    if any(seed not in policy['seeds'] for seed in seeds):
        raise GateError('Unregistered run seed')
    positions = [policy['seeds'].index(seed) for seed in seeds]
    if positions != sorted(set(positions)):
        raise GateError('Runs are repeated or outside the fixed seed order')
    if len(seeds) < policy['minimum_completed_runs']:
        raise GateError('Too few completed runs for the prespecified analysis')
    return seeds


def descriptive_policy(policy: dict, runs: int) -> dict:
    """The CT-UPDATE-016 definitions, applied to these runs."""
    return dict(policy['descriptive'], protocol_id='CT-UPDATE-016', runs=runs, cohorts=policy['cohorts'])


def summarize_strong(runs: dict[str, list[tuple[int, list[dict]]]], expected: dict[str, list[dict]],
                     policy: dict) -> dict:
    validate_strong_policy(policy)
    seeds = run_seeds(runs, policy)
    if set(expected) != set(COHORTS):
        raise GateError('Frozen cohorts are incomplete')
    analysis = policy['analysis']
    counts = {c: policy['cohorts'][c]['patients'] for c in COHORTS}
    run_weights, patient_weights = resampling_weights(analysis['bootstrap_draws'], analysis['bootstrap_seed'],
                                                      len(seeds), counts)
    family = analysis['primary_family_size']
    cohorts = {}
    for cohort in COHORTS:
        frozen = policy['cohorts'][cohort]
        if (len(expected[cohort]) != 2 * frozen['pairs']
                or len({r['patient'] for r in expected[cohort]}) != frozen['patients']):
            raise GateError('Frozen evaluation cohort size differs from the policy')
        grids, reference, boundary = defaultdict(list), None, []
        for seed, records in runs[cohort]:
            validate_replication_coverage(records, expected[cohort])
            patients, values = patient_values(records)
            if reference is None:
                reference = patients
            elif patients != reference:
                raise GateError('Patient membership differs across runs')
            for name, array in {**values, **contrast_values(values)}.items():
                grids[name].append(array)
            boundary.append(dict(seed=seed, models=_boundary(records)))
        import numpy as np
        matrices = {name: np.vstack(rows) for name, rows in grids.items()}
        cohorts[cohort] = dict(
            patients=len(reference), pairs=len(expected[cohort]) // 2,
            primary_contrasts={n: interval_summary(matrices[n], run_weights, patient_weights[cohort], family)
                               for n in PRIMARY},
            secondary_contrasts={n: interval_summary(matrices[n], run_weights, patient_weights[cohort])
                                 for n in SECONDARY},
            strategies={n: dict(mae_ml=float(matrices['mae:' + n].mean()),
                                signed_error_ml=float(matrices['signed:' + n].mean())) for n in STRATEGIES},
            followup_dice={m: float(matrices['dice:' + m].mean()) for m in MODELS},
            per_run_boundary=boundary)
    descriptive = summarize_posthoc(runs, expected, descriptive_policy(policy, len(seeds)))
    descriptive['protocol_definitions'] = 'CT-UPDATE-016'
    return dict(status='STRONG_REPLICATION_ANALYSIS_COMPLETE', protocol_id=PROTOCOL_ID,
                model='nnU-Net v2 ResEnc-M (3d_fullres) on the frozen lesion-centred patches', runs=len(seeds),
                seeds=seeds, missing_seeds=[s for s in policy['seeds'] if s not in seeds], models=list(MODELS),
                primary_contrasts=list(PRIMARY), primary_family_size=family, bootstrap=analysis['bootstrap'],
                bootstrap_draws=analysis['bootstrap_draws'], bootstrap_seed=analysis['bootstrap_seed'],
                weighting=analysis['weighting'], cohorts_pooled=False, unet_runs_pooled=False,
                patient_level_values_included=False, cohorts=cohorts, descriptive=descriptive)


def verify_strong(result: dict, runs: dict, expected: dict, policy: dict) -> dict:
    """Exact reproduction, then a separate scalar recomputation of every primary and secondary point estimate."""
    again = summarize_strong(runs, expected, policy)
    if json.dumps(again, sort_keys=True) != json.dumps(result, sort_keys=True):
        raise GateError('Strong-model result is not reproduced exactly')
    checked = 0

    def check(actual, wanted):
        nonlocal checked
        if not math.isclose(actual, wanted, rel_tol=TOLERANCE, abs_tol=TOLERANCE):
            raise GateError(f'Scalar recomputation differs: {actual!r} != {wanted!r}')
        checked += 1

    for cohort in COHORTS:
        per_run = defaultdict(list)
        for _, records in runs[cohort]:
            patients = _scalar_patient_values(records)
            contrasts = [_scalar_contrasts(v) for v in patients.values()]
            for name in (*PRIMARY, *SECONDARY):
                per_run[name].append(statistics.fmean(c[name] for c in contrasts))
            for model in MODELS:
                per_run['dice:' + model].append(statistics.fmean(v['dice:' + model] for v in patients.values()))
        stored = result['cohorts'][cohort]
        for group, names in (('primary_contrasts', PRIMARY), ('secondary_contrasts', SECONDARY)):
            for name in names:
                check(stored[group][name]['mean'], statistics.fmean(per_run[name]))
                for value, wanted in zip(stored[group][name]['per_run_means'], per_run[name]):
                    check(value, wanted)
        for model in MODELS:
            check(stored['followup_dice'][model], statistics.fmean(per_run['dice:' + model]))
    descriptive = {k: v for k, v in result['descriptive'].items() if k != 'protocol_definitions'}
    report = verify_posthoc(descriptive, runs, expected, descriptive_policy(policy, result['runs']))
    return dict(status='STRONG_REPLICATION_VERIFIED', protocol_id=PROTOCOL_ID,
                values_checked=checked + report['values_checked'], summary_reproduced_exactly=True,
                intervals_recomputed_with_same_code=True)
