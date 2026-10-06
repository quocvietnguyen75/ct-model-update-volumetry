"""CT-UPDATE-015 training-run replication: policy, arithmetic, scoring and verification."""
from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import re
import statistics

import numpy as np

from careport.ct_update_gate import GateError

PROTOCOL_ID = 'CT-UPDATE-015'
MODELS = ('old', 'new', 'continued')
COHORTS = ('internal', 'pantrack')
ORIGINAL_SEEDS = (20260915, 20260917, 20260919)
SEEDS = tuple(range(20261001, 20261021))
STRATEGIES = {'old_old': ('old', 'old'), 'old_new': ('old', 'new'), 'new_new': ('new', 'new'),
              'old_continued': ('old', 'continued'), 'continued_continued': ('continued', 'continued')}
PRIMARY = ('retaining_baseline_cost', 'ordinary_update_benefit', 'contour_benefit')
SECONDARY = ('baseline_version_shift_ml', 'same_version_change_benefit', 'added_data_mixed_benefit',
             'continued_retaining_baseline_cost', 'added_data_contour_benefit')
SHA256 = re.compile('[0-9a-f]{64}')

_FIXED = dict(
    protocol_id=PROTOCOL_ID, date='2026-09-25', question_id='IMG-ALT-01',
    role='training_run_replication_on_fixed_inspected_cohorts_no_tuning',
    amendment_file='newpaper/plan/CT_EXPERIMENT_AMENDMENT_015_2026-09-25.md',
    project_root_env='CT_UPDATE_REMOTE_ROOT', internal_dataset_root_env='CT_UPDATE_REMOTE_DATA_ROOT',
    pantrack_dataset_root_env='CT_UPDATE_PANTRACK_REMOTE_ROOT',
    source_config_file='config/preanalysis/ct_update_001_2026-09-15.json',
    internal_policy_file='config/preanalysis/ct_internal_holdout_014_2026-09-24.json',
    pantrack_policy_file='config/preanalysis/ct_pantrack_confirmation_010_2026-09-18.json',
    pantrack_registry_file='config/preanalysis/ct_pantrack_checkpoints_010_2026-09-18.json',
    development_cache_relative='derived/patches_001', run_root_relative='runs/replication_015',
    original_seeds=list(ORIGINAL_SEEDS), seeds=list(SEEDS), execution_id_prefix='CT-REPL-015-',
    training=dict(old_steps=20000, update_steps=10000, phases=list(MODELS),
                  eligible_training_patients=dict(old=113, new=225, continued=113)),
    models=list(MODELS),
    cohorts=dict(internal=dict(patients=29, pairs=277), pantrack=dict(patients=45, pairs=181)),
    analysis=dict(primary_contrasts=list(PRIMARY), primary_family_size=6,
                  bootstrap='two_way_independent_run_and_patient_percentile', bootstrap_draws=10000,
                  bootstrap_seed=20261021,
                  draw_order=['runs_shared_across_cohorts', 'internal_patients', 'pantrack_patients'],
                  weighting='pair_losses_within_patient_then_equal_patients_then_equal_runs',
                  primary_runs='new_seeds_only_original_runs_displayed_not_pooled',
                  minimum_completed_runs=15, clinical_margin_ml=None),
    execution=dict(partition='mig', gres='gpu:nvidia_h100_80gb_hbm3_3g.40gb:1', cpus_per_task=8, memory='48G',
                   time='0', concurrent_allocations=1, scoring_workers=6, max_cumulative_allocation_hours=80,
                   technical_resubmissions_per_seed=1),
    storage=dict(study_cap_bytes=268435456000, project_cap_bytes=966367641600,
                 planned_additional_bytes=5368709120),
)
_HASH_FIELDS = ('amendment_sha256', 'source_config_sha256', 'source_config_canonical_sha256',
                'internal_policy_sha256', 'internal_freeze_sha256', 'pantrack_policy_sha256',
                'pantrack_registry_sha256', 'pantrack_model_freeze_sha256')


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_replication_policy(policy: dict) -> None:
    if not isinstance(policy, dict) or set(policy) != set(_FIXED) | set(_HASH_FIELDS):
        raise GateError('Replication policy has missing or unapproved fields')
    if any(json.dumps(policy[k], sort_keys=True) != json.dumps(v, sort_keys=True) for k, v in _FIXED.items()):
        raise GateError('Unapproved replication design, seeds, cohorts or resources')
    if any(not isinstance(policy[k], str) or not SHA256.fullmatch(policy[k]) for k in _HASH_FIELDS):
        raise GateError('Replication policy lacks a valid binding hash')


def seed_for_index(policy: dict, index: int) -> int:
    validate_replication_policy(policy)
    if type(index) is not int or not 1 <= index <= len(policy['seeds']):
        raise GateError('Replication index is outside the fixed seed list')
    return policy['seeds'][index - 1]


def execution_policy(policy: dict, base: dict, index: int) -> dict:
    """Trainer execution policy for one fixed run; accepted unchanged by ct_update_train."""
    seed = seed_for_index(policy, index)
    if canonical_sha256(base) != policy['source_config_canonical_sha256']:
        raise GateError('Base configuration differs from the frozen source')
    return dict(execution_id=f"{policy['execution_id_prefix']}{index:02d}", amendment=PROTOCOL_ID,
                source_config_sha256=policy['source_config_canonical_sha256'], seed=seed,
                old_steps=policy['training']['old_steps'], update_steps=policy['training']['update_steps'])


def validate_replication_coverage(records: list[dict], expected: list[dict]) -> None:
    key = lambda r: (r['pair_id'], r['timepoint'])
    wanted, got = {key(r): r for r in expected}, {key(r): r for r in records}
    if not wanted or len(wanted) != len(expected) or len(got) != len(records) or set(wanted) != set(got):
        raise GateError('Replication predictions do not exactly cover the frozen cohort')
    for identity, record in got.items():
        source = wanted[identity]
        if (any(record[k] != source[k] for k in ('patient', 'scan', 'lesion_id', 'timepoint'))
                or not math.isclose(record['reference_volume_ml'], source['reference_volume_ml'],
                                    rel_tol=1e-12, abs_tol=1e-12)
                or set(record['predictions']) != set(MODELS)):
            raise GateError('Replication prediction identity, reference or model set differs')
        for measured in record['predictions'].values():
            volume, dice, hd = (measured[k] for k in ('volume_ml', 'dice', 'hd95_mm'))
            if (type(volume) not in (int, float) or not math.isfinite(volume) or volume < 0
                    or type(dice) not in (int, float) or not math.isfinite(dice) or not 0 <= dice <= 1
                    or measured['empty_prediction'] is not (volume == 0)
                    or (volume == 0 and (dice != 0 or hd is not None))
                    or (volume > 0 and (hd is None or not math.isfinite(hd) or hd < 0))):
                raise GateError('Invalid replication metric or empty-mask convention')


def patient_values(records: list[dict]) -> tuple[list[str], dict[str, np.ndarray]]:
    """Pair errors first, then equal-weight patient means, as in the frozen analyses."""
    pairs = defaultdict(dict)
    for row in records:
        key, timepoint = (row['patient'], row['pair_id']), row['timepoint']
        if timepoint not in ('bl', 'fu') or timepoint in pairs[key]:
            raise GateError('Duplicate or invalid paired role')
        pairs[key][timepoint] = row
    collected = defaultdict(lambda: defaultdict(list))
    for (patient, _), pair in pairs.items():
        if set(pair) != {'bl', 'fu'}:
            raise GateError('Replication pair lacks one timepoint')
        bl, fu = pair['bl'], pair['fu']
        target = fu['reference_volume_ml'] - bl['reference_volume_ml']
        b = {m: bl['predictions'][m]['volume_ml'] for m in MODELS}
        f = {m: fu['predictions'][m]['volume_ml'] for m in MODELS}
        for name, (baseline, followup) in STRATEGIES.items():
            error = f[followup] - b[baseline] - target
            collected[patient]['mae:'+name].append(abs(error))
            collected[patient]['signed:'+name].append(error)
        for model in MODELS:
            collected[patient]['dice:'+model].append(fu['predictions'][model]['dice'])
    patients = sorted(collected)
    if not patients:
        raise GateError('Empty replication cohort')
    return patients, {name: np.array([np.mean(collected[p][name]) for p in patients])
                      for name in sorted(collected[patients[0]])}


def contrast_values(values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    mae = lambda name: values['mae:'+name]
    return dict(retaining_baseline_cost=mae('old_new') - mae('new_new'),
                ordinary_update_benefit=mae('old_old') - mae('old_new'),
                contour_benefit=values['dice:new'] - values['dice:old'],
                baseline_version_shift_ml=values['signed:old_new'] - values['signed:new_new'],
                same_version_change_benefit=mae('old_old') - mae('new_new'),
                added_data_mixed_benefit=mae('old_continued') - mae('old_new'),
                continued_retaining_baseline_cost=mae('old_continued') - mae('continued_continued'),
                added_data_contour_benefit=values['dice:new'] - values['dice:continued'])


def resampling_indices(draws: int, seed: int, run_count: int,
                       patient_counts: dict[str, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fixed draw order: runs (shared by both cohorts), then each cohort's patients."""
    if (type(draws) is not int or draws < 1 or type(run_count) is not int or run_count < 2
            or set(patient_counts) != set(COHORTS)
            or any(type(n) is not int or n < 2 for n in patient_counts.values())):
        raise GateError('Invalid replication resampling request')
    rng = np.random.default_rng(seed)
    runs = rng.integers(0, run_count, size=(draws, run_count))
    return runs, {c: rng.integers(0, patient_counts[c], size=(draws, patient_counts[c])) for c in COHORTS}


def _counts(indices: np.ndarray, size: int) -> np.ndarray:
    weights = np.zeros((indices.shape[0], size))
    np.add.at(weights, (np.arange(indices.shape[0])[:, None], indices), 1.)
    return weights


def resampling_weights(draws: int, seed: int, run_count: int,
                       patient_counts: dict[str, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    runs, patients = resampling_indices(draws, seed, run_count, patient_counts)
    return _counts(runs, run_count), {c: _counts(v, patient_counts[c]) for c, v in patients.items()}


def interval_summary(matrix, run_weights: np.ndarray, patient_weights: np.ndarray,
                     family: int | None = None) -> dict:
    """Runs x patients contrast matrix; two-way, patient-only and run-only percentile intervals."""
    matrix = np.asarray(matrix, dtype=float)
    runs, patients = matrix.shape
    if (run_weights.shape[1] != runs or patient_weights.shape[1] != patients
            or run_weights.shape[0] != patient_weights.shape[0] or not np.isfinite(matrix).all()):
        raise GateError('Replication contrast matrix and resampling weights disagree')
    two_way = ((run_weights @ matrix) * patient_weights).sum(axis=1) / (runs * patients)
    patient_only = patient_weights @ matrix.mean(axis=0) / patients
    run_only = run_weights @ matrix.mean(axis=1) / runs
    per_run = matrix.mean(axis=1)

    def quantiles(values, low, high):
        return [float(v) for v in np.quantile(values, [low, high])]

    result = dict(mean=float(matrix.mean()), two_way_percentile_95=quantiles(two_way, .025, .975),
                  patient_only_percentile_95=quantiles(patient_only, .025, .975),
                  run_only_percentile_95=quantiles(run_only, .025, .975),
                  per_run_means=[float(v) for v in per_run],
                  run_standard_deviation=float(np.std(per_run, ddof=1)),
                  runs_positive=int(np.sum(per_run > 0)), runs_negative=int(np.sum(per_run < 0)))
    if family is not None:
        tail = .05 / (2 * family)
        adjusted = quantiles(two_way, tail, 1 - tail)
        result.update(two_way_family_interval=adjusted,
                      family_interval_excludes_zero=bool(adjusted[0] > 0 or adjusted[1] < 0))
    return result


def _boundary(records: list[dict]) -> dict:
    result = {}
    for model in MODELS:
        empty, distances = 0, defaultdict(list)
        for row in records:
            if row['timepoint'] != 'fu':
                continue
            measured = row['predictions'][model]
            empty += bool(measured['empty_prediction'])
            if measured['hd95_mm'] is not None:
                distances[row['patient']].append(measured['hd95_mm'])
        result[model] = dict(empty_followup_predictions=empty, patients_with_defined_hd95=len(distances),
                             patient_mean_hd95_mm_nonempty_only=float(np.mean([np.mean(v) for v in distances.values()]))
                             if distances else None)
    return result


def _run_order(runs: dict, policy: dict) -> list[int]:
    if set(runs) != set(COHORTS):
        raise GateError('Replication runs must cover exactly both cohorts')
    seeds = [seed for seed, _ in runs[COHORTS[0]]]
    if any([seed for seed, _ in runs[c]] != seeds for c in COHORTS):
        raise GateError('Cohort run sets differ')
    if any(seed not in policy['seeds'] for seed in seeds):
        raise GateError('Unregistered replication seed')
    positions = [policy['seeds'].index(seed) for seed in seeds]
    if positions != sorted(set(positions)):
        raise GateError('Replication runs are repeated or outside the fixed seed order')
    if len(seeds) < policy['analysis']['minimum_completed_runs']:
        raise GateError('Too few completed runs for the prespecified primary analysis')
    return seeds


def _check_frozen_size(expected: list[dict], frozen: dict) -> None:
    if len(expected) != 2*frozen['pairs'] or len({r['patient'] for r in expected}) != frozen['patients']:
        raise GateError('Frozen evaluation cohort size differs from the replication policy')


def summarize_replication(runs: dict[str, list[tuple[int, list[dict]]]], expected: dict[str, list[dict]],
                          policy: dict) -> dict:
    validate_replication_policy(policy)
    seeds = _run_order(runs, policy)
    if set(expected) != set(COHORTS):
        raise GateError('Frozen replication cohorts are incomplete')
    analysis = policy['analysis']
    counts = {c: policy['cohorts'][c]['patients'] for c in COHORTS}
    run_weights, patient_weights = resampling_weights(analysis['bootstrap_draws'], analysis['bootstrap_seed'],
                                                      len(seeds), counts)
    family = analysis['primary_family_size']
    cohorts = {}
    for cohort in COHORTS:
        _check_frozen_size(expected[cohort], policy['cohorts'][cohort])
        grids, reference, boundary = defaultdict(list), None, []
        for seed, records in runs[cohort]:
            validate_replication_coverage(records, expected[cohort])
            patients, values = patient_values(records)
            if reference is None:
                reference = patients
            elif patients != reference:
                raise GateError('Patient membership differs across replication runs')
            for name, array in {**values, **contrast_values(values)}.items():
                grids[name].append(array)
            boundary.append(dict(seed=seed, models=_boundary(records)))
        matrices = {name: np.vstack(rows) for name, rows in grids.items()}
        cohorts[cohort] = dict(
            patients=len(reference), pairs=len(expected[cohort]) // 2,
            primary_contrasts={n: interval_summary(matrices[n], run_weights, patient_weights[cohort], family)
                               for n in PRIMARY},
            secondary_contrasts={n: interval_summary(matrices[n], run_weights, patient_weights[cohort])
                                 for n in SECONDARY},
            strategies={n: dict(mae_ml=float(matrices['mae:'+n].mean()),
                                signed_error_ml=float(matrices['signed:'+n].mean())) for n in STRATEGIES},
            followup_dice={m: float(matrices['dice:'+m].mean()) for m in MODELS},
            per_run_boundary=boundary)
    return dict(status='REPLICATION_ANALYSIS_COMPLETE', protocol_id=PROTOCOL_ID, runs=len(seeds), seeds=seeds,
                missing_seeds=[s for s in policy['seeds'] if s not in seeds], models=list(MODELS),
                primary_contrasts=list(PRIMARY), primary_family_size=family,
                bootstrap=analysis['bootstrap'], bootstrap_draws=analysis['bootstrap_draws'],
                bootstrap_seed=analysis['bootstrap_seed'], weighting=analysis['weighting'],
                original_runs_pooled=False, cohorts_pooled=False, intervals_resample_training_runs=True,
                patient_level_values_included=False, cohorts=cohorts)


def _scalar_patient_values(records: list[dict]) -> dict[str, dict[str, float]]:
    """Separate scalar path for verification; does not call patient_values."""
    rows = {(r['patient'], r['pair_id'], r['timepoint']): r for r in records}
    per_patient = defaultdict(lambda: defaultdict(list))
    for (patient, pair_id, timepoint), fu in rows.items():
        if timepoint != 'fu':
            continue
        bl = rows[(patient, pair_id, 'bl')]
        target = fu['reference_volume_ml'] - bl['reference_volume_ml']
        for name, (a, b) in STRATEGIES.items():
            error = fu['predictions'][b]['volume_ml'] - bl['predictions'][a]['volume_ml'] - target
            per_patient[patient]['mae:'+name].append(abs(error))
            per_patient[patient]['signed:'+name].append(error)
        for model in MODELS:
            per_patient[patient]['dice:'+model].append(fu['predictions'][model]['dice'])
    return {p: {k: statistics.fmean(v) for k, v in metrics.items()} for p, metrics in per_patient.items()}


def _scalar_contrasts(v: dict[str, float]) -> dict[str, float]:
    return dict(retaining_baseline_cost=v['mae:old_new'] - v['mae:new_new'],
                ordinary_update_benefit=v['mae:old_old'] - v['mae:old_new'],
                contour_benefit=v['dice:new'] - v['dice:old'],
                baseline_version_shift_ml=v['signed:old_new'] - v['signed:new_new'],
                same_version_change_benefit=v['mae:old_old'] - v['mae:new_new'],
                added_data_mixed_benefit=v['mae:old_continued'] - v['mae:old_new'],
                continued_retaining_baseline_cost=v['mae:old_continued'] - v['mae:continued_continued'],
                added_data_contour_benefit=v['dice:new'] - v['dice:continued'])


def verify_replication(summary: dict, runs: dict, expected: dict, policy: dict) -> dict:
    """Recompute reported aggregates and primary intervals by separate scalar/sub-grid arithmetic."""
    validate_replication_policy(policy)
    seeds = _run_order(runs, policy)
    if summary.get('seeds') != seeds or summary.get('runs') != len(seeds):
        raise GateError('Verified runs differ from the reported replication runs')
    analysis = policy['analysis']
    family, draws = analysis['primary_family_size'], analysis['bootstrap_draws']
    counts = {c: policy['cohorts'][c]['patients'] for c in COHORTS}
    run_draws, patient_draws = resampling_indices(draws, analysis['bootstrap_seed'], len(seeds), counts)
    differences = []

    def close(actual, target):
        if (actual is None) != (target is None):
            raise GateError('Reported replication aggregate differs from scalar recalculation')
        if target is None:
            return
        delta = abs(float(actual) - float(target))
        if not math.isfinite(delta) or delta > 1e-9:
            raise GateError('Reported replication aggregate differs from scalar recalculation')
        differences.append(delta)

    for cohort in COHORTS:
        _check_frozen_size(expected[cohort], policy['cohorts'][cohort])
        reported = summary['cohorts'][cohort]
        contrasts = {name: [] for name in PRIMARY + SECONDARY}
        run_metrics = defaultdict(list)
        for position, (seed, records) in enumerate(runs[cohort]):
            validate_replication_coverage(records, expected[cohort])
            values = _scalar_patient_values(records)
            patients = sorted(values)
            scalar = [_scalar_contrasts(values[p]) for p in patients]
            for name in contrasts:
                contrasts[name].append([row[name] for row in scalar])
            for metric in values[patients[0]]:
                run_metrics[metric].append(statistics.fmean(values[p][metric] for p in patients))
            boundary = reported['per_run_boundary'][position]
            if boundary['seed'] != seed:
                raise GateError('Boundary diagnostics are out of run order')
            for model in MODELS:
                fu = [r for r in records if r['timepoint'] == 'fu']
                nonempty = defaultdict(list)
                for r in fu:
                    if r['predictions'][model]['hd95_mm'] is not None:
                        nonempty[r['patient']].append(r['predictions'][model]['hd95_mm'])
                close(boundary['models'][model]['empty_followup_predictions'],
                      sum(1 for r in fu if r['predictions'][model]['empty_prediction']))
                close(boundary['models'][model]['patients_with_defined_hd95'], len(nonempty))
                close(boundary['models'][model]['patient_mean_hd95_mm_nonempty_only'],
                      statistics.fmean(statistics.fmean(v) for v in nonempty.values()) if nonempty else None)
        for name, rows in contrasts.items():
            target = reported['primary_contrasts' if name in PRIMARY else 'secondary_contrasts'][name]
            per_run = [statistics.fmean(row) for row in rows]
            close(target['mean'], statistics.fmean(per_run))
            for actual, wanted in zip(target['per_run_means'], per_run, strict=True):
                close(actual, wanted)
            if name not in PRIMARY:
                continue
            grid = np.array(rows)
            boot = np.array([grid[np.ix_(run_draws[b], patient_draws[cohort][b])].mean() for b in range(draws)])
            patient_boot = np.array([grid.mean(axis=0)[patient_draws[cohort][b]].mean() for b in range(draws)])
            run_boot = np.array([grid.mean(axis=1)[run_draws[b]].mean() for b in range(draws)])
            tail = .05 / (2 * family)
            checks = [('two_way_percentile_95', boot, (.025, .975)),
                      ('two_way_family_interval', boot, (tail, 1 - tail)),
                      ('patient_only_percentile_95', patient_boot, (.025, .975)),
                      ('run_only_percentile_95', run_boot, (.025, .975))]
            for field, values, probabilities in checks:
                for actual, wanted in zip(target[field], np.quantile(values, probabilities), strict=True):
                    close(actual, wanted)
        for strategy in STRATEGIES:
            close(reported['strategies'][strategy]['mae_ml'], statistics.fmean(run_metrics['mae:'+strategy]))
            close(reported['strategies'][strategy]['signed_error_ml'],
                  statistics.fmean(run_metrics['signed:'+strategy]))
        for model in MODELS:
            close(reported['followup_dice'][model], statistics.fmean(run_metrics['dice:'+model]))
    return dict(status='REPLICATION_ARITHMETIC_REPRODUCED', numeric_values_checked=len(differences),
                max_absolute_difference=max(differences), primary_intervals_reproduced=True,
                secondary_intervals_recomputed=False, runs=len(seeds))


def load_native_mask(path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib
    volume = nib.load(path)
    return np.asanyarray(volume.dataobj), np.asarray(volume.affine)


def score_group(task: dict, loader=load_native_mask) -> list[tuple]:
    """Worker: score every record of one native mask with the unchanged native_metrics."""
    from careport.ct_update_evaluate import native_metrics
    path = Path(task['mask_path'])
    if file_sha256(path) != task['mask_sha256']:
        raise GateError('Native mask changed before replication scoring')
    labels, affine = loader(path)
    results = []
    for item in task['items']:
        if not np.allclose(affine, item['native_affine'], rtol=0, atol=1e-4):
            raise GateError('Replication scoring geometry differs from the frozen reference')
        results.append((item['key'], {m: native_metrics(item['logits'][m], item['patch_affine'], affine, labels,
                                                        item['lesion_id'], item['reference_voxels'])
                                      for m in MODELS}))
    return results


def evaluate_cohort(records: list[dict], cache: Path, locate_mask, infer, workers: int,
                    loader=load_native_mask) -> list[dict]:
    """GPU inference in this process; CPU scoring of whole native masks in spawned workers."""
    from careport.ct_pantrack_run import checked_path
    if type(workers) is not int or workers < 1:
        raise GateError('Invalid replication scoring worker count')
    groups = []
    for record in records:
        path, digest = locate_mask(record)
        mask = (str(path), digest)
        if groups and groups[-1][0] == mask:
            groups[-1][1].append(record)
        else:
            groups.append((mask, [record]))
    if len({mask for mask, _ in groups}) != len(groups):
        raise GateError('Records for one native mask are not contiguous')
    measured = {}

    def collect(future):
        for key, value in future.result():
            if key in measured:
                raise GateError('Duplicate replication prediction')
            measured[key] = value

    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        pending = deque()
        for (path, digest), group in groups:
            items = []
            for record in group:
                patch = checked_path(cache, record['patch_file'])
                if file_sha256(patch) != record['patch_sha256']:
                    raise GateError('Cached patch changed after preprocessing')
                with np.load(patch, allow_pickle=False) as payload:
                    image = payload['image'].astype(np.float32)
                logits = infer(image)
                if set(logits) != set(MODELS):
                    raise GateError('Inference returned an unexpected model set')
                items.append(dict(key=(record['pair_id'], record['timepoint']), patch_affine=record['patch_affine'],
                                  native_affine=record['native_affine'], lesion_id=record['lesion_id'],
                                  reference_voxels=record['reference_voxels'], logits=logits))
            pending.append(pool.submit(score_group, dict(mask_path=path, mask_sha256=digest, items=items), loader))
            while len(pending) > 2*workers:
                collect(pending.popleft())
        while pending:
            collect(pending.popleft())
    rows = []
    for record in records:
        row = {k: record[k] for k in ('patient', 'pair_id', 'scan', 'timepoint', 'lesion_id', 'reference_volume_ml')}
        row['predictions'] = measured[(record['pair_id'], record['timepoint'])]
        rows.append(row)
    return rows
