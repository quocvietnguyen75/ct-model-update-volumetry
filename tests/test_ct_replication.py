"""Synthetic contracts for the CT-UPDATE-015 training-run replication."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from careport.ct_pantrack_confirm import MODELS as FROZEN_MODELS, _patient_arrays
from careport.ct_replication import (
    COHORTS, MODELS, PRIMARY, canonical_sha256, contrast_values, evaluate_cohort, execution_policy,
    interval_summary, patient_values, resampling_weights, score_group, seed_for_index,
    summarize_replication, validate_replication_coverage, validate_replication_policy,
    verify_replication,
)
from careport.ct_update_data import patch_affine
from careport.ct_update_evaluate import native_metrics
from careport.ct_update_gate import GateError
from careport.ct_update_train import execution_config

ROOT = Path(__file__).resolve().parents[1]
POLICY_FILE = ROOT/'config/preanalysis/ct_replication_015_2026-09-25.json'
BASE_FILE = ROOT/'config/preanalysis/ct_update_001_2026-09-15.json'
PATCH_SHAPE = (12, 12, 12)

# Deterministic native volume, rebuilt identically inside spawned scoring workers.
AFFINE = np.diag([0.8, 0.8, 2.0, 1.0])
LABELS = np.zeros((40, 40, 16), dtype=np.int16)
LABELS[16:24, 15:25, 6:10] = 3
LABELS[5:9, 5:9, 2:4] = 7


def synthetic_loader(path):
    return LABELS, AFFINE


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy():
    return json.loads(POLICY_FILE.read_text(encoding='utf-8'))


def load_base():
    return json.loads(BASE_FILE.read_text(encoding='utf-8'))


def measured(rng, reference, empty=False):
    if empty:
        return dict(volume_ml=0., dice=0., hd95_mm=None, empty_prediction=True)
    return dict(volume_ml=float(max(.01, reference + rng.normal(0, 2))), dice=float(rng.uniform(.3, .9)),
                hd95_mm=float(rng.uniform(1, 9)), empty_prediction=False)


def cohort_records(seed, pairs_per_patient, models=MODELS, prefix='SECRETPATIENT'):
    """Rows mimicking frozen prediction records; reference values depend only on the pair."""
    rng = np.random.default_rng(seed)
    rows = []
    for number, count in enumerate(pairs_per_patient):
        patient = f'{prefix}{number:03d}'
        for pair in range(count):
            pair_id = hashlib.sha256(f'{patient}:{pair}'.encode()).hexdigest()
            for timepoint in ('bl', 'fu'):
                reference = float(np.random.default_rng([number, pair, int(timepoint == 'fu')]).uniform(.5, 30))
                predictions = {m: measured(rng, reference, empty=(m == 'continued' and pair == 1 and timepoint == 'fu'))
                               for m in models}
                rows.append(dict(patient=patient, pair_id=pair_id, scan=f'{patient}_{timepoint}_{pair:02d}',
                                 timepoint=timepoint, lesion_id=pair + 1, reference_volume_ml=reference,
                                 predictions=predictions))
    return rows


def expected_from(records):
    return [{k: r[k] for k in ('patient', 'pair_id', 'scan', 'timepoint', 'lesion_id', 'reference_volume_ml')}
            for r in records]


INTERNAL_PAIRS = [10]*27 + [4, 3]        # 29 patients, 277 pairs
PANTRACK_PAIRS = [4]*44 + [5]            # 45 patients, 181 pairs


class PolicyTests(unittest.TestCase):
    def test_committed_policy_binds_amendment_and_frozen_inputs(self):
        policy = load_policy()
        validate_replication_policy(policy)
        self.assertEqual(sha256(ROOT/policy['amendment_file']), policy['amendment_sha256'])
        for key in ('source_config', 'internal_policy', 'pantrack_policy', 'pantrack_registry'):
            self.assertEqual(sha256(ROOT/policy[key+'_file']), policy[key+'_sha256'])
        self.assertEqual(canonical_sha256(load_base()), policy['source_config_canonical_sha256'])

    def test_design_changes_are_rejected(self):
        mutations = [
            lambda p: p['seeds'].__setitem__(0, 20260915),
            lambda p: p['seeds'].append(20261021),
            lambda p: p['seeds'].__setitem__(1, p['seeds'][0]),
            lambda p: p['training'].__setitem__('old_steps', 30000),
            lambda p: p.__setitem__('models', ['old', 'new']),
            lambda p: p['analysis'].__setitem__('primary_family_size', 3),
            lambda p: p['analysis'].__setitem__('bootstrap_draws', 2000),
            lambda p: p['analysis'].__setitem__('minimum_completed_runs', 3),
            lambda p: p['execution'].__setitem__('concurrent_allocations', 4),
            lambda p: p['cohorts']['pantrack'].__setitem__('patients', 44),
        ]
        for mutate in mutations:
            policy = load_policy()
            mutate(policy)
            with self.assertRaises(GateError):
                validate_replication_policy(policy)

    def test_execution_policies_are_accepted_by_unchanged_trainer(self):
        policy, base = load_policy(), load_base()
        first, last = execution_policy(policy, base, 1), execution_policy(policy, base, 20)
        self.assertEqual((first['seed'], last['seed']), (20261001, 20261020))
        self.assertEqual((first['execution_id'], first['amendment']), ('CT-REPL-015-01', 'CT-UPDATE-015'))
        config = execution_config(base, first)
        self.assertEqual((config['seed'], config['training']['old_steps'], config['training']['update_steps']),
                         (20261001, 20000, 10000))
        self.assertEqual(seed_for_index(policy, 20), 20261020)
        for index in (0, 21, '1', True):
            with self.assertRaises(GateError):
                execution_policy(policy, base, index)
        with self.assertRaises(GateError):
            execution_policy(policy, dict(base, seed=1), 1)


class ArithmeticTests(unittest.TestCase):
    def test_patient_values_match_frozen_external_arithmetic(self):
        records = cohort_records(0, [1, 3, 2], FROZEN_MODELS)
        identity = dict(slope=1., intercept_ml=0., degenerate=False)
        frozen_patients, frozen, _ = _patient_arrays(
            records, {name: identity for name in ('old_to_new', 'new_to_old', 'old_to_continued', 'continued_to_old')})
        restricted = [dict(r, predictions={m: r['predictions'][m] for m in MODELS}) for r in records]
        patients, values = patient_values(restricted)
        self.assertEqual(patients, frozen_patients)
        for strategy in ('old_old', 'old_new', 'new_new', 'old_continued', 'continued_continued'):
            for kind in ('mae', 'signed'):
                np.testing.assert_allclose(values[f'{kind}:{strategy}'], frozen[f'{kind}:{strategy}'], rtol=0, atol=1e-12)
        for model in MODELS:
            np.testing.assert_allclose(values['dice:'+model], frozen['dice:'+model], rtol=0, atol=1e-12)
        shift = {}
        for row in restricted:
            if row['timepoint'] == 'bl':
                shift.setdefault(row['patient'], []).append(
                    row['predictions']['new']['volume_ml'] - row['predictions']['old']['volume_ml'])
        contrasts = contrast_values(values)
        np.testing.assert_allclose(contrasts['baseline_version_shift_ml'],
                                   [np.mean(shift[p]) for p in patients], rtol=0, atol=1e-12)
        np.testing.assert_allclose(contrasts['retaining_baseline_cost'],
                                   values['mae:old_new'] - values['mae:new_new'], rtol=0, atol=0)

    def test_two_way_bootstrap_reduces_to_one_way_for_constant_factors(self):
        runs, patients = resampling_weights(400, 7, 5, {'internal': 6, 'pantrack': 4})
        self.assertEqual(runs.shape, (400, 5))
        self.assertTrue(np.all(runs.sum(axis=1) == 5))
        weights = patients['internal']
        identical_runs = np.tile(np.linspace(-1, 2, 6), (5, 1))
        summary = interval_summary(identical_runs, runs, weights, family=6)
        np.testing.assert_allclose(summary['two_way_percentile_95'], summary['patient_only_percentile_95'],
                                   rtol=0, atol=1e-12)
        identical_patients = np.tile(np.array([[-.5], [.1], [.4], [.9], [1.3]]), (1, 6))
        summary = interval_summary(identical_patients, runs, weights, family=6)
        np.testing.assert_allclose(summary['two_way_percentile_95'], summary['run_only_percentile_95'],
                                   rtol=0, atol=1e-12)
        self.assertEqual((summary['runs_positive'], summary['runs_negative']), (4, 1))
        low, high = summary['two_way_percentile_95']
        adjusted_low, adjusted_high = summary['two_way_family_interval']
        self.assertTrue(adjusted_low <= low <= high <= adjusted_high)
        self.assertAlmostEqual(summary['mean'], identical_patients.mean(), places=12)
        self.assertNotIn('two_way_family_interval', interval_summary(identical_patients, runs, weights))
        repeat = resampling_weights(400, 7, 5, {'internal': 6, 'pantrack': 4})
        np.testing.assert_array_equal(repeat[0], runs)


class SummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()
        base = {'internal': cohort_records(100, INTERNAL_PAIRS), 'pantrack': cohort_records(200, PANTRACK_PAIRS)}
        cls.expected = {cohort: expected_from(rows) for cohort, rows in base.items()}
        cls.runs = {cohort: [] for cohort in COHORTS}
        for index, seed in enumerate(cls.policy['seeds'][:16]):
            for offset, cohort in enumerate(COHORTS):
                rows = cohort_records(1000*index + offset, INTERNAL_PAIRS if cohort == 'internal' else PANTRACK_PAIRS)
                cls.runs[cohort].append((seed, rows))
        cls.summary = summarize_replication(cls.runs, cls.expected, cls.policy)

    def test_summary_is_aggregate_complete_and_ordered(self):
        summary = self.summary
        self.assertEqual(summary['runs'], 16)
        self.assertEqual(summary['missing_seeds'], self.policy['seeds'][16:])
        self.assertFalse(summary['original_runs_pooled'])
        for cohort, (patients, pairs) in {'internal': (29, 277), 'pantrack': (45, 181)}.items():
            result = summary['cohorts'][cohort]
            self.assertEqual((result['patients'], result['pairs']), (patients, pairs))
            self.assertEqual(set(result['primary_contrasts']), set(PRIMARY))
            for contrast in result['primary_contrasts'].values():
                self.assertEqual(len(contrast['per_run_means']), 16)
                self.assertIn('two_way_family_interval', contrast)
            for contrast in result['secondary_contrasts'].values():
                self.assertNotIn('two_way_family_interval', contrast)
        text = json.dumps(summary)
        self.assertNotIn('SECRETPATIENT', text)
        self.assertNotIn(self.expected['internal'][0]['pair_id'], text)

    def test_fixed_order_minimum_runs_and_membership_are_enforced(self):
        short = {c: rows[:14] for c, rows in self.runs.items()}
        with self.assertRaises(GateError):
            summarize_replication(short, self.expected, self.policy)
        swapped = {c: [rows[1], rows[0], *rows[2:]] for c, rows in self.runs.items()}
        with self.assertRaises(GateError):
            summarize_replication(swapped, self.expected, self.policy)
        mismatched = dict(self.runs, pantrack=self.runs['pantrack'][:15])
        with self.assertRaises(GateError):
            summarize_replication(mismatched, self.expected, self.policy)

    def test_verification_reproduces_and_detects_tampering(self):
        receipt = verify_replication(self.summary, self.runs, self.expected, self.policy)
        self.assertGreater(receipt['numeric_values_checked'], 500)
        self.assertLess(receipt['max_absolute_difference'], 1e-9)
        tampered = copy.deepcopy(self.summary)
        tampered['cohorts']['internal']['primary_contrasts']['retaining_baseline_cost']['mean'] += 1e-6
        with self.assertRaises(GateError):
            verify_replication(tampered, self.runs, self.expected, self.policy)
        tampered = copy.deepcopy(self.summary)
        tampered['cohorts']['pantrack']['primary_contrasts']['ordinary_update_benefit']['two_way_family_interval'][0] -= 1e-6
        with self.assertRaises(GateError):
            verify_replication(tampered, self.runs, self.expected, self.policy)

    def test_coverage_requires_exact_membership_models_and_empty_convention(self):
        rows = cohort_records(5, [2, 1])
        expected = expected_from(rows)
        validate_replication_coverage(rows, expected)
        broken = [
            rows[:-1],
            [dict(rows[0], predictions={**rows[0]['predictions'], 'scratch': rows[0]['predictions']['old']}), *rows[1:]],
            [dict(rows[0], reference_volume_ml=rows[0]['reference_volume_ml'] + 1e-6), *rows[1:]],
            [dict(rows[0], predictions={**rows[0]['predictions'],
                                        'old': dict(volume_ml=0., dice=.5, hd95_mm=None, empty_prediction=True)}), *rows[1:]],
            [*rows, rows[0]],
        ]
        for records in broken:
            with self.assertRaises(GateError):
                validate_replication_coverage(records, expected)


def scoring_items(seed=3):
    rng = np.random.default_rng(seed)
    items = []
    for identity in (3, 7):
        coordinates = np.argwhere(LABELS == identity)
        centre = coordinates.mean(axis=0)
        geometry = patch_affine(AFFINE, centre, PATCH_SHAPE, (1.5, 1.5, 1.5))
        logits = {m: rng.normal(0, 2, PATCH_SHAPE).astype(np.float32) + 1.5 for m in MODELS}
        items.append(dict(key=(f'pair{identity}', 'bl'), patch_affine=geometry.tolist(), native_affine=AFFINE.tolist(),
                          lesion_id=identity, reference_voxels=int((LABELS == identity).sum()), logits=logits))
    return items


def direct(item, model):
    return native_metrics(item['logits'][model], item['patch_affine'], AFFINE, LABELS, item['lesion_id'],
                          item['reference_voxels'])


def fixed_inference(image):
    # Deterministic logits derived from the cached image, one map per model.
    return {m: (image*(i+1) - .2*i).astype(np.float32) for i, m in enumerate(MODELS)}


class ScoringTests(unittest.TestCase):
    def test_worker_scoring_equals_direct_native_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            mask = Path(tmp)/'mask.nii.gz'
            mask.write_bytes(b'synthetic native mask payload')
            items = scoring_items()
            results = score_group(dict(mask_path=str(mask), mask_sha256=sha256(mask), items=items), synthetic_loader)
            self.assertEqual([key for key, _ in results], [item['key'] for item in items])
            for (_, values), item in zip(results, items):
                for model in MODELS:
                    self.assertEqual(values[model], direct(item, model))
            with self.assertRaises(GateError):
                score_group(dict(mask_path=str(mask), mask_sha256='0'*64, items=items), synthetic_loader)
            moved = [dict(items[0], native_affine=np.diag([.8, .8, 2.5, 1.]).tolist())]
            with self.assertRaises(GateError):
                score_group(dict(mask_path=str(mask), mask_sha256=sha256(mask), items=moved), synthetic_loader)

    def test_parallel_cohort_evaluation_preserves_order_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)/'cache'
            (cache/'patches').mkdir(parents=True)
            masks = {}
            for scan in ('scanA', 'scanB'):
                masks[scan] = Path(tmp)/f'{scan}.nii.gz'
                masks[scan].write_bytes(scan.encode())
            rng = np.random.default_rng(11)
            records = []
            for number, (scan, item) in enumerate([('scanA', scoring_items(1)[0]), ('scanA', scoring_items(1)[1]),
                                                    ('scanB', scoring_items(2)[0])]):
                relative = f'patches/p{number}.npz'
                np.savez(cache/relative, image=rng.normal(0, 1, PATCH_SHAPE).astype(np.float16),
                         target=np.zeros(PATCH_SHAPE, dtype=np.uint8))
                records.append(dict(patient='SECRETPATIENT001', pair_id=f'pair{number}', scan=scan,
                                    timepoint='fu' if number % 2 else 'bl', lesion_id=item['lesion_id'],
                                    reference_volume_ml=1., reference_voxels=item['reference_voxels'],
                                    patch_file=relative, patch_sha256=sha256(cache/relative),
                                    patch_affine=item['patch_affine'], native_affine=item['native_affine']))

            def locate(record):
                return masks[record['scan']], sha256(masks[record['scan']])

            rows = evaluate_cohort(records, cache, locate, fixed_inference, 2, synthetic_loader)
            self.assertEqual([(r['pair_id'], r['timepoint']) for r in rows],
                             [(r['pair_id'], r['timepoint']) for r in records])
            for row, record in zip(rows, records):
                with np.load(cache/record['patch_file']) as payload:
                    logits = fixed_inference(payload['image'].astype(np.float32))
                for model in MODELS:
                    expected = native_metrics(logits[model], record['patch_affine'], AFFINE, LABELS,
                                              record['lesion_id'], record['reference_voxels'])
                    self.assertEqual(row['predictions'][model], expected)
            with self.assertRaises(GateError):
                evaluate_cohort([records[0], records[2], records[1]], cache, locate, fixed_inference, 2, synthetic_loader)
            changed = [dict(records[0], patch_sha256='0'*64), *records[1:]]
            with self.assertRaises(GateError):
                evaluate_cohort(changed, cache, locate, fixed_inference, 2, synthetic_loader)


if __name__ == '__main__':
    unittest.main()
