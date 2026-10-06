"""Synthetic coverage for external-cohort metadata gates; no clinical files."""

import copy
import json
import unittest

from careport.ct_pantrack_intake import (
    IntakeError, audit_metadata, load_metadata_json, validate_access, validate_headers,
)


def fixture():
    patients = {
        'PanTrack_001': ['PanTrack_001_20200101', 'PanTrack_001_20200201'],
        'PanTrack_002': ['PanTrack_002_20200101', 'PanTrack_002_20200201'],
    }
    tracking = {}
    for patient, scans in patients.items():
        tracking[patient] = [{'51': {
            'img_bl': scans[0], 'img_fu': scans[1],
            'bl_point': [1, 2, 3], 'fu_point': [4, 5, 6],
            'fu_point_prop': None, 'merged_lesions': [51],
        }}]
    split = [{'train': patients['PanTrack_001'], 'val': patients['PanTrack_002']}]
    files = {f'images/{s}_0000.nii.gz' for scans in patients.values() for s in scans}
    files |= {f'labels/{s}.nii.gz' for scans in patients.values() for s in scans}
    return patients, tracking, split, files


class PanTrackIntakeTests(unittest.TestCase):
    def test_valid_cohort_retains_missing_propagated_prompts(self):
        report, pairs = audit_metadata(*fixture())
        self.assertEqual(report['eligible_patients'], 2)
        self.assertEqual(report['eligible_lesion_pairs'], 2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(report['suggested_split_patient_overlap'], 0)
        self.assertFalse(report['image_voxels_read'])

    def test_repeated_visits_remain_in_one_patient(self):
        patients, tracking, split, files = fixture()
        scan = 'PanTrack_001_20200301'
        patients['PanTrack_001'].append(scan)
        extra = copy.deepcopy(tracking['PanTrack_001'][0])
        extra['51']['img_bl'] = 'PanTrack_001_20200201'
        extra['51']['img_fu'] = scan
        tracking['PanTrack_001'].append(extra)
        files |= {f'images/{scan}_0000.nii.gz', f'labels/{scan}.nii.gz'}
        report, pairs = audit_metadata(patients, tracking, split, files)
        self.assertEqual(report['eligible_patients'], 2)
        self.assertEqual(report['eligible_lesion_pairs'], 3)
        self.assertEqual(sum(pair.patient == 'PanTrack_001' for pair in pairs), 2)

    def test_scan_split_overlap_is_reported_and_never_adopted(self):
        patients, tracking, _, files = fixture()
        split = [{'train': [v[0] for v in patients.values()],
                  'val': [v[1] for v in patients.values()]}]
        report, _ = audit_metadata(patients, tracking, split, files)
        self.assertEqual(report['suggested_split_patient_overlap'], 2)
        self.assertFalse(report['suggested_split_adopted'])

    def test_reversed_or_invalid_calendar_dates_fail(self):
        for value in ['PanTrack_001_20191201', 'PanTrack_001_20200231']:
            patients, tracking, split, files = fixture()
            patients['PanTrack_001'][1] = value
            with self.assertRaises(IntakeError):
                audit_metadata(patients, tracking, split, files)

    def test_cross_patient_pair_fails(self):
        patients, tracking, split, files = fixture()
        tracking['PanTrack_001'][0]['51']['img_fu'] = patients['PanTrack_002'][1]
        with self.assertRaises(IntakeError):
            audit_metadata(patients, tracking, split, files)

    def test_duplicate_or_unsafe_scan_fails(self):
        for value in ['PanTrack_001_20200101', '../PanTrack_001_20200201']:
            patients, tracking, split, files = fixture()
            patients['PanTrack_001'][1] = value
            with self.assertRaises(IntakeError):
                audit_metadata(patients, tracking, split, files)

    def test_missing_image_and_orphan_label_fail(self):
        for remove in [True, False]:
            patients, tracking, split, files = fixture()
            if remove:
                files.remove('images/PanTrack_001_20200101_0000.nii.gz')
            else:
                files.add('labels/PanTrack_003_20200101.nii.gz')
            with self.assertRaises(IntakeError):
                audit_metadata(patients, tracking, split, files)

    def test_missing_followup_is_not_a_zero_volume_observation(self):
        for point in [None, float('nan'), [float('nan')] * 3]:
            patients, tracking, split, files = fixture()
            tracking['PanTrack_001'][0]['51']['fu_point'] = point
            tracking['PanTrack_001'][0]['51']['merged_lesions'] = []
            report, pairs = audit_metadata(patients, tracking, split, files)
            self.assertEqual(report['eligible_lesion_pairs'], 1)
            self.assertEqual(report['exclusions']['missing_followup_point'], 1)
            self.assertEqual(len(pairs), 1)

    def test_mixed_nan_or_boolean_prompt_is_invalid(self):
        for point in [[1, float('nan'), 3], [True, 2, 3], [1, 2]]:
            patients, tracking, split, files = fixture()
            tracking['PanTrack_001'][0]['51']['fu_point'] = point
            with self.assertRaises(IntakeError):
                audit_metadata(patients, tracking, split, files)

    def test_documented_absent_target_is_excluded_without_zero_volume_imputation(self):
        for point in [None, float('nan'), [1, 2, 3]]:
            with self.subTest(point=point):
                patients, tracking, split, files = fixture()
                entry = tracking['PanTrack_001'][0]['51']
                entry['merged_lesions'] = [0]
                entry['fu_point'] = point
                report, pairs = audit_metadata(patients, tracking, split, files)
                self.assertEqual(report['eligible_lesion_pairs'], 1)
                self.assertEqual(report['exclusions'], {'documented_absent_followup_target': 1})
                self.assertEqual([pair.patient for pair in pairs], ['PanTrack_002'])

    def test_multiple_absent_targets_do_not_collide_with_a_persistent_lesion(self):
        patients, tracking, split, files = fixture()
        for label in ['81', '91']:
            entry = copy.deepcopy(tracking['PanTrack_001'][0]['51'])
            entry.update(merged_lesions=[0], fu_point=None)
            tracking['PanTrack_001'][0][label] = entry
        report, pairs = audit_metadata(patients, tracking, split, files)
        self.assertEqual(report['eligible_lesion_pairs'], 2)
        self.assertEqual(report['exclusions'], {'documented_absent_followup_target': 2})
        self.assertEqual(len(pairs), 2)

    def test_malformed_followup_targets_still_fail(self):
        for targets in [[-1], [0, 51], [0, 0], [51, 51], [0.0], [False], [51.0], None]:
            with self.subTest(targets=targets):
                patients, tracking, split, files = fixture()
                tracking['PanTrack_001'][0]['51']['merged_lesions'] = targets
                with self.assertRaises(IntakeError):
                    audit_metadata(patients, tracking, split, files)

    def test_merge_split_and_target_collision_are_excluded(self):
        patients, tracking, split, files = fixture()
        entry = tracking['PanTrack_001'][0]['51']
        entry['merged_lesions'] = [51, 81]
        report, _ = audit_metadata(patients, tracking, split, files)
        self.assertEqual(report['exclusions']['non_one_to_one_or_relabelled'], 1)
        entry['merged_lesions'] = [51]
        tracking['PanTrack_001'][0]['81'] = copy.deepcopy(entry)
        report, _ = audit_metadata(patients, tracking, split, files)
        self.assertEqual(report['eligible_lesion_pairs'], 1)
        self.assertEqual(report['exclusions']['target_collision'], 2)

    def test_unknown_fields_and_missing_step_fail(self):
        for unknown in [True, False]:
            patients, tracking, split, files = fixture()
            if unknown:
                tracking['PanTrack_001'][0]['51']['outcome'] = 1
            else:
                tracking['PanTrack_001'] = []
            with self.assertRaises(IntakeError):
                audit_metadata(patients, tracking, split, files)

    def test_duplicate_json_keys_and_infinity_fail(self):
        for raw in [b'{"x": 1, "x": 2}', b'{"x": Infinity}']:
            with self.assertRaises(IntakeError):
                load_metadata_json(raw)
        self.assertIsNone(load_metadata_json(b'{"x": null}')['x'])

    def test_access_gate_rejects_holdout_or_unverified_inputs(self):
        policy = {'model_input_roles': ['ct_image', 'reference_centroid'],
                  'external_role': 'evaluation_only_no_tuning',
                  'inspect_image_voxels': False, 'inspect_label_voxels': False,
                  'inspect_model_outputs': False, 'internal_test_values_permitted': False}
        transfer = {'status': 'CONTENT_HASH_PASS', 'files_verified': 330,
                    'bytes_verified': 57820749445, 'metadata_sha256': 'abc'}
        policy.update(expected_files=330, expected_bytes=57820749445,
                      expected_metadata_sha256='abc')
        validate_access(policy, transfer)
        for key in ['inspect_image_voxels', 'inspect_label_voxels',
                    'inspect_model_outputs', 'internal_test_values_permitted']:
            changed = dict(policy, **{key: True})
            with self.assertRaises(IntakeError):
                validate_access(changed, transfer)
        with self.assertRaises(IntakeError):
            validate_access(policy, dict(transfer, status='PENDING'))

    def test_header_checks_respect_xyz_and_require_paired_geometry(self):
        _, pairs = audit_metadata(*fixture())
        affine = ((1.5, 0, 0, -12), (0, 2, 0, 4), (0, 0, 3, 8), (0, 0, 0, 1))
        header = {'shape': (9, 7, 5), 'affine': affine, 'units': 'mm', 'zooms': (1.5, 2, 3)}
        headers = {scan: (header, copy.deepcopy(header))
                   for pair in pairs for scan in (pair.baseline, pair.followup)}
        # FU z=6 is outside the actual 5-voxel z axis; swapping x/z must not pass.
        with self.assertRaises(IntakeError):
            validate_headers(headers, pairs)
        for image, label in headers.values():
            image['shape'] = label['shape'] = (9, 7, 8)
        report = validate_headers(headers, pairs)
        self.assertEqual(report['scan_headers_verified'], 4)
        self.assertFalse(report['label_voxels_read'])
        key = next(iter(headers))
        different = dict(headers[key][1], affine=((1.5, 0, 0, -11), *affine[1:]))
        headers[key] = (headers[key][0], different)
        with self.assertRaises(IntakeError):
            validate_headers(headers, pairs)

    def test_unknown_units_and_singular_affines_fail(self):
        _, pairs = audit_metadata(*fixture())
        base = {'shape': (10, 10, 10), 'affine': ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)),
                'units': 'mm', 'zooms': (1,1,1)}
        for header in [dict(base, units='unknown'),
                       dict(base, affine=((1,0,0,0),(0,0,0,0),(0,0,1,0),(0,0,0,1)))]:
            headers = {scan: (header, copy.deepcopy(header))
                       for pair in pairs for scan in (pair.baseline, pair.followup)}
            with self.assertRaises(IntakeError):
                validate_headers(headers, pairs)


if __name__ == '__main__':
    unittest.main()
