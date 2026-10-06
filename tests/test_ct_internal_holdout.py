"""Synthetic access, identity, reference and analysis tests for amendment 014."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from careport.ct_update_gate import GateError, validate_input_policy
from careport.ct_pantrack_confirm import MODELS, MAPS
from careport.ct_internal_holdout import (
    validate_holdout_policy, parse_content_manifest, make_test_records,
    validate_membership, resolve_reference_issues, validate_complete_seeds,
    summarize_holdout,
)


def contracts():
    return (json.loads(Path('config/preanalysis/ct_internal_holdout_014_2026-09-24.json').read_text()),
            json.loads(Path('config/preanalysis/ct_update_001_2026-09-15.json').read_text()))


def lesion(identity='1', clear='False', topology='UNCHANGED', merged=''):
    return dict(lesion_id=identity, topology_class=topology, linking_unclear=clear, merged_into=merged,
                img_id_bl='1', img_id_fu='2', cog_bl='[2,3,4]', cog_fu='[3,4,5]',
                volume_bl='10000', volume_fu='14000', target_lesion='False', use_for_challenge='False')


def prediction_fixture():
    expected, _ = make_test_records('0000000001', [lesion()], {'0000000001'})
    for r in expected:
        r['reference_volume_ml'] = 10. if r['timepoint'] == 'bl' else 12.
    records = []
    for r in expected:
        records.append({**r, 'predictions': {m: dict(volume_ml=14. if m == 'new' and r['timepoint']=='fu' else 10.,
                            dice=.8, hd95_mm=1., empty_prediction=False) for m in MODELS}})
    maps = {m: dict(slope=1., intercept_ml=0., degenerate=False) for m in MAPS}
    return expected, records, maps


class InternalHoldoutTests(unittest.TestCase):
    def test_policy_does_not_weaken_development_test_prohibition(self):
        policy, base = contracts()
        validate_holdout_policy(policy, base)
        self.assertFalse(base['test_values_permitted'])
        for key,value in [('training_permitted',True),('fitting_permitted',True),
                           ('seeds',[20260915]),('model_input_roles',['ct_image','reference_volume_ml'])]:
            with self.subTest(key=key), self.assertRaises(GateError):
                validate_holdout_policy(dict(policy,**{key:value}),base)
        with self.assertRaises(GateError):
            validate_holdout_policy(policy,dict(base,test_values_permitted=True))
        with self.assertRaises(GateError):
            validate_input_policy(dict(base,test_values_permitted=True))
        changed=copy.deepcopy(policy)
        changed['analysis']['primary_family_size']=3
        with self.assertRaises(GateError):
            validate_holdout_policy(changed,base)

    def test_content_manifest_rejects_path_escape_duplicates_and_omissions(self):
        raw=('a'*64+'  inputsTr/example.csv\n').encode()
        self.assertEqual(parse_content_manifest(raw, [('inputsTr/example.csv',20)]), {'inputsTr/example.csv':'a'*64})
        for bad in [raw+raw, raw.replace(b'inputsTr/example.csv',b'../example.csv'),
                    raw.replace(b'inputsTr/example.csv',b'/example.csv'), b'bad  inputsTr/example.csv\n']:
            with self.subTest(bad=bad), self.assertRaises(GateError):
                parse_content_manifest(bad,[('inputsTr/example.csv',20)])
        with self.assertRaises(GateError):
            parse_content_manifest(raw,[('inputsTr/example.csv',20),('missing',4)])

    def test_only_test_patients_with_correct_multivolume_pairing(self):
        records, counts=make_test_records('0000000001',[lesion()],{'0000000001'})
        self.assertEqual(counts['eligible'],1)
        self.assertEqual([r['image_index'] for r in records],[1,2])
        self.assertEqual([r['timepoint'] for r in records],['bl','fu'])
        self.assertEqual(len({r['pair_id'] for r in records}),1)
        self.assertTrue(all(r['split']=='test' for r in records))
        self.assertEqual(validate_membership(records,{'0000000001'}),dict(patients=1,pairs=1,timepoints=2))
        with self.assertRaises(GateError):
            make_test_records('0000000002',[lesion()],{'0000000001'})
        for bad in [records[:-1], records+[records[0]]]:
            with self.assertRaises(GateError):validate_membership(bad,{'0000000001'})
        for key,value in [('split','train'),('scan','wrong-role'),('pair_id','0'*64),('timepoint','unknown')]:
            bad=copy.deepcopy(records);bad[0][key]=value
            with self.subTest(key=key),self.assertRaises(GateError):validate_membership(bad,{'0000000001'})

    def test_unknown_links_and_merger_targets_preserve_original_exclusions(self):
        rows=[lesion(),lesion('2',clear=''),lesion('3',clear='True'),lesion('4',topology='MERGING',merged='1')]
        records,counts=make_test_records('0000000001',rows,{'0000000001'})
        self.assertEqual(records,[])
        self.assertEqual(counts['merger_destination'],1)
        self.assertEqual(counts['link_unknown'],1)
        self.assertEqual(counts['link_unclear'],1)
        with self.assertRaises(GateError):make_test_records('0000000001',[lesion(),lesion()],{'0000000001'})

    def test_only_csv_discrepancy_can_be_resolved_and_all_recounts_must_pass(self):
        good=dict(checked=2,count_mismatches=0,geometry_mismatches=0,full_vs_bounded_count_agreements=2)
        result=resolve_reference_issues({'csv_native_volume_mismatch':2},good)
        self.assertEqual(result['warnings'],{'csv_native_volume_mismatch':2})
        self.assertEqual(result['status'],'INTERNAL_REFERENCE_SCHEMA_PASS')
        for bad in [dict(good,checked=1),dict(good,count_mismatches=1),dict(good,geometry_mismatches=1),
                    dict(good,full_vs_bounded_count_agreements=1)]:
            with self.assertRaises(GateError):resolve_reference_issues({'csv_native_volume_mismatch':2},bad)
        with self.assertRaises(GateError):resolve_reference_issues({'centroid_xyz_mismatch':1},good)

    def test_complete_seed_gate_checks_states_hashes_and_coverage(self):
        policy,_=contracts()
        freeze=dict(policy_sha256='a'*64,source_manifest_sha256='b'*64,index_sha256='c'*64,
                    preprocessing_report_sha256='d'*64,patients=1,pairs=1,timepoints=2,
                    model_freeze_sha256=policy['model_freeze_sha256'],
                    checkpoint_registry_sha256=policy['checkpoint_registry_sha256'])
        states={seed:{m:'f'*64 for m in MODELS} for seed in policy['seeds']}
        receipts=[dict(status='INTERNAL_SEED_COMPLETE',protocol_id='CT-UPDATE-014',seed=seed,
                       **freeze,models=list(MODELS),model_state_sha256=states[seed],
                       private_predictions_sha256='e'*64,internal_test_values_read=True,
                       inference_only=True) for seed in policy['seeds']]
        validate_complete_seeds(receipts,policy,freeze,states)
        for bad in [receipts[:-1],receipts+[receipts[0]]]:
            with self.assertRaises(GateError):validate_complete_seeds(bad,policy,freeze,states)
        for key,value in [('index_sha256','0'*64),('pairs',2),('status','PARTIAL'),('model_state_sha256',{})]:
            bad=copy.deepcopy(receipts);bad[0][key]=value
            with self.subTest(key=key),self.assertRaises(GateError):validate_complete_seeds(bad,policy,freeze,states)

    def test_internal_analysis_and_csv_sensitivity_keep_original_references(self):
        policy,_=contracts();expected,records,maps=prediction_fixture()
        before=copy.deepcopy(records)
        result=summarize_holdout([records]*3,expected,[maps]*3,policy)
        self.assertEqual(result['cohort'],'Longitudinal-CT internal test')
        self.assertTrue(result['internal_test_values_read'])
        self.assertEqual(result['primary_family_size'],6)
        self.assertEqual(result['seed_averaged']['strategies']['old_old']['mae_ml'],2.)
        csv=result['csv_reference_sensitivity']['seed_averaged']
        self.assertEqual(csv['strategies']['old_old']['mae_ml'],4.)
        self.assertEqual(csv['strategies']['old_new']['mae_ml'],0.)
        self.assertEqual(csv['primary_contrast_points']['ordinary_update_benefit'],4.)
        self.assertEqual(records,before)
        self.assertFalse(result['csv_reference_sensitivity']['inferential_tests_performed'])

    def test_empty_masks_stay_in_errors_and_dice(self):
        policy,_=contracts();expected,records,maps=prediction_fixture()
        for r in records:
            r['predictions']['new']=dict(volume_ml=0.,dice=0.,hd95_mm=None,empty_prediction=True)
        result=summarize_holdout([records]*3,expected,[maps]*3,policy)
        self.assertEqual(result['seed_averaged']['patient_mean_followup_dice']['new'],0.)
        self.assertEqual(result['seed_averaged']['strategies']['old_new']['mae_ml'],12.)
        self.assertEqual(result['patients'],1)


if __name__=='__main__':unittest.main()
