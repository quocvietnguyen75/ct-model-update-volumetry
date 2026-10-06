from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from careport.ct_update_controls import interpolate_state, summarize_controls, validate_control_records
from careport.ct_update_control_run import load_checkpoint, validate_control_policy
from careport.ct_update_train import phase_plan, train_phase
from careport.ct_update_gate import GateError


def prediction_records():
    records = []
    for patient, count in [('a', 1), ('b', 3)]:
        for lesion in range(count):
            for timepoint in ('bl', 'fu'):
                followup = timepoint == 'fu'
                reference = 12. if followup else 10.
                volumes = {'old': reference, 'new': reference + 2.,
                           'continued': reference + 1., 'scratch': reference,
                           'bcwi_025': reference + 1.5, 'bcwi_050': reference + 1.,
                           'bcwi_075': reference + .5}
                if patient == 'b' and followup:
                    volumes['scratch'] += 2
                records.append(dict(patient=patient, lesion_id=lesion, timepoint=timepoint,
                                    reference_volume_ml=reference, csv_volume_mm3=reference*1000,
                                    predictions={name: dict(volume_ml=v, dice=.8, hd95_mm=1.,
                                                            empty_prediction=False) for name,v in volumes.items()}))
    return records


class CTUpdateControlsTests(unittest.TestCase):
    def test_scratch_routes_all_patients_both_stages_and_matches_optimizer_step_budgets(self):
        config={'training':{'old_steps':20000,'update_steps':10000}}
        self.assertEqual(phase_plan('old',config),(True,20000,None))
        self.assertEqual(phase_plan('new',config),(False,10000,'old'))
        self.assertEqual(phase_plan('continued',config),(True,10000,'old'))
        self.assertEqual(phase_plan('scratch_base',config),(False,20000,None))
        self.assertEqual(phase_plan('scratch',config),(False,10000,'scratch_base'))
        with self.assertRaises(GateError):
            phase_plan('unknown',config)
        for phase,parent in [('scratch',None),('old',Path('unused'))]:
            with self.assertRaises(GateError):
                train_phase(phase,config,None,[],None,parent)

    def test_control_policy_binds_seed_grid_and_source_configuration(self):
        base=json.loads(Path('config/preanalysis/ct_update_001_2026-09-15.json').read_text())
        execution=json.loads(Path('config/preanalysis/ct_update_execution_003_2026-09-15.json').read_text())
        policy=json.loads(Path('config/preanalysis/ct_update_controls_001_2026-09-16.json').read_text())
        validate_control_policy(policy,base,execution)
        for bad in [{**policy,'bcwi_old_weight_alphas':[.1]},
                    {**policy,'test_values_permitted':True},
                    {**policy,'source_config_sha256':'bad'}]:
            with self.assertRaises(GateError):
                validate_control_policy(bad,base,execution)
        with self.assertRaises(GateError):
            validate_control_policy(policy,base,{**execution,'seed':20260919})

    def test_saved_checkpoint_checks_both_receipt_and_internal_lineage_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            checkpoint=root/'scratch.pt'
            stored=dict(phase='scratch',config_sha256='config',seed=7,steps=10,
                        parent_sha256='parent',state_dict={'weight':torch.ones(2)})
            torch.save(stored,checkpoint)
            receipt={k:stored[k] for k in ('phase','config_sha256','seed','steps','parent_sha256')}
            receipt.update(status='TRAINING_COMPLETE',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            receipt_path=root/'scratch_receipt.json'
            receipt_path.write_text(json.dumps(receipt))
            state,got=load_checkpoint(root,'scratch','config','parent',7,10)
            torch.testing.assert_close(state['weight'],torch.ones(2))
            with self.assertRaises(GateError):
                load_checkpoint(root,'scratch','config','wrong-parent',7,10)
            receipt_path.write_text(json.dumps({**receipt,'seed':8}))
            with self.assertRaises(GateError):
                load_checkpoint(root,'scratch','config','parent',7,10)
            receipt_path.write_text(json.dumps(receipt))
            checkpoint.write_bytes(checkpoint.read_bytes()+b'changed')
            with self.assertRaises(GateError):
                load_checkpoint(root,'scratch','config','parent',7,10)

    def test_interpolation_matches_endpoints_and_midpoint_without_mutating_sources(self):
        old={'weight':torch.tensor([2.,4.]),'count':torch.tensor(3)}
        new={'weight':torch.tensor([6.,8.]),'count':torch.tensor(3)}
        for alpha,expected in [(0,[6.,8.]),(.5,[4.,6.]),(1,[2.,4.])]:
            state=interpolate_state(old,new,alpha)
            torch.testing.assert_close(state['weight'],torch.tensor(expected))
            state['weight'].add_(100)
        torch.testing.assert_close(old['weight'],torch.tensor([2.,4.]))
        torch.testing.assert_close(new['weight'],torch.tensor([6.,8.]))
        for bad in [{'weight':new['weight']},
                    {**new,'count':torch.tensor(4)},
                    {**new,'weight':torch.ones(3)},
                    {**new,'weight':torch.ones(2,dtype=torch.float64)},
                    {**new,'weight':torch.tensor([float('nan'),1.])}]:
            with self.subTest(bad=bad),self.assertRaises(GateError):
                interpolate_state(old,bad,.5)
        for alpha in (-.1,1.1,float('nan')):
            with self.assertRaises(GateError):
                interpolate_state(old,new,alpha)

    def test_patient_balanced_error_contrasts_and_csv_sensitivity(self):
        records=prediction_records()
        result=summarize_controls(records,200,7)
        self.assertEqual(result['patients'],2)
        self.assertEqual(result['lesions'],4)
        self.assertAlmostEqual(result['native']['strategies']['old_scratch']['mae_ml'],1.)
        self.assertAlmostEqual(result['native']['strategies']['old_new']['mae_ml'],2.)
        self.assertAlmostEqual(result['native']['candidates']['scratch']['mixed_benefit_vs_new']['mean'],1.)
        self.assertAlmostEqual(result['native']['candidates']['scratch']['refresh_benefit_vs_new']['mean'],-1.)
        self.assertEqual(result['native'],result['csv_reference_sensitivity'])
        altered=deepcopy(records)
        for r in altered:
            if r['timepoint']=='fu':
                r['csv_volume_mm3']+=1000
        other=summarize_controls(altered,200,7)
        self.assertEqual(result['native'],other['native'])
        self.assertNotEqual(other['native'],other['csv_reference_sensitivity'])

    def test_metrics_reject_extra_missing_duplicate_or_nonfinite_model_values(self):
        records=prediction_records()
        bad_nan=deepcopy(records);bad_nan[0]['predictions']['scratch']['volume_ml']=float('nan')
        for bad in [records[:-1],records+[records[0]],bad_nan]:
            with self.subTest(bad=bad),self.assertRaises(GateError):
                summarize_controls(bad,20,1)
        missing=deepcopy(records);missing[0]['predictions'].pop('scratch')
        with self.assertRaises(GateError):
            summarize_controls(missing,20,1)

    def test_boundary_coverage_retains_empty_predictions_in_dice(self):
        records=prediction_records()
        for r in records:
            if r['timepoint']=='fu' and r['patient']=='a':
                r['predictions']['scratch'].update(dice=0.,volume_ml=0.,hd95_mm=None,empty_prediction=True)
        result=summarize_controls(records,20,1)
        boundary=result['boundary_diagnostics']['scratch']
        self.assertEqual(boundary['empty_followup_predictions'],1)
        self.assertEqual(boundary['patients_with_defined_hd95'],1)
        self.assertAlmostEqual(result['patient_mean_followup_dice']['scratch'],.4)

    def test_control_records_must_exactly_match_protected_validation_keys(self):
        records=prediction_records()
        index=[{**r,'split':'val'} for r in records]
        validate_control_records(records,index)
        for bad in [records[:-1],records+[records[0]], [{**r,'patient':'unexpected'} for r in records]]:
            with self.assertRaises(GateError):
                validate_control_records(bad,index)
        with self.assertRaises(GateError):
            validate_control_records(records,[{**r,'split':'test'} for r in index])


if __name__=='__main__':
    unittest.main()
