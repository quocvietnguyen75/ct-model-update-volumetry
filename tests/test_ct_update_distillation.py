from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from careport.ct_update_distillation import (
    binary_distillation_loss, distillation_step, frozen_teacher,
    load_distilled_checkpoint, validate_distillation_policy,
)
from careport.ct_update_controls import summarize_controls, validate_control_records
from careport.ct_update_gate import GateError


class CTUpdateDistillationTests(unittest.TestCase):
    def test_binary_kl_direction_temperature_scaling_and_teacher_detachment(self):
        temperature = 2.
        student = torch.tensor([[[[temperature * math.log(1.5)]]]], requires_grad=True)
        teacher = torch.tensor([[[[-temperature * math.log(3.)]]]], requires_grad=True)
        loss = binary_distillation_loss(student, teacher, temperature)
        expected = temperature**2 * (.25 * math.log(.25/.6) + .75 * math.log(.75/.4))
        self.assertAlmostEqual(loss.item(), expected, places=6)
        loss.backward()
        self.assertAlmostEqual(student.grad.item(), temperature * (.6 - .25), places=6)
        self.assertIsNone(teacher.grad)
        equal = binary_distillation_loss(student.detach(), student.detach(), temperature)
        self.assertEqual(equal.item(), 0.)

    def test_kl_remains_finite_for_confident_opposite_predictions_and_rejects_invalid_inputs(self):
        student = torch.tensor([[[[-1000., 1000.]]]], requires_grad=True)
        teacher = -student.detach()
        loss = binary_distillation_loss(student, teacher, 2.)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(student.grad).all())
        for temperature in (0., -1., float('nan')):
            with self.assertRaises(GateError):
                binary_distillation_loss(student, teacher, temperature)
        for bad in [torch.zeros(2), torch.full_like(student, float('nan'))]:
            with self.assertRaises(GateError):
                binary_distillation_loss(student, bad, 2.)

    def test_teacher_is_an_independent_frozen_copy_and_receives_the_same_image(self):
        class RecordingConv(torch.nn.Conv2d):
            def forward(self, image):
                self.seen = image
                return super().forward(image)
        student = RecordingConv(1, 1, 1)
        teacher = frozen_teacher(student)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        before = deepcopy(teacher.state_dict())
        image = torch.tensor([[[[1., 2.], [3., 4.]]]])
        target = torch.ones_like(image)
        optimizer = torch.optim.SGD(student.parameters(), lr=.1)
        report = distillation_step(student, teacher, image, target,
                                   torch.nn.BCEWithLogitsLoss(), optimizer, 1., 2.)
        self.assertIs(student.seen, teacher.seen)
        self.assertTrue(math.isfinite(report['total_loss']))
        self.assertTrue(any(not torch.equal(student.state_dict()[k], v) for k, v in before.items()))
        for key, value in before.items():
            torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)
            self.assertNotEqual(teacher.state_dict()[key].data_ptr(), student.state_dict()[key].data_ptr())
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        teacher.train()
        with self.assertRaises(GateError):
            distillation_step(student, teacher, image, target,
                              torch.nn.BCEWithLogitsLoss(), optimizer, 1., 2.)

    def test_zero_weight_preserves_the_supervised_optimizer_step(self):
        first = torch.nn.Conv2d(1, 1, 1)
        second = deepcopy(first)
        teacher = frozen_teacher(first)
        image = torch.ones(1, 1, 2, 2)
        target = torch.zeros_like(image)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(first.parameters(), lr=.01)
        other_optimizer = torch.optim.SGD(second.parameters(), lr=.01)
        distillation_step(first, teacher, image, target, criterion, optimizer, 0., 2.)
        other_optimizer.zero_grad(set_to_none=True)
        criterion(second(image), target).backward()
        other_optimizer.step()
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)

    def test_policy_binds_the_registered_teacher_strengths_and_seed(self):
        base = json.loads(Path('config/preanalysis/ct_update_001_2026-09-15.json').read_text())
        execution = json.loads(Path('config/preanalysis/ct_update_execution_002_2026-09-15.json').read_text())
        policy = json.loads(Path('config/preanalysis/ct_update_distillation_001_2026-09-16.json').read_text())
        validate_distillation_policy(policy, base, execution)
        for bad in [{**policy, 'temperature': 3.}, {**policy, 'test_values_permitted': True},
                    {**policy, 'candidates': {'kd_001': 2.}},
                    {**policy, 'teacher_input': 'other_timepoint'}]:
            with self.assertRaises(GateError):
                validate_distillation_policy(bad, base, execution)
        with self.assertRaises(GateError):
            validate_distillation_policy(policy, base, {**execution, 'seed': 20260919})

    def test_distilled_checkpoint_binds_policy_and_teacher_in_both_internal_and_receipt_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fields = dict(phase='kd_001', config_sha256='config', parent_sha256='old',
                          seed=7, steps=10000, teacher_sha256='old',
                          distillation_policy_sha256='policy', coefficient=1., temperature=2.)
            path = root/'kd_001.pt'
            torch.save({**fields, 'state_dict': {'w': torch.ones(1)}}, path)
            receipt = {**fields, 'status': 'TRAINING_COMPLETE', 'teacher_unchanged': True,
                       'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            (root/'kd_001_receipt.json').write_text(json.dumps(receipt))
            args = (root, 'kd_001', 'config', 'old', 7, 'policy', 1., 2.)
            state, _ = load_distilled_checkpoint(*args)
            self.assertEqual(state['w'].item(), 1.)
            (root/'kd_001_receipt.json').write_text(json.dumps({**receipt, 'teacher_sha256': 'wrong'}))
            with self.assertRaises(GateError):
                load_distilled_checkpoint(*args)
            torch.save({**fields, 'distillation_policy_sha256': 'wrong',
                        'state_dict': {'w': torch.ones(1)}}, path)
            receipt['checkpoint_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root/'kd_001_receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(GateError):
                load_distilled_checkpoint(*args)

    def test_candidate_metrics_preserve_patient_weighting_and_exact_validation_cohort(self):
        from test_ct_update_controls import prediction_records
        records = prediction_records()
        candidates = ('kd_001', 'kd_010')
        for row in records:
            old = row['predictions']
            row['predictions'] = {m: old[m] for m in ['old', 'new', 'continued']}
            row['predictions'].update(kd_001=old['scratch'], kd_010=old['bcwi_025'])
        index = [{**r, 'split': 'val'} for r in records]
        validate_control_records(records, index, candidates=candidates)
        result = summarize_controls(records, 20, 7, candidates=candidates)
        self.assertEqual(set(result['native']['candidates']), set(candidates))
        self.assertAlmostEqual(result['native']['strategies']['old_kd_001']['mae_ml'], 1.)
        self.assertEqual(len(result['native']['strategies']), 9)
        with self.assertRaises(GateError):
            validate_control_records(records[:-1], index, candidates=candidates)
        for bad in [('old',), ('kd_001', 'kd_001'), ()]:
            with self.assertRaises(GateError):
                summarize_controls(records, 20, 7, candidates=bad)


if __name__ == '__main__':
    unittest.main()
