"""Synthetic contracts for the CT-UPDATE-017 nnU-Net adapter; skipped where nnU-Net v2 is not installed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from careport.ct_update_gate import GateError

try:
    import nnunetv2  # noqa: F401
    HAVE_NNUNET = True
except ImportError:
    HAVE_NNUNET = False

ROOT = Path(__file__).resolve().parents[1]


def record(patient, lesion, timepoint, split='train', old=True, patch_file='p.npz'):
    return dict(patient=patient, lesion_id=lesion, timepoint=timepoint, split=split, old_training=old,
                patch_file=patch_file)


def write_patch(cache: Path, patient, lesion, timepoint, image, target) -> str:
    """As in the frozen cache, a patch file is named by the SHA-256 of its patient:lesion:timepoint key."""
    name = hashlib.sha256(f'{patient}:{lesion}:{timepoint}'.encode()).hexdigest() + '.npz'
    (cache/'patches').mkdir(exist_ok=True)
    np.savez(cache/'patches'/name, image=image.astype(np.float16), target=target.astype(np.uint8))
    return f'patches/{name}'


@unittest.skipUnless(HAVE_NNUNET, 'nnU-Net v2 not installed')
class DatasetTests(unittest.TestCase):
    def setUp(self):
        from careport import ct_nnunet
        self.m = ct_nnunet
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_written_case_reads_back_unchanged_with_the_nnunet_reader(self):
        from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
        rng = np.random.default_rng(1)
        image = rng.normal(size=(12, 10, 8)).astype(np.float32)
        target = (rng.random((12, 10, 8)) > .8).astype(np.uint8)
        self.m.write_case(image, target, self.tmp/'c_0000.nii.gz', self.tmp/'c.nii.gz')
        data, properties = SimpleITKIO().read_images([str(self.tmp/'c_0000.nii.gz')])
        seg, _ = SimpleITKIO().read_seg(str(self.tmp/'c.nii.gz'))
        np.testing.assert_array_equal(data[0], image)
        np.testing.assert_array_equal(seg[0], target)
        self.assertEqual(tuple(properties['spacing']), (1.5, 1.5, 1.5))

    def test_training_records_follow_the_frozen_old_and_expanded_membership(self):
        records = [record('a', 1, 'bl'), record('a', 1, 'fu'), record('b', 2, 'fu', old=False),
                   record('b', 2, 'bl', old=False), record('c', 3, 'bl', split='val'), record('c', 3, 'fu', split='val')]
        old = self.m.training_records(records, 'old')
        expanded = self.m.training_records(records, 'expanded')
        self.assertEqual([(r['patient'], r['timepoint']) for r in old], [('a', 'bl'), ('a', 'fu')])
        self.assertEqual([(r['patient'], r['timepoint']) for r in expanded],
                         [('a', 'bl'), ('a', 'fu'), ('b', 'bl'), ('b', 'fu')])
        with self.assertRaises(GateError):
            self.m.training_records(records, 'test')

    def test_dataset_description_uses_no_normalization_and_one_lesion_label(self):
        description = self.m.dataset_json(7)
        self.assertEqual(description['channel_names'], {'0': 'noNorm'})
        self.assertEqual(description['labels'], {'background': 0, 'lesion': 1})
        self.assertEqual(description['numTraining'], 7)
        self.assertEqual(self.m.dataset_name('old'), 'Dataset901_CTUpdateOld')
        self.assertEqual(self.m.dataset_name('expanded'), 'Dataset902_CTUpdateExpanded')

    def test_dataset_writing_binds_each_record_to_its_own_patch_and_records_content_hashes(self):
        cache = self.tmp/'cache'; cache.mkdir()
        cube = np.zeros((8, 8, 8)); cube[3:5, 3:5, 3:5] = 1
        keys = [('pat-a', 1, 'bl', True), ('pat-a', 1, 'fu', True), ('pat-b', 2, 'bl', False), ('pat-b', 2, 'fu', False)]
        records = [record(p, l, t, old=o, patch_file=write_patch(cache, p, l, t, cube * (k + 1), cube))
                   for k, (p, l, t, o) in enumerate(keys)]
        manifest = self.m.write_dataset(cache, records, 'expanded', self.tmp/'raw')
        folder = self.tmp/'raw'/'Dataset902_CTUpdateExpanded'
        self.assertEqual((manifest['cases'], manifest['patients']), (4, 2))
        self.assertRegex(manifest['patch_content_digest'], '^[0-9a-f]{64}$')
        names = sorted(p.name for p in (folder/'imagesTr').iterdir())
        self.assertEqual(names[0], 'ctpatch_00000_0000.nii.gz')
        self.assertFalse(any('pat-' in n for n in names))
        self.assertNotIn('pat-', json.dumps(manifest))
        # The same patches give the same digest; a record pointing at another lesion's patch is refused.
        self.assertEqual(self.m.write_dataset(cache, records, 'expanded', self.tmp/'raw2')['patch_content_digest'],
                         manifest['patch_content_digest'])
        swapped = [dict(r) for r in records]
        swapped[0]['patch_file'], swapped[1]['patch_file'] = records[1]['patch_file'], records[0]['patch_file']
        with self.assertRaises(GateError):
            self.m.write_dataset(cache, swapped, 'expanded', self.tmp/'raw3')

    def test_folder_digest_covers_names_sizes_and_contents(self):
        folder = self.tmp/'f'; (folder/'a').mkdir(parents=True)
        (folder/'a'/'x.bin').write_bytes(b'one'); (folder/'y.bin').write_bytes(b'two')
        first = self.m.folder_digest(folder)
        self.assertEqual(first['files'], 2)
        self.assertEqual(self.m.folder_digest(folder), first)
        (folder/'y.bin').write_bytes(b'tw0')
        self.assertNotEqual(self.m.folder_digest(folder)['sha256'], first['sha256'])


@unittest.skipUnless(HAVE_NNUNET, 'nnU-Net v2 not installed')
class CorrectionHelperTests(unittest.TestCase):
    """CT-UPDATE-017 correction A: the predictor must hold the trained weights, checked before any evaluation."""

    def setUp(self):
        from careport import ct_nnunet
        self.m = ct_nnunet
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_weights_match_only_when_every_tensor_and_name_agrees(self):
        import torch
        network = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 3), torch.nn.InstanceNorm3d(2, affine=True))
        state = {k: v.clone() for k, v in network.state_dict().items()}
        self.assertTrue(self.m.weights_match(network, state))
        changed = dict(state, **{'0.weight': state['0.weight'] + 1})
        self.assertFalse(self.m.weights_match(network, changed))
        self.assertFalse(self.m.weights_match(network, {k: v for k, v in state.items() if k != '0.bias'}))

    def test_binary_dice_counts_overlap_of_two_masks(self):
        prediction = np.zeros((4, 4, 4), bool)
        target = np.zeros((4, 4, 4), np.uint8)
        prediction[:2] = True
        target[1:3] = 1
        self.assertAlmostEqual(self.m.binary_dice(prediction, target), 2 * 16 / (32 + 32))
        self.assertEqual(self.m.binary_dice(np.zeros_like(prediction), target), 0.0)
        with self.assertRaises(GateError):
            self.m.binary_dice(prediction, np.zeros_like(target))

    def test_sanity_patches_are_the_first_nonempty_old_training_patches_in_membership_order(self):
        rng = np.random.default_rng(3)
        records = []
        for patient, lesion, old, filled in (('a', 1, True, False), ('b', 2, True, True), ('c', 3, False, True),
                                             ('d', 4, True, True), ('e', 5, True, True)):
            for timepoint in ('bl', 'fu'):
                target = np.zeros((6, 6, 6), np.uint8)
                if filled and timepoint == 'fu':
                    target[2:4, 2:4, 2:4] = 1
                name = write_patch(self.tmp, patient, lesion, timepoint, rng.normal(size=(6, 6, 6)), target)
                records.append(record(patient, lesion, timepoint, old=old, patch_file=name))
        chosen = self.m.sanity_patches(self.tmp, records, 2)
        self.assertEqual([(r['patient'], r['timepoint']) for r, _, _ in chosen], [('b', 'fu'), ('d', 'fu')])
        self.assertTrue(all(t.any() and image.dtype == np.float32 for _, image, t in chosen))
        with self.assertRaises(GateError):
            self.m.sanity_patches(self.tmp, records, 4)


@unittest.skipUnless(HAVE_NNUNET, 'nnU-Net v2 not installed')
class EndToEndTests(unittest.TestCase):
    def test_plan_train_transfer_and_predict_on_a_tiny_synthetic_dataset(self):
        """Runs in a subprocess, because nnU-Net reads its folder settings when it is first imported."""
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, nnUNet_raw=str(Path(tmp)/'raw'), nnUNet_preprocessed=str(Path(tmp)/'pre'),
                       nnUNet_results=str(Path(tmp)/'results'), nnUNet_n_proc_DA='0', nnUNet_compile='false',
                       PYTHONPATH=os.pathsep.join([str(ROOT/'src')] + sys.path))
            completed = subprocess.run([sys.executable, '-m', 'careport.ct_nnunet', 'synthetic-check', '--workdir', tmp],
                                       env=env, capture_output=True, text=True, timeout=900)
            self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
            report = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(report['status'], 'NNUNET_SYNTHETIC_CHECK_PASS')
        self.assertTrue(report['all_weights_transferred'])
        self.assertTrue(report['segmentation_head_transferred'])
        # Correction A: the predictor must carry the checkpoint's weights, not a fresh initialization.
        self.assertTrue(report['predictor_weights_loaded'])
        self.assertEqual(report['logit_shape'], [32, 32, 32])
        self.assertTrue(report['threshold_matches_softmax'])
        self.assertEqual(report['epochs'], dict(old=1, new=1, continued=1))
        self.assertEqual(report['epoch_seconds_logged'], 1)


if __name__ == '__main__':
    unittest.main()
