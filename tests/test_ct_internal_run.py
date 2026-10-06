"""Synthetic integration checks for ordered internal holdout data gates."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from careport.ct_update_gate import GateError, inventory_files, metadata_fingerprint
from careport.ct_internal_run import metadata_gate, load_verified_cache, prepare
from test_ct_internal_holdout import contracts, prediction_fixture, lesion


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset_fixture(project,extra_header=False):
    root=project/'data';(root/'inputsTr').mkdir(parents=True);(root/'targetsTr').mkdir()
    splits=dict(train=[],val=[],test=['0000000001'])
    split_hash=write_json(root/'data_split.json',splits)
    header=b'lesion_id,linking_unclear'+(b',hospital_id' if extra_header else b'')
    (root/'inputsTr/0000000001.csv').write_bytes(header+b'\nSECRET,\xff\n')
    for tp in ('BL','FU'):
        (root/f'inputsTr/0000000001_{tp}_img_00.nii.gz').write_bytes(b'opaque image')
        folder='inputsTr' if tp=='BL' else 'targetsTr'
        (root/f'{folder}/0000000001_{tp}_mask_00.nii.gz').write_bytes(b'opaque label')
    files=inventory_files(root)
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name,_ in files}
    manifest=''.join(f'{v}  {k}\n' for k,v in sorted(hashes.items())).encode()
    policy,base=contracts()
    policy['nominal_test_patients']=1
    # This synthetic dataset produces two ~2.7 MB patches, not a clinical cache.
    # Production policy validation retains the original 16 GiB reservation.
    policy['storage']=dict(policy['storage'],planned_additional_bytes=16*1024*1024,
                           cache_cap_bytes=8*1024*1024,artifact_cap_bytes=8*1024*1024)
    base.update(expected_files=len(files),expected_bytes=sum(n for _,n in files),
        expected_metadata_sha256=metadata_fingerprint(files),expected_split_sha256=split_hash,
        expected_content_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        expected_split_counts=dict(train=0,val=0,test=1),
        required_csv_columns=['lesion_id'],optional_csv_columns=['linking_unclear'])
    policy['source_manifest_sha256']=base['expected_content_manifest_sha256']
    transfer=dict(status='CONTENT_HASH_PASS',content_manifest_sha256=base['expected_content_manifest_sha256'],
                  metadata_sha256=base['expected_metadata_sha256'],files_verified=len(files))
    return root,base,policy,transfer,manifest


class InternalRunTests(unittest.TestCase):
    def test_synthetic_metadata_fixture_fits_a_64_mib_temporary_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);root,base,policy,transfer,manifest=dataset_fixture(project)
            with patch('careport.ct_pantrack_run.shutil.disk_usage',return_value=SimpleNamespace(free=64*1024*1024)):
                report,_,_=metadata_gate(root,project,base,policy,transfer,manifest)
            self.assertEqual(report['status'],'INTERNAL_METADATA_PASS')

    def test_prepare_end_to_end_preserves_native_references_and_freezes_all_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);root,base,policy,transfer,_=dataset_fixture(project)
            row=lesion()
            row.update(img_id_bl='0',img_id_fu='0',cog_bl='[2.5,2.5,2.5]',cog_fu='[2.5,2.5,2.5]',
                       volume_bl='12',volume_fu='12')
            with (root/'inputsTr/0000000001.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            files=inventory_files(root)
            manifest=''.join(f'{hashlib.sha256((root/name).read_bytes()).hexdigest()}  {name}\n' for name,_ in sorted(files)).encode()
            base.update(required_csv_columns=list(row),optional_csv_columns=[],expected_bytes=sum(n for _,n in files),
                        expected_metadata_sha256=metadata_fingerprint(files),
                        expected_content_manifest_sha256=hashlib.sha256(manifest).hexdigest())
            policy['source_manifest_sha256']=base['expected_content_manifest_sha256']
            transfer.update(metadata_sha256=base['expected_metadata_sha256'],
                            content_manifest_sha256=base['expected_content_manifest_sha256'])
            control=project/'control';control.mkdir()
            policy['transfer_receipt_sha256']=write_json(control/'transfer_verified.json',transfer)
            (control/'source_files.sha256').write_bytes(manifest)
            ctx=SimpleNamespace(policy=policy,base=base,project=project,dataset=root,registry=dict(runs=[]),
                                policy_sha='a'*64,source_sha='b'*64)
            labels=np.zeros((8,8,8),dtype=np.uint8);labels[2:4,2:4,2:4]=1
            header=SimpleNamespace(get_xyzt_units=lambda:('mm','unknown'),get_zooms=lambda:(1.,1.,1.))
            def load(path):
                data=np.zeros_like(labels,dtype=np.float32) if '_img_' in str(path) else labels
                return SimpleNamespace(shape=data.shape,affine=np.eye(4),header=header,dataobj=data)
            with patch.dict('sys.modules',{'nibabel':SimpleNamespace(load=load)}):prepare(ctx)
            cache=project/policy['cache_relative']
            freeze=json.loads((project/policy['freeze_relative']).read_text())
            self.assertEqual(freeze['pairs'],1)
            self.assertFalse(freeze['internal_model_outputs_read'])
            schema=json.loads((cache/'reference_schema.json').read_text())
            self.assertEqual(schema['warnings'],{'csv_native_volume_mismatch':2})
            self.assertEqual(schema['native_recount']['checked'],2)
            self.assertEqual(json.loads((cache/'reference_schema_original.json').read_text())['status'],
                             'INTERNAL_REFERENCE_SCHEMA_FAIL')
            _,records=load_verified_cache(project,policy,freeze,{'0000000001'})
            self.assertEqual(len(records),2)
            self.assertEqual([r['reference_volume_ml'] for r in records],[.008,.008])
            self.assertTrue(all((cache/r['patch_file']).is_file() for r in records))
            with patch.dict('sys.modules',{'nibabel':SimpleNamespace(load=load)}),self.assertRaises(GateError):prepare(ctx)

    def test_metadata_reads_headers_and_hashes_but_does_not_parse_body_or_decode_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);root,base,policy,transfer,manifest=dataset_fixture(project)
            original=copy.deepcopy(base)
            result,hashes,splits=metadata_gate(root,project,base,policy,transfer,manifest)
            self.assertEqual(result['status'],'INTERNAL_METADATA_PASS')
            self.assertEqual(result['nominal_test_patients'],1)
            self.assertFalse(result['csv_body_read'])
            self.assertFalse(result['test_values_read'])
            self.assertEqual(result['test_files_hashed'],6)
            self.assertNotIn('SECRET',json.dumps(result))
            self.assertNotIn('0000000001',json.dumps(result))
            self.assertEqual(base,original)

    def test_same_size_source_change_is_caught_before_reference_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);root,base,policy,transfer,manifest=dataset_fixture(project)
            (root/'inputsTr/0000000001_BL_img_00.nii.gz').write_bytes(b'opaque IMAGE')
            with self.assertRaises(GateError):metadata_gate(root,project,base,policy,transfer,manifest)

    def test_unapproved_header_and_failed_transfer_do_not_open_test_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);root,base,policy,transfer,manifest=dataset_fixture(project,True)
            with self.assertRaises(GateError):metadata_gate(root,project,base,policy,transfer,manifest)
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);root,base,policy,transfer,manifest=dataset_fixture(project)
            with self.assertRaises(GateError):metadata_gate(root,project,base,policy,dict(transfer,status='PENDING'),manifest)

    def test_cache_requires_bound_passed_reports_and_exact_reference_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            project=Path(tmp);policy,_=contracts()
            cache=project/policy['cache_relative'];cache.mkdir(parents=True)
            expected,_,_=prediction_fixture()
            for r in expected:r['centroid']=list(r['centroid'])
            reference=cache/'reference_records_private.jsonl'
            reference.write_text('\n'.join(json.dumps(r) for r in expected)+'\n')
            refs_hash=hashlib.sha256(reference.read_bytes()).hexdigest()
            common=dict(policy_sha256='a'*64,source_manifest_sha256='b'*64)
            metadata_hash=write_json(cache/'metadata_report.json',dict(status='INTERNAL_METADATA_PASS',**common))
            reference_hash=write_json(cache/'reference_schema.json',dict(status='INTERNAL_REFERENCE_SCHEMA_PASS',
                private_manifest_sha256=refs_hash,metadata_report_sha256=metadata_hash,**common))
            rows=[dict(r,patch_file='patches/'+r['timepoint']+'.npz',patch_sha256='f'*64) for r in expected]
            index=cache/'cache_index_private.jsonl';index.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
            index_hash=hashlib.sha256(index.read_bytes()).hexdigest()
            preparation=dict(status='INTERNAL_PREPROCESSING_PASS',index_sha256=index_hash,
                reference_report_sha256=reference_hash,reference_manifest_sha256=refs_hash,metadata_report_sha256=metadata_hash,**common)
            prep_hash=write_json(cache/'preprocessing_report.json',preparation)
            freeze=dict(status='INTERNAL_COHORT_AND_MODELS_FROZEN',patients=1,pairs=1,timepoints=2,
                metadata_report_sha256=metadata_hash,reference_report_sha256=reference_hash,
                reference_manifest_sha256=refs_hash,preprocessing_report_sha256=prep_hash,index_sha256=index_hash,**common)
            _,actual=load_verified_cache(project,policy,freeze,{'0000000001'})
            self.assertEqual(actual,rows)
            with self.assertRaises(GateError):load_verified_cache(project,policy,dict(freeze,status='PENDING'),{'0000000001'})
            index.write_text(index.read_text().replace('"test"','"train"'))
            with self.assertRaises(GateError):load_verified_cache(project,policy,freeze,{'0000000001'})


if __name__=='__main__':unittest.main()
