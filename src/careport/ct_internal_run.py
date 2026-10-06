"""Execute ordered CT-UPDATE-014 gates, fixed-model inference and complete analysis."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from careport.ct_internal_holdout import (
    validate_holdout_policy, parse_content_manifest, make_test_records, validate_membership,
    resolve_reference_issues, validate_complete_seeds, summarize_holdout,
)
from careport.ct_pantrack_confirm import MODELS, validate_prediction_coverage
from careport.ct_pantrack_run import checked_path, digest, read_bound_json, write_json, write_records, storage_gate, load_states
from careport.ct_update_data import inspect_volume, volume_paths
from careport.ct_update_gate import GateError, audit_metadata, inventory_files
from careport.ct_update_preprocess import extract_volume, projected_cache_bytes
from careport.ct_update_recount import independent_count


def metadata_gate(dataset: Path, project: Path, base: dict, policy: dict,
                  transfer: dict, manifest: bytes) -> tuple[dict,dict,dict]:
    storage = storage_gate(project,policy)
    report = audit_metadata(dataset,{**base,'storage':policy['storage']},transfer,
                            storage['current_study_bytes'],storage['shared_free_bytes'])
    if hashlib.sha256(manifest).hexdigest() != policy['source_manifest_sha256']:
        raise GateError('Internal source manifest differs from the verified transfer')
    files = inventory_files(dataset)
    hashes = parse_content_manifest(manifest,files)
    splits = json.loads((dataset/'data_split.json').read_bytes())
    if len(splits['test']) != policy['nominal_test_patients']:
        raise GateError('Nominal internal holdout population changed')
    selected = [(name,size) for name,size in files if name=='data_split.json' or Path(name).name[:10] in set(splits['test'])]
    for name,_ in selected:
        if digest(checked_path(dataset,name)) != hashes[name]:
            raise GateError('A selected internal source file changed since transfer')
    report.update(status='INTERNAL_METADATA_PASS',protocol_id=policy['protocol_id'],
                  nominal_test_patients=len(splits['test']),test_files_hashed=len(selected),
                  test_file_bytes=sum(size for _,size in selected),
                  test_csv_bytes=sum(size for name,size in selected if name.endswith('.csv')),
                  source_manifest_sha256=policy['source_manifest_sha256'],source_file_hashes_verified=True)
    return report,hashes,splits


def load_verified_cache(project: Path, policy: dict, freeze: dict, test_patients: set[str]) -> tuple[Path,list[dict]]:
    if freeze.get('status') != 'INTERNAL_COHORT_AND_MODELS_FROZEN':
        raise GateError('Internal outcome access requires a completed cohort and model freeze')
    cache = checked_path(project,policy['cache_relative'])
    metadata = read_bound_json(cache/'metadata_report.json',freeze['metadata_report_sha256'])
    reference = read_bound_json(cache/'reference_schema.json',freeze['reference_report_sha256'])
    prep = read_bound_json(cache/'preprocessing_report.json',freeze['preprocessing_report_sha256'])
    for report,status in [(metadata,'INTERNAL_METADATA_PASS'),(reference,'INTERNAL_REFERENCE_SCHEMA_PASS'),
                          (prep,'INTERNAL_PREPROCESSING_PASS')]:
        if (report.get('status') != status or report.get('policy_sha256') != freeze['policy_sha256']
                or report.get('source_manifest_sha256') != freeze['source_manifest_sha256']):
            raise GateError('Internal ordered-stage receipt failed or belongs to another execution')
    if (reference['metadata_report_sha256'] != freeze['metadata_report_sha256']
            or reference['private_manifest_sha256'] != freeze['reference_manifest_sha256']
            or any(prep.get(k) != freeze[k] for k in ('metadata_report_sha256','reference_report_sha256',
                                                     'reference_manifest_sha256','index_sha256'))):
        raise GateError('Internal reference and preparation receipts disagree')
    paths = [cache/'reference_records_private.jsonl',cache/'cache_index_private.jsonl']
    raw = [p.read_bytes() for p in paths]
    for data,key in zip(raw,['reference_manifest_sha256','index_sha256']):
        if hashlib.sha256(data).hexdigest() != freeze[key]:
            raise GateError('Internal private reference/cache manifest changed')
    references,records = [[json.loads(line) for line in data.splitlines()] for data in raw]
    counts = validate_membership(records,test_patients)
    if any(counts[k] != freeze[k] for k in counts):
        raise GateError('Internal cohort counts changed')
    key = lambda r:(r['pair_id'],r['timepoint'])
    expected,actual = {key(r):r for r in references},{key(r):r for r in records}
    if len(expected)!=len(references) or len(actual)!=len(records) or set(expected)!=set(actual):
        raise GateError('Internal patch coverage differs from native references')
    for identity,row in expected.items():
        if any(actual[identity].get(k)!=v for k,v in row.items()):
            raise GateError('Internal cache changed a frozen native reference field')
        checked_path(cache,actual[identity]['patch_file'])
    return cache,records


def load_context(args) -> SimpleNamespace:
    policy = read_bound_json(args.policy,os.environ['CT_INTERNAL_POLICY_SHA256'])
    code = Path.cwd().resolve()
    base = read_bound_json(code/policy['source_config_file'],policy['source_config_sha256'])
    validate_holdout_policy(policy,base)
    if digest(code/policy['amendment_file']) != policy['amendment_sha256']:
        raise GateError('Internal scientific amendment changed')
    source_sha = digest(code/'EXECUTION_SHA256SUMS')
    if source_sha != os.environ['CT_INTERNAL_SOURCE_SHA256']:
        raise GateError('Internal source manifest differs from submitted execution')
    project = Path(os.environ[policy['project_root_env']]).resolve()
    dataset = Path(os.environ[policy['dataset_root_env']]).resolve()
    if not dataset.is_relative_to(project) or dataset==project:
        raise GateError('Internal dataset is outside its study storage root')
    registry = read_bound_json(code/policy['checkpoint_registry_file'],policy['checkpoint_registry_sha256'])
    maps = read_bound_json(project/'control/pantrack_models_010.json',policy['model_freeze_sha256'])
    if (maps['status']!='EXTERNAL_MODELS_AND_CALIBRATION_FROZEN'
            or maps['checkpoint_registry_sha256']!=policy['checkpoint_registry_sha256']
            or maps['internal_test_values_read'] is not False
            or [r['seed'] for r in registry['runs']] != policy['seeds']
            or [r['seed'] for r in maps['runs']] != policy['seeds']):
        raise GateError('Internal model or development-map lineage changed')
    states = {}
    for seed in policy['seeds']:
        receipt = read_bound_json(project/f'runs/pantrack_010_seed_{seed}/seed_complete.json',
                                  policy['reference_seed_receipt_sha256'][str(seed)])
        if (receipt['status']!='EXTERNAL_SEED_COMPLETE' or receipt['seed']!=seed
                or receipt['model_freeze_sha256']!=policy['model_freeze_sha256']
                or receipt['checkpoint_registry_sha256']!=policy['checkpoint_registry_sha256']
                or set(receipt['model_state_sha256'])!=set(MODELS)):
            raise GateError('Previously frozen nine-model state receipt is incomplete')
        states[seed] = receipt['model_state_sha256']
    return SimpleNamespace(policy=policy,base=base,code=code,project=project,dataset=dataset,
                           registry=registry,maps=maps,states=states,policy_sha=digest(args.policy),source_sha=source_sha)


def common(ctx) -> dict:
    return dict(protocol_id=ctx.policy['protocol_id'],policy_sha256=ctx.policy_sha,
                source_manifest_sha256=ctx.source_sha)


def prepare(ctx) -> None:
    import nibabel as nib
    policy,project,dataset = ctx.policy,ctx.project,ctx.dataset
    output = checked_path(project,policy['cache_relative'])
    freeze_path = checked_path(project,policy['freeze_relative'])
    if output.exists() or freeze_path.exists():
        raise GateError('Internal preparation/freeze already exists; inspect preserved artifacts')
    # All trained files are bound before any held-out reference values are read.
    for run in ctx.registry['runs']:
        for artifact in run['checkpoints'].values():
            if digest(checked_path(project,artifact['path'])) != artifact['receipt']['checkpoint_sha256']:
                raise GateError('Internal checkpoint file differs from registered development state')
    transfer = read_bound_json(project/'control/transfer_verified.json',policy['transfer_receipt_sha256'])
    metadata,hashes,splits = metadata_gate(dataset,project,ctx.base,policy,transfer,
                                          (project/'control/source_files.sha256').read_bytes())
    metadata.update(common(ctx))
    metadata['raw_content_manifest_sha256'] = policy['source_manifest_sha256']
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'metadata_report.json',metadata)
    print(json.dumps(dict(status=metadata['status'],test_files_hashed=metadata['test_files_hashed'],
                          test_file_bytes=metadata['test_file_bytes'])),flush=True)
    test_patients = set(splits['test'])
    candidates, exclusions = [],Counter()
    for patient in sorted(test_patients):
        with (dataset/'inputsTr'/f'{patient}.csv').open(newline='',encoding='utf-8-sig') as stream:
            records,counts = make_test_records(patient,list(csv.DictReader(stream)),test_patients)
        candidates.extend(records);exclusions.update(counts)
    cohort = validate_membership(candidates,test_patients)
    groups = defaultdict(list)
    for record in candidates:
        groups[(record['patient'],record['timepoint'],record['image_index'])].append(record)
    references,issues = [],Counter()
    tasks = []
    for number,((patient,tp,index),entries) in enumerate(sorted(groups.items())):
        task = dict(patient=patient,split='test',timepoint=tp,index=index,entries=entries,root=str(dataset))
        audited = inspect_volume(task)
        image_path,mask_path = volume_paths(dataset,patient,tp,index)
        for row in audited['records']:
            row.update(image_relative=image_path.relative_to(dataset).as_posix(),
                       mask_relative=mask_path.relative_to(dataset).as_posix(),
                       native_image_sha256=hashes[image_path.relative_to(dataset).as_posix()],
                       native_mask_sha256=hashes[mask_path.relative_to(dataset).as_posix()])
        references.extend(audited['records']);issues.update(audited['issues']);tasks.append(audited)
        if (number+1)%10==0:print(json.dumps(dict(reference_volumes_checked=number+1,total=len(groups))),flush=True)
    references.sort(key=lambda r:(r['scan'],r['lesion_id']))
    write_records(output/'reference_records_private.jsonl',references)
    reference_sha = digest(output/'reference_records_private.jsonl')
    original = dict(**common(ctx),status='INTERNAL_REFERENCE_SCHEMA_FAIL' if issues else 'INTERNAL_REFERENCE_SCHEMA_PASS',
                    **cohort,nominal_test_patients=len(test_patients),
                    patients_without_eligible_pairs=len(test_patients)-cohort['patients'],
                    exclusions_and_row_counts=dict(exclusions),issues=dict(issues),
                    private_manifest_sha256=reference_sha,
                    metadata_report_sha256=digest(output/'metadata_report.json'),
                    internal_test_references_read=True,internal_model_outputs_read=False)
    write_json(output/'reference_schema_original.json',original)
    recount = dict(checked=0,count_mismatches=0,geometry_mismatches=0,full_vs_bounded_count_agreements=0)
    if set(issues)=={'csv_native_volume_mismatch'}:
        for audited in tasks:
            flagged = [r for r in audited['records'] if abs(r['reference_volume_ml']*1000-r['csv_volume_mm3'])>
                       max(r['reference_volume_ml']*1000/r['reference_voxels'],.005*r['reference_volume_ml']*1000)]
            if not flagged:continue
            path = checked_path(dataset,flagged[0]['mask_relative'])
            if digest(path)!=flagged[0]['native_mask_sha256']:raise GateError('Mask changed before internal native recount')
            volume = nib.load(path);labels=np.asanyarray(volume.dataobj)
            for record in flagged:
                measured=independent_count(labels,volume.affine,volume.header.get_zooms(),record)
                recount['checked']+=1
                recount['count_mismatches']+=not measured['count_matches']
                recount['geometry_mismatches']+=not measured['physical_volume_matches']
                recount['full_vs_bounded_count_agreements']+=measured['box_count_matches']
    resolution = resolve_reference_issues(dict(issues),recount)
    if validate_membership(references,test_patients)!=cohort:raise GateError('Native reference audit lost eligible records')
    schema = {**original,**resolution,'issues':{},'native_recount':recount,
              'original_report_sha256':digest(output/'reference_schema_original.json')}
    write_json(output/'reference_schema.json',schema)
    print(json.dumps(dict(status=schema['status'],**cohort,warnings=schema['warnings'])),flush=True)
    projected = projected_cache_bytes(len(references),policy['patch']['shape'],policy['storage']['cache_cap_bytes'])
    storage = storage_gate(project,policy)
    (output/'patches').mkdir()
    counts,cache_records = Counter(),[]
    for number,audited in enumerate(tasks):
        for record in audited['records'][:1]:
            for path_key,hash_key in [('image_relative','native_image_sha256'),('mask_relative','native_mask_sha256')]:
                if digest(checked_path(dataset,record[path_key]))!=record[hash_key]:
                    raise GateError('Native source changed between reference and preprocessing gates')
        task = dict(audited,root=str(dataset),output=str(output),patch_policy=policy['patch'])
        extracted = extract_volume(task)
        counts.update(extracted['counts'])
        for row in extracted['records']:
            row['patch_sha256']=digest(checked_path(output,row['patch_file']))
            cache_records.append(row)
        if counts['bytes']>projected:raise GateError('Internal cache exceeded the frozen storage projection')
        if (number+1)%10==0:print(json.dumps(dict(patches_written=len(cache_records),total=len(references))),flush=True)
    cache_records.sort(key=lambda r:(r['scan'],r['lesion_id']))
    write_records(output/'cache_index_private.jsonl',cache_records)
    prep = dict(**common(ctx),status='INTERNAL_PREPROCESSING_PASS',**cohort,counts=dict(counts),
                projected_cache_bytes=projected,storage=storage,
                metadata_report_sha256=digest(output/'metadata_report.json'),
                reference_report_sha256=digest(output/'reference_schema.json'),reference_manifest_sha256=reference_sha,
                index_sha256=digest(output/'cache_index_private.jsonl'),
                internal_test_references_read=True,internal_model_outputs_read=False)
    write_json(output/'preprocessing_report.json',prep)
    freeze = dict(**common(ctx),status='INTERNAL_COHORT_AND_MODELS_FROZEN',**cohort,
                  frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  preprocessing_report_sha256=digest(output/'preprocessing_report.json'),
                  model_freeze_sha256=policy['model_freeze_sha256'],
                  checkpoint_registry_sha256=policy['checkpoint_registry_sha256'],
                  expected_split_sha256=ctx.base['expected_split_sha256'],
                  internal_model_outputs_read=False)
    for key in ('metadata_report_sha256','reference_report_sha256','reference_manifest_sha256','index_sha256'):
        freeze[key]=prep[key]
    load_verified_cache(project,policy,freeze,test_patients)
    noncache_bytes=sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if noncache_bytes>policy['storage']['artifact_cap_bytes']:raise GateError('Internal metadata outputs exceeded their cap')
    write_json(freeze_path,freeze)
    print(json.dumps(dict(status=freeze['status'],**cohort,freeze_sha256=digest(freeze_path),
                          cache_bytes=counts['bytes'],internal_model_outputs_read=False)),flush=True)


def evaluation_inputs(ctx) -> tuple[dict,Path,list[dict]]:
    freeze = read_bound_json(checked_path(ctx.project,ctx.policy['freeze_relative']),os.environ['CT_INTERNAL_COHORT_FREEZE_SHA256'])
    for key,value in [('policy_sha256',ctx.policy_sha),('source_manifest_sha256',ctx.source_sha),
                      ('model_freeze_sha256',ctx.policy['model_freeze_sha256']),
                      ('checkpoint_registry_sha256',ctx.policy['checkpoint_registry_sha256']),
                      ('expected_split_sha256',ctx.base['expected_split_sha256'])]:
        if freeze.get(key)!=value:raise GateError('Internal execution freeze changed or belongs to another source')
    split = read_bound_json(ctx.dataset/'data_split.json',ctx.base['expected_split_sha256'])
    storage_gate(ctx.project,ctx.policy)
    cache,records = load_verified_cache(ctx.project,ctx.policy,freeze,set(split['test']))
    return freeze,cache,records


def evaluate(ctx, seed: int) -> None:
    import nibabel as nib
    import torch
    from careport.ct_update_control_run import state_fingerprint
    from careport.ct_update_evaluate import native_metrics
    from careport.ct_update_train import make_model, seed_runtime
    freeze,cache,records = evaluation_inputs(ctx)
    selected = [r for r in ctx.registry['runs'] if r['seed']==seed]
    if len(selected)!=1:raise GateError('Unregistered internal inference seed')
    states,config = load_states(ctx.project,ctx.code,ctx.registry,selected[0])
    fingerprints = {m:state_fingerprint(s) for m,s in states.items()}
    if fingerprints!=ctx.states[seed]:raise GateError('Internal inference states differ from external frozen models')
    output = checked_path(ctx.project,f'runs/internal_holdout_014_seed_{seed}')
    output.mkdir(parents=True,exist_ok=False)
    if not torch.cuda.is_available() or 'H100' not in torch.cuda.get_device_name(0):
        raise GateError('Internal evaluation requires one allocated H100')
    seed_runtime(seed)
    models = {}
    for name in MODELS:
        model=make_model(config);model.load_state_dict(states[name],strict=True)
        models[name]=model.cuda().eval().requires_grad_(False)
    del states
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        image=torch.zeros((1,1,*ctx.policy['patch']['shape']),device='cuda')
        for model in models.values():
            prediction=model(image)
            if prediction.shape!=image.shape or not torch.isfinite(prediction).all():
                raise GateError('Internal synthetic GPU inference failed')
    print(json.dumps(dict(status='INTERNAL_GPU_RUNTIME_PASS',seed=seed,models=len(models))),flush=True)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    predictions=[];previous=None
    with (output/'predictions_private.jsonl').open('x',encoding='utf-8') as stream:
        for number,record in enumerate(records):
            path=checked_path(cache,record['patch_file'])
            if digest(path)!=record['patch_sha256']:raise GateError('Internal cached patch changed')
            with np.load(path,allow_pickle=False) as payload:
                image=torch.from_numpy(payload['image'].astype(np.float32)[None,None]).cuda()
            if record['scan']!=previous:
                path=checked_path(ctx.dataset,record['mask_relative'])
                if digest(path)!=record['native_mask_sha256']:raise GateError('Internal native reference changed before inference')
                volume=nib.load(path);labels=np.asanyarray(volume.dataobj);affine=volume.affine
                previous=record['scan']
            if not np.allclose(affine,record['native_affine'],rtol=0,atol=1e-4):raise GateError('Internal inference geometry changed')
            measured={}
            for name,model in models.items():
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(image)[0,0].float().cpu().numpy()
                measured[name]=native_metrics(logits,record['patch_affine'],affine,labels,
                                             record['lesion_id'],record['reference_voxels'])
            row={k:record[k] for k in ('patient','pair_id','scan','timepoint','lesion_id','reference_volume_ml')}
            row['predictions']=measured;predictions.append(row)
            stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            if stream.tell()>ctx.policy['storage']['artifact_cap_bytes']:raise GateError('Internal prediction artifact exceeded its cap')
            if (number+1)%10==0:print(json.dumps(dict(stage='inference',seed=seed,completed=number+1,total=len(records))),flush=True)
    validate_prediction_coverage(predictions,records)
    receipt=dict(**common(ctx),status='INTERNAL_SEED_COMPLETE',seed=seed,models=list(MODELS),
                 model_state_sha256=fingerprints,private_predictions_sha256=digest(output/'predictions_private.jsonl'),
                 elapsed_seconds=time.monotonic()-started,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                 internal_test_values_read=True,inference_only=True)
    for key in ('patients','pairs','timepoints','index_sha256','preprocessing_report_sha256',
                'model_freeze_sha256','checkpoint_registry_sha256'):
        receipt[key]=freeze[key]
    write_json(output/'seed_complete.json',receipt)
    print(json.dumps(dict(status=receipt['status'],seed=seed,timepoints=len(records),
                          receipt_sha256=digest(output/'seed_complete.json'))),flush=True)


def analyze(ctx) -> None:
    freeze,_,expected = evaluation_inputs(ctx)
    folders=[ctx.project/f'runs/internal_holdout_014_seed_{s}' for s in ctx.policy['seeds']]
    receipts=[json.loads((p/'seed_complete.json').read_bytes()) for p in folders]
    validate_complete_seeds(receipts,ctx.policy,freeze,ctx.states)
    for folder,receipt in zip(folders,receipts):
        if digest(folder/'predictions_private.jsonl')!=receipt['private_predictions_sha256']:
            raise GateError('Internal completed prediction payload changed')
    records=[[json.loads(line) for line in (p/'predictions_private.jsonl').read_bytes().splitlines()] for p in folders]
    result=summarize_holdout(records,expected,[r['maps'] for r in ctx.maps['runs']],ctx.policy)
    result.update(common(ctx),completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  cohort_freeze_sha256=digest(checked_path(ctx.project,ctx.policy['freeze_relative'])),
                  seed_receipt_sha256={str(s):digest(p/'seed_complete.json') for s,p in zip(ctx.policy['seeds'],folders)},
                  checkpoint_registry_sha256=ctx.policy['checkpoint_registry_sha256'],
                  model_freeze_sha256=ctx.policy['model_freeze_sha256'],arithmetic_recheck_pending=True)
    output=checked_path(ctx.project,'runs/internal_holdout_014_analysis')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'internal_results.json',result)
    print(json.dumps(dict(status=result['status'],patients=result['patients'],pairs=result['pairs'],
                          result_sha256=digest(output/'internal_results.json'))),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','evaluate','analyze'])
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();os.umask(0o077)
    ctx=load_context(args)
    if args.stage=='prepare':prepare(ctx)
    elif args.stage=='evaluate':evaluate(ctx,args.seed)
    else:analyze(ctx)


if __name__=='__main__':main()
