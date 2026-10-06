"""Frozen identity, completion and analysis contracts for CT-UPDATE-014."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

import numpy as np

from careport.ct_pantrack_confirm import MODELS, PRIMARY, _patient_arrays, summarize_external
from careport.ct_update_data import classify_rows
from careport.ct_update_gate import GateError, PATIENT, validate_input_policy


def validate_holdout_policy(policy: dict, base: dict) -> None:
    validate_input_policy(base)
    fixed = dict(protocol_id='CT-UPDATE-014', cohort='Longitudinal-CT internal test',
                 role='internal_evaluation_only_no_tuning', nominal_test_patients=30,
                 project_root_env='CT_UPDATE_REMOTE_ROOT', dataset_root_env='CT_UPDATE_REMOTE_DATA_ROOT',
                 seeds=[20260915,20260917,20260919], models=list(MODELS),
                 training_permitted=False, fitting_permitted=False,
                 test_references_permitted_after_metadata=True, test_predictions_permitted_after_freeze=True,
                 model_input_roles=['ct_image','reference_centroid'],
                 time_policy='supplied_BL_FU_pair_only_no_elapsed_time_claim',
                 eligibility=base['eligibility'], patch=base['patch'],
                 schema_csv_resolution='amendment_002_exact_native_recount_only',
                 cache_relative='derived/internal_holdout_014',
                 freeze_relative='control/internal_holdout_014_freeze.json',
                 analysis=dict(bootstrap_draws=10000,bootstrap_seed=20260924,primary_family_size=6,
                               precision_target_halfwidth_ml=5.,primary_contrasts=list(PRIMARY),
                               weighting='pair_losses_within_patient_then_seed_average_then_equal_patients',
                               clinical_margin_ml=None,csv_reference_sensitivity=True),
                 storage=dict(study_cap_bytes=268435456000,project_cap_bytes=966367641600,
                              planned_additional_bytes=17179869184,cache_cap_bytes=8589934592,
                              artifact_cap_bytes=536870912))
    if any(type(policy.get(k)) is not type(v) or policy[k] != v for k,v in fixed.items()):
        raise GateError('Unapproved internal holdout scientific or resource policy')


def parse_content_manifest(raw: bytes, files: list[tuple[str,int]]) -> dict[str,str]:
    result = {}
    for line in raw.decode('utf-8-sig').splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch('[0-9a-f]{64}',parts[0]):
            raise GateError('Malformed content hash manifest')
        name = parts[1]
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name
                or name != path.as_posix() or name in result):
            raise GateError('Unsafe or duplicate content manifest path')
        result[name] = parts[0]
    if set(result) != {name for name,_ in files}:
        raise GateError('Content manifest and metadata inventory do not agree')
    return result


def _pair_id(patient: str, identity: int, bl_index: int, fu_index: int) -> str:
    return hashlib.sha256(json.dumps([patient,identity,bl_index,fu_index],separators=(',',':')).encode()).hexdigest()


def make_test_records(patient: str, rows: list[dict], test_patients: set[str]) -> tuple[list[dict],dict]:
    if patient not in test_patients or not PATIENT.fullmatch(patient):
        raise GateError('Non-test patient requested by internal cohort builder')
    eligible, counts = classify_rows(rows)
    result = []
    for lesion in eligible:
        pair_id = _pair_id(patient,lesion['lesion_id'],lesion['bl']['image_index'],lesion['fu']['image_index'])
        for timepoint in ('bl','fu'):
            index = lesion[timepoint]['image_index']
            if not 0 <= index <= 99:
                raise GateError('Image index is outside the frozen filename schema')
            result.append(dict(patient=patient,split='test',pair_id=pair_id,lesion_id=lesion['lesion_id'],
                               timepoint=timepoint,scan=f'{patient}_{timepoint.upper()}_{index:02d}',
                               **lesion[timepoint]))
    return result,dict(counts)


def validate_membership(records: list[dict], test_patients: set[str]) -> dict[str,int]:
    grouped = defaultdict(dict)
    if not records:
        raise GateError('No eligible paired test references')
    for record in records:
        p, identity, tp, index = (record[k] for k in ('patient','lesion_id','timepoint','image_index'))
        if (p not in test_patients or record['split'] != 'test' or tp not in {'bl','fu'}
                or type(identity) is not int or identity <= 0 or type(index) is not int or not 0 <= index <= 99
                or record['scan'] != f'{p}_{tp.upper()}_{index:02d}' or tp in grouped[(p,identity)]):
            raise GateError('Internal membership, role, index or uniqueness gate failed')
        if 'reference_volume_ml' in record and (not math.isfinite(record['reference_volume_ml']) or record['reference_volume_ml'] <= 0):
            raise GateError('Internal reference volume is not finite and positive')
        grouped[(p,identity)][tp] = record
    for (patient, identity), pair in grouped.items():
        if set(pair) != {'bl','fu'}:
            raise GateError('Internal lesion is missing a paired timepoint')
        expected = _pair_id(patient,identity,pair['bl']['image_index'],pair['fu']['image_index'])
        if any(r['pair_id'] != expected for r in pair.values()):
            raise GateError('Internal pair fingerprint differs from its BL/FU image mapping')
    return dict(patients=len({r['patient'] for r in records}),pairs=len(grouped),timepoints=len(records))


def resolve_reference_issues(issues: dict, recount: dict) -> dict:
    if not issues:
        return dict(status='INTERNAL_REFERENCE_SCHEMA_PASS',warnings={},resolved_issue_types=[])
    required = issues.get('csv_native_volume_mismatch',0)
    if (set(issues) != {'csv_native_volume_mismatch'} or required <= 0
            or recount.get('checked') != required or recount.get('count_mismatches') != 0
            or recount.get('geometry_mismatches') != 0 or recount.get('full_vs_bounded_count_agreements') != required):
        raise GateError('Internal schema has an unresolved native-reference discrepancy')
    return dict(status='INTERNAL_REFERENCE_SCHEMA_PASS',warnings=dict(issues),
                resolved_issue_types=['csv_native_volume_mismatch'])


def validate_complete_seeds(receipts: list[dict], policy: dict, freeze: dict, states: dict[int,dict]) -> None:
    if len(receipts) != 3 or {r['seed'] for r in receipts} != set(policy['seeds']):
        raise GateError('Internal outcomes require all three completed seeds')
    for receipt in receipts:
        seed = receipt['seed']
        required = dict(status='INTERNAL_SEED_COMPLETE',protocol_id='CT-UPDATE-014',models=list(MODELS),
                        internal_test_values_read=True,inference_only=True,model_state_sha256=states[seed])
        for key in ('policy_sha256','source_manifest_sha256','index_sha256','preprocessing_report_sha256',
                    'model_freeze_sha256','checkpoint_registry_sha256','patients','pairs','timepoints'):
            required[key] = freeze[key]
        if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k,v in required.items()):
            raise GateError('Internal completion, cohort or model-state binding changed')
        if not re.fullmatch('[0-9a-f]{64}',receipt.get('private_predictions_sha256','')):
            raise GateError('Completed internal predictions lack a valid content hash')


def _points(arrays: dict) -> dict:
    mae = {key[4:]:float(values.mean()) for key,values in arrays.items() if key.startswith('mae:')}
    primary = dict(ordinary_update_benefit=mae['old_old']-mae['old_new'],
                   retaining_baseline_cost=mae['old_new']-mae['new_new'],
                   contour_benefit=float((arrays['dice:new']-arrays['dice:old']).mean()))
    return dict(strategies={name:dict(mae_ml=value,signed_error_ml=float(arrays['signed:'+name].mean()))
                            for name,value in mae.items()},primary_contrast_points=primary)


def summarize_holdout(seed_records: list[list[dict]], expected: list[dict], maps: list[dict], policy: dict) -> dict:
    analysis = policy['analysis']
    result = summarize_external(seed_records,expected,maps,analysis['bootstrap_draws'],analysis['bootstrap_seed'],
                                analysis['primary_family_size'],analysis['precision_target_halfwidth_ml'])
    result.update(status='INTERNAL_EVALUATION_COMPLETE',protocol_id='CT-UPDATE-014',
                  cohort=policy['cohort'],internal_test_values_read=True,fitting_performed=False,
                  design_informed_by_external_results=True)
    references = {(r['pair_id'],r['timepoint']):r['csv_volume_mm3']/1000 for r in expected}
    csv_arrays,native_arrays,summaries = [],[],[]
    for seed,records,coefficients,summary in zip(policy['seeds'],seed_records,maps,result['per_seed']):
        summary['training_seed'] = seed
        changed = [dict(r,reference_volume_ml=references[(r['pair_id'],r['timepoint'])]) for r in records]
        _,native,_ = _patient_arrays(records,coefficients)
        _,csv,_ = _patient_arrays(changed,coefficients)
        native_arrays.append(native);csv_arrays.append(csv)
        row = _points(csv)
        row['training_seed'] = seed
        row['primary_difference_from_native'] = {k:row['primary_contrast_points'][k]-summary['primary_contrasts'][k]['mean'] for k in PRIMARY}
        summaries.append(row)
    averaged = {k:np.mean([a[k] for a in csv_arrays],axis=0) for k in csv_arrays[0]}
    average = _points(averaged)
    average['primary_difference_from_native'] = {k:average['primary_contrast_points'][k]-result['seed_averaged']['primary_contrasts'][k]['mean'] for k in PRIMARY}
    changes = [abs(float(csv[k].mean()-native[k].mean())) for csv,native in zip(csv_arrays,native_arrays)
               for k in csv if k.startswith('mae:')]
    result['csv_reference_sensitivity'] = dict(reference='CSV_mm3_divided_by_1000',
        per_seed=summaries,seed_averaged=average,inferential_tests_performed=False,
        maximum_absolute_per_seed_strategy_mae_difference_ml=max(changes),native_Dice_and_HD95_unchanged=True)
    return result
