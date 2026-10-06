"""Run CT-UPDATE-013 against hash-bound, completed external predictions only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from careport.ct_pantrack_confirm import MODELS, validate_confirmation_policy
from careport.ct_pantrack_run import checked_path, digest, load_cache, read_bound_json, storage_gate
from careport.ct_update_diagnostics import diagnose, verify_reproduction
from careport.ct_update_gate import GateError


def validate_policy(policy: dict) -> None:
    fixed = dict(protocol_id='CT-UPDATE-013', post_hoc=True,
                 project_root_env='CT_UPDATE_REMOTE_ROOT', seeds=[20260915, 20260917, 20260919],
                 expected_patients=45, expected_pairs=181, internal_test_values_permitted=False,
                 training_permitted=False, downloads_permitted=False, new_inferential_tests=False,
                 output_relative='runs/pantrack_diagnostic_013', size_bins_ml=[1, 10],
                 storage=dict(study_cap_bytes=268435456000, project_cap_bytes=966367641600,
                              planned_additional_bytes=67108864), output_cap_bytes=33554432)
    if any(type(policy.get(k)) is not type(v) or policy[k] != v for k, v in fixed.items()):
        raise GateError('Unapproved diagnostic scope or storage policy')
    if set(policy['seed_receipt_sha256']) != {str(seed) for seed in fixed['seeds']}:
        raise GateError('Incomplete diagnostic seed bindings')


def validate_seed_receipt(receipt: dict, seed: int, policy: dict) -> None:
    fixed = dict(status='EXTERNAL_SEED_COMPLETE', protocol_id='CT-UPDATE-010', seed=seed,
                 patients=policy['expected_patients'], pairs=policy['expected_pairs'],
                 timepoints=2*policy['expected_pairs'], models=list(MODELS),
                 policy_sha256=policy['original_policy_sha256'],
                 internal_test_values_read=False)
    for key in ('checkpoint_registry_sha256', 'model_freeze_sha256',
                'preprocessing_report_sha256', 'index_sha256'):
        fixed[key] = policy[key]
    if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k, v in fixed.items()):
        raise GateError('Diagnostic input lacks the frozen completion and lineage gates')


def read_prediction_records(path: Path, expected_hash: str) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise GateError('Private prediction payload differs from its frozen hash')
    return [json.loads(line) for line in raw.splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--confirmation-policy', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    policy = read_bound_json(args.policy, os.environ['CT_UPDATE_DIAGNOSTIC_POLICY_SHA256'])
    validate_policy(policy)
    project = Path(os.environ[policy['project_root_env']]).resolve()
    output = checked_path(project, policy['output_relative'])
    if output.exists():
        raise GateError('Diagnostic output already exists; original artifacts cannot be overwritten')
    storage = storage_gate(project, policy)
    original_policy = read_bound_json(args.confirmation_policy, policy['original_policy_sha256'])
    validate_confirmation_policy(original_policy)
    read_bound_json(args.registry, policy['checkpoint_registry_sha256'])
    read_bound_json(project/'control/pantrack_models_010.json', policy['model_freeze_sha256'])
    read_bound_json(project/'derived/pantrack_010/preprocessing_report.json', policy['preprocessing_report_sha256'])
    if os.environ.get('CT_UPDATE_MODEL_FREEZE_SHA256') != policy['model_freeze_sha256']:
        raise GateError('Execution model freeze differs from the diagnostic specification')
    source_manifest = Path('EXECUTION_SHA256SUMS')
    if digest(source_manifest) != os.environ['CT_UPDATE_DIAGNOSTIC_SOURCE_SHA256']:
        raise GateError('Execution source manifest changed')
    receipts, prediction_files = {}, {}
    for seed in policy['seeds']:
        folder = project/f'runs/pantrack_010_seed_{seed}'
        receipt = read_bound_json(folder/'seed_complete.json', policy['seed_receipt_sha256'][str(seed)])
        validate_seed_receipt(receipt, seed, policy)
        prediction_files[seed] = folder/'predictions_private.jsonl'
        if digest(prediction_files[seed]) != receipt['private_predictions_sha256']:
            raise GateError('Diagnostic prediction hash preflight failed')
        receipts[seed] = receipt
    # Reuse the original candidate/time-order, schema, cache and identity gates.
    cache_args = SimpleNamespace(policy=args.confirmation_policy, registry=args.registry)
    _, expected, _ = load_cache(cache_args, original_policy, project)
    if (len(expected) != 2*policy['expected_pairs']
            or len({r['patient'] for r in expected}) != policy['expected_patients']):
        raise GateError('Diagnostic cohort cardinality differs from the frozen cohort')
    original = read_bound_json(project/'runs/pantrack_010_analysis/external_results.json',
                               policy['original_result_sha256'])
    records = {seed: read_prediction_records(prediction_files[seed], receipts[seed]['private_predictions_sha256'])
               for seed in policy['seeds']}
    public, private = diagnose(records, expected)
    public['original_result_reproduction'] = verify_reproduction(public, original)
    public.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), post_hoc=True,
                  policy_sha256=digest(args.policy), source_manifest_sha256=digest(source_manifest),
                  input_result_sha256=policy['original_result_sha256'], storage=storage,
                  seed_receipt_sha256=policy['seed_receipt_sha256'],
                  private_prediction_sha256={str(s): r['private_predictions_sha256'] for s, r in receipts.items()})
    private_bytes = (json.dumps(private, allow_nan=False) + '\n').encode()
    public['private_contributions_sha256'] = hashlib.sha256(private_bytes).hexdigest()
    public['private_contributions_bytes'] = len(private_bytes)
    public_bytes = (json.dumps(public, indent=2, allow_nan=False) + '\n').encode()
    if len(private_bytes) + len(public_bytes) > policy['output_cap_bytes']:
        raise GateError('Diagnostic output would exceed its storage reservation')
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in [('contributions_private.json', private_bytes), ('diagnostic_results.json', public_bytes)]:
        with (output/name).open('xb') as stream:
            stream.write(payload)
    print(json.dumps(dict(status=public['status'], patients=public['patients'], pairs=public['pairs'],
                          result_sha256=digest(output/'diagnostic_results.json'),
                          internal_test_values_read=False)), flush=True)


if __name__ == '__main__':
    main()
