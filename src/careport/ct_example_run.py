"""Execute CT-UPDATE-018 on the cluster: choose the example pair, re-infer it, check it and save private crops.

One stage on an H100 profile. The pair is chosen through the verified results' receipt and prediction hashes, the
selected run's registered old and updated checkpoints are run once on its two cached patches, and the native-grid
masks must agree with the stored evaluation before anything is saved. The crops, masks and the pair's identity stay
in the private output folder; the receipt holds hashes, the selected seed and the check only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from careport.ct_example import (MODELS, PROTOCOL_ID, agreement, canonical, change_table, choose_pair, choose_run,
                                 crop_windows, example_pairs, largest_area_slice, radiological,
                                 validate_example_policy, window_limits)
from careport.ct_pantrack_run import checked_path, read_bound_json, storage_gate, write_json
from careport.ct_replication import file_sha256
from careport.ct_update_gate import GateError

PANTRACK_MANIFEST = 'control/pantrack_remote_manifest_2026-09-18.json'


def check_strong_bindings(policy: dict, environment, source_sha: str) -> None:
    """The strong-model gates must run on the registered 017 policy and correction, and on this bundle."""
    if (environment.get('CT_STRONG_POLICY_SHA256') != policy['strong_policy_sha256']
            or environment.get('CT_STRONG_FIX_SHA256') != policy['strong_fix_sha256']
            or environment.get('CT_STRONG_SOURCE_SHA256') != source_sha):
        raise GateError('The strong-model bindings differ from the example policy')


def load_example_context(policy_path: Path, code: Path | None = None) -> SimpleNamespace:
    from careport.ct_strong_run import load_context
    code = (code or Path.cwd()).resolve()
    policy = read_bound_json(policy_path, os.environ['CT_EXAMPLE_POLICY_SHA256'])
    validate_example_policy(policy)
    if file_sha256(code/policy['amendment_file']) != policy['amendment_sha256']:
        raise GateError('Example amendment changed')
    source_sha = file_sha256(code/'EXECUTION_SHA256SUMS')
    if source_sha != os.environ['CT_EXAMPLE_SOURCE_SHA256']:
        raise GateError('Example source manifest differs from the submitted bundle')
    check_strong_bindings(policy, os.environ, source_sha)
    for name in ('strong_policy', 'strong_fix'):
        if file_sha256(code/policy[f'{name}_file']) != policy[f'{name}_sha256']:
            raise GateError(f'The registered {name} file changed')
    strong = load_context(code/policy['strong_policy_file'], code)
    results = read_bound_json(code/policy['strong_results_file'], policy['strong_results_sha256'])
    if results.get('policy_sha256') != strong.policy_sha or results.get('correction_sha256') != strong.fix_sha:
        raise GateError('The verified results belong to another policy or correction')
    return SimpleNamespace(policy=policy, policy_sha=file_sha256(policy_path), source_sha=source_sha, code=code,
                           strong=strong, results=results,
                           output=checked_path(strong.project, policy['output_relative']))


def select_example(results: dict, policy: dict, run_root: Path) -> dict:
    """The registered run and pair, read only through the verified results' receipt and prediction hashes."""
    descriptive, stratum = results['descriptive'], policy['stratum']
    if results.get('seeds') != descriptive.get('seeds'):
        raise GateError('The primary and descriptive results list different runs')
    entries = [e for e in descriptive['cohorts'][policy['cohort']]['size_strata'] if e.get('stratum') == stratum['name']]
    if (len(entries) != 1 or entries[0].get('lower_ml') != stratum['lower_ml']
            or entries[0].get('upper_ml') != stratum['upper_ml']):
        raise GateError('The reported stratum differs from the registered one')
    summary = entries[0]['retaining_baseline_cost']
    seed, value, mean = choose_run(summary['per_run_means'], descriptive['seeds'])
    if not math.isclose(mean, summary['mean'], rel_tol=0, abs_tol=1e-9):
        raise GateError('The per-run values do not reproduce the reported stratum mean')
    folder = Path(run_root)/f'run_{seed}'/'evaluation'
    receipt_sha = file_sha256(folder/'seed_complete.json')
    if receipt_sha != results['seed_receipt_sha256'].get(str(seed)):
        raise GateError('The selected run receipt differs from the verified results')
    receipt = json.loads((folder/'seed_complete.json').read_bytes())
    if receipt.get('status') != 'STRONG_SEED_COMPLETE' or receipt.get('seed') != seed:
        raise GateError('The selected run receipt is not a completed evaluation of that seed')
    predictions = folder/f"{policy['cohort']}_predictions_private.jsonl"
    predictions_sha = file_sha256(predictions)
    if predictions_sha != receipt['cohorts'][policy['cohort']]['predictions_sha256']:
        raise GateError('The selected run predictions changed after evaluation')
    rows = [json.loads(line) for line in predictions.read_bytes().splitlines()]
    pairs = example_pairs(rows, stratum)
    pair = choose_pair(pairs, value)
    key = (pair['patient'], pair['pair_id'])
    stored = {(r['timepoint'], m): (r['predictions'][m]['volume_ml'], r['predictions'][m]['dice'])
              for r in rows if (r['patient'], r['pair_id']) == key for m in MODELS}
    return dict(seed=seed, run_value=value, run_mean=mean, pair=pair, stored=stored, pairs_in_stratum=len(pairs),
                eligible_pairs=sum(not p['empty'] for p in pairs), receipt_sha256=receipt_sha,
                predictions_sha256=predictions_sha)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _native_views(ctx, choice: dict, checkpoints: dict, device) -> tuple[dict, dict, dict]:
    """Re-infer the pair on its cached patches; return the stored-format measurements, native views and records."""
    import nibabel as nib
    from careport.ct_nnunet import foreground_logit, make_predictor
    from careport.ct_replication import load_native_mask
    from careport.ct_replication_run import load_cohorts, mask_locator
    from careport.ct_update_data import native_prediction
    from careport.ct_update_evaluate import native_metrics
    replication, cohort = ctx.strong.replication, ctx.policy['cohort']
    cache, records = load_cohorts(replication)[cohort]
    key = (choice['pair']['patient'], choice['pair']['pair_id'])
    chosen = {r['timepoint']: r for r in records if (r['patient'], r['pair_id']) == key}
    if set(chosen) != {'bl', 'fu'}:
        raise GateError('The selected pair is not one frozen cohort pair')
    manifest = read_bound_json(replication.project/PANTRACK_MANIFEST, replication.pantrack_policy['file_manifest_sha256'])
    file_hashes = {entry['path']: entry['sha256'] for entry in manifest['files']}
    locate = mask_locator(replication, cohort)
    predictors = {m: make_predictor(checkpoints[m], device) for m in MODELS}
    measured, views = {}, {}
    for timepoint in ('bl', 'fu'):
        record = chosen[timepoint]
        patch = checked_path(cache, record['patch_file'])
        if file_sha256(patch) != record['patch_sha256']:
            raise GateError('Cached patch changed after preprocessing')
        with np.load(patch, allow_pickle=False) as payload:
            image = payload['image'].astype(np.float32)
        mask_path, digest = locate(record)
        if file_sha256(mask_path) != digest:
            raise GateError('Native mask changed')
        labels, affine = load_native_mask(mask_path)
        if not np.allclose(affine, record['native_affine'], rtol=0, atol=1e-4):
            raise GateError('Native geometry differs from the frozen record')
        relative = f"images/{record['scan']}_0000.nii.gz"
        ct_path = checked_path(replication.roots[cohort], relative)
        if file_sha256(ct_path) != file_hashes.get(relative):
            raise GateError('The CT image differs from the verified transfer')
        scan = nib.load(ct_path)
        if scan.shape != labels.shape or not np.allclose(scan.affine, affine, rtol=0, atol=1e-4):
            raise GateError('The CT image and its labels disagree in geometry')
        masks = {'reference': labels == record['lesion_id']}
        for model in MODELS:
            logits = foreground_logit(predictors[model], image)
            metrics = native_metrics(logits, record['patch_affine'], affine, labels, record['lesion_id'],
                                     record['reference_voxels'])
            measured[(timepoint, model)] = (metrics['volume_ml'], metrics['dice'])
            slices, prediction = native_prediction(logits, record['patch_affine'], affine, labels.shape)
            full = np.zeros(labels.shape, dtype=bool)
            full[slices] = prediction
            masks[model] = full
        views[timepoint] = dict(image=np.asanyarray(scan.dataobj).astype(np.float32), masks=masks, affine=affine)
    return measured, views, chosen


def display_crops(views: dict, display: dict) -> tuple[dict, dict, float]:
    """Radiological axial slices with the largest reference area, cropped by the registered rule."""
    shown = {}
    for timepoint, view in views.items():
        reference, spacing = canonical(view['masks']['reference'], view['affine'])
        index = largest_area_slice(reference, axis=2)
        layers = {'image': view['image'], **view['masks']}
        shown[timepoint] = dict(slice=index, spacing=(spacing[1], spacing[0]),
                                layers={name: radiological(canonical(array, view['affine'])[0][:, :, index])
                                        for name, array in layers.items()})
    order = ('bl', 'fu')
    side, windows = crop_windows([shown[t]['layers']['reference'] for t in order], [shown[t]['spacing'] for t in order],
                                 display['crop_minimum_mm'], display['crop_factor'])
    arrays, placement = {}, {}
    for timepoint, (r0, r1, c0, c1) in zip(order, windows):
        for name, layer in shown[timepoint]['layers'].items():
            crop = layer[r0:r1, c0:c1]
            arrays[f'{timepoint}_{name}'] = crop.astype(np.float32) if name == 'image' else crop.astype(bool)
        arrays[f'{timepoint}_spacing_mm'] = np.asarray(shown[timepoint]['spacing'], dtype=float)
        placement[timepoint] = dict(slice=shown[timepoint]['slice'], window=[r0, r1, c0, c1])
    return arrays, placement, side


def run(ctx) -> None:
    from careport.ct_strong_run import _require_h100, _sanity_receipt, _schedule, _trained_checkpoints
    policy, strong = ctx.policy, ctx.strong
    if ctx.output.exists():
        raise GateError('The example output already exists')
    device, device_name = _require_h100()
    choice = select_example(ctx.results, policy, strong.root)
    _, schedule_sha = _schedule(strong)
    sanity_sha = _sanity_receipt(strong)
    folder = strong.root/f"run_{choice['seed']}"
    training, checkpoints = _trained_checkpoints(folder, choice['seed'], schedule_sha)
    os.environ['nnUNet_results'] = str(folder/'results')
    storage_gate(strong.project, strong.policy)
    measured, views, chosen = _native_views(ctx, choice, checkpoints, device)
    check = policy['check']
    report = agreement(choice['stored'], measured, check['volume_relative_tolerance'], check['dice_absolute_tolerance'])
    ctx.output.mkdir(parents=False, exist_ok=False)
    if not report['passed']:
        write_json(ctx.output/'example_check_failed.json', dict(status='EXAMPLE_CHECK_FAILED', protocol_id=PROTOCOL_ID,
                                                                 seed=choice['seed'], agreement=report,
                                                                 completed_at_utc=_now()))
        raise GateError('The re-inferred masks disagree with the stored evaluation; nothing is rendered')
    arrays, placement, side = display_crops(views, policy['display'])
    pair = choice['pair']
    values = dict(volumes={'reference': dict(bl=pair['ref_bl'], fu=pair['ref_fu']),
                           'old': dict(bl=pair['bl']['old'], fu=pair['fu']['old']),
                           'new': dict(bl=pair['bl']['new'], fu=pair['fu']['new'])},
                  table=change_table(pair), crop_side_mm=side, seed=choice['seed'], cohort=policy['cohort'],
                  window_hu=list(window_limits(policy['display']['window_width_hu'],
                                               policy['display']['window_level_hu'])))
    arrays['values_json'] = np.array(json.dumps(values))
    np.savez_compressed(ctx.output/'example_private.npz', **arrays)
    identity = dict(patient=pair['patient'], pair_id=pair['pair_id'],
                    scans={t: chosen[t]['scan'] for t in chosen}, lesion_ids={t: chosen[t]['lesion_id'] for t in chosen})
    write_json(ctx.output/'selection_private.json', dict(identity, placement=placement, values=values, agreement=report,
                                                          measured={f'{t}:{m}': list(v) for (t, m), v in measured.items()}))
    names = ('example_private.npz', 'selection_private.json')
    size = sum((ctx.output/name).stat().st_size for name in names)
    if size > policy['storage_cap_bytes']:
        raise GateError('The example output exceeds its storage cap')
    receipt = dict(status='EXAMPLE_READY', protocol_id=PROTOCOL_ID, policy_sha256=ctx.policy_sha,
                   source_manifest_sha256=ctx.source_sha, amendment_sha256=policy['amendment_sha256'],
                   strong_results_sha256=policy['strong_results_sha256'], seed=choice['seed'],
                   run_value_ml=choice['run_value'], run_mean_ml=choice['run_mean'],
                   pairs_in_stratum=choice['pairs_in_stratum'], eligible_pairs=choice['eligible_pairs'],
                   seed_receipt_sha256=choice['receipt_sha256'], predictions_sha256=choice['predictions_sha256'],
                   schedule_sha256=schedule_sha, sanity_receipt_sha256=sanity_sha,
                   checkpoint_sha256={m: training['checkpoint_sha256'][m] for m in MODELS}, agreement=report,
                   crop_side_mm=side, outputs={name: file_sha256(ctx.output/name) for name in names},
                   output_bytes=size, device=device_name, patient_level_values_in_receipt=False,
                   completed_at_utc=_now())
    write_json(ctx.output/'example_receipt.json', receipt)
    print(json.dumps(receipt), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True, type=Path)
    run(load_example_context(parser.parse_args().policy))


if __name__ == '__main__':
    main()
