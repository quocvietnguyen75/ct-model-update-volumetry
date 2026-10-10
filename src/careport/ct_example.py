"""CT-UPDATE-018: choose one example lesion pair and prepare its display (an illustration, not an analysis).

Pure functions for the selection rules, the slice and crop rules, the change table and the agreement check. The
cluster runner (careport.ct_example_run) applies them to private data; nothing here reads files.
"""
from __future__ import annotations

import math
import re

import numpy as np

from careport.ct_posthoc import STRATEGIES, pair_rows, strategy_errors
from careport.ct_update_gate import GateError

PROTOCOL_ID = 'CT-UPDATE-018'
MODELS = ('old', 'new')
SHA256 = re.compile('[0-9a-f]{64}')
MAX_TOLERANCE = 0.05


def validate_example_policy(policy: dict) -> None:
    """The registered decisions: PanTrack, the nnU-Net old and updated models, the 10 mL stratum and fixed rules."""
    hashes = [policy.get(k) for k in ('amendment_sha256', 'strong_policy_sha256', 'strong_fix_sha256',
                                      'strong_results_sha256')]
    check, display, stratum = policy.get('check') or {}, policy.get('display') or {}, policy.get('stratum') or {}
    tolerances = [check.get('volume_relative_tolerance'), check.get('dice_absolute_tolerance')]
    numbers = [display.get(k) for k in ('crop_minimum_mm', 'crop_factor', 'window_width_hu')]
    if (policy.get('protocol_id') != PROTOCOL_ID or policy.get('cohort') != 'pantrack'
            or policy.get('models') != list(MODELS)
            or stratum != dict(name='ge10', lower_ml=10.0, upper_ml=None)
            or not all(isinstance(h, str) and SHA256.fullmatch(h) for h in hashes)
            or not all(type(t) in (int, float) and 0 < t <= MAX_TOLERANCE for t in tolerances)
            or not all(type(n) in (int, float) and n > 0 for n in numbers)
            or type(display.get('window_level_hu')) not in (int, float)
            or display.get('orientation') != 'radiological'
            or (policy.get('selection') or {}).get('only_selected_pair_viewed') is not True
            or policy.get('permission_required_for_submission') is not True
            or policy.get('patient_level_values_in_record') is not False
            or not str(policy.get('output_relative', '')).startswith('runs/')):
        raise GateError('Example policy differs from the registered CT-UPDATE-018 decisions')


def choose_run(per_run: list[float], seeds: list[int]) -> tuple[int, float, float]:
    """Seed and value of the run closest to the mean over runs, and that mean; ties go to the earlier seed."""
    if not seeds or len(per_run) != len(seeds) or len(set(seeds)) != len(seeds):
        raise GateError('Per-run values and seeds do not correspond')
    mean = math.fsum(per_run) / len(per_run)
    best = min(range(len(seeds)), key=lambda i: (abs(per_run[i] - mean), seeds[i]))
    return seeds[best], per_run[best], mean


def _in_stratum(volume: float, stratum: dict) -> bool:
    return volume >= stratum['lower_ml'] and (stratum['upper_ml'] is None or volume < stratum['upper_ml'])


def example_pairs(records: list[dict], stratum: dict) -> list[dict]:
    """One run's pairs in the stratum, in patient and pair order, with each pair's retention cost.

    The cost uses the analysis's errors (measured change minus reference change): |error old/new| - |error new/new|.
    'empty' marks a pair with an empty prediction from either model at either timepoint.
    """
    empty = {}
    for row in records:
        key = (row['patient'], row['pair_id'])
        flags = [bool(row['predictions'][m].get('empty_prediction')) or row['predictions'][m]['volume_ml'] <= 0
                 for m in MODELS]
        empty[key] = empty.get(key, False) or any(flags)
    pairs = []
    for pair in pair_rows(records):
        if not _in_stratum(pair['ref_bl'], stratum):
            continue
        errors = strategy_errors(pair)
        pairs.append(dict(pair, errors=errors, cost=abs(errors['old_new']) - abs(errors['new_new']),
                          empty=empty[(pair['patient'], pair['pair_id'])]))
    return pairs


def choose_pair(pairs: list[dict], target: float) -> dict:
    """The pair whose cost is closest to the target, skipping empty predictions; ties keep patient and pair order."""
    eligible = [p for p in pairs if not p['empty']]
    if not eligible:
        raise GateError('No pair in the stratum has nonempty predictions')
    return min(eligible, key=lambda p: abs(p['cost'] - target))


def largest_area_slice(mask: np.ndarray, axis: int = 2) -> int:
    """Index along `axis` with the largest mask area; ties go to the index nearest the mask's centroid."""
    mask = np.asarray(mask, dtype=bool)
    areas = mask.sum(axis=tuple(a for a in range(mask.ndim) if a != axis))
    if areas.max() == 0:
        raise GateError('The reference mask is empty')
    centroid = float(np.average(np.arange(areas.size), weights=areas))
    candidates = np.flatnonzero(areas == areas.max())
    return int(min(candidates, key=lambda k: (abs(k - centroid), k)))


def canonical(array: np.ndarray, affine) -> tuple[np.ndarray, tuple[float, float, float]]:
    """The array reoriented so its indices increase toward the patient's right, anterior and superior (RAS+),
    with the voxel spacing along those axes. Each array axis must map to its own world axis."""
    columns = np.asarray(affine, dtype=float)[:3, :3]
    world = [int(np.argmax(np.abs(columns[:, i]))) for i in range(3)]
    if sorted(world) != [0, 1, 2] or any(not np.any(columns[:, i]) for i in range(3)):
        raise GateError('The image axes do not map one to one onto patient axes')
    order = [world.index(w) for w in range(3)]
    result = np.transpose(np.asarray(array), order)
    for w, i in enumerate(order):
        if columns[w, i] < 0:
            result = np.flip(result, axis=w)
    spacing = tuple(float(np.linalg.norm(columns[:, i])) for i in order)
    return result, spacing


def radiological(axial: np.ndarray) -> np.ndarray:
    """An axial slice indexed (right, anterior) shown with anterior up and the patient's right on the viewer's left."""
    return np.asarray(axial)[::-1, ::-1].T


def _span(centre: float, length: float, size: int) -> tuple[int, int]:
    count = min(int(math.floor(length + 0.5)), size)
    start = min(max(int(math.floor(centre - count / 2 + 0.5)), 0), size - count)
    return start, start + count


def crop_windows(masks: list[np.ndarray], spacings: list[tuple[float, float]], minimum_mm: float,
                 factor: float) -> tuple[float, list[tuple[int, int, int, int]]]:
    """Square crops (rows, columns) centred on each mask: side max(minimum, factor x the largest in-slice extent
    at either timepoint), in millimetres, the same at every timepoint, moved (or, if needed, cut) inside the image."""
    extents = []
    for mask, (row_mm, col_mm) in zip(masks, spacings, strict=True):
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            raise GateError('The displayed reference slice is empty')
        extents.append(max((rows.max() - rows.min() + 1) * row_mm, (cols.max() - cols.min() + 1) * col_mm))
    side = max(float(minimum_mm), float(factor) * max(extents))
    windows = []
    for mask, (row_mm, col_mm) in zip(masks, spacings):
        rows, cols = np.nonzero(mask)
        fitted = min(side, mask.shape[0] * row_mm, mask.shape[1] * col_mm)
        windows.append((*_span(float(rows.mean()), fitted / row_mm, mask.shape[0]),
                        *_span(float(cols.mean()), fitted / col_mm, mask.shape[1])))
    return side, windows


def window_limits(width: float, level: float) -> tuple[float, float]:
    return level - width / 2, level + width / 2


def change_table(pair: dict) -> dict:
    """Reference change, and each strategy's measured change and error, as the analysis defines them."""
    errors = strategy_errors(pair)
    rows = [dict(strategy=name, baseline_model=a, followup_model=b, change=pair['fu'][b] - pair['bl'][a],
                 error=errors[name]) for name, (a, b) in STRATEGIES.items()]
    return dict(reference_change=pair['ref_fu'] - pair['ref_bl'], rows=rows,
                retention_cost=abs(errors['old_new']) - abs(errors['new_new']))


def agreement(stored: dict, measured: dict, volume_tolerance: float, dice_tolerance: float) -> dict:
    """Re-inferred (volume, Dice) per (timepoint, model) against the stored evaluation."""
    if set(stored) != set(measured) or not stored:
        return dict(passed=False, reason='different measurements', max_relative_volume_difference=None,
                    max_dice_difference=None)
    volume = max(abs(measured[k][0] - stored[k][0]) / stored[k][0] for k in stored)
    dice = max(abs(measured[k][1] - stored[k][1]) for k in stored)
    return dict(passed=bool(volume <= volume_tolerance and dice <= dice_tolerance), reason=None,
                max_relative_volume_difference=volume, max_dice_difference=dice)
