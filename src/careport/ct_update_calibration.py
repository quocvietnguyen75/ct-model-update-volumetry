"""Patient-held-out affine volume correction with refitted bootstrap intervals."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from careport.ct_update_gate import GateError


CONTRASTS = {
    "old_to_new_correction_benefit": ("old_new", "mapped_old_new"),
    "new_to_old_correction_benefit": ("old_new", "old_mapped_new"),
    "old_to_continued_correction_benefit": ("old_continued", "mapped_old_continued"),
    "continued_to_old_correction_benefit": ("old_continued", "old_mapped_continued"),
    "mapped_baseline_cost_vs_new_reprocessing": ("mapped_old_new", "new_new"),
    "mapped_baseline_cost_vs_continued_reprocessing": ("mapped_old_continued", "continued_continued"),
}


@dataclass(frozen=True)
class AffineMap:
    slope: float
    intercept: float
    degenerate: bool

    def predict(self, volume: np.ndarray) -> np.ndarray:
        return np.maximum(0., self.intercept + self.slope * volume)


def fit_affine(x, y, weights) -> AffineMap:
    x, y, weights = (np.asarray(a, dtype=float) for a in (x, y, weights))
    if (x.ndim != 1 or not x.size or x.shape != y.shape or x.shape != weights.shape
            or any(not np.isfinite(a).all() or np.any(a < 0) for a in (x, y, weights))
            or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise GateError("Invalid affine calibration inputs or weights")
    weights = weights / weights.sum()
    mean_x, mean_y = float(weights @ x), float(weights @ y)
    variance = float(weights @ ((x - mean_x) ** 2))
    degenerate = variance <= 1e-12 * max(1., float(weights @ (x ** 2)))
    slope = 1. if degenerate else max(0., float(weights @ ((x - mean_x) * (y - mean_y))) / variance)
    intercept = mean_y - slope * mean_x
    if not np.isfinite([slope, intercept, variance]).all():
        raise GateError("Nonfinite affine calibration fit")
    return AffineMap(slope, intercept, degenerate)


def assign_folds(patients: list[str], count: int, salt: str) -> dict[str, int]:
    if (type(count) is not int or count < 2 or len(patients) < count
            or len(set(patients)) != len(patients)
            or any(not isinstance(p, str) or not p for p in patients)):
        raise GateError("Invalid patient-fold assignment inputs")
    ranked = sorted(patients, key=lambda p: (hashlib.sha256((salt + p).encode()).hexdigest(), p))
    return {patient: rank % count for rank, patient in enumerate(ranked)}


@dataclass(frozen=True)
class CalibrationData:
    patients: tuple[str, ...]
    patient_index: np.ndarray
    patient_folds: np.ndarray
    lesions_per_patient: np.ndarray
    values: dict[str, np.ndarray]


def _prepare(rows: list[dict], folds: dict[str, int]) -> CalibrationData:
    if not rows:
        raise GateError("Empty calibration cohort")
    patients = tuple(sorted({r['patient'] for r in rows}))
    if (set(patients) != set(folds) or any(type(f) is not int for f in folds.values())
            or set(folds.values()) != set(range(len(set(folds.values()))))
            or len(set(folds.values())) < 2):
        raise GateError("Calibration folds do not match the patient cohort")
    if len({(r['patient'], r['lesion_id']) for r in rows}) != len(rows):
        raise GateError("Duplicate calibration lesion")
    positions = {p: i for i, p in enumerate(patients)}
    patient_index = np.array([positions[r['patient']] for r in rows])
    fields = [m + '_' + t for m in ('old', 'new', 'continued', 'reference', 'csv_reference')
              for t in ('bl', 'fu')]
    values = {name: np.array([r[name] for r in rows], dtype=float) for name in fields}
    if any(not np.isfinite(v).all() or np.any(v < 0) for v in values.values()):
        raise GateError("Invalid calibration or reference volume")
    return CalibrationData(patients, patient_index, np.array([folds[p] for p in patients]),
                           np.bincount(patient_index, minlength=len(patients)), values)


def _point(data: CalibrationData, multiplicities: np.ndarray) -> dict:
    if (multiplicities.shape != (len(data.patients),) or np.any(multiplicities < 0)
            or not np.isfinite(multiplicities).all() or multiplicities.sum() <= 0):
        raise GateError("Invalid patient bootstrap multiplicities")
    weights = multiplicities[data.patient_index] / data.lesions_per_patient[data.patient_index]
    row_folds = data.patient_folds[data.patient_index]
    values = data.values
    changes = {
        'old_old': values['old_fu'] - values['old_bl'],
        'old_new': values['new_fu'] - values['old_bl'],
        'new_new': values['new_fu'] - values['new_bl'],
        'old_continued': values['continued_fu'] - values['old_bl'],
        'continued_continued': values['continued_fu'] - values['continued_bl'],
    }
    coefficients = []
    for model in ('new', 'continued'):
        mapped_baseline, mapped_followup = np.zeros(len(weights)), np.zeros(len(weights))
        for fold in sorted(set(data.patient_folds)):
            fitting, held = row_folds != fold, row_folds == fold
            forward = fit_affine(values['old_bl'][fitting], values[model + '_bl'][fitting], weights[fitting])
            reverse = fit_affine(values[model + '_bl'][fitting], values['old_bl'][fitting], weights[fitting])
            mapped_baseline[held] = forward.predict(values['old_bl'][held])
            mapped_followup[held] = reverse.predict(values[model + '_fu'][held])
            for source, target, fitted in [('old', model, forward), (model, 'old', reverse)]:
                coefficients.append({'fold': int(fold), 'source': source, 'target': target,
                                     'slope': fitted.slope, 'intercept_ml': fitted.intercept,
                                     'degenerate': fitted.degenerate})
        changes['mapped_old_' + model] = values[model + '_fu'] - mapped_baseline
        changes['old_mapped_' + model] = mapped_followup - values['old_bl']
    result = {'coefficients': coefficients}
    for reference, prefix in [('native', 'reference'), ('csv', 'csv_reference')]:
        target = values[prefix + '_fu'] - values[prefix + '_bl']
        strategies = {}
        for name, change in changes.items():
            errors = change - target
            strategies[name] = {'mae_ml': float(weights @ np.abs(errors) / weights.sum()),
                                'signed_error_ml': float(weights @ errors / weights.sum())}
        result[reference] = {'strategies': strategies,
                             'contrasts': {name: {'mean_ml': strategies[a]['mae_ml'] - strategies[b]['mae_ml']}
                                           for name, (a, b) in CONTRASTS.items()}}
    return result


def crossfit_estimates(rows: list[dict], folds: dict[str, int]) -> dict:
    data = _prepare(rows, folds)
    return _point(data, np.ones(len(data.patients)))


def calibration_analysis(rows: list[dict], folds: dict[str, int], bootstrap_draws: int,
                         bootstrap_seed: int) -> dict:
    if type(bootstrap_draws) is not int or bootstrap_draws < 1:
        raise GateError("Invalid calibration bootstrap count")
    data = _prepare(rows, folds)
    result = _point(data, np.ones(len(data.patients)))
    strata = [np.flatnonzero(data.patient_folds == f) for f in sorted(set(data.patient_folds))]
    draws = {ref: {name: [] for name in CONTRASTS} for ref in ('native', 'csv')}
    rng = np.random.default_rng(bootstrap_seed)
    for _ in range(bootstrap_draws):
        selected = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in strata])
        sampled = _point(data, np.bincount(selected, minlength=len(data.patients)))
        for reference in draws:
            for name in CONTRASTS:
                draws[reference][name].append(sampled[reference]['contrasts'][name]['mean_ml'])
    for reference in draws:
        for name in CONTRASTS:
            interval = np.quantile(draws[reference][name], [.025, .975])
            result[reference]['contrasts'][name]['exploratory_95_percentile_interval_ml'] = interval.tolist()
    result.update(status='DEVELOPMENT_ONLY', patients=len(data.patients), lesions=len(rows),
                  fold_patient_counts=[len(indices) for indices in strata], bootstrap_draws=bootstrap_draws,
                  bootstrap_seed=bootstrap_seed, bootstrap_refits_calibration=True,
                  bootstrap_stratified_by_fixed_fold=True, test_evaluated=False,
                  confirmatory_inference=False, segmentation_outputs_changed=False)
    return result
