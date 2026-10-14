"""LaTeX tables and a number-consistency check for the CT manuscript (aggregates only)."""
from __future__ import annotations

import csv
import io
import re

from careport.ct_replication_report import COHORT_NAMES, COHORTS, LABELS, PRIMARY
from careport.ct_update_gate import GateError

# One contrast order for every manuscript table and figure: the replicated compatibility cost first.
MANUSCRIPT_CONTRASTS = PRIMARY
# The manuscript calls the prespecified retention cost (protocol key retaining_baseline_cost) the compatibility
# cost (authors' decision D3, 12 October 2026); the frozen records keep their original labels.
MANUSCRIPT_LABELS = {**LABELS, 'retaining_baseline_cost': 'Compatibility cost (mL)'}
# A table cell with no value, such as the interval of a stratum too small for one.
NO_VALUE = '---'
# An unsigned decimal that is not part of a version string such as 1.5.1.
_DECIMAL = re.compile(r'(?<![\w.])\d+\.\d+(?!\w|\.\d)')
# A reported interval or range written as "X to Y", optionally with a unit after X.
_PAIR = re.compile(r'([+-]?\d+\.\d+)\s*(?:mL\s*)?to\s*([+-]?\d+\.\d+)')


def latex_table(header: list[str], rows: list[list[str]], align: str) -> str:
    """A tabular with rules around the header and after the last row; cells are already LaTeX."""
    if len(align) != len(header) or any(len(row) != len(header) for row in rows):
        raise GateError('Table header, alignment and rows disagree')
    lines = [rf'\begin{{tabular}}{{{align}}}', r'\hline', ' & '.join(header) + r' \\', r'\hline',
             *(' & '.join(row) + r' \\' for row in rows), r'\hline', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def latex_number(value, digits: int, sign: bool = False) -> str:
    """A fixed-precision number; a negative sign is typeset as a minus, and `sign` adds an explicit plus."""
    text = f'{float(value):+.{digits}f}' if sign else f'{float(value):.{digits}f}'
    return '$-$' + text[1:] if text.startswith('-') else text


def latex_interval(low, high, digits: int, sign: bool = False) -> str:
    return f'[{latex_number(low, digits, sign)}, {latex_number(high, digits, sign)}]'


def contrast_digits(name: str) -> int:
    """Dice contrasts take four decimals; millilitre contrasts three."""
    return 4 if name == 'contour_benefit' else 3


def _volume_label(low: float, high: float | None) -> str:
    if high is None:
        return rf'$\geq{low:g}$'
    return rf'$<{high:g}$' if low == 0 else rf'{low:g} to $<{high:g}$'


def posthoc_stratum_rows(results: dict, contrast: str, with_mae: bool = True) -> list[list[str]]:
    """One row per cohort and lesion-size stratum (CT-UPDATE-016), in policy order: volume range, patients / pairs,
    optionally the mean change MAE of old/old, old/new and new/new, then the contrast's mean, its two-way interval
    (absent when the stratum was too small for one) and the runs on each side of zero."""
    rows = []
    for cohort in COHORTS:
        for entry in results['cohorts'][cohort]['size_strata']:
            row = [COHORT_NAMES[cohort], _volume_label(entry['lower_ml'], entry['upper_ml']),
                   f"{entry['patients']} / {entry['pairs']}"]
            if with_mae:
                mae = entry.get('mae_ml')
                row += [latex_number(mae[s], 3) if mae else NO_VALUE for s in ('old_old', 'old_new', 'new_new')]
            summary = entry.get(contrast)
            if summary is None:
                row += [NO_VALUE] * 3
            else:
                bounds = summary.get('two_way_percentile_95')
                row += [latex_number(summary['mean'], 3, True), latex_interval(*bounds, 3, True) if bounds else NO_VALUE,
                        f"{summary['runs_positive']} / {summary['runs_negative']}"]
            rows.append(row)
    return rows


def posthoc_sensitivity_rows(published: dict, recomputed: dict) -> list[list[str]]:
    """Each primary contrast's published family-wise interval (10 000 draws) beside its CT-UPDATE-016 recomputation
    (100 000 draws), whether it is established under both, the run signs and the one-sided sign probability."""
    rows = []
    for cohort in COHORTS:
        for name in MANUSCRIPT_CONTRASTS:
            old = published['cohorts'][cohort]['primary_contrasts'][name]
            new = recomputed['cohorts'][cohort]['interval_sensitivity'][name]
            digits = contrast_digits(name)
            same = old['family_interval_excludes_zero'] == new['family_interval_excludes_zero']
            probability = new['sign_probability']
            rows.append([COHORT_NAMES[cohort], MANUSCRIPT_LABELS[name], latex_interval(*old['two_way_family_interval'], digits, True),
                         latex_interval(*new['two_way_family_interval'], digits, True),
                         ('Yes' if new['family_interval_excludes_zero'] else 'No') if same else 'Changed',
                         f"{new['runs_positive']} / {new['runs_negative']}",
                         '$<$0.001' if probability < 0.001 else f'{probability:.3f}'])
    return rows


def primary_contrast_rows(results: dict) -> list[list[str]]:
    """One row per cohort and primary contrast of a training-run replication: the mean, its two-way and family-wise
    intervals, whether it is established (family-wise interval excludes zero) and the runs on each side of zero."""
    rows = []
    for cohort in COHORTS:
        for name in MANUSCRIPT_CONTRASTS:
            summary, digits = results['cohorts'][cohort]['primary_contrasts'][name], contrast_digits(name)
            rows.append([COHORT_NAMES[cohort], MANUSCRIPT_LABELS[name], latex_number(summary['mean'], digits, True),
                         latex_interval(*summary['two_way_percentile_95'], digits, True),
                         latex_interval(*summary['two_way_family_interval'], digits, True),
                         'Yes' if summary['family_interval_excludes_zero'] else 'No',
                         f"{summary['runs_positive']} / {summary['runs_negative']}"])
    return rows


def interval_component_rows(unet: dict, nnunet: dict) -> list[list[str]]:
    """Each primary contrast of both replications with its two-way, patient-only and run-only percentile 95%
    intervals, the components the replication methods report descriptively (ARS R5, 12 October 2026)."""
    rows = []
    for model, results in (('U-Net', unet), ('nnU-Net', nnunet)):
        for cohort in COHORTS:
            for name in MANUSCRIPT_CONTRASTS:
                summary, digits = results['cohorts'][cohort]['primary_contrasts'][name], contrast_digits(name)
                rows.append([model, COHORT_NAMES[cohort], MANUSCRIPT_LABELS[name],
                             latex_number(summary['mean'], digits, True),
                             *(latex_interval(*summary[key], digits, True) for key in
                               ('two_way_percentile_95', 'patient_only_percentile_95', 'run_only_percentile_95'))])
    return rows


# CT-UPDATE-019 columns: each segmenter and cohort, never pooled.
COMPAT_COLUMNS = (('unet', 'internal'), ('unet', 'pantrack'), ('nnunet', 'internal'), ('nnunet', 'pantrack'))


def _with_interval(value, bounds, digits: int, sign: bool = False, scale: float = 1.0) -> str:
    return (f'{latex_number(scale * value, digits, sign)} '
            f'{latex_interval(scale * bounds[0], scale * bounds[1], digits, sign)}')


def _compat_cells(result: dict, render) -> list[str]:
    return [render(result['segmenters'][model]['cohorts'][cohort]) for model, cohort in COMPAT_COLUMNS]


def _share(name: str):
    return lambda c: _with_interval(c['patient_response'][name]['mean'],
                                    c['patient_response'][name]['two_way_percentile_95'], 1, scale=100)


def _share_range(name: str):
    def render(c):
        entry = c['patient_response'][name]
        return (f"{latex_number(100 * entry['mean'], 1)} ({latex_number(100 * entry['run_min'], 1)} to "
                f"{latex_number(100 * entry['run_max'], 1)})")
    return render


_SUMMARY = (
    (r'Relative compatibility cost (\%)', lambda c: _with_interval(
        c['relative_compatibility_cost']['mean_percent'],
        c['relative_compatibility_cost']['two_way_percentile_95'], 1, True)),
    (r'Version difference, bias (\%)', lambda c: _with_interval(
        c['version_agreement']['bias_percent'], c['version_agreement']['bias_percent_two_way_95'], 1, True)),
    (r'Version difference, 95\% limits of agreement (\%)', lambda c: (
        f"{latex_number(c['version_agreement']['limits_percent'][0], 1, True)} to "
        f"{latex_number(c['version_agreement']['limits_percent'][1], 1, True)}")),
    (r'Compatibility index $\kappa$', lambda c: _with_interval(
        c['decomposition']['kappa'], c['decomposition']['kappa_two_way_95'], 2)),
    (r'Increase in RMS log change error (\%)', lambda c: _with_interval(
        c['decomposition']['observed_inflation_percent'],
        c['decomposition']['observed_inflation_percent_two_way_95'], 1, True)),
    (r'Predicted without the cross term (\%)', lambda c: _with_interval(
        c['decomposition']['predicted_inflation_percent'],
        c['decomposition']['predicted_inflation_percent_two_way_95'], 1, True)),
    ('Patient-level assessments', lambda c: str(c['patient_response']['assessments'])),
    (r'Category changed by mixing (\%)', _share('differs')),
    (r'Increase versus no increase changed (\%)', _share('progression_split_differs')),
    (r'Agreement with reference, new/new (\%)', _share('agrees:new_new')),
    (r'Agreement with reference, old/new (\%)', _share('agrees:old_new')),
)
_FULL_EXTRA = (
    (r'Agreement with reference, old/old (\%)', _share('agrees:old_old')),
    (r'Mixing loses the reference category (\%)', _share('mixing_loses_reference')),
    (r'Mixing gains the reference category (\%)', _share('mixing_gains_reference')),
    (r'Mixing-only increase (\%)', _share_range('spurious_increase')),
    (r'Reprocessing-only increase (\%)', _share_range('masked_increase')),
    (r'Mixing-only decrease (\%)', _share_range('spurious_decrease')),
    (r'Reprocessing-only decrease (\%)', _share_range('masked_decrease')),
    ('Reference categories: decrease / stable / increase', lambda c: ' / '.join(
        str(c['patient_response']['reference_category_counts'][k]) for k in ('decrease', 'stable', 'increase'))),
    ('Cross-term ratio', lambda c: _with_interval(c['decomposition']['cross_ratio'],
                                                  c['decomposition']['cross_ratio_two_way_95'], 3, True)),
    ('Valid pair-runs (log scale)', lambda c: str(c['version_agreement']['valid_pair_runs'])),
    ('Excluded pair-runs (log scale)', lambda c: str(c['version_agreement']['excluded_pair_runs'])),
)


def compat_summary_rows(result: dict) -> list[list[str]]:
    """The main-text summary of CT-UPDATE-019: one row per quantity, one column per segmenter and cohort; each
    estimate is followed by its two-way 95% interval where one exists. Shares are percentages of assessments."""
    return [[label, *_compat_cells(result, render)] for label, render in _SUMMARY]


def compat_full_rows(result: dict) -> list[list[str]]:
    """Every CT-UPDATE-019 output for the supplement: the summary rows, then the remaining shares (direction shares as
    the mean and range over runs), the reference categories, the cross-term ratio and the log-scale exclusions."""
    return compat_summary_rows(result) + [[label, *_compat_cells(result, render)] for label, render in _FULL_EXTRA]


# Rows of the U-Net/nnU-Net comparison: (kind, name, label).
COMPARISON_ROWS = (('dice', 'old', 'Follow-up Dice, old model'), ('dice', 'new', 'Follow-up Dice, updated model'),
                   *(('primary', name, MANUSCRIPT_LABELS[name]) for name in MANUSCRIPT_CONTRASTS),
                   ('secondary', 'baseline_version_shift_ml', 'Baseline version shift (mL)'))


def model_comparison_rows(unet: dict, nnunet: dict) -> list[list[str]]:
    """The two replications side by side, per cohort (descriptive; the studies are never pooled): follow-up Dice of
    the old and updated models, each primary contrast with its family-wise interval and run signs, and the baseline
    version shift with its two-way 95% interval."""
    rows = []
    for cohort in COHORTS:
        for kind, name, label in COMPARISON_ROWS:
            row = [COHORT_NAMES[cohort], label]
            for results in (unet, nnunet):
                entry = results['cohorts'][cohort]
                if kind == 'dice':
                    row += [latex_number(entry['followup_dice'][name], 4), NO_VALUE]
                    continue
                summary = entry['primary_contrasts' if kind == 'primary' else 'secondary_contrasts'][name]
                bounds = summary['two_way_family_interval' if kind == 'primary' else 'two_way_percentile_95']
                digits = contrast_digits(name)
                row += [f"{latex_number(summary['mean'], digits, True)} {latex_interval(*bounds, digits, True)}",
                        f"{summary['runs_positive']} / {summary['runs_negative']}"]
            rows.append(row)
    return rows


def decimal_tokens(text: str) -> list[tuple[str, float, int]]:
    """Every unsigned decimal in the text with its value and number of decimals."""
    return [(m.group(), float(m.group()), len(m.group().split('.')[1])) for m in _DECIMAL.finditer(text)]


def source_values(json_objects: list, csv_texts: list[str]) -> list[float]:
    """Every numeric value in nested JSON-like objects and in CSV cells; booleans are not numbers."""
    values: list[float] = []

    def walk(item) -> None:
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            values.append(float(item))
        elif isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)

    for item in json_objects:
        walk(item)
    for text in csv_texts:
        for row in csv.reader(io.StringIO(text)):
            for cell in row:
                try:
                    values.append(float(cell))
                except ValueError:
                    pass
    return values


def percent_values(json_objects: list) -> list[float]:
    """Numbers stored in percent units: every number under a key named 'percent', ending in '_percent' or containing
    '_percent_', and the 'two_way_percentile_95' interval of an estimate stored as 'mean_percent'."""
    values: list[float] = []

    def walk(item, in_percent: bool) -> None:
        if isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            if in_percent:
                values.append(float(item))
        elif isinstance(item, dict):
            interval_in_percent = 'mean_percent' in item
            for key, child in item.items():
                name = str(key)
                walk(child, in_percent or name == 'percent' or name.endswith('_percent') or '_percent_' in name
                     or (interval_in_percent and name == 'two_way_percentile_95'))
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child, in_percent)

    for item in json_objects:
        walk(item, False)
    return values


def unmatched_percentages(text: str, fractions: list[float], percents: list[float] = ()) -> list[str]:
    """Decimal percentages (written N.N\\%) that are neither the rounding of a source fraction at two more decimals
    nor the rounding of a value stored in percent units at their own precision, ignoring sign."""
    missing = []
    for token in re.findall(r'(\d+\.\d+)\\%', text):
        digits = len(token.split('.')[1])
        fraction = f'{float(token) / 100:.{digits + 2}f}'
        if unmatched_numbers(fraction, fractions) and token not in {f'{abs(v):.{digits}f}' for v in percents}:
            missing.append(token + '%')
    return missing


def _reported_intervals(item) -> list[tuple[float, float]]:
    """Every two-value interval and every per-run range in nested result summaries."""
    found = []
    if isinstance(item, dict):
        for key, value in item.items():
            if (('interval' in key or 'percentile' in key) and isinstance(value, list) and len(value) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)):
                found.append((float(value[0]), float(value[1])))
            elif key == 'per_run_means' and isinstance(value, list) and value:
                found.append((float(min(value)), float(max(value))))
            else:
                found.extend(_reported_intervals(value))
    elif isinstance(item, list):
        for child in item:
            found.extend(_reported_intervals(child))
    return found


def _rounds_to(value: float, token: str) -> bool:
    return f'{value:.{len(token.split(".")[1])}f}' == token.lstrip('+')


def unmatched_intervals(text: str, summaries) -> list[str]:
    """'X to Y' pairs that are not both bounds of one reported interval or run range, in either sign."""
    candidates = _reported_intervals(summaries)
    plain = text.replace('$', '').replace(r'\,', ' ').replace('−', '-')
    missing = []
    for low, high in _PAIR.findall(plain):
        if not any((_rounds_to(lo, low) and _rounds_to(hi, high)) or (_rounds_to(-hi, low) and _rounds_to(-lo, high))
                   for lo, hi in candidates):
            missing.append(f'{low} to {high}')
    return missing


def unmatched_numbers(text: str, sources: list[float], exempt: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Decimal tokens that are not the rounding of any source value at their displayed precision, ignoring sign."""
    rounded: dict[int, set[str]] = {}
    missing = []
    for token, _, digits in decimal_tokens(text):
        if token in exempt:
            continue
        if digits not in rounded:
            rounded[digits] = {f'{abs(v):.{digits}f}' for v in sources}
        if token not in rounded[digits]:
            missing.append(token)
    return missing
