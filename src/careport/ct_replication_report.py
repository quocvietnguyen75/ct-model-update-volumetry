"""Manuscript figure and table data for the verified CT-UPDATE-015 replication (aggregates only)."""
from __future__ import annotations

from careport.ct_update_gate import GateError

PRIMARY = ('retaining_baseline_cost', 'ordinary_update_benefit', 'contour_benefit')
COHORTS = ('internal', 'pantrack')
ORIGINAL_SEEDS = ('20260915', '20260917', '20260919')
ORIGINAL_COHORT_LABEL = {'internal': 'internal', 'pantrack': 'PanTrack'}
LABELS = {'retaining_baseline_cost': 'Retention cost (mL)', 'ordinary_update_benefit': 'Update benefit (mL)',
          'contour_benefit': 'Contour benefit (Dice)'}
COHORT_NAMES = {'internal': 'Internal', 'pantrack': 'PanTrack'}
RUN_COLUMNS = (('primary_contrasts', 'retaining_baseline_cost'), ('primary_contrasts', 'ordinary_update_benefit'),
               ('primary_contrasts', 'contour_benefit'), ('secondary_contrasts', 'baseline_version_shift_ml'),
               ('secondary_contrasts', 'continued_retaining_baseline_cost'))


# The U-Net replication (CT-UPDATE-015) and the nnU-Net replication (CT-UPDATE-017) share one result layout.
COMPLETE_STATUSES = ('REPLICATION_ANALYSIS_COMPLETE', 'STRONG_REPLICATION_ANALYSIS_COMPLETE')


def _require_complete(results: dict) -> None:
    if results.get('status') not in COMPLETE_STATUSES or len(results.get('seeds', [])) != results.get('runs'):
        raise GateError('Replication result is incomplete')


def panel_data(results: dict, original_rows: list[dict]) -> dict:
    """Per contrast and cohort: 20 replication run means, 3 original run means and the replication estimate."""
    _require_complete(results)
    panels = {}
    for name in PRIMARY:
        panels[name] = {}
        for cohort in COHORTS:
            summary = results['cohorts'][cohort]['primary_contrasts'][name]
            replication = [float(v) for v in summary['per_run_means']]
            if len(replication) != results['runs']:
                raise GateError('Run-level values do not cover every replication run')
            original = {r['seed']: float(r['mean']) for r in original_rows
                        if r['cohort'] == ORIGINAL_COHORT_LABEL[cohort] and r['contrast'] == name
                        and r['seed'] in ORIGINAL_SEEDS}
            if set(original) != set(ORIGINAL_SEEDS):
                raise GateError('Original run values are incomplete')
            panels[name][cohort] = dict(replication=replication, original=[original[s] for s in ORIGINAL_SEEDS],
                                        mean=float(summary['mean']), family=list(summary['two_way_family_interval']),
                                        two_way=list(summary['two_way_percentile_95']),
                                        runs_positive=summary['runs_positive'], runs_negative=summary['runs_negative'],
                                        established=bool(summary['family_interval_excludes_zero']))
    return panels


def _interval(values, digits):
    return f'[{values[0]:+.{digits}f}, {values[1]:+.{digits}f}]'


def markdown_table(results: dict) -> str:
    rows = ['| Cohort | Contrast | Mean | Two-way 95% interval | Family interval | Established | Runs + / - |',
            '|---|---|---:|---|---|---|---|']
    for cohort in COHORTS:
        for name in PRIMARY:
            s = results['cohorts'][cohort]['primary_contrasts'][name]
            digits = 4 if name == 'contour_benefit' else 3
            rows.append(f"| {COHORT_NAMES[cohort]} | {LABELS[name]} | {s['mean']:+.{digits}f} | "
                        f"{_interval(s['two_way_percentile_95'], digits)} | {_interval(s['two_way_family_interval'], digits)} | "
                        f"{'Yes' if s['family_interval_excludes_zero'] else 'No'} | {s['runs_positive']} / {s['runs_negative']} |")
    return '\n'.join(rows)


def run_rows(results: dict) -> list[dict]:
    """One row per cohort and replication seed; each value is already a mean over that cohort's patients."""
    _require_complete(results)
    rows = []
    for cohort in COHORTS:
        columns = {name: [float(v) for v in results['cohorts'][cohort][section][name]['per_run_means']]
                   for section, name in RUN_COLUMNS}
        if any(len(values) != results['runs'] for values in columns.values()):
            raise GateError('Run-level values do not cover every replication run')
        rows.extend(dict(cohort=cohort, seed=int(seed), **{name: values[i] for name, values in columns.items()})
                    for i, seed in enumerate(results['seeds']))
    return rows


def _signed(value: float, digits: int) -> str:
    text = f'{value:+.{digits}f}'
    return text if text[0] == '+' else '$-$' + text[1:]


def run_table_rows(rows: list[dict], cohort: str) -> list[list[str]]:
    """LaTeX cells for one cohort's runs in seed order, followed by the mean over runs."""
    mine = [r for r in rows if r['cohort'] == cohort]
    if not mine:
        raise GateError('No replication runs for this cohort')
    digits = [4 if name == 'contour_benefit' else 3 for _, name in RUN_COLUMNS]

    def cells(values):
        return [_signed(v, d) for v, d in zip(values, digits)]

    means = [sum(r[name] for r in mine) / len(mine) for _, name in RUN_COLUMNS]
    return [[str(r['seed']), *cells([r[name] for _, name in RUN_COLUMNS])] for r in mine] + [['Mean', *cells(means)]]
