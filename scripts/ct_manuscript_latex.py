"""Write the CT manuscript's LaTeX tables, the per-run replication export and figure copies.

Run scripts/ct_manuscript_figures.py first; this script copies its print-size figures.

Run from the repository root with PYTHONPATH=src. Reads verified aggregate files only. LaTeX
outputs go to newpaper/ct_manuscript/pmb_submission/ without subfolders, as the IOP submission system
requires; the per-run export goes to newpaper/ct_manuscript/tables/.
"""
import csv
import json
import re
import shutil
from pathlib import Path

from careport.ct_manuscript import (MANUSCRIPT_CONTRASTS, MANUSCRIPT_LABELS, compat_full_rows, compat_summary_rows,
                                    contrast_digits as digits_for, interval_component_rows,
                                    latex_interval as interval,
                                    latex_number as num, latex_table, model_comparison_rows, posthoc_sensitivity_rows,
                                    posthoc_stratum_rows, primary_contrast_rows)
from careport.ct_replication_report import COHORT_NAMES, RUN_COLUMNS, run_rows, run_table_rows

RECORD = Path('newpaper/research_selection_2026-09-13')
MANUSCRIPT = Path('newpaper/ct_manuscript')
TABLES = MANUSCRIPT/'tables'
LATEX = MANUSCRIPT/'pmb_submission'
SEEDS = ('20260915', '20260917', '20260919')
CONTRASTS = MANUSCRIPT_CONTRASTS
SHORT = {'retaining_baseline_cost': 'Compatibility|cost', 'ordinary_update_benefit': 'Update|benefit',
         'contour_benefit': 'Contour|benefit'}
FIGURES = MANUSCRIPT/'figures'/'print'
EXAMPLE_FIGURE = 'figure4_example_pair.pdf'
# The verified CT-UPDATE-019 aggregates; their tables and Figure 5 are written once this record exists.
COMPAT_RESULTS = RECORD/'CT_COMPAT_RESULTS_019_2026-10-12.json'
COMPAT_FIGURE = ('figure5_compatibility.pdf', 'figure5.pdf')
COMPAT_HEAD = ('Quantity', 'U-Net,|internal', 'U-Net,|PanTrack', 'nnU-Net,|internal', 'nnU-Net,|PanTrack')
# The three original runs moved to the supplement on 12 October (decision D4): their figure is now figure S1 and
# the PanTrack diagnostics figure S2; the print files keep their earlier names.
FIGURE_COPIES = (('figure1_update_explainer.pdf', 'figure1.pdf'), ('figure2_replication_runs.pdf', 'figure2.pdf'),
                 ('figure3_model_comparison.pdf', 'figure3.pdf'), (EXAMPLE_FIGURE, 'figure4.pdf'),
                 ('figure5_three_run_contrasts.pdf', 'figureS1.pdf'),
                 ('figureS1_pantrack_diagnostics.pdf', 'figureS2.pdf'))
STRATEGIES = ('old_old', 'old_new', 'new_new', 'old_continued', 'continued_continued', 'old_scratch',
              'scratch_scratch', 'old_bcwi_025', 'bcwi_025_bcwi_025', 'old_bcwi_050', 'bcwi_050_bcwi_050',
              'old_bcwi_075', 'bcwi_075_bcwi_075', 'old_kd_001', 'kd_001_kd_001', 'old_kd_010', 'kd_010_kd_010',
              'mapped_old_new', 'old_mapped_new', 'mapped_old_continued', 'old_mapped_continued')
MODELS = ('old', 'new', 'continued', 'scratch', 'bcwi_025', 'bcwi_050', 'bcwi_075', 'kd_001', 'kd_010')
RUN_HEADER = ('Seed', 'Compatibility|cost (mL)', 'Update|benefit (mL)', 'Contour|benefit (Dice)',
              'Baseline version|shift (mL)', 'Continuation|compatibility|cost (mL)')


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def code(name):
    return r'\texttt{' + name.replace('_', r'\_') + '}'


def heads(*texts):
    """Header cells; a '|' breaks a cell onto a new line."""
    return [r'\begin{tabular}[b]{@{}c@{}}' + r'\\'.join(t.split('|')) + r'\end{tabular}' if '|' in t else t
            for t in texts]


def core_strategies():
    body = [[r['cohort'], r['strategy'].replace('_', ' / '), num(r['mae_ml'], 3), num(r['signed_error_ml'], 3),
             num(r['followup_dice'], 4)] for r in read_csv(TABLES/'table2_core_strategies.csv')]
    return latex_table(heads('Cohort', 'Baseline /|follow-up', 'Change|MAE (mL)', 'Signed change|error (mL)',
                             'Follow-up|Dice'), body, 'llrrr')


def primary_contrasts():
    rows = {(r['cohort'], r['contrast']): r for r in read_csv(TABLES/'table3_primary_contrasts.csv')}
    body = []
    for cohort in COHORT_NAMES.values():
        for name in CONTRASTS:
            r, d = rows[(cohort, name)], digits_for(name)
            body.append([cohort, MANUSCRIPT_LABELS[name], num(r['mean'], d),
                         interval(r['percentile_95_low'], r['percentile_95_high'], d),
                         interval(r['bonferroni_family_low'], r['bonferroni_family_high'], d)])
    return latex_table(heads('Cohort', 'Contrast', 'Estimate', r'Ordinary 95\%|interval', 'Family-wise|interval'),
                       body, 'llrrr')


def model_controls():
    rows = read_csv(TABLES/'table4_model_controls.csv')
    by = {(r['cohort'], r['model']): r for r in rows}
    body = []
    for model in MODELS:
        internal, external = by[('Internal', model)], by[('PanTrack', model)]
        label = re.sub(r'(alpha|lambda)=(\S+)', r'$\\\1=\2$', internal['label'])
        body.append([label, *(v for r in (internal, external)
                              for v in (num(r['followup_dice'], 4), num(r['mixed_mae_ml'], 3), num(r['same_mae_ml'], 3)))])
    header = heads('Model setting', 'Internal|Dice', 'Internal|mixed MAE', 'Internal|same MAE', 'PanTrack|Dice',
                   'PanTrack|mixed MAE', 'PanTrack|same MAE')
    return latex_table(header, body, 'lrrrrrr')


def replication_primary(results):
    return latex_table(heads('Cohort', 'Contrast', 'Mean', r'Two-way 95\%|interval', 'Family-wise|interval', 'Established',
                             'Runs|$+$ / $-$'), primary_contrast_rows(results), 'llrrrcc')


def all_strategies():
    rows = {(r['cohort'], r['strategy']): r for r in read_csv(TABLES/'s1_all_strategies.csv')
            if r['seed'] == 'fixed_seed_average'}
    body = [[code(s), *(num(rows[(c, s)][f], 3) for c in COHORT_NAMES.values() for f in ('mae_ml', 'signed_error_ml'))]
            for s in STRATEGIES]
    return latex_table(heads('Strategy', 'Internal|MAE (mL)', 'Internal signed|error (mL)', 'PanTrack|MAE (mL)',
                             'PanTrack signed|error (mL)'), body, 'lrrrr')


def seed_core():
    strategy = {(r['cohort'], r['seed'], r['strategy']): r['mae_ml'] for r in read_csv(TABLES/'s1_all_strategies.csv')}
    dice = {(r['cohort'], r['seed'], r['model']): r['patient_mean_followup_dice']
            for r in read_csv(TABLES/'s2_all_model_metrics.csv')}
    contrast = {(r['cohort'], r['seed'], r['contrast']): r['mean'] for r in read_csv(TABLES/'s3_all_contrasts.csv')}
    body = []
    for cohort in COHORT_NAMES.values():
        for seed in SEEDS:
            key = (cohort, seed)
            body.append([cohort, seed, *(num(strategy[key + (s,)], 3) for s in ('old_old', 'old_new', 'new_new')),
                         *(num(dice[key + (m,)], 4) for m in ('old', 'new')),
                         *(num(contrast[key + (c,)], digits_for(c)) for c in CONTRASTS)])
    header = heads('Cohort', 'Seed', 'Old/old|MAE', 'Old/new|MAE', 'New/new|MAE', 'Old|Dice', 'New|Dice',
                   *(SHORT[c] for c in CONTRASTS))
    return latex_table(header, body, 'llrrrrrrrr')


def model_metrics():
    rows = {(r['cohort'], r['seed'], r['model']): r for r in read_csv(TABLES/'s2_all_model_metrics.csv')}
    body = [[cohort, seed, code(model), num(r['patient_mean_followup_dice'], 4),
             num(r['patient_mean_hd95_mm_nonempty_only'], 2), r['empty_followup_predictions'],
             r['patients_with_defined_hd95']]
            for cohort in COHORT_NAMES.values() for seed in SEEDS for model in MODELS
            for r in (rows[(cohort, seed, model)],)]
    return latex_table(heads('Cohort', 'Seed', 'Model', 'Follow-up|Dice', 'HD95 (mm),|nonempty', 'Empty|follow-ups',
                             'Defined|patients'), body, 'lllrrrr')


def replication_runs(rows, cohort):
    body = run_table_rows(rows, cohort)
    body[-1][0] = r'\hline Mean'
    return latex_table(heads(*RUN_HEADER), body, 'lrrrrr')


STRATUM_HEADER = ('Cohort', 'Volume|(mL)', 'Patients /|pairs')


def posthoc_strata(posthoc):
    return latex_table(heads(*STRATUM_HEADER, 'Old/old|MAE (mL)', 'Old/new|MAE (mL)', 'New/new|MAE (mL)',
                             'Compatibility|cost (mL)', r'Two-way 95\%|interval', 'Runs|$+$ / $-$'),
                       posthoc_stratum_rows(posthoc, 'retaining_baseline_cost'), 'llcrrrrrc')


def posthoc_update(posthoc):
    return latex_table(heads(*STRATUM_HEADER, 'Update|benefit (mL)', r'Two-way 95\%|interval', 'Runs|$+$ / $-$'),
                       posthoc_stratum_rows(posthoc, 'ordinary_update_benefit', with_mae=False), 'llcrrc')


def posthoc_sensitivity(results, posthoc):
    return latex_table(heads('Cohort', 'Contrast', r'Family-wise,|10\,000 draws',
                             r'Family-wise,|100\,000 draws', 'Established', 'Runs|$+$ / $-$', 'Sign|probability'),
                       posthoc_sensitivity_rows(results, posthoc), 'llrrccr')


def stacked(cell):
    """An 'estimate [interval]' cell as two right-aligned lines; '\\{}' keeps '[' from being read as a length."""
    if ' [' not in cell:
        return cell
    estimate, interval_text = cell.split(' [', 1)
    return r'\begin{tabular}[t]{@{}r@{}}' + estimate + r'\\{}[' + interval_text + r'\end{tabular}'


def strong_comparison(results, strong):
    rows = [[cell if i < 2 else stacked(cell) for i, cell in enumerate(row)]
            for row in model_comparison_rows(results, strong)]
    return latex_table(heads('Cohort', 'Quantity', f"U-Net,|{results['runs']} runs", 'Runs|$+$ / $-$',
                             f"nnU-Net,|{strong['runs']} runs", 'Runs|$+$ / $-$'), rows, 'llrcrc')


def interval_components(results, strong):
    """Grouped by segmenter under a spanning heading, so the table fits the text width."""
    rows = interval_component_rows(results, strong)
    header = heads('Cohort', 'Contrast', 'Mean', r'Two-way 95\%|interval', r'Patients only,|95\% interval',
                   r'Runs only,|95\% interval')
    lines = [r'\begin{tabular}{llrrrr}', r'\hline', ' & '.join(header) + r' \\', r'\hline']
    for model, runs in (('U-Net', results['runs']), ('nnU-Net', strong['runs'])):
        lines.append(rf'\multicolumn{{6}}{{l}}{{\textit{{{model}, {runs} training runs}}}} \\')
        lines += [' & '.join(row[1:]) + r' \\' for row in rows if row[0] == model]
    lines += [r'\hline', r'\end{tabular}']
    return '\n'.join(lines) + '\n'


def compat_table(result, full=False):
    """CT-UPDATE-019: one row per quantity, one column per segmenter and cohort (never pooled)."""
    rows = compat_full_rows(result) if full else compat_summary_rows(result)
    body = [[r'\raggedright ' + row[0], *(stacked(cell) for cell in row[1:])] for row in rows]
    table = latex_table(heads(*COMPAT_HEAD), body, 'lrrrr')
    # The quantity labels wrap, ragged right, in a fixed-width first column so that the table fits the text width.
    # \raggedright redefines \\ inside the cell, so these rows end with \tabularnewline (no extra package needed).
    lines = [line[:-2] + r'\tabularnewline' if line.endswith(' ' + '\\' * 2) else line for line in table.splitlines()]
    return '\n'.join(lines).replace(r'\begin{tabular}{lrrrr}', r'\begin{tabular}{p{0.27\textwidth}rrrr}', 1) + '\n'


def write_run_export(rows, filename='s6_replication_runs.csv'):
    fields = ['cohort', 'seed', *(name for _, name in RUN_COLUMNS)]
    with (TABLES/filename).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({**r, 'cohort': COHORT_NAMES[r['cohort']]} for r in rows)


def copy_figures(source, target, copies=FIGURE_COPIES):
    """Copy the print figures under their manuscript names and return the names written."""
    written = []
    for name, manuscript_name in copies:
        if name == EXAMPLE_FIGURE and not (source/name).exists():
            continue  # drawn only from private crops (CT-UPDATE-018); absent outside the authors' repository
        shutil.copyfile(source/name, target/manuscript_name)
        written.append(manuscript_name)
    return written


def main():
    results = json.loads((RECORD/'CT_REPLICATION_RESULTS_2026-09-27.json').read_text(encoding='utf-8'))
    posthoc = json.loads((RECORD/'CT_POSTHOC_RESULTS_2026-10-02.json').read_text(encoding='utf-8'))
    strong = json.loads((RECORD/'CT_STRONG_RESULTS_2026-10-06.json').read_text(encoding='utf-8'))
    rows, strong_rows = run_rows(results), run_rows(strong)
    LATEX.mkdir(exist_ok=True)
    fragments = {'tab_core_strategies': core_strategies(), 'tab_primary_contrasts': primary_contrasts(),
                 'tab_model_controls': model_controls(), 'tab_replication': replication_primary(results),
                 'tabS_strategies': all_strategies(), 'tabS_seed_core': seed_core(),
                 'tabS_model_metrics': model_metrics(), 'tabS_runs_internal': replication_runs(rows, 'internal'),
                 'tabS_runs_pantrack': replication_runs(rows, 'pantrack'), 'tab_posthoc_strata': posthoc_strata(posthoc),
                 'tabS_posthoc_update': posthoc_update(posthoc),
                 'tabS_posthoc_sensitivity': posthoc_sensitivity(results, posthoc),
                 'tab_strong_comparison': strong_comparison(results, strong),
                 'tabS_strong_primary': replication_primary(strong),
                 'tabS_strong_runs_internal': replication_runs(strong_rows, 'internal'),
                 'tabS_strong_runs_pantrack': replication_runs(strong_rows, 'pantrack'),
                 'tabS_strong_strata': posthoc_strata(strong['descriptive']),
                 'tabS_interval_components': interval_components(results, strong)}
    copies = FIGURE_COPIES
    if COMPAT_RESULTS.exists():
        compat = json.loads(COMPAT_RESULTS.read_text(encoding='utf-8'))
        fragments.update(tab_compat=compat_table(compat), tabS_compat=compat_table(compat, full=True))
        copies = FIGURE_COPIES + (COMPAT_FIGURE,)
    for name, text in fragments.items():
        (LATEX/f'{name}.tex').write_text(text, encoding='utf-8')
    write_run_export(rows)
    write_run_export(strong_rows, 's7_strong_replication_runs.csv')
    figures = copy_figures(FIGURES, LATEX, copies)
    print(f'wrote {len(fragments)} table fragments, {len(figures)} figures, s6 ({len(rows)} rows) and s7 ({len(strong_rows)} rows)')


if __name__ == '__main__':
    main()
