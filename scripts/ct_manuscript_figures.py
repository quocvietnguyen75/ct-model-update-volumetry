"""Write the print-size CT manuscript figures from the verified aggregate files.

Run from the repository root with PYTHONPATH=src. Reads aggregates only, except that the example-pair figure
(amendment CT-UPDATE-018) is drawn when CT_EXAMPLE_PRIVATE_NPZ names its private crop file; without it the figure
is skipped. Writes new PDFs to newpaper/ct_manuscript/figures/print/; the earlier hash-bound figures and receipts are
not touched. scripts/ct_manuscript_latex.py then copies these files into the submission folder.
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from careport.ct_manuscript_figures import (REPLICATION_LEGEND, THREE_RUN_LEGEND, contrast_figure,
                                            diagnostic_figure, example_figure, example_from_arrays,
                                            model_comparison_figure, model_comparison_panels, save_pdf,
                                            update_explainer)
from careport.ct_replication_report import COHORT_NAMES, COHORTS, ORIGINAL_SEEDS, PRIMARY, panel_data

RECORD = Path('newpaper/research_selection_2026-09-13')
MANUSCRIPT = Path('newpaper/ct_manuscript')
OUT = MANUSCRIPT/'figures'/'print'
DIAGNOSTIC_SHA256 = '176640452e16d49e102622f1798b806339ee03235fe082ddab932b1b45e070e9'


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def cohort_labels(results):
    return tuple((c, f"{COHORT_NAMES[c]}\n{results['cohorts'][c]['patients']} patients\n"
                     f"{results['cohorts'][c]['pairs']} pairs") for c in COHORTS)


def replication_panels(results):
    data = panel_data(results, read_csv(RECORD/'CT_HOLDOUT_PRIMARY_CONTRASTS_2026-09-25.csv'))
    return {name: {c: dict(estimate=p['mean'], interval=tuple(p['family']), original=p['original'],
                           runs=p['replication'], positive=p['runs_positive'])
                   for c, p in data[name].items()} for name in PRIMARY}


def three_run_panels():
    average = {(r['cohort'], r['contrast']): r for r in read_csv(MANUSCRIPT/'tables'/'table3_primary_contrasts.csv')}
    seeds = {(r['cohort'], r['seed'], r['contrast']): float(r['mean'])
             for r in read_csv(MANUSCRIPT/'tables'/'s3_all_contrasts.csv')}
    panels = {}
    for name in PRIMARY:
        panels[name] = {}
        for cohort in COHORTS:
            row = average[(COHORT_NAMES[cohort], name)]
            if row['seed'] != 'fixed_seed_average' or row['reporting_role'] != 'primary_family':
                raise ValueError(f'Unexpected three-run summary row for {cohort} / {name}')
            panels[name][cohort] = dict(estimate=float(row['mean']),
                                        interval=(float(row['bonferroni_family_low']),
                                                  float(row['bonferroni_family_high'])),
                                        original=[seeds[(COHORT_NAMES[cohort], s, name)] for s in ORIGINAL_SEEDS])
    return panels


def diagnostic_inputs():
    raw = (RECORD/'CT_DIAGNOSTIC_RESULTS_2026-09-24.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIAGNOSTIC_SHA256:
        raise ValueError('Diagnostic input differs from the verified result')
    results = json.loads(raw)
    batches = []
    for batch in results['per_seed'] + [results['seed_averaged']]:
        new = batch['models']['new']
        batches.append(dict(components={'old_new_minus_old_old': new['mixed_minus_old_ml'],
                                        'old_new_minus_new_new': new['mixed_minus_same_ml'],
                                        'new_new_minus_old_old': new['same_minus_old_ml']},
                            strata={k: batch['size_strata'][k]['models']['new']['mixed_minus_old_ml']
                                    for k in ('lt1', '1to10', 'ge10')}))
    patients = {k: results['seed_averaged']['size_strata'][k]['patients'] for k in ('lt1', '1to10', 'ge10')}
    labels = [str(seed) for seed in results['seeds']] + ['Three-run\naverage']
    return batches, labels, patients


def main():
    results = json.loads((RECORD/'CT_REPLICATION_RESULTS_2026-09-27.json').read_text(encoding='utf-8'))
    posthoc = json.loads((RECORD/'CT_POSTHOC_RESULTS_2026-10-02.json').read_text(encoding='utf-8'))
    strong = json.loads((RECORD/'CT_STRONG_RESULTS_2026-10-06.json').read_text(encoding='utf-8'))
    replication, three_run = replication_panels(results), three_run_panels()
    for name in PRIMARY:  # Both figures must show the same three original runs.
        for cohort in COHORTS:
            a, b = replication[name][cohort]['original'], three_run[name][cohort]['original']
            if any(abs(x - y) > 1e-9 for x, y in zip(a, b)):
                raise ValueError(f'Original-run values disagree for {cohort} / {name}')
    labels = cohort_labels(results)
    models = model_comparison_panels(results, posthoc, strong, strong['descriptive'])
    figures = {'figure1_update_explainer.pdf': update_explainer(),
               'figure2_replication_runs.pdf': contrast_figure(replication, labels, REPLICATION_LEGEND),
               'figure3_model_comparison.pdf': model_comparison_figure(models, labels,
                                                                       {'unet': 'U-Net', 'nnunet': 'nnU-Net'}),
               'figure5_three_run_contrasts.pdf': contrast_figure(three_run, labels, THREE_RUN_LEGEND),
               'figureS1_pantrack_diagnostics.pdf': diagnostic_figure(*diagnostic_inputs())}
    private = os.environ.get('CT_EXAMPLE_PRIVATE_NPZ')
    if private:  # Patient-level crops: read from outside the repository, never written into it except as this figure.
        with np.load(private, allow_pickle=False) as arrays:
            figures['figure4_example_pair.pdf'] = example_figure(example_from_arrays(arrays))
    else:
        print('example-pair figure skipped: CT_EXAMPLE_PRIVATE_NPZ is not set')
    for name, fig in figures.items():
        save_pdf(fig, OUT/name)
    print(f'wrote {len(figures)} figures to {OUT}')


if __name__ == '__main__':
    main()
