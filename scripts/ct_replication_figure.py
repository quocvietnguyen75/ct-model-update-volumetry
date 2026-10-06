"""Render the CT-UPDATE-015 manuscript figure and primary table from verified aggregate files.

Run from the repository root with PYTHONPATH=src. Reads aggregates only.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from careport.ct_replication_report import COHORTS, PRIMARY, markdown_table, panel_data

RECORD = Path('newpaper/research_selection_2026-09-13')
OUT = Path('newpaper/ct_manuscript')
TITLES = {'retaining_baseline_cost': 'Baseline-retention cost',
          'ordinary_update_benefit': 'Ordinary-update benefit',
          'contour_benefit': 'Contour benefit'}
XLABELS = {'retaining_baseline_cost': 'mL; > 0 favours\nreprocessing the baseline',
           'ordinary_update_benefit': 'mL; > 0 favours updating\nwith the old baseline kept',
           'contour_benefit': 'Dice; > 0 favours\nthe updated model'}
ROWS = {'internal': 1.0, 'pantrack': 0.0}
INK, MUTED, RUN_FILL, GRID = '#1a1a1a', '#555555', '#8c8c8c', '#e6e6e6'


def main():
    results = json.loads((RECORD/'CT_REPLICATION_RESULTS_2026-09-27.json').read_text())
    with (RECORD/'CT_HOLDOUT_PRIMARY_CONTRASTS_2026-09-25.csv').open(newline='', encoding='utf-8') as stream:
        original = list(csv.DictReader(stream))
    panels = panel_data(results, original)
    plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED,
                         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), gridspec_kw=dict(wspace=0.12))
    offsets = np.linspace(-0.13, 0.13, results['runs'])
    for ax, name in zip(axes, PRIMARY):
        for cohort in COHORTS:
            p, y = panels[name][cohort], ROWS[cohort]
            ax.scatter(p['replication'], y + offsets, s=14, facecolor=RUN_FILL, edgecolor='white', linewidth=0.4,
                       zorder=3)
            ax.scatter(p['original'], [y + 0.26]*3, s=18, marker='D', facecolor='white', edgecolor=INK,
                       linewidth=0.8, zorder=4)
            lo, hi = p['family']
            ax.plot([lo, hi], [y - 0.27]*2, color=INK, linewidth=1.6, solid_capstyle='butt', zorder=4)
            ax.plot([lo, lo], [y - 0.32, y - 0.22], color=INK, linewidth=1.0)
            ax.plot([hi, hi], [y - 0.32, y - 0.22], color=INK, linewidth=1.0)
            ax.scatter([p['mean']], [y - 0.27], s=26, marker='s', color=INK, zorder=5)
            ax.annotate(f"{p['runs_positive']}/20 > 0", xy=(1.0, y + 0.44), xycoords=('axes fraction', 'data'),
                        ha='right', va='center', fontsize=7, color=MUTED, zorder=6,
                        bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
        ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
        ax.grid(axis='x', color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([ROWS['internal'], ROWS['pantrack']])
        ax.set_yticklabels(['Internal\n29 patients', 'PanTrack\n45 patients'] if name == PRIMARY[0] else ['', ''])
        ax.tick_params(axis='y', length=0)
        ax.set_title(TITLES[name], fontsize=8.5, color=INK, loc='left')
        ax.set_xlabel(XLABELS[name], fontsize=7, color=MUTED)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    handles = [Line2D([], [], marker='o', linestyle='', markerfacecolor=RUN_FILL, markeredgecolor='white',
                      markersize=5, label='Replication run (20 new seeds)'),
               Line2D([], [], marker='D', linestyle='', markerfacecolor='white', markeredgecolor=INK,
                      markersize=4.5, label='Original run (3 seeds; not pooled)'),
               Line2D([], [], marker='s', color=INK, linewidth=1.6, markersize=4,
                      label='Replication mean with six-contrast family interval')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.27)
    (OUT/'figures').mkdir(exist_ok=True)
    for suffix, kwargs in (('pdf', {}), ('png', {'dpi': 300})):
        fig.savefig(OUT/'figures'/f'figure2_replication_runs.{suffix}', bbox_inches='tight', **kwargs)
    (OUT/'tables').mkdir(exist_ok=True)
    (OUT/'tables'/'table5_replication_primary.md').write_text(markdown_table(results) + '\n', encoding='utf-8')
    print('wrote figure2_replication_runs.pdf/.png and table5_replication_primary.md')


if __name__ == '__main__':
    main()
