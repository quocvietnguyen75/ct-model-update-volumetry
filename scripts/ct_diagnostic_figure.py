"""Render aggregate-only CT-UPDATE-013 figures; no patient records are read."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--output-stem', type=Path, required=True)
args = parser.parse_args()
raw = args.results.read_bytes()
if hashlib.sha256(raw).hexdigest() != '176640452e16d49e102622f1798b806339ee03235fe082ddab932b1b45e070e9':
    raise ValueError('Figure input differs from the verified diagnostic result')
results = json.loads(raw)
batches = results['per_seed'] + [results['seed_averaged']]
labels = ['20260915', '20260917', '20260919', 'Fixed-seed\naverage']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), layout='constrained')
colors = ['#273c75', '#d28a25', '#598a7c']
metrics = ['mixed_minus_old_ml', 'mixed_minus_same_ml', 'same_minus_old_ml']
names = ['Mixed − old/old', 'Mixed − new/new', 'New/new − old/old']
x = np.arange(4)
for i, (metric, name, color) in enumerate(zip(metrics, names, colors)):
    values = [b['models']['new'][metric] for b in batches]
    bars = axes[0].bar(x+(i-1)*.25, values, width=.23, label=name, color=color)
    axes[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
axes[0].set_title('A  Error differences include two components', loc='left', fontsize=11)
axes[0].set_xticks(x, labels)
axes[0].set_ylabel('Patient-weighted MAE difference (mL)')
axes[0].set_ylim(-2.1, 8.1)
axes[0].legend(loc='upper left', frameon=False, fontsize=8)

for i, (label, name, color) in enumerate(zip(['lt1','1to10','ge10'],
        ['<1 mL (6 patients)', '1–<10 mL (24 patients)', '≥10 mL (36 patients)'], colors)):
    values = [b['size_strata'][label]['models']['new']['mixed_minus_old_ml'] for b in batches]
    bars = axes[1].bar(x+(i-1)*.25, values, width=.23, label=name, color=color)
    axes[1].bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
axes[1].set_title('B  The larger-lesion stratum carries the increase', loc='left', fontsize=11)
axes[1].set_xticks(x, labels)
axes[1].set_ylabel('Mixed minus old/old MAE (mL)')
axes[1].set_ylim(-2.5, 11.4)
axes[1].legend(loc='upper left', frameon=False, fontsize=8)
for ax in axes:
    ax.axhline(0, color='#5b6270', linewidth=.7)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.16)
    ax.set_xlabel('Training seed')
fig.suptitle('PanTrack: 45 patients, 181 pairs — post-hoc descriptive diagnostics', fontsize=12)
fig.supxlabel('Positive differences mean higher error. Patients may appear in several size strata. No new inference or patient exclusions.', fontsize=9)
args.output_stem.parent.mkdir(parents=True, exist_ok=True)
for suffix in ('.png', '.pdf'):
    destination = args.output_stem.with_suffix(suffix)
    if destination.exists():
        raise FileExistsError(destination)
    fig.savefig(destination, dpi=220)
plt.close(fig)
