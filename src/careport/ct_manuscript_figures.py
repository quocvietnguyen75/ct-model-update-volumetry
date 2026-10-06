"""Print-size figures for the CT manuscript, drawn from verified aggregate values only.

Each figure is drawn at the IOP text width (153 mm) and saved uncropped, so it is placed at 100% scale
and its text prints at the stated sizes, never below 8 pt. The callers pass aggregate values; nothing
here reads patient-level data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Annotation, Text

from careport.ct_manuscript import MANUSCRIPT_CONTRASTS

TEXT_WIDTH_IN = 153 / 25.4
MIN_TEXT_PT = 8.0
CONTRASTS = MANUSCRIPT_CONTRASTS
TITLES = {'retaining_baseline_cost': 'Retention cost', 'ordinary_update_benefit': 'Update benefit',
          'contour_benefit': 'Contour benefit'}
XLABELS = {'retaining_baseline_cost': 'mL; > 0 favours\nreprocessing the baseline',
           'ordinary_update_benefit': 'mL; > 0 favours updating\nwith the old baseline kept',
           'contour_benefit': 'Dice; > 0 favours\nthe updated model'}
MARKERS = {'runs': 'o', 'original': 'D', 'estimate': 's'}
REPLICATION_LEGEND = {'runs': 'Replication run (20 new)', 'original': 'Original run (3; not pooled)',
                      'estimate': 'Replication mean, family-wise interval'}
THREE_RUN_LEGEND = {'original': 'Original run', 'estimate': 'Three-run average, family-wise interval'}
COMPONENTS = (('old_new_minus_old_old', 'Old/new − old/old'), ('old_new_minus_new_new', 'Old/new − new/new'),
              ('new_new_minus_old_old', 'New/new − old/old'))
STRATA = (('lt1', '<1 mL'), ('1to10', '1–<10 mL'), ('ge10', '≥10 mL'))
BAR_COLOURS = ('#273c75', '#d28a25', '#9cc9bc')
INK, MUTED, RUN_FILL, GRID, UPDATED_FILL = '#1a1a1a', '#555555', '#8c8c8c', '#e6e6e6', '#4d4d4d'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'axes.edgecolor': MUTED,
         'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.unicode_minus': True}


def _minus(value: float, digits: int = 2) -> str:
    return f'{value:.{digits}f}'.replace('-', '−')


def _require_panels(panels: dict, cohorts, legend: dict) -> None:
    needed = {'estimate', 'interval', 'original'} | ({'runs'} if 'runs' in legend else set())
    for name in CONTRASTS:
        for cohort, _ in cohorts:
            entry = panels.get(name, {}).get(cohort)
            if entry is None or not needed <= set(entry):
                raise ValueError(f'Missing figure values for {name} / {cohort}')


def contrast_figure(panels: dict, cohorts, legend: dict) -> Figure:
    """Three contrast panels (retention cost first) with one row per cohort.

    panels[contrast][cohort] holds 'estimate', 'interval' (family-wise), 'original' (original-run values)
    and, for the replication figure, 'runs' and 'positive'. The legend keys select what is drawn.
    """
    _require_panels(panels, cohorts, legend)
    with_runs = 'runs' in legend
    rows = {cohort: float(len(cohorts) - 1 - k) for k, (cohort, _) in enumerate(cohorts)}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.75), layout='constrained', sharey=True)
        for ax, name in zip(axes, CONTRASTS):
            for cohort, _ in cohorts:
                p, y = panels[name][cohort], rows[cohort]
                if with_runs:
                    runs = list(p['runs'])
                    step = 0.26 / max(len(runs) - 1, 1)
                    ax.scatter(runs, [y - 0.13 + step * i for i in range(len(runs))], s=14, marker=MARKERS['runs'],
                               facecolor=RUN_FILL, edgecolor='white', linewidth=0.4, zorder=3)
                    original_y, estimate_y = y + 0.26, y - 0.27
                else:
                    original_y, estimate_y = y + 0.12, y - 0.15
                ax.scatter(p['original'], [original_y] * len(p['original']), s=18, marker=MARKERS['original'],
                           facecolor='white', edgecolor=INK, linewidth=0.8, zorder=4)
                low, high = p['interval']
                ax.plot([low, high], [estimate_y] * 2, color=INK, linewidth=1.6, solid_capstyle='butt', zorder=4)
                for end in (low, high):
                    ax.plot([end, end], [estimate_y - 0.05, estimate_y + 0.05], color=INK, linewidth=1.0, zorder=4)
                ax.scatter([p['estimate']], [estimate_y], s=26, marker=MARKERS['estimate'], color=INK, zorder=5)
                if with_runs and 'positive' in p:
                    ax.annotate(f"{p['positive']}/{len(p['runs'])} > 0", xy=(1.0, y + 0.44),
                                xycoords=('axes fraction', 'data'), ha='right', va='center', fontsize=8,
                                color=MUTED, zorder=6, bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
            ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
            ax.grid(axis='x', color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            ax.set_ylim(-0.55, len(cohorts) - 0.45)
            ax.set_yticks(list(rows.values()), [label for _, label in cohorts])
            ax.tick_params(axis='y', length=0, labelsize=8)
            ax.tick_params(axis='x', labelsize=8)
            ax.set_title(TITLES[name], loc='left', fontsize=9, color=INK)
            ax.set_xlabel(XLABELS[name], fontsize=8, color=MUTED)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        handles = []
        if with_runs:
            handles.append(Line2D([], [], marker=MARKERS['runs'], linestyle='', markerfacecolor=RUN_FILL,
                                  markeredgecolor='white', markersize=5, label=legend['runs']))
        handles.append(Line2D([], [], marker=MARKERS['original'], linestyle='', markerfacecolor='white',
                              markeredgecolor=INK, markersize=4.5, label=legend['original']))
        handles.append(Line2D([], [], marker=MARKERS['estimate'], color=INK, linewidth=1.6, markersize=4,
                              label=legend['estimate']))
        # Three entries do not fit on one line at 8 pt within the text width.
        fig.legend(handles=handles, loc='outside lower center', ncol=min(len(handles), 2), frameon=False,
                   fontsize=8)
    return fig


def strategy_schematic() -> Figure:
    """Which model measures each scan under the three core strategies, and the primary contrasts."""
    height = 2.3
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(TEXT_WIDTH_IN, height))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, TEXT_WIDTH_IN)
        ax.set_ylim(0, height)
        ax.axis('off')
        centres, box_w, box_h = {'baseline': 2.05, 'followup': 3.25}, 1.05, 0.34
        ax.text(centres['baseline'], height - 0.2, 'Baseline scan', ha='center', va='center', fontsize=9,
                fontweight='bold', color=INK)
        ax.text(centres['followup'], height - 0.2, 'Follow-up scan', ha='center', va='center', fontsize=9,
                fontweight='bold', color=INK)
        strategies = (('old/old', 'Old model at both visits', ('old', 'old')),
                      ('old/new', 'Retain the old baseline', ('old', 'new')),
                      ('new/new', 'Reprocess the baseline', ('new', 'new')))
        ys = (height - 0.62, height - 1.1, height - 1.58)
        for (code, description, models), y in zip(strategies, ys):
            ax.text(0.08, y + 0.08, code, va='center', fontsize=9, fontweight='bold', color=INK)
            ax.text(0.08, y - 0.1, description, va='center', fontsize=8, color=INK)
            for column, model in zip(('baseline', 'followup'), models):
                x = centres[column]
                fill, ink = ('white', INK) if model == 'old' else (UPDATED_FILL, 'white')
                ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                            boxstyle='round,pad=0.02,rounding_size=0.05', facecolor=fill,
                                            edgecolor=INK, linewidth=0.8))
                ax.text(x, y, 'Old model' if model == 'old' else 'Updated model', ha='center', va='center',
                        fontsize=8, color=ink)
        x0 = 3.92

        def bracket(top: float, bottom: float, label: str) -> None:
            ax.plot([x0, x0 + 0.08, x0 + 0.08, x0], [top, top, bottom, bottom], color=INK, linewidth=0.8)
            ax.text(x0 + 0.16, (top + bottom) / 2, label, va='center', fontsize=8, color=INK)

        bracket(ys[0], ys[1] + 0.04, 'Update benefit\nMAE(old/old) − MAE(old/new)')
        bracket(ys[1] - 0.04, ys[2], 'Retention cost\nMAE(old/new) − MAE(new/new)')
        ax.text((centres['baseline'] + centres['followup']) / 2, 0.24,
                'Contour benefit: follow-up Dice of the updated model minus that of the old model',
                ha='center', va='center', fontsize=8, color=INK)
    return fig


def diagnostic_figure(batches: list[dict], labels: list[str], strata_patients: dict) -> Figure:
    """Post hoc external diagnostics: the old/new − old/old difference, its two parts, and size strata."""
    if len(batches) != len(labels):
        raise ValueError('One label is needed per seed or average')
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN, 4.8), layout='constrained')
        panels = (('components', COMPONENTS, 'A  Old/new − old/old and its two parts'),
                  ('strata', tuple((key, f'{name} ({strata_patients[key]} patients)') for key, name in STRATA),
                   'B  Old/new − old/old by baseline-size stratum (post hoc)'))
        for ax, (field, series, title) in zip(axes, panels):
            values_all = []
            for i, ((key, label), colour) in enumerate(zip(series, BAR_COLOURS)):
                values = [float(b[field][key]) for b in batches]
                values_all += values
                bars = ax.bar([k + (i - 1) * 0.26 for k in range(len(batches))], values, width=0.24, color=colour,
                              label=label, zorder=3)
                ax.bar_label(bars, labels=[_minus(v) for v in values], padding=2, fontsize=8, rotation=90)
            low, high = min(values_all + [0.0]), max(values_all + [0.0])
            span = high - low
            # Rotated value labels need room beyond the longest bar in each direction.
            ax.set_ylim(low - (0.5 if low < 0 else 0.05) * span, high + 0.45 * span)
            ax.axhline(0, color=MUTED, linewidth=0.7, zorder=2)
            ax.grid(axis='y', color=GRID, linewidth=0.6, zorder=0)
            ax.set_xticks(range(len(batches)), labels)
            ax.tick_params(labelsize=8)
            ax.set_ylabel('MAE difference (mL)', fontsize=8)
            ax.set_title(title, loc='left', fontsize=9, color=INK)
            ax.legend(loc='upper left', frameon=False, fontsize=8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
        axes[-1].set_xlabel('Original training run', fontsize=8)
    return fig


def _visible_texts(fig: Figure) -> list[Text]:
    """Texts that are actually drawn; tick labels outside an axis's view limits are never drawn."""
    fig.canvas.draw()
    undrawn = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            span = high - low
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                location = tick.get_loc()
                if location is None or not low - 1e-9 * span <= location <= high + 1e-9 * span:
                    undrawn.update({id(tick.label1), id(tick.label2)})
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in undrawn]


def text_sizes(fig: Figure) -> list[float]:
    """Font sizes (pt) of every visible, nonempty text element after layout."""
    return [float(t.get_fontsize()) for t in _visible_texts(fig)]


def figure_texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in _visible_texts(fig)]


def text_outside_bounds(fig: Figure, tolerance_px: float = 1.0) -> list[str]:
    """Texts cut by the page edge, and value annotations that leave their own axes."""
    texts = _visible_texts(fig)
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in texts:
        box = text.get_window_extent(renderer)
        limits = [fig.bbox]
        if isinstance(text, Annotation) and text.axes is not None:
            limits.append(text.axes.bbox)
        for limit in limits:
            if (box.x0 < limit.x0 - tolerance_px or box.x1 > limit.x1 + tolerance_px
                    or box.y0 < limit.y0 - tolerance_px or box.y1 > limit.y1 + tolerance_px):
                outside.append(text.get_text())
                break
    return outside


def save_pdf(fig: Figure, path: Path, overwrite: bool = True) -> None:
    """Uncropped PDF at the figure's exact size, with embedded TrueType fonts and no creation date."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({'pdf.fonttype': 42}):
        fig.savefig(path, format='pdf', metadata={'CreationDate': None})
