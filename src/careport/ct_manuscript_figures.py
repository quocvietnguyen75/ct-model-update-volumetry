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
from matplotlib.patches import Circle, FancyBboxPatch
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
# Model versions in the explainer (old: amber, updated: navy) and segmenters in the model comparison (U-Net: black
# squares, nnU-Net: teal circles); the two encodings never share a colour, and each pair differs in greyscale too.
VERSION_INK = {'old': '#b8731a', 'new': '#273c75'}
VERSION_FILL = {'old': '#fbe9cf', 'new': '#273c75'}
MODEL_STYLE = {'unet': dict(ink=INK, runs=RUN_FILL, marker='s'), 'nnunet': dict(ink='#16706b', runs='#9cc9bc', marker='o')}
MODELS = tuple(MODEL_STYLE)
COHORT_KEYS = ('internal', 'pantrack')
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


def update_explainer() -> Figure:
    """Illustration, not data: (a) how a measurement that mixes model versions takes up the between-version
    difference at baseline, and (b) which model measures each scan under the three core strategies, with the
    primary contrasts."""
    width, height = TEXT_WIDTH_IN, 4.5
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width, height))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.axis('off')
        renderer = fig.canvas.get_renderer()

        def heading(y: float, letter: str, title: str) -> None:
            ax.text(0.04, y, letter, va='center', fontsize=10, fontweight='bold', color=INK)
            ax.text(0.26, y, title, va='center', fontsize=9, fontweight='bold', color=INK)

        # (a) One lesion pair; circles are drawn to scale only within each scan and carry no study data.
        heading(height - 0.16, 'a', 'How mixing model versions enters the measured change')
        tile, top = 1.12, height - 0.38
        radii = {'Baseline scan': (0.27, 0.17, 0.25), 'Follow-up scan': (0.39, 0.27, 0.36)}
        for x0, (label, (r_ref, r_old, r_new)) in zip((0.3, 1.62), radii.items()):
            centre = (x0 + tile / 2, top - tile / 2)
            ax.add_patch(FancyBboxPatch((x0, top - tile), tile, tile, boxstyle='round,pad=0,rounding_size=0.08',
                                        facecolor='#dedede', edgecolor='none'))
            ax.add_patch(Circle(centre, r_ref, facecolor='#a9a9a9', edgecolor=INK, linewidth=1.0, linestyle=(0, (3, 2))))
            ax.add_patch(Circle(centre, r_old, facecolor='none', edgecolor=VERSION_INK['old'], linewidth=1.6))
            ax.add_patch(Circle(centre, r_new, facecolor='none', edgecolor=VERSION_INK['new'], linewidth=1.6))
            ax.text(x0 + tile / 2, top - tile - 0.15, label, ha='center', va='center', fontsize=9, color=INK)
        # Contour key under the tiles, too wide for one row there: the reference, then the two model versions.
        key_y, key_row = top - tile - 0.42, 0.19
        for row, entries in enumerate(((('Reference', INK, (0, (3, 2))),),
                                       (('Old model', VERSION_INK['old'], '-'),
                                        ('Updated model', VERSION_INK['new'], '-')))):
            x, y = 0.3, key_y - key_row * row
            for text, colour, style in entries:
                ax.plot([x, x + 0.22], [y] * 2, color=colour, linewidth=1.6 if style == '-' else 1.0,
                        linestyle=style, gid=f'key:{text}:handle')
                label = ax.text(x + 0.29, y, text, va='center', fontsize=8, color=INK, gid=f'key:{text}:label')
                x += 0.29 + label.get_window_extent(renderer).width / fig.dpi + 0.25
        lines = (('One version (new/new)', 'bold'), ('change = FU(new) − BL(new)', 'normal'), ('', 'normal'),
                 ('Mixed versions (old/new)', 'bold'), ('change = FU(new) − BL(old)', 'normal'),
                 ('= [FU(new) − BL(new)] + ', 'normal'), ('', 'normal'),
                 ('Any difference between the versions at baseline,', 'normal'),
                 ('in either direction, enters the change error.', 'normal'))
        for k, (text, weight) in enumerate(lines):
            if not text:
                continue
            y = top - 0.08 - 0.165 * k
            shown = ax.text(3.08, y, text, va='center', fontsize=8, fontweight=weight, color=INK)
            if text.startswith('= ['):
                # The between-version term, in the updated model's colour, follows the same-version change.
                width = shown.get_window_extent(renderer).width / fig.dpi
                ax.text(3.08 + width, y, '[BL(new) − BL(old)]', va='center', fontsize=8, fontweight='bold',
                        color=VERSION_INK['new'])

        # (b) The three core strategies and the primary contrasts.
        b_top = key_y - key_row - 0.36
        heading(b_top, 'b', 'Measurement strategies and primary contrasts')
        centres, box_w, box_h = {'baseline': 2.05, 'followup': 3.25}, 1.05, 0.3
        for column, label in (('baseline', 'Baseline scan'), ('followup', 'Follow-up scan')):
            ax.text(centres[column], b_top - 0.3, label, ha='center', va='center', fontsize=8, color=MUTED)
        strategies = (('old/old', 'Old model at both visits', ('old', 'old')),
                      ('old/new', 'Retain the old baseline', ('old', 'new')),
                      ('new/new', 'Reprocess the baseline', ('new', 'new')))
        ys = [b_top - 0.66 - 0.42 * k for k in range(3)]
        for (code, description, models), y in zip(strategies, ys):
            ax.text(0.08, y + 0.08, code, va='center', fontsize=9, fontweight='bold', color=INK)
            ax.text(0.08, y - 0.1, description, va='center', fontsize=8, color=INK)
            for column, model in zip(('baseline', 'followup'), models):
                x = centres[column]
                ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                            boxstyle='round,pad=0.02,rounding_size=0.05',
                                            facecolor=VERSION_FILL[model], edgecolor=VERSION_INK[model],
                                            linewidth=0.9))
                ax.text(x, y, 'Old model' if model == 'old' else 'Updated model', ha='center', va='center',
                        fontsize=8, color=INK if model == 'old' else 'white')
        x0 = 3.92

        def bracket(upper: float, lower: float, label: str) -> None:
            ax.plot([x0, x0 + 0.08, x0 + 0.08, x0], [upper, upper, lower, lower], color=INK, linewidth=0.8)
            ax.text(x0 + 0.16, (upper + lower) / 2, label, va='center', fontsize=8, color=INK)

        bracket(ys[0], ys[1] + 0.04, 'Update benefit\nMAE(old/old) − MAE(old/new)')
        bracket(ys[1] - 0.04, ys[2], 'Retention cost\nMAE(old/new) − MAE(new/new)')
        ax.text((centres['baseline'] + centres['followup']) / 2, ys[2] - 0.38,
                'Contour benefit: follow-up Dice of the updated model minus that of the old model',
                ha='center', va='center', fontsize=8, color=INK)
    return fig


def _volume_text(lower: float, upper: float | None) -> str:
    if upper is None:
        return f'≥{lower:g} mL'
    return f'<{upper:g} mL' if lower == 0 else f'{lower:g}–<{upper:g} mL'


def model_comparison_panels(unet: dict, unet_strata: dict, nnunet: dict, nnunet_strata: dict) -> dict:
    """Values for the U-Net/nnU-Net figure, taken unchanged from the verified result files.

    'primary' holds each primary contrast per cohort and model: the mean, its family-wise interval and the per-run
    means. 'strata' holds, per cohort, the descriptive retention cost by baseline-volume stratum (mean, two-way 95%
    interval and per-run means) with a label giving the volume range and patients / pairs. The two replications
    measured the same lesions, so their strata must agree exactly.
    """
    primary = {name: {cohort: {} for cohort in COHORT_KEYS} for name in CONTRASTS}
    for model, results in (('unet', unet), ('nnunet', nnunet)):
        for cohort in COHORT_KEYS:
            for name in CONTRASTS:
                summary = results['cohorts'][cohort]['primary_contrasts'][name]
                primary[name][cohort][model] = dict(estimate=summary['mean'],
                                                    interval=tuple(summary['two_way_family_interval']),
                                                    runs=list(summary['per_run_means']))
    strata = {}
    for cohort in COHORT_KEYS:
        first, second = unet_strata['cohorts'][cohort]['size_strata'], nnunet_strata['cohorts'][cohort]['size_strata']
        identity = [[(e['lower_ml'], e['upper_ml'], e['patients'], e['pairs']) for e in rows] for rows in (first, second)]
        if identity[0] != identity[1]:
            raise ValueError(f'Size strata differ between the two replications in {cohort}')
        rows = []
        for u_entry, n_entry in zip(first, second):
            row = dict(label=f"{_volume_text(u_entry['lower_ml'], u_entry['upper_ml'])}\n"
                             f"{u_entry['patients']} / {u_entry['pairs']}")
            for model, entry in (('unet', u_entry), ('nnunet', n_entry)):
                summary = entry.get('retaining_baseline_cost') or {}
                if summary.get('two_way_percentile_95') is None:
                    raise ValueError(f"No retention-cost interval for {cohort} {row['label']!r} ({model})")
                row[model] = dict(estimate=summary['mean'], interval=tuple(summary['two_way_percentile_95']),
                                  runs=list(summary.get('per_run_means', [])))
            rows.append(row)
        strata[cohort] = rows
    return dict(primary=primary, strata=strata)


def _model_band(ax, y: float, values: dict, model: str, gid: str, with_runs: bool = True) -> None:
    """One model's runs (small dots), interval and estimate on a horizontal band centred at y."""
    style = MODEL_STYLE[model]
    runs = values['runs']
    if with_runs and runs:
        step = 0.16 / max(len(runs) - 1, 1)
        ax.scatter(runs, [y - 0.08 + step * i for i in range(len(runs))], s=9, marker='o', facecolor=style['runs'],
                   edgecolor='white', linewidth=0.3, zorder=3, gid=f'{gid}:runs')
    low, high = values['interval']
    ax.plot([low, high], [y, y], color=style['ink'], linewidth=1.5, solid_capstyle='butt', zorder=4,
            gid=f'{gid}:interval')
    for end in (low, high):
        ax.plot([end, end], [y - 0.045, y + 0.045], color=style['ink'], linewidth=1.0, zorder=4)
    ax.scatter([values['estimate']], [y], s=24, marker=style['marker'], color=style['ink'], edgecolor='white',
               linewidth=0.5, zorder=5, gid=f'{gid}:estimate')


def _tidy(ax, title: str, xlabel: str) -> None:
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.grid(axis='x', color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', length=0, labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title(title, loc='left', fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def model_comparison_figure(panels: dict, cohorts, legend: dict) -> Figure:
    """(a) The three primary contrasts of both replications; (b) the descriptive retention cost by baseline volume.

    cohorts: ((key, row label), ...); legend: model key -> display name. Every drawn estimate, interval and run
    carries a gid naming its panel, cohort, row and model, so tests can compare the drawing with the values.
    """
    offset = {'unet': 0.17, 'nnunet': -0.17}
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(TEXT_WIDTH_IN, 5.6), layout='constrained')
        top_fig, bottom_fig = fig.subfigures(2, 1, height_ratios=(1.0, 1.08))
        top_fig.suptitle('a   Primary contrasts of the two replications', x=0.0, ha='left', fontsize=9,
                         fontweight='bold', color=INK)
        top_axes = top_fig.subplots(1, 3, sharey=True)
        rows = {cohort: float(len(cohorts) - 1 - k) for k, (cohort, _) in enumerate(cohorts)}
        for ax, name in zip(top_axes, CONTRASTS):
            for cohort, _ in cohorts:
                for model in MODELS:
                    _model_band(ax, rows[cohort] + offset[model], panels['primary'][name][cohort][model], model,
                                f'primary:{name}:{cohort}:{model}')
            ax.set_ylim(-0.5, len(cohorts) - 0.5)
            ax.set_yticks(list(rows.values()), [label for _, label in cohorts])
            _tidy(ax, TITLES[name], XLABELS[name])
        # Post hoc for the U-Net (CT-UPDATE-016), but specified before nnU-Net training (CT-UPDATE-017).
        bottom_fig.suptitle('b   Retention cost by reference baseline volume (descriptive)', x=0.0, ha='left',
                            fontsize=9, fontweight='bold', color=INK)
        bottom_axes = bottom_fig.subplots(1, 2)
        for ax, (cohort, label) in zip(bottom_axes, cohorts):
            strata = panels['strata'][cohort]
            ys = [float(len(strata) - 1 - k) for k in range(len(strata))]
            for k, (row, y) in enumerate(zip(strata, ys)):
                for model in MODELS:
                    # Single runs within a stratum can be extreme and would set the axis; panel (b) shows the
                    # stratum means and their intervals only.
                    _model_band(ax, y + offset[model], row[model], model, f'strata:{cohort}:{k}:{model}',
                                with_runs=False)
            ax.set_ylim(-0.5, len(strata) - 0.5)
            ax.set_yticks(ys, [row['label'] for row in strata])
            _tidy(ax, label.split('\n')[0], XLABELS['retaining_baseline_cost'])
        handles = []
        for model in MODELS:
            style = MODEL_STYLE[model]
            handles.append(Line2D([], [], marker='o', linestyle='', markerfacecolor=style['runs'],
                                  markeredgecolor='white', markersize=4.5, label=f'{legend[model]}: training run (a)'))
            handles.append(Line2D([], [], marker=style['marker'], color=style['ink'], linewidth=1.5, markersize=4.5,
                                  label=f'{legend[model]}: mean and interval'))
        fig.legend(handles=handles, loc='outside lower center', ncol=2, frameon=False, fontsize=8)
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


def text_outside_bounds(fig: Figure, tolerance_px: float = 0.5) -> list[str]:
    """Texts cut by the page edge, and value annotations that leave their own axes.

    Text flush with the edge is whole; a box even 1 px past it loses its descenders in print.
    """
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
