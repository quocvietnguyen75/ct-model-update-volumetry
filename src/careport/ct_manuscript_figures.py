"""Print-size figures for the CT manuscript, drawn from verified aggregate values only.

Each figure is drawn at the IOP text width (153 mm) and saved uncropped, so it is placed at 100% scale
and its text prints at the stated sizes, never below 8 pt. The callers pass aggregate values; nothing
here reads patient-level data.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
# Model versions share one palette in Figures 1 and 4: amber for the old model and sky blue for the updated one
# (Okabe-Ito hues, kept apart under colour-vision deficiency). Full hues draw outlines; light tints carry text. The
# segmenters in the model comparison (U-Net: black squares, nnU-Net: teal circles) never share these colours.
VERSION_LINE = {'old': '#f5a623', 'new': '#56b4e9'}
VERSION_FILL = {'old': '#fdecd2', 'new': '#dcf0fb'}
VERSION_TERM = '#1d5f8a'  # the version-difference term of the identity, readable on its highlight
BAND = (*matplotlib.colors.to_rgb(VERSION_LINE['new']), 0.65)  # the difference between versions, translucent
SCAN = {'background': '#1c2025', 'organ': '#7d848c', 'lesion': '#4d535a'}  # stylised scan tiles (Figure 1)
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


def _radius(theta, radius: float, harmonics) -> np.ndarray:
    """Radius of a smooth closed shape: radius * (1 + sum of a*cos(k*theta + phase)) over (k, a, phase)."""
    return radius * (1 + sum(a * np.cos(k * theta + phase) for k, a, phase in harmonics))


def _shape(centre, radius: float, harmonics, scale: float = 1.0, smooth: float = 1.0,
           shift=(0.0, 0.0)) -> tuple:
    """A shape as (centre, radius, harmonics), optionally scaled, smoothed (weaker harmonics) and moved."""
    return ((centre[0] + shift[0], centre[1] + shift[1]), radius * scale,
            tuple((k, a * smooth, phase) for k, a, phase in harmonics))


def _outline(shape, offset=(0.0, 0.0), points: int = 360) -> np.ndarray:
    (cx, cy), radius, harmonics = shape
    theta = np.linspace(0, 2 * np.pi, points, endpoint=False)
    r = _radius(theta, radius, harmonics)
    return np.column_stack([offset[0] + cx + r * np.cos(theta), offset[1] + cy + r * np.sin(theta)])


def _scan_tile(size: float, organ, lesion, pixels: int = 360) -> np.ndarray:
    """A stylised scan (an illustration, not data): a softly shaded organ on a dark field, with a darker lesion."""
    half = size / 2
    u = np.linspace(-half, half, pixels)
    x, y = np.meshgrid(u, -u)  # image row 0 is the top of the tile

    def weight(shape, softness):
        (cx, cy), radius, harmonics = shape
        rho, theta = np.hypot(x - cx, y - cy), np.arctan2(y - cy, x - cx)
        return 1 / (1 + np.exp(np.clip((rho - _radius(theta, radius, harmonics)) / softness, -50, 50)))

    colour = {k: np.array(matplotlib.colors.to_rgb(v)) for k, v in SCAN.items()}
    (ox, oy), oradius, _ = organ
    shade = 0.84 + 0.16 * np.clip(1 - ((x - ox) ** 2 + (y - oy) ** 2) / oradius ** 2, 0, 1)
    in_organ, in_lesion = weight(organ, 0.006)[..., None], weight(lesion, 0.012)[..., None]
    image = colour['background'] * (1 - in_organ) + colour['organ'] * shade[..., None] * in_organ
    image = image * (1 - in_lesion * in_organ) + colour['lesion'] * in_lesion * in_organ
    return np.clip(image, 0, 1)


def update_explainer() -> Figure:
    """Illustration, not data: (a) how a measurement that mixes model versions takes up the between-version
    difference at baseline, and (b) which model measures each scan under the three core strategies, with the
    primary contrasts."""
    from matplotlib import patheffects
    from matplotlib.patches import PathPatch, Rectangle
    from matplotlib.path import Path as MplPath
    width, height = TEXT_WIDTH_IN, 4.5
    edge = [patheffects.withStroke(linewidth=2.6, foreground='#111111')]
    styles = {'reference': dict(color='white', linewidth=1.1, linestyle=(0, (3, 2))),
              'old': dict(color=VERSION_LINE['old'], linewidth=1.4), 'new': dict(color=VERSION_LINE['new'], linewidth=1.4)}
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

        # (a) One lesion pair on stylised scans; the shapes are drawn to scale within each scan and carry no data.
        heading(height - 0.16, 'a', 'How mixing model versions enters the measured change')
        tile, top = 1.15, height - 0.38
        organ = ((0.16, -0.06), 0.58, ((2, 0.07, 0.5), (3, 0.04, 2.0)))
        lesions = {'baseline': ((-0.04, 0.03), 0.23, ((2, 0.09, 0.4), (3, 0.06, 2.3), (5, 0.025, 1.1))),
                   'followup': ((-0.04, 0.03), 0.30, ((2, 0.10, 0.6), (3, 0.05, 2.6), (5, 0.03, 0.8)))}
        # The models trace smoother outlines than the reference; the old model's lies inside the updated one's.
        fits = {'baseline': (0.93, 0.74), 'followup': (0.95, 0.80)}
        outlines = {}
        for x0, (scan, label) in zip((0.3, 1.75), (('baseline', 'Baseline scan'), ('followup', 'Follow-up scan'))):
            centre = (x0 + tile / 2, top - tile / 2)
            frame = FancyBboxPatch((x0, top - tile), tile, tile, boxstyle='round,pad=0,rounding_size=0.08',
                                   facecolor='none', edgecolor='none')
            ax.add_patch(frame)
            image = ax.imshow(_scan_tile(tile, organ, lesions[scan]), extent=(x0, x0 + tile, top - tile, top),
                              interpolation='bilinear', aspect='auto', zorder=1, gid=f'explainer:scan:{scan}')
            image.set_clip_path(frame)
            new_scale, old_scale = fits[scan]
            shapes = {'reference': lesions[scan],
                      'new': _shape(*lesions[scan], scale=new_scale, smooth=0.75),
                      'old': _shape(*lesions[scan], scale=new_scale * old_scale, smooth=0.525, shift=(0.01, -0.008))}
            outlines[scan] = {layer: _outline(shape, centre) for layer, shape in shapes.items()}
            if scan == 'baseline':
                # The version difference: the band between the old and the updated model's baseline outlines.
                outer, inner = outlines[scan]['new'], outlines[scan]['old'][::-1]
                vertices = np.vstack([outer, outer[:1], inner, inner[:1]])
                codes = ([MplPath.MOVETO] + [MplPath.LINETO] * (len(outer) - 1) + [MplPath.CLOSEPOLY]
                         + [MplPath.MOVETO] + [MplPath.LINETO] * (len(inner) - 1) + [MplPath.CLOSEPOLY])
                ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=BAND, edgecolor='none', zorder=2,
                                       gid='explainer:version_difference'))
            for layer in ('reference', 'old', 'new'):
                points = np.vstack([outlines[scan][layer], outlines[scan][layer][:1]])
                line = ax.plot(points[:, 0], points[:, 1], zorder=3, solid_joinstyle='round',
                               gid=f'explainer:outline:{scan}:{layer}', **styles[layer])[0]
                line.set_path_effects(edge)
            ax.text(x0 + tile / 2, top - tile - 0.15, label, ha='center', va='center', fontsize=9, color=INK)
        ax.annotate('', xy=(1.75 - 0.05, top - tile / 2), xytext=(0.3 + tile + 0.05, top - tile / 2),
                    arrowprops=dict(arrowstyle='-|>', color=MUTED, linewidth=1.0, shrinkA=0, shrinkB=0))
        # Key under the tiles, in two rows: the reference and the version difference, then the two model versions.
        # Each handle sits on a chip of the scan's dark field, so it looks exactly as it does on the scans.
        key_y, key_row = top - tile - 0.42, 0.2
        for row, entries in enumerate((('Reference', 'Version difference'), ('Old model', 'Updated model'))):
            x, y = 0.34, key_y - key_row * row
            for text in entries:
                ax.add_patch(FancyBboxPatch((x - 0.04, y - 0.065), 0.30, 0.13, boxstyle='round,pad=0,rounding_size=0.03',
                                            facecolor=SCAN['background'], edgecolor='none', zorder=1))
                if text == 'Version difference':
                    ax.add_patch(Rectangle((x, y - 0.04), 0.22, 0.08, facecolor=BAND, edgecolor='none', zorder=2,
                                           gid=f'key:{text}:handle'))
                else:
                    layer = {'Reference': 'reference', 'Old model': 'old', 'Updated model': 'new'}[text]
                    handle = ax.plot([x, x + 0.22], [y] * 2, zorder=3, gid=f'key:{text}:handle', **styles[layer])[0]
                    handle.set_path_effects(edge)
                label = ax.text(x + 0.34, y, text, va='center', fontsize=8, color=INK, gid=f'key:{text}:label')
                x += 0.34 + label.get_window_extent(renderer).width / fig.dpi + 0.28
        lines = (('One version (new/new)', 'bold'), ('change = FU(new) − BL(new)', 'normal'), ('', 'normal'),
                 ('Mixed versions (old/new)', 'bold'), ('change = FU(new) − BL(old)', 'normal'),
                 ('= [FU(new) − BL(new)] + ', 'normal'), ('', 'normal'),
                 ('Any difference between the versions at baseline,', 'normal'),
                 ('in either direction, enters the change error.', 'normal'))
        for k, (text, weight) in enumerate(lines):
            if not text:
                continue
            y = top - 0.08 - 0.165 * k
            shown = ax.text(3.12, y, text, va='center', fontsize=8, fontweight=weight, color=INK)
            if text.startswith('= ['):
                # The between-version term, highlighted in the band's colour, follows the same-version change.
                width = shown.get_window_extent(renderer).width / fig.dpi
                ax.text(3.12 + width + 0.03, y, '[BL(new) − BL(old)]', va='center', fontsize=8, fontweight='bold',
                        color=VERSION_TERM, bbox=dict(boxstyle='round,pad=0.15', facecolor=BAND, edgecolor='none'))

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
        badges = {'old': 0, 'new': 0}
        for (code, description, models), y in zip(strategies, ys):
            ax.text(0.08, y + 0.08, code, va='center', fontsize=9, fontweight='bold', color=INK)
            ax.text(0.08, y - 0.1, description, va='center', fontsize=8, color=INK)
            for column, model in zip(('baseline', 'followup'), models):
                x = centres[column]
                ax.add_patch(FancyBboxPatch((x - box_w / 2, y - box_h / 2), box_w, box_h,
                                            boxstyle='round,pad=0.02,rounding_size=0.05',
                                            facecolor=VERSION_FILL[model], edgecolor=VERSION_LINE[model],
                                            linewidth=1.2, gid=f'explainer:badge:{model}:{badges[model]}'))
                badges[model] += 1
                ax.text(x, y, 'Old model' if model == 'old' else 'Updated model', ha='center', va='center',
                        fontsize=8, color=INK)
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


# The example pair (CT-UPDATE-018) uses Figure 1's version palette; a black outline keeps each contour visible on
# dark and bright tissue alike, and in the legend.
EXAMPLE_LAYERS = (('reference', 'Reference', '#ffffff', (0, (3, 2))), ('old', 'Old model', VERSION_LINE['old'], '-'),
                  ('new', 'Updated model', VERSION_LINE['new'], '-'))
EXAMPLE_STRATEGIES = (('old_old', 'old/old'), ('old_new', 'old/new'), ('new_new', 'new/new'))


def _signed(value: float, digits: int = 2) -> str:
    return f'{value:+.{digits}f}'.replace('-', '−')


def example_from_arrays(arrays) -> dict:
    """The example_figure input from the arrays careport.ct_example_run saves (example_private.npz)."""
    import json
    needed = {f'{t}_{k}' for t in ('bl', 'fu') for k in ('image', 'reference', 'old', 'new', 'spacing_mm')}
    if not needed | {'values_json'} <= set(arrays):
        raise ValueError('The example arrays are incomplete')
    values = json.loads(str(np.asarray(arrays['values_json'])))
    timepoints = {}
    for timepoint in ('bl', 'fu'):
        view = {k: np.asarray(arrays[f'{timepoint}_{k}']) for k in ('image', 'reference', 'old', 'new')}
        view['spacing'] = tuple(float(v) for v in np.asarray(arrays[f'{timepoint}_spacing_mm']))
        timepoints[timepoint] = view
    return dict(timepoints=timepoints, volumes=values['volumes'], table=values['table'],
                window_hu=tuple(values['window_hu']))


def example_figure(example: dict) -> Figure:
    """One real lesion pair (amendment CT-UPDATE-018): the baseline and follow-up crops with the reference,
    old-model and updated-model contours, the stored volumes, and each strategy's measured change and error.

    example['timepoints'][t] holds 'image' (HU), the boolean 'reference', 'old' and 'new' crops and 'spacing'
    (row, column mm); 'volumes', 'table' (from careport.ct_example.change_table) and 'window_hu' give the rest.
    """
    from matplotlib import patheffects
    for timepoint in ('bl', 'fu'):
        view = example['timepoints'].get(timepoint) or {}
        shape = np.shape(view.get('image'))
        if len(shape) != 2:
            raise ValueError(f'The {timepoint} image is not one slice')
        for layer, *_ in EXAMPLE_LAYERS:
            mask = np.asarray(view.get(layer))
            if mask.shape != shape or not mask.any():
                raise ValueError(f'The {layer} layer at {timepoint} is empty or has another shape')
    outline = [patheffects.withStroke(linewidth=2.6, foreground='black')]
    width, height, size = TEXT_WIDTH_IN, 2.95, 1.9
    vmin, vmax = example['window_hu']
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width, height))
        base = fig.add_axes((0, 0, 1, 1))
        base.set_xlim(0, width)
        base.set_ylim(0, height)
        base.axis('off')
        bottom = height - 0.3 - size
        for left, (timepoint, title) in zip((0.08, 0.08 + size + 0.2), (('bl', 'Baseline scan'),
                                                                        ('fu', 'Follow-up scan'))):
            view = example['timepoints'][timepoint]
            rows, cols = np.shape(view['image'])
            row_mm, col_mm = view['spacing']
            ax = fig.add_axes((left / width, bottom / height, size / width, size / height))
            ax.imshow(view['image'], cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest',
                      extent=(0, cols * col_mm, rows * row_mm, 0), gid=f'example:image:{timepoint}')
            x, y = (np.arange(cols) + 0.5) * col_mm, (np.arange(rows) + 0.5) * row_mm
            for layer, _, colour, style in EXAMPLE_LAYERS:
                contour = ax.contour(x, y, np.asarray(view[layer], dtype=float), levels=[0.5], colors=[colour],
                                     linewidths=1.2, linestyles=[style])
                contour.set_path_effects(outline)
                contour.set_gid(f'example:contour:{timepoint}:{layer}')
            ax.set_xlim(0, cols * col_mm)
            ax.set_ylim(rows * row_mm, 0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, loc='left', fontsize=9, color=INK)
            if timepoint == 'bl':
                start, level = 0.07 * cols * col_mm, 0.92 * rows * row_mm
                bar = ax.plot([start, start + 10], [level, level], color='white', linewidth=2.0, solid_capstyle='butt')
                bar[0].set_path_effects(outline)
                label = ax.text(start + 5, level - 0.04 * rows * row_mm, '10 mm', ha='center', va='bottom',
                                fontsize=8, color='white')
                label.set_path_effects([patheffects.withStroke(linewidth=2.0, foreground='black')])
            for k, (layer, name, _, _) in enumerate(EXAMPLE_LAYERS):
                level = bottom - 0.17 - 0.16 * k
                base.text(left, level, name, va='center', fontsize=8, color=INK)
                base.text(left + size, level, f"{example['volumes'][layer][timepoint]:.2f} mL", ha='right',
                          va='center', fontsize=8, color=INK, gid=f'example:volume:{timepoint}:{layer}')
        # Each strategy's measured change and its error against the reference change, in mL.
        table, columns = example['table'], (4.42, 5.38, 5.94)
        top = height - 0.42
        base.text(columns[0], top, 'Change and error (mL)', va='center', fontsize=8, fontweight='bold', color=INK)
        for x, text in zip(columns[1:], ('Change', 'Error')):
            base.text(x, top - 0.22, text, ha='right', va='center', fontsize=8, color=MUTED)
        rows = {r['strategy']: r for r in table['rows']}
        for k, (name, label) in enumerate(EXAMPLE_STRATEGIES):
            level = top - 0.42 - 0.18 * k
            base.text(columns[0], level, label, va='center', fontsize=8, color=INK, gid=f'example:strategy:{name}')
            base.text(columns[1], level, _signed(rows[name]['change']), ha='right', va='center', fontsize=8,
                      color=INK, gid=f'example:change:{name}')
            base.text(columns[2], level, _signed(rows[name]['error']), ha='right', va='center', fontsize=8,
                      color=INK, gid=f'example:error:{name}')
        level = top - 0.42 - 0.18 * len(EXAMPLE_STRATEGIES)
        base.text(columns[0], level, 'Reference', va='center', fontsize=8, color=INK)
        base.text(columns[1], level, _signed(table['reference_change']), ha='right', va='center', fontsize=8,
                  color=INK, gid='example:reference_change')
        base.plot([columns[0], columns[2]], [level + 0.1] * 2, color=GRID, linewidth=0.8)
        base.text(columns[0], level - 0.32, f"Retention cost of this pair:\n{_signed(table['retention_cost'])} mL",
                  va='top', fontsize=8, color=INK, linespacing=1.4, gid='example:retention_cost')
        handles = []
        for _, name, colour, style in EXAMPLE_LAYERS:
            handle = Line2D([], [], color=colour, linestyle=style, linewidth=1.2, label=name)
            handle.set_path_effects(outline)
            handles.append(handle)
        base.legend(handles=handles, loc='lower left', bbox_to_anchor=(columns[0] - 0.1, 0.12),
                    bbox_transform=base.transData, frameon=False, fontsize=8, handlelength=2.2)
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
