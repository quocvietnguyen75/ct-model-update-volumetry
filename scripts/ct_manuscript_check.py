"""Check the PMB LaTeX sources against the verified aggregate results and count main-text words.

Run from the repository root with PYTHONPATH=src. Every decimal number in the manuscript, supplement
and generated tables must be the rounding of a value in the verified result files, unless it is a
listed design constant. Prints unmatched numbers with context and exits nonzero if any remain.
"""
import json
import re
import sys
from pathlib import Path

from careport.ct_manuscript import (decimal_tokens, percent_values, source_values, unmatched_intervals, unmatched_numbers,
                                    unmatched_percentages)

RECORD = Path('newpaper/research_selection_2026-09-13')
MANUSCRIPT = Path('newpaper/ct_manuscript')
LATEX = MANUSCRIPT/'pmb_submission'
JSON_SOURCES = ['CT_REPLICATION_RESULTS_2026-09-27.json', 'CT_REPLICATION_VERIFICATION_2026-09-27.json',
                'CT_COHORT_DESCRIPTION_2026-09-25.json', 'CT_DIAGNOSTIC_RESULTS_2026-09-24.json',
                'CT_PANTRACK_EXTERNAL_RESULTS_2026-09-19.json', 'CT_INTERNAL_RESULTS_2026-09-25.json',
                'CT_PILOT_001_RESULTS_2026-09-15.json', 'CT_PILOT_002_RESULTS_2026-09-15.json',
                'CT_PILOT_003_RESULTS_2026-09-16.json', 'CT_PILOT_004_RESULTS_2026-09-16.json',
                'CT_POSTHOC_RESULTS_2026-10-02.json', 'CT_STRONG_RESULTS_2026-10-06.json',
                'CT_STRONG_SCHEDULE_2026-10-02.json', 'CT_STRONG_SANITY_2026-10-06.json']
# The CT-UPDATE-019 result joins the sources once it is recorded (12 October 2026 revision).
COMPAT = 'CT_COMPAT_RESULTS_019_2026-10-12.json'
# Design constants and descriptors stated in Methods, not results.
DESIGN = {'1.05': 'centroid tolerance (voxels)', '1.5': 'isotropic spacing (mm)', '0.5': 'flip probability; 0.5% rule; compatibility index for a 25% increase in mean squared error (individual-agreement bound)',
          '0.25': 'interpolation weight', '0.50': 'interpolation weight', '0.75': 'interpolation weight',
          '0.05': 'family error rate', '3.6': 'verification difference 3.6e-15',
          '0.001': 'display threshold for sign probabilities', '4.0': 'Creative Commons licence version (CC BY-NC 4.0)',
          '0.32': 'compatibility index for a 5% increase, sqrt(1.05^2 - 1)',
          '0.46': 'compatibility index for a 10% increase, sqrt(1.1^2 - 1)', '1.96': 'limits-of-agreement multiplier'}
# Values quoted from cited literature, with their sources (checked in the 12 October citation audit).
LITERATURE = {'12.1': 'Zhao et al 2009, lower 95% limit of agreement, same-day repeat CT',
              '13.4': 'Zhao et al 2009, upper 95% limit of agreement, same-day repeat CT',
              '9.0': 'Athelogou et al 2016, repeatability coefficient within one algorithm',
              '22.1': 'Athelogou et al 2016, reproducibility coefficient across algorithms',
              '13.9': 'Zhao et al 2025, lesion categories changed by volume instead of diameter thresholds',
              '16.7': 'Adhikary et al 2026 (preprint), patient-level target-lesion categories changed',
              '17.7': 'Goldmacher et al 2026, investigator-central progression discordance',
              '11.8': 'Goldmacher et al 2026, the same in melanoma',
              '97.7': 'Kuhl et al 2019, agreement on progression with identical target lesions',
              '1.7': 'Rinaldi et al 2010, mean variability between two software versions (1.66%, abstract)'}
MAIN_SECTIONS = ('Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion')


def sources():
    names = JSON_SOURCES + ([COMPAT] if (RECORD/COMPAT).exists() else [])
    objects = [json.loads((RECORD/name).read_text(encoding='utf-8')) for name in names]
    objects.append(json.loads(Path('config/preanalysis/ct_replication_015_2026-09-25.json').read_text(encoding='utf-8')))
    texts = [p.read_text(encoding='utf-8') for p in sorted((MANUSCRIPT/'tables').glob('*.csv'))]
    texts += [p.read_text(encoding='utf-8') for p in sorted(RECORD.glob('CT_HOLDOUT_*.csv'))]
    return source_values(objects, texts), percent_values(objects)


def main_text_words(tex: str) -> int:
    """Words from Introduction to the end of Conclusion, without floats, equations or LaTeX commands."""
    body = tex.split(r'\section{Introduction}', 1)[1].split(r'\ack{', 1)[0]
    body = re.sub(r'\\begin\{(table|figure|equation)\}.*?\\end\{\1\}', ' ', body, flags=re.S)
    body = re.sub(r'\\(citep|citet|citealp|ref|label|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}', ' ', body)
    body = re.sub(r'\$[^$]*\$', ' x ', body)
    body = re.sub(r'\\[a-zA-Z]+\*?', ' ', body)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’.,/%-]*", body))


def replication_sections(values_json, strong_json, holdout_csv, tex, compat_json=None):
    """The abstract and the two replication Results subsections may only use replication values (the abstract also
    original-run contrasts); each 'X to Y' there must be both bounds of one reported interval or run range."""
    def subsection(heading):
        return tex.split(heading, 1)[1].split(r'\subsection{', 1)[0]

    sections = {
        'abstract': (tex.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0],
                     source_values([values_json, strong_json] + ([compat_json] if compat_json else []), [holdout_csv]),
                     [values_json, strong_json] + ([compat_json] if compat_json else [])),
        'replication results': (subsection(r'\subsection{Training-run replication}'),
                                source_values([values_json], [holdout_csv]), values_json),
        'stronger-model results': (subsection(r'\subsection{Stronger-model replication}\label{sec:strong}'),
                                   source_values([strong_json, values_json], []), [strong_json, values_json])}
    failures = 0
    for name, (text, allowed, summaries) in sections.items():
        text = re.sub(r'\\begin\{table\}.*?\\end\{table\}', ' ', text, flags=re.S)
        # Decimal percentages are verified against every source by unmatched_percentages in main().
        text = re.sub(r'\d+\.\d+\\%', ' ', text)
        missing = unmatched_numbers(text, allowed, exempt=set(DESIGN)) + unmatched_intervals(text, summaries)
        print(f'{name} against replication results only: {len(missing)} unmatched {missing}')
        failures += len(missing)
    return failures


def main():
    values, percents = sources()
    tex = Path(sys.argv[1]).read_text(encoding='utf-8') if len(sys.argv) > 1 else (LATEX/'main.tex').read_text(encoding='utf-8')
    failures = replication_sections(
        json.loads((RECORD/'CT_REPLICATION_RESULTS_2026-09-27.json').read_text(encoding='utf-8')),
        json.loads((RECORD/'CT_STRONG_RESULTS_2026-10-06.json').read_text(encoding='utf-8')),
        (RECORD/'CT_HOLDOUT_PRIMARY_CONTRASTS_2026-09-25.csv').read_text(encoding='utf-8'), tex,
        json.loads((RECORD/COMPAT).read_text(encoding='utf-8')) if (RECORD/COMPAT).exists() else None)
    for path in [LATEX/'main.tex', LATEX/'supplement.tex', *sorted(LATEX.glob('tab*.tex'))]:
        text = path.read_text(encoding='utf-8')
        text = re.sub(r'^%.*$', '', text, flags=re.M)
        text = re.sub(r'\\url\{[^}]*\}', ' ', text)
        for token in LITERATURE:  # quoted literature values are not study results
            text = re.sub(r'(?<![\d.])' + re.escape(token) + r'(?=\\%)', ' ', text)
        missing = unmatched_percentages(text, values, percents)
        text = re.sub(r'\d+\.\d+\\%', ' ', text)
        allowed = values
        if 'compat' in path.name and (RECORD/COMPAT).exists():
            # The CT-UPDATE-019 tables show shares (stored as fractions) as percentages in percent-labelled rows.
            compat = json.loads((RECORD/COMPAT).read_text(encoding='utf-8'))
            allowed = values + [100 * v for v in source_values([compat], [])]
        missing += unmatched_numbers(text, allowed, exempt=set(DESIGN) | set(LITERATURE))
        print(f'{path.name}: {len(decimal_tokens(text))} decimals, {len(missing)} unmatched')
        for token in missing:
            # A percentage is reported as 'N.N%' but written 'N.N\%' in the source.
            where = text.find(token.rstrip('%'))
            where = max(where, 0)
            print(f'   {token}: ...{" ".join(text[max(0, where - 70):where + 20].split())}...')
        failures += len(missing)
    words = main_text_words((LATEX/'main.tex').read_text(encoding='utf-8'))
    print(f'main text words (Introduction to Conclusion, excluding floats and equations): {words}')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
