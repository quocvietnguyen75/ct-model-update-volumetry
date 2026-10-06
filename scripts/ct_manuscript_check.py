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
# Design constants and descriptors stated in Methods, not results.
DESIGN = {'1.05': 'centroid tolerance (voxels)', '1.5': 'isotropic spacing (mm)', '0.5': 'flip probability; 0.5% rule',
          '0.25': 'interpolation weight', '0.50': 'interpolation weight', '0.75': 'interpolation weight',
          '0.05': 'family error rate', '3.6': 'verification difference 3.6e-15',
          '0.001': 'display threshold for sign probabilities'}
MAIN_SECTIONS = ('Introduction', 'Methods', 'Results', 'Discussion', 'Conclusion')


def sources():
    objects = [json.loads((RECORD/name).read_text(encoding='utf-8')) for name in JSON_SOURCES]
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


def replication_sections(values_json, strong_json, holdout_csv, tex):
    """The abstract and the two replication Results subsections may only use replication values (the abstract also
    original-run contrasts); each 'X to Y' there must be both bounds of one reported interval or run range."""
    def subsection(heading):
        return tex.split(heading, 1)[1].split(r'\subsection{', 1)[0]

    sections = {
        'abstract': (tex.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0],
                     source_values([values_json, strong_json], [holdout_csv]), [values_json, strong_json]),
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
        (RECORD/'CT_HOLDOUT_PRIMARY_CONTRASTS_2026-09-25.csv').read_text(encoding='utf-8'), tex)
    for path in [LATEX/'main.tex', LATEX/'supplement.tex', *sorted(LATEX.glob('tab*.tex'))]:
        text = path.read_text(encoding='utf-8')
        text = re.sub(r'^%.*$', '', text, flags=re.M)
        text = re.sub(r'\\url\{[^}]*\}', ' ', text)
        missing = unmatched_percentages(text, values, percents)
        text = re.sub(r'\d+\.\d+\\%', ' ', text)
        missing += unmatched_numbers(text, values, exempt=set(DESIGN))
        print(f'{path.name}: {len(decimal_tokens(text))} decimals, {len(missing)} unmatched')
        for token in missing:
            where = text.index(token)
            print(f'   {token}: ...{" ".join(text[max(0, where - 70):where + 20].split())}...')
        failures += len(missing)
    words = main_text_words((LATEX/'main.tex').read_text(encoding='utf-8'))
    print(f'main text words (Introduction to Conclusion, excluding floats and equations): {words}')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
