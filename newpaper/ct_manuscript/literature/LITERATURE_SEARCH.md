# Literature search records

These files are the records behind the statements about prior work in *Better contours, incompatible measurements: the cost of mixing segmentation model versions in longitudinal CT tumour volumetry* (supplementary section "Literature searches"). Four searches were run; each has its query log and, where applicable, its screening records.

| Search | Date | Scope | Sources | Files |
|---|---|---|---|---|
| 1. Scoping review | through 20 September 2026 | Measurement incompatibility across software versions; update regression and its remedies; longitudinal lesion segmentation; training-run variation | Web search, official and author sources, targeted title checks, citation chaining; 38 queries | `2026-09-20/SEARCH_LOG.json` (queries, batches and source records; 23 included papers); `2026-09-20/EVIDENCE.csv` (the evidence register) |
| 2. Competing-work check | 27 September 2026 | Controlled evaluations of a segmentation model update on longitudinal volume change | Web search, the arXiv API, Europe PMC; 21 queries | `2026-09-27/REFRESH.md` (queries by source, candidates found, and why each was or was not relevant) |
| 3. Six-part search | 12 October 2026 | Machine-learning update compatibility; software-version dependence of quantitative imaging; the metrology of change measurement; response assessment; foundation and promptable segmentation models; governance and quality assurance of updates | PubMed, Europe PMC, OpenAlex, Semantic Scholar, the arXiv API, Crossref and official sources, with forward and backward citation chasing | `2026-10-12/SEARCH_LOG.json` (every query, with hits, screened and kept counts); `2026-10-12/PAPERS.csv` (one card per kept work: what it studied and found, and what it does not establish); `2026-10-12/CLAIM_CHECKS.csv` (each manuscript claim against its cited source); `2026-10-12/AUDIT.csv` and `AUDIT_INPUT.csv` (the citation audit of the proposed references); `2026-10-12/references_candidates.bib`; the per-part files under `2026-10-12/parts/` |
| 4. Reviewer-perspective check and full-text retrieval | 13 October 2026 | Prior work a critical reviewer would expect to see cited; full texts of sources previously read only as abstracts | Crossref, OpenAlex, Europe PMC, the arXiv API, publishers' pages | `2026-10-13/SEARCH_LOG.json` (retrieval log); `2026-10-13/oa_status.csv` (open-access status of the sources); `2026-10-13/reviewer_pass/SEARCH_LOG.json` and `PAPERS.csv` (the reviewer-perspective candidates, with cards) |

## How to read the logs

- Each entry of a `SEARCH_LOG.json` records the tool, the query, the number of hits, how many were screened and which were kept, with a note on the screening decision. The `parts/` folders of the 12 October search hold the same records for each of its six parts.
- Each row of a `PAPERS.csv` is one work: its citation, year, venue and status, problem, method, data, main result, relevance to the manuscript, what it does not establish, how deeply it was read (full text, abstract or summary), and how it was found.
- Each row of a `CLAIM_CHECKS.csv` is one manuscript sentence that cites a source, with the passage of the source it rests on and a verdict.

The searches covered English-language sources, and some sources were available only as abstracts; the logs record which. An absence of a found study does not show that none exists. Reference metadata were checked against the Crossref and DataCite registries before submission.
