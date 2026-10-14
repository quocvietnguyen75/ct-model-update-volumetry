# CT manuscript: targeted literature refresh before submission

27 September 2026. IMG-ALT-01. This refresh checks for directly competing work and changed citations since the [20 September review](CT_POSTRESULT_LITERATURE_REVIEW_2026-09-20.md) (23 papers). It is targeted, not a new scoping review.

**Result: no study was found that evaluates how updating a segmentation model affects longitudinal change measurement when the baseline was measured by the previous version. One citation changed: the PanTrack paper was published in the MICCAI 2026 proceedings on 24 September 2026.** The Introduction's bounded absence statement stands. It now says the search was updated on 27 September. No manuscript claim changed.

## Question screened

Is there published or preprinted work that:
- updates or retrains a segmentation model (or other quantitative-imaging model);
- and measures the effect on longitudinal change, comparing a historical baseline measurement with reprocessing?

Adjacent evidence was also noted:
- software or version dependence of volumetry;
- update-compatibility methods;
- governance of model updates.

## Coverage

| Source | Queries (date filter) | Screening |
|---|---|---|
| Web search | 9 queries: segmentation model update with longitudinal volume and baseline reprocessing; backward-compatible medical segmentation and volumetry; software version change with tumour volume follow-up; model update with predetermined change control plans; software updates in AI brain volumetry; updating auto-segmentation models; lung-nodule volumetry software updates; algorithm version with follow-up re-analysis; LMID 2026 workshop | Titles and snippets; selected pages opened |
| arXiv API | 8 queries: "model update"+segmentation; "backward compatible"+segmentation; "model updates"+medical; "negative flips"+segmentation; "model versions"+segmentation; longitudinal+volumetry+consistency; "update regression"; "backward compatibility"+model+update (newest first; items from June 2025 screened) | Titles |
| Europe PMC REST API | 4 queries: software version with volumetry and deep learning or AI (2024–2026); model, software or algorithm update with segmentation and longitudinal or follow-up (2025–2026); backward compatibility, update regression or negative flips in medical settings (2024–2026); retraining with segmentation and volume change (2025–2026) | Titles; one abstract read in full |
| Crossref and DataCite APIs | Title and DOI lookups for every cited work (see the [citation audit](../ct_manuscript/CITATION_AUDIT_2026-09-27.md)) | Metadata |

Access limits:
- PubMed and DBLP blocked automated retrieval (browser challenge), so Europe PMC substituted for PubMed.
- Springer and Nature landing pages redirected to login.
- Google Scholar was not used.
- Screening was by title, snippet or abstract; no full text was read.

## Items screened

| Item | Relation to this paper | Disposition |
|---|---|---|
| Kirchhoff et al., *Exploiting longitudinal context in clinician-verified interactive lesion tracking*, MICCAI 2026, LNCS pp. 259–269, doi:10.1007/978-3-032-38172-9_25 (published 24 September 2026) | PanTrack source and a longitudinal tracking method; does not compare model versions | Citation updated from the arXiv preprint |
| *Lung nodule detection and potential impact on guideline-based management: a retrospective post-market evaluation of three commercial software systems*, Eur Radiol 2026, doi:10.1007/s00330-026-12702-5 (abstract read) | Small measurement differences between three commercial tools changed guideline-based follow-up recommendations. These are different products, not a version update | Optional citation for the Discussion sentence on clinical relevance; not inserted |
| *Comparative analysis of four commercially available artificial intelligence software for brain volumetry*, Diagnostics 2026, doi:10.3390/diagnostics16152383 (title and snippet only) | Between-software variability limits interchangeability of absolute volumes. Adds to the software-dependence strand; no update experiment | Optional; the manuscript already cites stronger primary evidence (Ashraf 2010, Buckler 2015, Knights 2024) |
| *From plug-and-play to institution-calibrated radiology AI*, Front Radiol 2026, doi:10.3389/fradi.2026.1902217 (page read) | Governance framework that lists software version changes for monitoring; does not address comparability of longitudinal measurements | Not needed |
| *Validation of open-source deep learning segmentation tools for automated glioma volumetry*, Front Neurol 2026, doi:10.3389/fneur.2026.1797487 (snippets only) | Narrative review arguing longitudinal consistency matters beyond Dice; no update experiment | Optional background; not inserted |
| *Keeping medical AI healthy and trustworthy*, IEEE Trans Biomed Eng 2026, doi:10.1109/TBME.2025.3642706 (title only) | Review of degradation detection and correction | Background only |
| AEGIS, arXiv:2603.22322; *Regulating radiology AI medical devices that evolve in their lifecycle*, arXiv:2412.20498 (titles and snippets) | Governance and regulation of adaptive or updated devices, including predetermined change control plans | Possible support for the Introduction's first paragraph if the authors want a regulatory citation; not inserted |

The arXiv queries on model updates returned mostly federated-learning, language-model and compatible-representation work. None evaluated longitudinal measurement.

## Limits

- The search is targeted and non-systematic, and depends on search-engine ranking and English-language indexing.
- The absence of a found study does not show that none exists. The manuscript states this.
- Optional items were not read in full. The authors should read any item before citing it (IOP requires that authors check all referenced material).
