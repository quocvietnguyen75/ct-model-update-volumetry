# CT-UPDATE-018: one real lesion pair as an illustration

9 October 2026. Question: IMG-ALT-01. Specified before any example image, per-pair value or new prediction is viewed. Machine-readable decisions: `config/preanalysis/ct_example_018_2026-10-09.json`.

## Why this exists

After the 9 October figure upgrade the manuscript still shows no CT image: Figure 1 draws the mechanism with illustrative contours. The user approved one real example, a single lesion pair at baseline and follow-up with the reference contour and the contours of the old and updated models ("I approve. We can delay."). The figure illustrates; it tests nothing, and no result or conclusion depends on it. The record is in the [revision plan](CT_REVISION_PLAN_2026-10-02.md).

Two conditions come with it:

- **Permission.** PanTrack is licensed CC BY-NC 4.0, and IOP requires written permission for material whose licence does not allow commercial reuse. The figure may circulate among the authors before permission arrives. It enters a submitted manuscript only with the PanTrack providers' written permission; otherwise it is removed and Figure 1's drawing remains the only illustration.
- **Git.** The user approved a narrow exception to the aggregates-only rule: this figure and the compiled manuscript that contains it, from public de-identified data, with the source and licence in the caption. The private crops, masks and the pair's identity never enter Git, and the public code repository never holds the image.

## Selection, fixed now

- **Cohort and model:** PanTrack and the nnU-Net replication (CT-UPDATE-017 with correction 017A), comparing its old and updated models. The nnU-Net is closer to a segmenter a hospital would deploy, and in PanTrack its baseline version shift was positive in all ten runs.
- **Stratum:** reference baseline volume of 10 mL or more, where the PanTrack retention cost was concentrated (+1.678 mL, positive in 10 of 10 runs).
- **Run:** the run whose retention cost in that stratum, as recorded per run in the verified results (`CT_STRONG_RESULTS_2026-10-06.json`), is closest to the ten-run mean. Ties go to the earlier seed.
- **Pair:** within that run and stratum, the pair whose retention cost, |error(old/new)| − |error(new/new)|, is closest to that run's stratum value. Errors are the analysis's: measured change minus reference change. Pairs with an empty prediction from either model at either timepoint are skipped. Ties go to the first pair in patient and pair order.
- **Nothing else is viewed.** No other run, pair, cohort, model or stratum is rendered, and the chosen pair is shown whatever it looks like; it is not replaced by a more striking one.

## The image, fixed now

- **Inference:** the selected run's registered old and updated checkpoints, each run once on the pair's two cached patches with the evaluation's predictor settings. Logits are mapped to the native grid and thresholded exactly as in the evaluation.
- **Check before rendering:** the re-inferred volume and Dice of each model at each timepoint must agree with the stored evaluation within 0.5% of the stored volume and 0.005 Dice. Otherwise the job stops, nothing is rendered, and any repair needs a dated addendum. The printed numbers are the stored evaluation values.
- **Slice:** at each timepoint, the axial slice, in the patient orientation given by the image affine, with the largest reference lesion area. Ties go to the slice nearest the lesion's centroid.
- **Crop:** a square centred on the reference lesion's in-slice centroid. Its side is the larger of 40 mm and 1.6 times the largest in-slice reference extent at either timepoint, the same at both timepoints. It is moved inside the image if needed and shown in radiological orientation (anterior up, the patient's right on the viewer's left).
- **Window:** soft tissue, width 400 HU and level 40 HU.
- **Shown:** each slice with the reference, old-model and updated-model contours; the stored volumes at both timepoints; and the reference change, with the measured change and its error under old/old, old/new and new/new.

## Gates, outputs and storage

- **Verification before use:** the bundle's `EXECUTION_SHA256SUMS`, this amendment and its policy, the 017 policy and correction 017A, the verified results file, the selected run's seed receipt (against the hash in the results), its prediction records and checkpoints, the cohort gates, the patch hash and the native mask hash. The CT unit tests run first.
- **Outputs:** a new private folder `runs/example_018/` on the cluster, holding the selection with the pair's identity, the crops and masks (`example_private.npz`) and a receipt. Only hashes, the selected seed and the check results enter the record. Storage is under 50 MB, inside the study cap.
- **Execution:** one job on the `mig` partition with one 3g.40gb slice, 8 CPUs and 48 GiB, unlimited wall time; it is expected to take minutes.

## Interpretation boundary

The example is one pair, chosen to be typical of the retention cost in one stratum of one cohort under one model. It is not evidence for the effect, which rests on the analyses already reported, and the caption states the rule that chose it.
