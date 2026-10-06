# CT model-update amendment 002: reference-volume provenance

Date: 15 September 2026. No model outcome has been inspected.

The first development schema audit passed image/mask geometry, explicit mm
units, positive label identity and native XYZ centroid agreement. It flagged
15 of 4,444 eligible lesion-timepoint records whose CSV volume differs from
the native-mask count by more than one voxel or 0.5%. These records affect
nine development patients (eight records in train and seven in validation).
Absolute discrepancies range from 0.00572 to 0.19542 mL; the largest relative
discrepancy is about 6.13%. Their cause is unresolved. Do not describe them
as proven rounding or annotation errors.

Amendment 001 already fixed complete native masks as the primary reference.
Retain that endpoint, all eligible lesions and the original CSV files. A CSV
summary is not allowed to replace the contour-derived reference selectively.
The [data descriptor](https://www.nature.com/articles/s41597-026-07466-y)
describes native-resolution images and corresponding reference segmentations;
it does not explain these specific discrepancies.

Before promoting the audit, independently recount every flagged native label
with a direct equality/count operation, verify agreement with a count confined
to its stored bounding box, and compare physical volumes computed using affine
determinant versus header spacing. Each must agree with the original audit.
Bind that independent receipt to the unchanged private schema manifest.

If all recount checks pass and CSV-volume disagreement is the only remaining
issue, create a separate schema-resolution receipt. Keep the original failed
report and its hash. Carry every discrepancy forward as a reference-provenance
warning; do not erase it, raise the comparison tolerance or remove affected
patients. Any other geometry, label, centroid or split failure remains fatal.

Before final claims, report a sensitivity analysis using the CSV volume changes
and quantify how reference choice changes the paired contrasts. Fix that
analysis before held-out evaluation. No equivalence claim is introduced.

This is a resolution of redundant reference metadata, not a changed primary
endpoint. The test partition remains unopened and the initial study budget,
training choices and primary population remain fixed.
