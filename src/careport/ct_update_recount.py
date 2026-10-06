"""Reproduce amendment 002's direct native-label recount from schema records."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from careport.ct_update_data import volume_paths
from careport.ct_update_gate import GateError


def independent_count(labels, affine, zooms, record):
    count = int(np.count_nonzero(labels == record["lesion_id"]))
    box = tuple(slice(a, b) for a, b in record["native_lesion_box"])
    boxed = int(np.count_nonzero(labels[box] == record["lesion_id"]))
    affine_volume = count * abs(float(np.linalg.det(np.asarray(affine)[:3, :3])))
    spacing_volume = count * float(np.prod(zooms))
    return {"count_matches": count == record["reference_voxels"], "box_count_matches": boxed == count,
            "physical_volume_matches": abs(affine_volume - spacing_volume) <= max(1e-3, affine_volume * 1e-5)}


def recount(root, schema, output):
    import nibabel as nib

    manifest = (schema / "development_records_private.jsonl").read_bytes()
    report = json.loads((schema / "schema_report.json").read_text())
    if (hashlib.sha256(manifest).hexdigest() != report["private_manifest_sha256"]
            or set(report["issues"]) != {"csv_native_volume_mismatch"}):
        raise GateError("Recount requires the unchanged CSV-discrepancy schema report")
    totals = {"checked": 0, "count_mismatches": 0, "geometry_mismatches": 0,
              "full_vs_bounded_count_agreements": 0}
    for line in manifest.splitlines():
        group = json.loads(line)
        if group["split"] not in {"train", "val"}:
            raise GateError("Recount cannot access test references")
        for record in group["records"]:
            native = record["reference_volume_ml"] * 1000
            voxel = native / record["reference_voxels"]
            if abs(native - record["csv_volume_mm3"]) <= max(voxel, .005 * native):
                continue
            _, path = volume_paths(root, group["patient"], group["timepoint"], group["index"])
            image = nib.load(path)
            result = independent_count(np.asanyarray(image.dataobj), image.affine, image.header.get_zooms(), record)
            totals["checked"] += 1
            totals["count_mismatches"] += not result["count_matches"]
            totals["geometry_mismatches"] += not result["physical_volume_matches"]
            totals["full_vs_bounded_count_agreements"] += result["box_count_matches"]
    passed = (totals["checked"] == report["issues"]["csv_native_volume_mismatch"]
              and totals["count_mismatches"] == totals["geometry_mismatches"] == 0
              and totals["full_vs_bounded_count_agreements"] == totals["checked"])
    receipt = {"status": "INDEPENDENT_NATIVE_COUNT_PASS" if passed else "FAIL", **totals,
               "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(), "test_values_read": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)
    if not passed:
        raise GateError("Independent native recount failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    recount(Path(os.environ[config["dataset_root_env"]]), args.schema, args.output)


if __name__ == "__main__":
    main()
