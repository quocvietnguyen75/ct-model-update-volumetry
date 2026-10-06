"""Development-only lesion linkage, geometry audit and fixed patch geometry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

from careport.ct_update_gate import GateError, choose_old_patients, validate_input_policy


def integer(value: str) -> int:
    try:
        number = float(value)
        if not math.isfinite(number) or number != int(number) or number < 0:
            raise ValueError
        return int(number)
    except (ValueError, TypeError, OverflowError) as error:
        raise GateError("Invalid nonnegative integer field") from error


def coordinate(value: str) -> tuple[float, float, float]:
    try:
        parts = re.split(r"[\s,]+", value.strip().strip("[]()").strip())
        result = tuple(float(p) for p in parts)
        if len(result) != 3 or not all(math.isfinite(p) for p in result):
            raise ValueError
        return result
    except (ValueError, TypeError, AttributeError) as error:
        raise GateError("Invalid finite voxel coordinate") from error


def classify_rows(rows: list[dict]) -> tuple[list[dict], Counter]:
    ids = [integer(row["lesion_id"]) for row in rows]
    if len(set(ids)) != len(ids) or any(i <= 0 for i in ids):
        raise GateError("Duplicate or nonpositive lesion identities")
    merge_targets = {integer(r["merged_into"]) for r in rows
                     if r["topology_class"].strip().upper() == "MERGING"
                     and str(r.get("merged_into", "")).strip().lower() not in {"", "nan", "none"}}
    counts: Counter = Counter()
    eligible = []
    for row, label in zip(rows, ids):
        counts["rows"] += 1
        if row["topology_class"].strip().upper() != "UNCHANGED":
            counts["non_one_to_one_topology"] += 1
            continue
        link = str(row.get("linking_unclear", "")).strip().lower()
        if link in {"", "nan", "none"}:
            counts["link_unknown"] += 1
            continue
        if link in {"true", "1", "1.0"}:
            counts["link_unclear"] += 1
            continue
        if link not in {"false", "0", "0.0"}:
            raise GateError("Unrecognized linkage flag encoding")
        if label in merge_targets:
            counts["merger_destination"] += 1
            continue
        record = {"lesion_id": label}
        for timepoint in ("bl", "fu"):
            volume = float(row["volume_" + timepoint])
            if not math.isfinite(volume) or volume <= 0:
                raise GateError("One-to-one lesion lacks positive finite reference volume")
            record[timepoint] = {"image_index": integer(row["img_id_" + timepoint]),
                                 "centroid": coordinate(row["cog_" + timepoint]),
                                 "csv_volume_mm3": volume}
        eligible.append(record)
        counts["eligible"] += 1
    return eligible, counts


def patch_affine(native_affine, center, shape, spacing):
    native = np.asarray(native_affine, dtype=float)
    scales = np.linalg.norm(native[:3, :3], axis=0)
    if not np.isfinite(native).all() or np.any(scales <= 0):
        raise GateError("Invalid image affine")
    direction = native[:3, :3] / scales
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-4):
        raise GateError("Sheared grid needs an explicit processing decision")
    patch = np.eye(4)
    patch[:3, :3] = direction @ np.diag(spacing)
    world_center = native @ np.r_[center, 1.]
    patch[:3, 3] = world_center[:3] - patch[:3, :3] @ ((np.asarray(shape) - 1) / 2)
    return patch


def sample_patch(array, native_affine, target_affine, shape, order, background):
    mapping = np.linalg.inv(native_affine) @ target_affine
    return ndimage.affine_transform(array, mapping[:3, :3], mapping[:3, 3],
                                    output_shape=tuple(shape), order=order, mode="constant",
                                    cval=background, prefilter=False,
                                    output=np.float32 if order else None)


def native_prediction(logits, patch_geometry, native_geometry, native_shape):
    # Limit allocation to a conservative native bounding box of the fixed patch.
    mapping = np.linalg.inv(native_geometry) @ patch_geometry
    corners = np.array(list(itertools.product(*[(0, n - 1) for n in logits.shape])))
    points = corners @ mapping[:3, :3].T + mapping[:3, 3]
    low = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
    high = np.minimum(np.ceil(points.max(axis=0)).astype(int) + 1, native_shape)
    if np.any(high <= low):
        raise GateError("Prediction patch does not intersect the native image")
    slices = tuple(slice(int(a), int(b)) for a, b in zip(low, high))
    roi_affine = np.array(native_geometry, dtype=float, copy=True)
    roi_affine[:3, 3] = np.asarray(native_geometry)[:3, :3] @ low + np.asarray(native_geometry)[:3, 3]
    sampled = sample_patch(np.asarray(logits, dtype=np.float32), patch_geometry,
                           roi_affine, high - low, 1, -20.)
    return slices, sampled >= 0


def volume_paths(root: Path, patient: str, timepoint: str, index: int):
    prefix = f"{patient}_{timepoint.upper()}"
    image = root / "inputsTr" / f"{prefix}_img_{index:02d}.nii.gz"
    folder = "inputsTr" if timepoint.lower() == "bl" else "targetsTr"
    mask = root / folder / f"{prefix}_mask_{index:02d}.nii.gz"
    return image, mask


def inspect_volume(task: dict) -> dict:
    import nibabel as nib

    image_path, mask_path = volume_paths(Path(task["root"]), task["patient"], task["timepoint"], task["index"])
    image, mask = nib.load(image_path), nib.load(mask_path)
    if (len(image.shape) != 3 or image.shape != mask.shape
            or not np.isfinite(image.affine).all()
            or not np.allclose(image.affine, mask.affine, rtol=0, atol=1e-4)):
        raise GateError("Image/mask native geometry mismatch")
    if image.header.get_xyzt_units()[0] != "mm" or mask.header.get_xyzt_units()[0] != "mm":
        raise GateError("NIfTI physical units are not explicitly millimetres")
    patch_affine(image.affine, (0, 0, 0), (3, 3, 3), (1, 1, 1))
    labels = np.asanyarray(mask.dataobj)
    if labels.dtype.kind not in "iu" or labels.min() < 0 or labels.max() > 100000:
        raise GateError("Reference mask is not bounded nonnegative integer labels")
    counts = np.bincount(labels.ravel(order="K"))
    boxes = ndimage.find_objects(labels)
    voxel_mm3 = abs(float(np.linalg.det(image.affine[:3, :3])))
    records = []
    issues: Counter = Counter()
    for entry in task["entries"]:
        identity = entry["lesion_id"]
        if identity >= len(counts) or counts[identity] == 0:
            issues["missing_positive_label"] += 1
            continue
        box = boxes[identity - 1]
        coordinates = np.nonzero(labels[box] == identity)
        actual_center = np.array([float(c.mean()) + s.start for c, s in zip(coordinates, box)])
        center = np.asarray(entry["centroid"], dtype=float)
        if np.any(center < 0) or np.any(center >= image.shape):
            issues["centroid_out_of_bounds"] += 1
        if np.max(np.abs(center - actual_center)) > 1.05:
            issues["centroid_xyz_mismatch"] += 1
            if np.max(np.abs(center[::-1] - actual_center)) <= 1.05:
                issues["centroid_matches_reversed_order"] += 1
        volume_mm3 = int(counts[identity]) * voxel_mm3
        if abs(volume_mm3 - entry["csv_volume_mm3"]) > max(voxel_mm3, 0.005 * volume_mm3):
            issues["csv_native_volume_mismatch"] += 1
        records.append({**entry, "reference_voxels": int(counts[identity]),
                        "reference_volume_ml": volume_mm3 / 1000,
                        "native_shape": list(image.shape), "native_affine": image.affine.tolist(),
                        "native_lesion_box": [[s.start, s.stop] for s in box]})
    return {"patient": task["patient"], "split": task["split"], "timepoint": task["timepoint"],
            "index": task["index"], "records": records, "issues": dict(issues)}


def audit_development(root: Path, config: dict, metadata: dict, output: Path, workers: int) -> dict:
    validate_input_policy(config)
    if metadata.get("status") != "METADATA_GATE_PASS" or metadata.get("metadata_sha256") != config["expected_metadata_sha256"]:
        raise GateError("Metadata gate does not bind this dataset")
    splits = json.loads((root / "data_split.json").read_text())
    old = set(choose_old_patients(splits["train"], config["old_training_patients"], config["subset_salt"]))
    tasks = []
    cohort = {}
    totals: Counter = Counter()
    for split in ("train", "val"):
        split_counts: Counter = Counter()
        patients_with_lesions = 0
        for patient in sorted(splits[split]):
            with (root / "inputsTr" / f"{patient}.csv").open(newline="", encoding="utf-8-sig") as stream:
                eligible, counts = classify_rows(list(csv.DictReader(stream)))
            split_counts.update(counts)
            patients_with_lesions += bool(eligible)
            groups = defaultdict(list)
            for lesion in eligible:
                for timepoint in ("bl", "fu"):
                    record = {"lesion_id": lesion["lesion_id"], **lesion[timepoint], "old_training": patient in old}
                    groups[(timepoint, record["image_index"])].append(record)
            for (timepoint, index), entries in groups.items():
                tasks.append({"root": str(root), "patient": patient, "split": split,
                              "timepoint": timepoint, "index": index, "entries": entries})
        cohort[split] = {**dict(split_counts), "patients_with_eligible_lesions": patients_with_lesions,
                         "patients_in_manifest": len(splits[split])}
    output.mkdir(parents=True, exist_ok=False)
    private = output / "development_records_private.jsonl"
    with private.open("x", encoding="utf-8") as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        done = 0
        for result in pool.map(inspect_volume, tasks):
            totals.update(result["issues"])
            stream.write(json.dumps(result) + "\n")
            done += 1
            if done % 25 == 0:
                print(json.dumps({"volumes_audited": done, "total": len(tasks)}), flush=True)
    report = {"date": config["date"], "protocol_id": config["protocol_id"],
              "status": "SCIENTIFIC_SCHEMA_PASS" if not totals else "SCIENTIFIC_SCHEMA_FAIL",
              "cohort": cohort, "issues": dict(totals), "development_volumes_audited": len(tasks),
              "old_subset_patient_count": len(old), "test_csv_bodies_read": False,
              "test_masks_decoded": False, "test_images_decoded": False,
              "ct_voxels_decoded": False, "development_reference_masks_decoded": True,
              "input_coordinate_convention": "native_NIfTI_xyz_requires_centroid_agreement",
              "private_manifest_sha256": hashlib.sha256(private.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()}
    (output / "schema_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    report = audit_development(Path(os.environ[config["dataset_root_env"]]), config,
                               json.loads(args.metadata_gate.read_text()), args.output, args.workers)
    if report["status"] != "SCIENTIFIC_SCHEMA_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
