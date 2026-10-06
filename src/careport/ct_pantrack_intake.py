"""Metadata-only PanTrack intake; external images, masks and scores stay sealed."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Mapping

from careport.ct_update_gate import inventory_files, metadata_fingerprint, validate_storage


class IntakeError(ValueError):
    """A metadata contract failed; messages never contain patient identifiers."""


PATIENT = re.compile(r"PanTrack_\d{3}")
SCAN = re.compile(r"(PanTrack_\d{3})_(\d{8})")
FIELDS = {"img_bl", "img_fu", "bl_point", "fu_point", "fu_point_prop", "merged_lesions"}
REQUIRED_FIELDS = FIELDS - {"fu_point_prop"}


@dataclass(frozen=True)
class CandidatePair:
    patient: str
    baseline: str
    followup: str
    label: int
    baseline_point: tuple[float, ...]
    followup_point: tuple[float, ...]


def load_metadata_json(raw: bytes) -> object:
    """Permit the documented missing-point NaN, but reject duplicate JSON keys."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise IntakeError("Duplicate metadata key")
            result[key] = value
        return result

    def constant(value):
        if value == "NaN":
            return float("nan")
        raise IntakeError("Infinite metadata value")

    try:
        return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs,
                          parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntakeError("Invalid metadata encoding or JSON") from exc


def validate_access(policy: Mapping, transfer: Mapping) -> None:
    if (sorted(policy.get("model_input_roles", [])) != ["ct_image", "reference_centroid"]
            or policy.get("external_role") != "evaluation_only_no_tuning"
            or any(policy.get(key) is not False for key in (
                "inspect_image_voxels", "inspect_label_voxels",
                "inspect_model_outputs", "internal_test_values_permitted"))):
        raise IntakeError("Unapproved input or outcome access")
    if (transfer.get("status") != "CONTENT_HASH_PASS"
            or transfer.get("files_verified") != policy["expected_files"]
            or transfer.get("bytes_verified") != policy["expected_bytes"]
            or transfer.get("metadata_sha256") != policy["expected_metadata_sha256"]):
        raise IntakeError("Transfer verification missing or belongs to another snapshot")


def _point(value: object) -> tuple[float, ...] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise IntakeError("Invalid point shape")
    if all(item is None or (isinstance(item, float) and math.isnan(item)) for item in value):
        return None
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise IntakeError("Invalid point coordinates")
    return tuple(float(item) for item in value)


def _scan_owner(scan: object) -> tuple[str, datetime]:
    match = SCAN.fullmatch(scan) if isinstance(scan, str) else None
    if not match:
        raise IntakeError("Invalid scan identifier")
    try:
        return match[1], datetime.strptime(match[2], "%Y%m%d")
    except ValueError as exc:
        raise IntakeError("Invalid scan calendar date") from exc


def _audit_split(split: object, scans: set[str]) -> int:
    if not isinstance(split, list) or len(split) != 1 or not isinstance(split[0], dict):
        raise IntakeError("Unexpected suggested split structure")
    fold = split[0]
    if set(fold) != {"train", "val"}:
        raise IntakeError("Unexpected suggested split names")
    seen = set()
    owners = {}
    for name, ids in fold.items():
        if (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids)
                or len(ids) != len(set(ids)) or seen.intersection(ids)):
            raise IntakeError("Duplicate or overlapping scan IDs in suggested split")
        seen.update(ids)
        owners[name] = {_scan_owner(item)[0] for item in ids}
    if seen != scans:
        raise IntakeError("Suggested split does not cover the same scans")
    return len(owners["train"] & owners["val"])


def audit_metadata(patients: object, tracking: object, split: object,
                   files: set[str]) -> tuple[dict, tuple[CandidatePair, ...]]:
    if not isinstance(patients, dict) or not patients:
        raise IntakeError("Missing patient manifest")
    scans = set()
    for patient, visits in patients.items():
        if not isinstance(patient, str) or not PATIENT.fullmatch(patient):
            raise IntakeError("Invalid patient identifier")
        if not isinstance(visits, list) or len(visits) < 2:
            raise IntakeError("Patient has fewer than two visits")
        previous = None
        for visit in visits:
            owner, date = _scan_owner(visit)
            if owner != patient or visit in scans or (previous is not None and date <= previous):
                raise IntakeError("Patient ownership, uniqueness or chronology failure")
            scans.add(visit)
            previous = date
    expected = {f"images/{scan}_0000.nii.gz" for scan in scans}
    expected |= {f"labels/{scan}.nii.gz" for scan in scans}
    observed = {name for name in files if name.startswith(("images/", "labels/"))}
    if observed != expected:
        raise IntakeError("Image and label coverage does not match manifest")
    overlap = _audit_split(split, scans)
    if not isinstance(tracking, dict) or set(tracking) != set(patients):
        raise IntakeError("Tracking patients do not match manifest")
    exclusions = Counter()
    candidates = []
    steps = entries = 0
    for patient, visits in patients.items():
        sequence = tracking[patient]
        if not isinstance(sequence, list) or len(sequence) != len(visits) - 1:
            raise IntakeError("Tracking does not cover each consecutive visit step")
        for index, step in enumerate(sequence):
            if not isinstance(step, dict) or not step:
                raise IntakeError("Empty or malformed tracking step")
            steps += 1
            targets = Counter()
            parsed = []
            for label_key, entry in step.items():
                if (not isinstance(label_key, str) or not label_key.isdecimal()
                        or int(label_key) <= 0 or str(int(label_key)) != label_key):
                    raise IntakeError("Invalid baseline lesion label")
                if not isinstance(entry, dict) or not REQUIRED_FIELDS <= set(entry) <= FIELDS:
                    raise IntakeError("Unexpected tracking fields")
                if (entry["img_bl"] != visits[index]
                        or entry["img_fu"] != visits[index + 1]):
                    raise IntakeError("Tracking ownership, chronology or step-order failure")
                destination = entry["merged_lesions"]
                if (not isinstance(destination, list)
                        or any(type(item) is not int or item < 0 for item in destination)
                        or (0 in destination and destination != [0])
                        or len(set(destination)) != len(destination)):
                    raise IntakeError("Invalid follow-up target labels")
                # Zero is the documented absent-target sentinel, not a label.
                targets.update(item for item in destination if item > 0)
                baseline = _point(entry["bl_point"])
                followup = _point(entry["fu_point"])
                parsed.append((int(label_key), entry, destination, baseline, followup))
            for label, entry, destination, baseline, followup in parsed:
                entries += 1
                if baseline is None:
                    exclusions["missing_baseline_point"] += 1
                elif destination == [0]:
                    exclusions["documented_absent_followup_target"] += 1
                elif followup is None:
                    exclusions["missing_followup_point"] += 1
                elif any(targets[item] > 1 for item in destination):
                    exclusions["target_collision"] += 1
                elif destination != [label]:
                    exclusions["non_one_to_one_or_relabelled"] += 1
                else:
                    candidates.append(CandidatePair(patient, entry["img_bl"], entry["img_fu"],
                                                    label, baseline, followup))
    report = {
        "status": "METADATA_GATE_PASS", "patients": len(patients), "scans": len(scans),
        "consecutive_scan_pairs": steps, "lesion_tracking_entries": entries,
        "eligible_patients": len({pair.patient for pair in candidates}),
        "eligible_lesion_pairs": len(candidates),
        "exclusions": dict(sorted(exclusions.items())),
        "suggested_split_patient_overlap": overlap, "suggested_split_adopted": False,
        "image_voxels_read": False, "label_voxels_read": False,
        "model_outputs_read": False, "internal_test_values_read": False,
        "geometry_and_instance_label_gate_closed": False,
        "cohort_role": "external_evaluation_only", "patient_weighting": "equal",
    }
    assert entries == len(candidates) + sum(exclusions.values())
    return report, tuple(candidates)


def validate_headers(headers: Mapping, candidates: tuple[CandidatePair, ...]) -> dict:
    """Check native image/label agreement without loading voxel arrays."""
    shapes = {}
    for scan, pair in headers.items():
        image, label = pair
        for header in (image, label):
            shape, matrix = header['shape'], header['affine']
            if (len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
                    or header['units'] != 'mm' or len(header['zooms']) != 3
                    or any(not math.isfinite(x) or x <= 0 for x in header['zooms'])
                    or len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                    or any(not math.isfinite(x) for row in matrix for x in row)):
                raise IntakeError('Invalid native image or label geometry')
            a, b, c = (row[:3] for row in matrix[:3])
            determinant = (a[0] * (b[1]*c[2] - b[2]*c[1])
                           - a[1] * (b[0]*c[2] - b[2]*c[0])
                           + a[2] * (b[0]*c[1] - b[1]*c[0]))
            if abs(determinant) <= 1e-12 or tuple(matrix[3]) != (0, 0, 0, 1):
                raise IntakeError('Singular or non-affine native transform')
        if (image['shape'] != label['shape']
                or any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-5)
                       for left, right in zip(image['affine'], label['affine'])
                       for a, b in zip(left, right))):
            raise IntakeError('Image and label native grids differ')
        shapes[scan] = image['shape']
    for pair in candidates:
        for scan, point in ((pair.baseline, pair.baseline_point),
                            (pair.followup, pair.followup_point)):
            if scan not in shapes or any(x < 0 or x > n-1 for x, n in zip(point, shapes[scan])):
                raise IntakeError('Reference prompt lies outside its XYZ image grid')
    return {'scan_headers_verified': len(headers), 'candidate_prompt_bounds_verified': 2*len(candidates),
            'image_label_native_grid_agreement': True, 'spatial_units': 'mm',
            'coordinate_convention': 'voxel_XYZ_nibabel_axes',
            'image_voxels_read': False, 'label_voxels_read': False,
            'label_instance_presence_verified': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--transfer-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--check-headers", action="store_true")
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8-sig"))
    transfer = json.loads(args.transfer_receipt.read_text(encoding="utf-8-sig"))
    validate_access(policy, transfer)
    root = Path(os.environ[policy["dataset_root_env"]])
    project = Path(os.environ[policy["project_root_env"]])
    for path in (args.output, args.private_output):
        if path.exists() or not path.resolve().is_relative_to(project.resolve()):
            raise IntakeError("Output must be new and inside the study directory")
    files = inventory_files(root)
    if (len(files) != policy["expected_files"]
            or sum(size for _, size in files) != policy["expected_bytes"]
            or metadata_fingerprint(files) != policy["expected_metadata_sha256"]):
        raise IntakeError("Transferred file metadata changed")
    for name, _ in files:
        if "/" not in name:
            if name not in policy["root_filenames"]:
                raise IntakeError("Unexpected root file")
        elif not name.startswith(("images/", "labels/")):
            raise IntakeError("Unexpected dataset directory")
    usage = sum(p.stat().st_size for p in project.rglob("*") if p.is_file() and not p.is_symlink())
    storage = validate_storage(policy["storage"], usage, shutil.disk_usage(project).free)
    metadata = {}
    for name, expected_hash in policy["manifest_hashes"].items():
        path = root / name
        if path.stat().st_size > policy["max_metadata_file_bytes"]:
            raise IntakeError("Metadata file exceeds limit")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise IntakeError("Metadata content changed")
        metadata[name] = load_metadata_json(raw)
    report, candidates = audit_metadata(metadata["patients.json"], metadata["tracking.json"],
                                        metadata["recommended_split.json"], {name for name, _ in files})
    if report["patients"] != policy["expected_patients"] or report["scans"] != policy["expected_scans"]:
        raise IntakeError("Dataset population counts differ from declared release")
    if args.check_headers:
        import nibabel as nib
        headers = {}
        for visits in metadata['patients.json'].values():
            for scan in visits:
                pair = []
                for relative in (f'images/{scan}_0000.nii.gz', f'labels/{scan}.nii.gz'):
                    image = nib.load(root / relative)
                    pair.append({'shape': tuple(int(n) for n in image.shape),
                                 'affine': tuple(tuple(float(x) for x in row) for row in image.affine),
                                 'units': image.header.get_xyzt_units()[0],
                                 'zooms': tuple(float(x) for x in image.header.get_zooms())})
                headers[scan] = pair
        report['headers'] = validate_headers(headers, candidates)
    os.umask(0o077)
    private = {"protocol_id": policy["protocol_id"], "pairs": [asdict(pair) for pair in candidates]}
    args.private_output.parent.mkdir(parents=True, exist_ok=True)
    with args.private_output.open("x", encoding="utf-8") as stream:
        json.dump(private, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report.update(protocol_id=policy["protocol_id"], storage=storage,
                  metadata_sha256=policy["expected_metadata_sha256"],
                  policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
                  transfer_receipt_sha256=hashlib.sha256(args.transfer_receipt.read_bytes()).hexdigest(),
                  private_candidates_sha256=hashlib.sha256(args.private_output.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
