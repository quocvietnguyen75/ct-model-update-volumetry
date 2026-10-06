"""Metadata-only access gate for the controlled CT model-update study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable, Mapping


class GateError(ValueError):
    """A declared access or data contract has not passed."""


PATIENT = re.compile(r"^[a-f0-9]{10}$")
VOLUME = re.compile(r"^([a-f0-9]{10})_(BL|FU)_(img|mask)_(\d{2})\.nii\.gz$")
PROMPT = re.compile(r"^([a-f0-9]{10})_(BL|FU)_(\d{2})\.json$")


def metadata_fingerprint(files: Iterable[tuple[str, int]]) -> str:
    encoded = "".join(f"{name}\t{size}\n" for name, size in sorted(files)).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_splits(splits: Mapping, patients: set[str]) -> dict[str, int]:
    if set(splits) != {"train", "val", "test"}:
        raise GateError("Unexpected split names")
    seen: set[str] = set()
    counts = {}
    for name, ids in splits.items():
        if not isinstance(ids, list) or any(not isinstance(p, str) or not PATIENT.fullmatch(p) for p in ids):
            raise GateError("Invalid patient identifiers in split manifest")
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise GateError("Duplicate or overlapping patient groups")
        seen.update(ids)
        counts[name] = len(ids)
    if seen != patients:
        raise GateError("Unassigned or missing patient groups")
    return counts


def choose_old_patients(patients: Iterable[str], count: int, salt: str) -> tuple[str, ...]:
    ids = tuple(patients)
    if len(set(ids)) != len(ids) or not 0 < count <= len(ids):
        raise GateError("Invalid old-data subset size or duplicate patients")
    return tuple(sorted(ids, key=lambda p: (hashlib.sha256((salt + p).encode()).hexdigest(), p))[:count])


def validate_input_policy(config: Mapping) -> None:
    if sorted(config["model_input_roles"]) != ["ct_image", "reference_centroid"]:
        raise GateError("Unapproved model input role")
    if config["time_policy"] != "supplied_BL_FU_pair_only_no_elapsed_time_claim":
        raise GateError("Unvalidated timestamp/elapsed-time interpretation")
    if config["test_values_permitted"] is not False or set(config["permitted_development_splits"]) != {"train", "val"}:
        raise GateError("Held-out values are not permitted in development")


def validate_storage(policy: Mapping, current_bytes: int, free_bytes: int) -> dict:
    additional = policy["planned_additional_bytes"]
    if min(current_bytes, free_bytes, additional) < 0:
        raise GateError("Invalid storage measurement")
    projected = current_bytes + additional
    if projected > min(policy["study_cap_bytes"], policy["project_cap_bytes"]) or additional > free_bytes:
        raise GateError("Storage projection exceeds a cap or current free space")
    return {"current_study_bytes": current_bytes, "planned_additional_bytes": additional,
            "projected_study_bytes": projected, "shared_free_bytes": free_bytes,
            "personal_quota_verified": False}


def inventory_files(root: Path) -> list[tuple[str, int]]:
    if not root.is_dir() or root.is_symlink():
        raise GateError("Dataset root is absent or a symlink")
    files = []
    for base, directories, names in os.walk(root):
        for name in directories + names:
            p = Path(base) / name
            if p.is_symlink():
                raise GateError("Symlink in dataset tree")
        for name in names:
            p = Path(base) / name
            if p.stat().st_size <= 0:
                raise GateError("Empty data file")
            files.append((p.relative_to(root).as_posix(), p.stat().st_size))
    return files


def audit_metadata(root: Path, config: Mapping, transfer: Mapping,
                   current_project_bytes: int, free_bytes: int) -> dict:
    validate_input_policy(config)
    storage = validate_storage(config["storage"], current_project_bytes, free_bytes)
    files = inventory_files(root)
    fingerprint = metadata_fingerprint(files)
    if (len(files) != config["expected_files"] or sum(s for _, s in files) != config["expected_bytes"]
            or fingerprint != config["expected_metadata_sha256"]):
        raise GateError("Received metadata differs from the frozen source snapshot")
    if (transfer.get("status") != "CONTENT_HASH_PASS"
            or transfer.get("content_manifest_sha256") != config["expected_content_manifest_sha256"]
            or transfer.get("metadata_sha256") != fingerprint
            or transfer.get("files_verified") != len(files)):
        raise GateError("Successful content verification is missing or belongs to another snapshot")
    split_bytes = (root / "data_split.json").read_bytes()
    if hashlib.sha256(split_bytes).hexdigest() != config["expected_split_sha256"]:
        raise GateError("Split manifest changed")
    splits = json.loads(split_bytes)
    patients: set[str] = set()
    images: set[tuple[str, str, str]] = set()
    masks: set[tuple[str, str, str]] = set()
    prompts: set[tuple[str, str, str]] = set()
    csv_patients: set[str] = set()
    variants: dict[tuple[str, ...], int] = {}
    for relative, _ in files:
        p = Path(relative)
        if relative == "data_split.json":
            continue
        if p.parent.as_posix() not in {"inputsTr", "targetsTr"}:
            raise GateError("Unexpected dataset directory")
        volume = VOLUME.fullmatch(p.name)
        prompt = PROMPT.fullmatch(p.name)
        if volume:
            patient, timepoint, role, index = volume.groups()
            expected_dir = "targetsTr" if role == "mask" and timepoint == "FU" else "inputsTr"
            if p.parent.as_posix() != expected_dir:
                raise GateError("Volume in unexpected role directory")
            (images if role == "img" else masks).add((patient, timepoint, index))
        elif prompt:
            patient, timepoint, index = prompt.groups()
            if p.parent.as_posix() != "inputsTr":
                raise GateError("Prompt in unexpected directory")
            prompts.add((patient, timepoint, index))
        elif p.suffix == ".csv" and PATIENT.fullmatch(p.stem) and p.parent.as_posix() == "inputsTr":
            patient = p.stem
            csv_patients.add(patient)
            # Read exactly one binary line; invalid body bytes must remain unread.
            with (root / p).open("rb") as stream:
                line = stream.readline(65537)
            if len(line) > 65536:
                raise GateError("CSV header exceeds bound")
            columns = next(csv.reader([line.decode("utf-8-sig").strip()]))
            if len(set(columns)) != len(columns):
                raise GateError("Duplicate CSV column")
            required = set(config["required_csv_columns"])
            allowed = required | set(config["optional_csv_columns"])
            if not required <= set(columns) <= allowed:
                raise GateError("Unrecognized CSV schema")
            key = tuple(sorted(columns))
            variants[key] = variants.get(key, 0) + 1
        else:
            raise GateError("Unexpected dataset filename")
        patients.add(patient)
    counts = validate_splits(splits, patients)
    if counts != config["expected_split_counts"]:
        raise GateError("Patient split counts changed")
    if images != masks or not prompts <= images or csv_patients != patients:
        raise GateError("Image, mask, prompt or CSV structural coverage is invalid")
    if any((p, t, "00") not in images for p in patients for t in ("BL", "FU")):
        raise GateError("Missing baseline or follow-up index-zero volume")
    return {"protocol_id": config["protocol_id"], "status": "METADATA_GATE_PASS",
            "files": len(files), "bytes": sum(s for _, s in files), "metadata_sha256": fingerprint,
            "split_counts": counts, "patient_count": len(patients), "image_count": len(images),
            "mask_count": len(masks), "images_without_prompt_json": len(images - prompts),
            "csv_header_variants": [{"columns": list(k), "files": v} for k, v in sorted(variants.items())],
            "csv_body_read": False, "image_voxels_read": False, "test_values_read": False,
            "timestamp_policy_checked": True, "input_roles_checked": True,
            "storage": storage, "scientific_schema_gate_closed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--transfer-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(os.environ[config["dataset_root_env"]])
    project = Path(os.environ[config["project_root_env"]])
    usage = sum(p.stat().st_size for p in project.rglob("*") if p.is_file() and not p.is_symlink())
    report = audit_metadata(root, config, json.loads(args.transfer_receipt.read_text()), usage,
                            shutil.disk_usage(project).free)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "files": report["files"], "patient_count": report["patient_count"]}))


if __name__ == "__main__":
    main()
