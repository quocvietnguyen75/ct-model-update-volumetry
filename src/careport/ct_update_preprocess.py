"""Bounded, fixed-field-of-view development patch extraction after schema pass."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil

import numpy as np

from careport.ct_update_data import patch_affine, sample_patch, volume_paths
from careport.ct_update_gate import GateError, validate_input_policy, validate_storage


def resolve_csv_discrepancy(report: dict, independent: dict) -> dict:
    count = report.get("issues", {}).get("csv_native_volume_mismatch", 0)
    if (report.get("status") != "SCIENTIFIC_SCHEMA_FAIL"
            or set(report.get("issues", {})) != {"csv_native_volume_mismatch"}
            or independent.get("status") != "INDEPENDENT_NATIVE_COUNT_PASS"
            or independent.get("checked") != count or count <= 0
            or independent.get("count_mismatches") != 0 or independent.get("geometry_mismatches") != 0
            or independent.get("full_vs_bounded_count_agreements") != count
            or independent.get("source_manifest_sha256") != report.get("private_manifest_sha256")):
        raise GateError("Reference discrepancy lacks a matching independent resolution")
    return {**report, "status": "SCIENTIFIC_SCHEMA_PASS", "issues": {},
            "warnings": dict(report["issues"]), "resolved_issue_types": ["csv_native_volume_mismatch"],
            "resolution_amendment": "CT-UPDATE-002", "primary_reference": "complete_native_mask",
            "original_failed_report_sha256": hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
            "independent_count_receipt": independent}


def require_schema(report: dict, manifest_bytes: bytes, config: dict) -> None:
    if (report.get("status") != "SCIENTIFIC_SCHEMA_PASS"
            or report.get("private_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest()
            or report.get("config_sha256") != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()):
        raise GateError("Successful scientific schema receipt is missing, stale or changed")


def projected_cache_bytes(patches: int, shape, cap: int) -> int:
    projected = patches * (int(np.prod(shape)) * 3 + 16384)
    if patches <= 0 or projected > cap:
        raise GateError("Patch cache projection exceeds its cap or has no patches")
    return projected


def make_patch_pair(image, labels, record, policy):
    native = np.asarray(record["native_affine"])
    geometry = patch_affine(native, record["centroid"], policy["shape"], policy["spacing_mm"])
    values = sample_patch(image, native, geometry, policy["shape"], 1, -1000)
    low, high = policy["hu_clip"]
    values = (np.clip(values, low, high) / 1000).astype(np.float16)
    target = (sample_patch(labels, native, geometry, policy["shape"], 0, 0) == record["lesion_id"]).astype(np.uint8)
    corners = np.array(list(itertools.product(*[(a, b - 1) for a, b in record["native_lesion_box"]])))
    mapping = np.linalg.inv(geometry) @ native
    mapped = corners @ mapping[:3, :3].T + mapping[:3, 3]
    truncated = bool(np.any(mapped < 0) or np.any(mapped > np.asarray(policy["shape"]) - 1))
    if not np.isfinite(values).all():
        raise GateError("Nonfinite CT patch")
    return values, target, geometry, truncated


def extract_volume(task: dict) -> dict:
    import nibabel as nib

    image_path, mask_path = volume_paths(Path(task["root"]), task["patient"], task["timepoint"], task["index"])
    image_obj, mask_obj = nib.load(image_path), nib.load(mask_path)
    image = np.asanyarray(image_obj.dataobj)
    labels = np.asanyarray(mask_obj.dataobj)
    records = []
    counts: Counter = Counter()
    for record in task["records"]:
        if (list(image.shape) != record["native_shape"] or labels.shape != image.shape
                or not np.allclose(image_obj.affine, record["native_affine"], rtol=0, atol=1e-4)
                or not np.allclose(mask_obj.affine, image_obj.affine, rtol=0, atol=1e-4)):
            raise GateError("Native geometry changed since schema audit")
        patch, target, geometry, truncated = make_patch_pair(image, labels, record, task["patch_policy"])
        key = f"{task['patient']}:{record['lesion_id']}:{task['timepoint']}"
        filename = hashlib.sha256(key.encode()).hexdigest() + ".npz"
        output = Path(task["output"]) / "patches" / filename
        with output.open("xb") as stream:
            np.savez(stream, image=patch, target=target)
        counts["patches"] += 1
        counts["bytes"] += output.stat().st_size
        counts["truncated_reference_boxes"] += truncated
        counts["empty_resampled_targets"] += not bool(target.any())
        records.append({**record, "patient": task["patient"], "split": task["split"],
                        "timepoint": task["timepoint"], "image_index": task["index"],
                        "patch_file": "patches/" + filename, "patch_affine": geometry.tolist(),
                        "reference_box_truncated": truncated})
    return {"records": records, "counts": dict(counts), "split": task["split"]}


def preprocess(root: Path, project: Path, schema: Path, output: Path, config: dict, workers: int,
               volume_recount: dict | None = None) -> dict:
    validate_input_policy(config)
    manifest_bytes = (schema / "development_records_private.jsonl").read_bytes()
    report = json.loads((schema / "schema_report.json").read_text())
    if volume_recount is not None:
        report = resolve_csv_discrepancy(report, volume_recount)
    require_schema(report, manifest_bytes, config)
    tasks = [json.loads(line) for line in manifest_bytes.splitlines()]
    permitted_warnings = set(report.get("resolved_issue_types", []))
    if any(t["split"] not in {"train", "val"} or set(t["issues"]) - permitted_warnings for t in tasks):
        raise GateError("Unexpected partition or unresolved schema issue")
    total_patches = sum(len(t["records"]) for t in tasks)
    projection = projected_cache_bytes(total_patches, config["patch"]["shape"], config["storage"]["cache_cap_bytes"])
    usage = sum(p.stat().st_size for p in project.rglob("*") if p.is_file() and not p.is_symlink())
    storage = validate_storage(config["storage"], usage, shutil.disk_usage(project).free)
    output.mkdir(parents=True, exist_ok=False)
    (output / "schema_resolution_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "patches").mkdir()
    for task in tasks:
        task.update(root=str(root), output=str(output), patch_policy=config["patch"])
    counts: Counter = Counter()
    by_split = {"train": Counter(), "val": Counter()}
    index = output / "cache_index_private.jsonl"
    with index.open("x", encoding="utf-8") as stream, ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(extract_volume, tasks):
            counts.update(result["counts"])
            by_split[result["split"]].update(result["counts"])
            for record in result["records"]:
                stream.write(json.dumps(record) + "\n")
            if counts["bytes"] > projection:
                raise GateError("Actual cache exceeded the preflight projection")
            print(json.dumps({"patches_written": counts["patches"], "total": total_patches}), flush=True)
    final = {"status": "PREPROCESSING_PASS", "date": config["date"], "protocol_id": config["protocol_id"],
             "counts": dict(counts), "by_split": {k: dict(v) for k, v in by_split.items()},
             "projected_cache_bytes": projection, "storage": storage, "test_values_read": False,
             "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
             "schema_manifest_sha256": report["private_manifest_sha256"],
             "config_sha256": report["config_sha256"]}
    (output / "preprocessing_report.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final), flush=True)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--volume-recount", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    preprocess(Path(os.environ[config["dataset_root_env"]]), Path(os.environ[config["project_root_env"]]),
               args.schema, args.output, config, args.workers,
               json.loads(args.volume_recount.read_text()) if args.volume_recount else None)


if __name__ == "__main__":
    main()
