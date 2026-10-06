"""Native-grid validation of old, updated and same-data continuation models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
import torch

from careport.ct_update_data import native_prediction, volume_paths
from careport.ct_update_gate import GateError, validate_input_policy
from careport.ct_update_metrics import summarize_predictions
from careport.ct_update_train import execution_config, make_model, seed_runtime, validate_cache_receipt


def native_metrics(logits, patch_geometry, native_geometry, labels, identity, reference_voxels):
    if not np.isfinite(logits).all():
        raise GateError("Nonfinite model prediction")
    reference = labels == identity
    if int(np.count_nonzero(reference)) != reference_voxels or reference_voxels <= 0:
        raise GateError("Reference mask changed or is empty")
    slices, prediction = native_prediction(logits, patch_geometry, native_geometry, labels.shape)
    predicted_voxels = int(prediction.sum())
    intersection = int(np.count_nonzero(prediction & reference[slices]))
    voxel_ml = abs(float(np.linalg.det(np.asarray(native_geometry)[:3, :3]))) / 1000
    hd95 = None
    if predicted_voxels:
        # Full reference support is included, even outside the prediction crop.
        reference_box = ndimage.find_objects(reference.astype(np.uint8))[0]
        union = tuple(slice(min(a.start, b.start), max(a.stop, b.stop)) for a, b in zip(slices, reference_box))
        region = np.zeros(tuple(s.stop - s.start for s in union), dtype=bool)
        offset = tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(slices, union))
        region[offset] = prediction
        predicted = np.pad(region, 1)
        target = np.pad(reference[union], 1)
        structure = ndimage.generate_binary_structure(3, 1)
        ps = predicted & ~ndimage.binary_erosion(predicted, structure, border_value=0)
        ts = target & ~ndimage.binary_erosion(target, structure, border_value=0)
        spacing = np.linalg.norm(np.asarray(native_geometry)[:3, :3], axis=0)
        dt = ndimage.distance_transform_edt(~ts, sampling=spacing)[ps]
        dp = ndimage.distance_transform_edt(~ps, sampling=spacing)[ts]
        hd95 = max(float(np.quantile(dt, .95)), float(np.quantile(dp, .95)))
    return {"volume_ml": predicted_voxels * voxel_ml,
            "dice": 2 * intersection / (predicted_voxels + reference_voxels),
            "hd95_mm": hd95, "empty_prediction": predicted_voxels == 0}


def paired_prediction_rows(records, use_csv_reference):
    groups = defaultdict(dict)
    for record in records:
        pair = groups[(record["patient"], record["lesion_id"])]
        if record["timepoint"] not in {"bl", "fu"} or record["timepoint"] in pair:
            raise GateError("Duplicate/invalid evaluation timepoint")
        pair[record["timepoint"]] = record
    result = []
    for (patient, identity), pair in sorted(groups.items()):
        if set(pair) != {"bl", "fu"}:
            raise GateError("Evaluation lesion is missing a paired timepoint")
        row = {"patient": patient, "lesion_id": identity}
        for timepoint, record in pair.items():
            row["reference_" + timepoint] = (record["csv_volume_mm3"] / 1000 if use_csv_reference
                                             else record["reference_volume_ml"])
            for phase in ("old", "new", "continued"):
                row[phase + "_" + timepoint] = record["predictions"][phase]["volume_ml"]
                if timepoint == "fu":
                    row[phase + "_dice"] = record["predictions"][phase]["dice"]
        result.append(row)
    return result


def evaluate(config, cache, training, output, policy=None):
    import nibabel as nib

    validate_input_policy(config)
    index_bytes = (cache / "cache_index_private.jsonl").read_bytes()
    validate_cache_receipt(json.loads((cache / "preprocessing_report.json").read_text()), index_bytes, config)
    all_records = [json.loads(line) for line in index_bytes.splitlines()]
    if any(r["split"] not in {"train", "val"} for r in all_records):
        raise GateError("Unexpected partition in development cache")
    records = sorted((r for r in all_records if r["split"] == "val"),
                     key=lambda r: (r["patient"], r["timepoint"], r["image_index"], r["lesion_id"]))
    if not records or not (training / "training_complete.json").is_file():
        raise GateError("Training is incomplete or validation is empty")
    config = execution_config(config, policy)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    old_hash = hashlib.sha256((training / "old.pt").read_bytes()).hexdigest()
    models = {}
    receipts = {}
    seed_runtime(config["seed"])
    for phase in ("old", "new", "continued"):
        checkpoint = training / f"{phase}.pt"
        receipt = json.loads((training / f"{phase}_receipt.json").read_text())
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != receipt["checkpoint_sha256"]:
            raise GateError("Checkpoint differs from its training receipt")
        stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (stored["config_sha256"] != config_hash or stored["phase"] != phase
                or stored["parent_sha256"] != (None if phase == "old" else old_hash)):
            raise GateError("Checkpoint configuration or lineage mismatch")
        model = make_model(config)
        model.load_state_dict(stored["state_dict"], strict=True)
        models[phase] = model.cuda().eval()
        receipts[phase] = receipt
    output.mkdir(parents=True, exist_ok=False)
    root = Path(os.environ[config["dataset_root_env"]])

    @lru_cache(maxsize=1)
    def mask_for(patient, timepoint, index):
        _, path = volume_paths(root, patient, timepoint, index)
        volume = nib.load(path)
        return np.asanyarray(volume.dataobj), volume.affine

    predictions = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with (output / "predictions_private.jsonl").open("x", encoding="utf-8") as stream:
        for i, record in enumerate(records):
            patch_path = cache / record["patch_file"]
            if not patch_path.resolve().is_relative_to(cache.resolve()):
                raise GateError("Patch path escapes the cache")
            with np.load(patch_path, allow_pickle=False) as payload:
                image = torch.from_numpy(payload["image"].astype(np.float32)[None, None]).cuda()
            labels, geometry = mask_for(record["patient"], record["timepoint"], record["image_index"])
            if not np.allclose(geometry, record["native_affine"], rtol=0, atol=1e-4):
                raise GateError("Reference geometry changed after schema audit")
            measured = {}
            for phase, model in models.items():
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(image)[0, 0].float().cpu().numpy()
                measured[phase] = native_metrics(logits, record["patch_affine"], geometry,
                                                 labels, record["lesion_id"], record["reference_voxels"])
            result = {k: record[k] for k in ("patient", "lesion_id", "timepoint", "reference_volume_ml", "csv_volume_mm3")}
            result["predictions"] = measured
            predictions.append(result)
            stream.write(json.dumps(result) + "\n")
            stream.flush()
            if (i + 1) % 20 == 0:
                print(json.dumps({"validation_timepoints_evaluated": i + 1, "total": len(records)}), flush=True)
    summary = summarize_predictions(paired_prediction_rows(predictions, False), config["bootstrap_draws"], config["bootstrap_seed"])
    sensitivity = summarize_predictions(paired_prediction_rows(predictions, True), config["bootstrap_draws"], config["bootstrap_seed"])
    boundaries = {}
    for phase in models:
        patient_hd = defaultdict(list)
        empty = 0
        for result in predictions:
            if result["timepoint"] == "fu":
                measured = result["predictions"][phase]
                empty += measured["empty_prediction"]
                if measured["hd95_mm"] is not None:
                    patient_hd[result["patient"]].append(measured["hd95_mm"])
        boundaries[phase] = {"patient_mean_hd95_mm_nonempty_only": float(np.mean([np.mean(v) for v in patient_hd.values()])) if patient_hd else None,
                             "empty_followup_predictions": empty, "patients_with_defined_hd95": len(patient_hd)}
    summary.update(protocol_id=config["protocol_id"], date=config["date"], checkpoint_receipts=receipts,
                   execution_id=config.get("execution_policy", {}).get("execution_id", "CT-PILOT-001"),
                   csv_reference_sensitivity=sensitivity, boundary_diagnostics=boundaries,
                   elapsed_seconds=time.monotonic() - started, config_sha256=config_hash,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   private_predictions_sha256=hashlib.sha256((output / "predictions_private.jsonl").read_bytes()).hexdigest())
    (output / "development_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-policy", type=Path)
    args = parser.parse_args()
    evaluate(json.loads(args.config.read_text()), args.cache, args.training, args.output,
             json.loads(args.execution_policy.read_text()) if args.execution_policy else None)


if __name__ == "__main__":
    main()
