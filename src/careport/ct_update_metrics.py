"""Patient-balanced development contrasts and protected training grouping."""

from collections import defaultdict
import math

import numpy as np

from careport.ct_update_gate import GateError


def training_groups(records: list[dict], old_only: bool) -> list[list[list[dict]]]:
    groups = defaultdict(lambda: defaultdict(dict))
    if any(r["split"] not in {"train", "val"} for r in records):
        raise GateError("Unexpected/test partition in development cache")
    for record in records:
        if record["split"] != "train" or (old_only and not record["old_training"]):
            continue
        pair = groups[record["patient"]][record["lesion_id"]]
        if record["timepoint"] in pair:
            raise GateError("Duplicate lesion/timepoint in training index")
        pair[record["timepoint"]] = record
    result = []
    for patient in sorted(groups):
        lesions = []
        for identity in sorted(groups[patient]):
            pair = groups[patient][identity]
            if set(pair) != {"bl", "fu"}:
                raise GateError("Training lesion is missing a paired timepoint")
            lesions.append([pair["bl"], pair["fu"]])
        result.append(lesions)
    if not result:
        raise GateError("Empty training cohort")
    return result


def summarize_predictions(rows: list[dict], bootstrap_draws: int, seed: int) -> dict:
    if not rows or bootstrap_draws < 1:
        raise GateError("Empty evaluation or invalid bootstrap count")
    grouped = defaultdict(list)
    seen = set()
    strategies = {"old_old": ("old", "old"), "old_new": ("old", "new"),
                  "new_new": ("new", "new"), "old_continued": ("old", "continued"),
                  "continued_continued": ("continued", "continued")}
    for row in rows:
        key = row["patient"], row["lesion_id"]
        if key in seen:
            raise GateError("Duplicate evaluation lesion")
        seen.add(key)
        values = [row["reference_" + t] for t in ("bl", "fu")]
        values += [row[m + "_" + t] for m in ("old", "new", "continued") for t in ("bl", "fu")]
        dice = [row[m + "_dice"] for m in ("old", "new", "continued")]
        if any(not math.isfinite(v) or v < 0 for v in values) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in dice):
            raise GateError("Invalid volume or contour metric")
        change = row["reference_fu"] - row["reference_bl"]
        errors = {name: row[b + "_fu"] - row[a + "_bl"] - change for name, (a, b) in strategies.items()}
        grouped[row["patient"]].append({**errors, **{m + "_dice": row[m + "_dice"] for m in ("old", "new", "continued")}})
    patients = sorted(grouped)
    absolute = {name: np.array([np.mean([abs(r[name]) for r in grouped[p]]) for p in patients]) for name in strategies}
    signed = {name: np.array([np.mean([r[name] for r in grouped[p]]) for p in patients]) for name in strategies}
    samples = np.random.default_rng(seed).integers(0, len(patients), size=(bootstrap_draws, len(patients)))
    def estimate(array):
        low, high = np.quantile(array[samples].mean(axis=1), [.025, .975])
        return {"mean_ml": float(array.mean()), "exploratory_95_percentile_interval_ml": [float(low), float(high)]}
    contrasts = {"ordinary_update_benefit": absolute["old_old"] - absolute["old_new"],
                 "retaining_baseline_cost": absolute["old_new"] - absolute["new_new"],
                 "refresh_benefit": absolute["old_old"] - absolute["new_new"],
                 "added_data_vs_same_data_mixed": absolute["old_continued"] - absolute["old_new"]}
    return {"status": "DEVELOPMENT_ONLY", "patients": len(patients), "lesions": len(rows),
            "strategies": {name: {"mae_ml": float(absolute[name].mean()), "signed_error_ml": float(signed[name].mean())}
                           for name in strategies},
            "contrasts": {name: estimate(array) for name, array in contrasts.items()},
            "patient_mean_followup_dice": {m: float(np.mean([np.mean([r[m + "_dice"] for r in grouped[p]])
                                                           for p in patients])) for m in ("old", "new", "continued")},
            "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed,
            "test_evaluated": False, "confirmatory_inference": False}
