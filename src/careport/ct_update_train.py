"""Fixed-budget development pilot with exact old-model checkpoint lineage."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from careport.ct_update_gate import GateError, validate_input_policy, validate_storage
from careport.ct_update_metrics import training_groups


def execution_config(base: dict, policy: dict | None) -> dict:
    result = deepcopy(base)
    if policy is None:
        return result
    expected = {"execution_id", "amendment", "source_config_sha256", "seed", "old_steps", "update_steps"}
    if (set(policy) != expected
            or policy["source_config_sha256"] != hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
            or not isinstance(policy["execution_id"], str) or not policy["execution_id"]
            or not isinstance(policy["amendment"], str) or not policy["amendment"]):
        raise GateError("Execution policy is stale or has unauthorized fields")
    for field in ("seed", "old_steps", "update_steps"):
        if type(policy[field]) is not int or not 0 < policy[field] < 2**32:
            raise GateError("Invalid execution seed or training budget")
    result["seed"] = policy["seed"]
    for field in ("old_steps", "update_steps"):
        result["training"][field] = policy[field]
    result["execution_policy"] = deepcopy(policy)
    return result


def validate_cache_receipt(report, index_bytes, config):
    if (report.get("status") != "PREPROCESSING_PASS" or report.get("test_values_read") is not False
            or report.get("index_sha256") != hashlib.sha256(index_bytes).hexdigest()
            or report.get("config_sha256") != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()):
        raise GateError("Preprocessing receipt is failed, stale or changed")


class PatchSampler:
    def __init__(self, cache: Path, records: list[dict], old_only: bool, seed: int):
        self.cache = cache
        self.groups = training_groups(records, old_only)
        self.rng = np.random.default_rng(seed)

    def select_record(self) -> dict:
        lesions = self.groups[int(self.rng.integers(len(self.groups)))]
        timepoints = lesions[int(self.rng.integers(len(lesions)))]
        return timepoints[int(self.rng.integers(2))]

    def batch(self, size: int):
        images, masks = [], []
        for _ in range(size):
            record = self.select_record()
            path = self.cache / record["patch_file"]
            if not path.resolve().is_relative_to(self.cache.resolve()):
                raise GateError("Patch path escapes the cache")
            with np.load(path, allow_pickle=False) as payload:
                image = payload["image"].astype(np.float32)
                target = payload["target"].astype(np.float32)
            for axis in range(3):
                if self.rng.integers(2):
                    image, target = np.flip(image, axis), np.flip(target, axis)
            images.append(np.ascontiguousarray(image[None]))
            masks.append(np.ascontiguousarray(target[None]))
        return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(masks))


def checkpoint_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_optimizer(model, policy):
    return torch.optim.AdamW(model.parameters(), lr=policy["learning_rate"], weight_decay=policy["weight_decay"])


def make_model(config):
    from monai.networks.nets import UNet

    policy = dict(config["model"])
    if policy.pop("family") != "monai.networks.nets.UNet":
        raise GateError("Unapproved model family")
    return UNet(**policy)


def seed_runtime(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_loss():
    from monai.losses import DiceCELoss

    return DiceCELoss(sigmoid=True, squared_pred=False, smooth_nr=1e-5, smooth_dr=1e-5)


def runtime_smoke(config, output):
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
        raise GateError("The pilot requires an allocated H100")
    seed_runtime(config["seed"])
    model = make_model(config).cuda()
    loss_function = make_loss()
    optimizer = make_optimizer(model, config["training"])
    shape = (config["training"]["batch_size"], 1, *config["patch"]["shape"])
    image = torch.zeros(shape, device="cuda")
    target = torch.zeros_like(image)
    target[:, :, 32:64, 32:64, 32:64] = 1
    image.copy_(target)
    before = next(model.parameters()).detach().clone()
    started = time.monotonic()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(image)
            loss = loss_function(prediction, target)
        if prediction.shape != target.shape or not torch.isfinite(loss):
            raise GateError("Synthetic forward/loss failed")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise GateError("Nonfinite synthetic gradient")
        optimizer.step()
    torch.cuda.synchronize()
    if torch.equal(before, next(model.parameters()).detach()):
        raise GateError("Synthetic optimizer did not update parameters")
    report = {"status": "GPU_RUNTIME_PASS", "gpu": torch.cuda.get_device_name(0),
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "synthetic_two_step_seconds": time.monotonic() - started,
              "versions": {p: version(p) for p in ("torch", "numpy", "scipy", "monai", "nibabel", "importlib_resources")},
              "autocast_dtype": "bfloat16", "patient_data_used": False}
    (output / "gpu_runtime_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    del model, image, target, optimizer, prediction, loss
    torch.cuda.empty_cache()


def phase_plan(phase: str, config: dict) -> tuple[bool, int, str | None]:
    plans = {"old": (True, "old_steps", None),
             "new": (False, "update_steps", "old"),
             "continued": (True, "update_steps", "old"),
             "scratch_base": (False, "old_steps", None),
             "scratch": (False, "update_steps", "scratch_base")}
    if phase not in plans:
        raise GateError("Unknown training phase")
    old_only, budget, parent_phase = plans[phase]
    return old_only, config["training"][budget], parent_phase


def train_phase(phase, config, cache, records, output, parent):
    old_only, steps, parent_phase = phase_plan(phase, config)
    if (parent is None) != (parent_phase is None):
        raise GateError("Training phase has the wrong parent requirement")
    seed_runtime(config["seed"])
    model = make_model(config)
    parent_hash = None
    if parent is not None:
        parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
        stored = torch.load(parent, map_location="cpu", weights_only=True)
        if stored["phase"] != parent_phase or stored["config_sha256"] != hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest():
            raise GateError("Wrong parent checkpoint lineage")
        model.load_state_dict(stored["state_dict"], strict=True)
    model = model.cuda().train()
    optimizer = make_optimizer(model, config["training"])
    sampler = PatchSampler(cache, records, old_only, config["seed"])
    loss_function = make_loss()
    started = time.monotonic()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    with (output / f"{phase}_progress.jsonl").open("x", encoding="utf-8") as progress:
        for step in range(1, steps + 1):
            image, target = sampler.batch(config["training"]["batch_size"])
            image, target = image.cuda(), target.cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(image)
                loss = loss_function(prediction, target)
            if not torch.isfinite(loss):
                raise GateError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step == 1 or step % 100 == 0:
                report = {"phase": phase, "step": step, "steps": steps,
                          "elapsed_seconds": round(time.monotonic() - started, 3),
                          "mean_training_loss_since_log": float(np.mean(losses)),
                          "gradient_norm": float(norm), "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                progress.write(json.dumps(report) + "\n")
                progress.flush()
                print(json.dumps(report), flush=True)
                losses.clear()
    path = output / f"{phase}.pt"
    with path.open("xb") as stream:
        torch.save({"state_dict": checkpoint_state(model), "phase": phase, "seed": config["seed"],
                    "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                    "parent_sha256": parent_hash, "steps": steps}, stream)
    receipt = {"phase": phase, "steps": steps, "seed": config["seed"], "parent_sha256": parent_hash,
               "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
               "execution_id": config.get("execution_policy", {}).get("execution_id", "CT-PILOT-001"),
               "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "checkpoint_bytes": path.stat().st_size, "eligible_training_patients": len(sampler.groups),
               "elapsed_seconds": time.monotonic() - started,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "status": "TRAINING_COMPLETE"}
    (output / f"{phase}_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    del model, optimizer, loss, prediction, image, target
    torch.cuda.empty_cache()
    return path


def main():
    import shutil

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-policy", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_input_policy(config)
    index_bytes = (args.cache / "cache_index_private.jsonl").read_bytes()
    validate_cache_receipt(json.loads((args.cache / "preprocessing_report.json").read_text()), index_bytes, config)
    records = [json.loads(line) for line in index_bytes.splitlines()]
    training_groups(records, False)
    project = Path(os.environ[config["project_root_env"]])
    usage = sum(p.stat().st_size for p in project.rglob("*") if p.is_file() and not p.is_symlink())
    validate_storage(config["storage"], usage, shutil.disk_usage(project).free)
    config = execution_config(config, json.loads(args.execution_policy.read_text()) if args.execution_policy else None)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "execution_config.json").write_text(json.dumps(config, indent=2) + "\n")
    runtime_smoke(config, args.output)
    old = train_phase("old", config, args.cache, records, args.output, None)
    train_phase("new", config, args.cache, records, args.output, old)
    train_phase("continued", config, args.cache, records, args.output, old)
    (args.output / "training_complete.json").write_text(json.dumps({"status": "THREE_PHASES_COMPLETE", "test_used": False}) + "\n")


if __name__ == "__main__":
    main()
