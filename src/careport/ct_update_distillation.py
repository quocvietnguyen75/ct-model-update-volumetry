"""Registered same-image old-teacher distillation with explicit checkpoint lineage."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from careport.ct_update_control_run import load_checkpoint, state_fingerprint
from careport.ct_update_gate import GateError
from careport.ct_update_train import (
    PatchSampler, checkpoint_state, execution_config, make_loss, make_model,
    make_optimizer, seed_runtime,
)


def validate_distillation_policy(policy: dict, base: dict, execution: dict) -> None:
    expected = {
        'analysis_id': 'CT-DISTILLATION-001', 'amendment': 'CT-UPDATE-006',
        'source_config_sha256': hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
        'execution_ids': ['CT-PILOT-002', 'CT-PILOT-003', 'CT-PILOT-004'],
        'training_seeds': [20260915, 20260917, 20260919], 'student_steps': 10000,
        'temperature': 2.0, 'candidates': {'kd_001': 1.0, 'kd_010': 10.0},
        'teacher_phase': 'old', 'teacher_input': 'same_augmented_image_as_student',
        'loss': 'supervised_DiceCE_plus_lambda_temperature_squared_binary_KL_teacher_to_student_mean_all_voxels',
        'training_data': 'all_eligible_training_patients', 'checkpoint_selection': 'final',
        'bootstrap_draws': 2000, 'bootstrap_seed': 20260916,
        'test_values_permitted': False, 'confirmatory_inference': False,
    }
    if json.dumps(policy, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise GateError('Unsupported or stale distillation policy')
    config = execution_config(base, execution)
    seeds = dict(zip(expected['execution_ids'], expected['training_seeds']))
    if (seeds.get(execution['execution_id']) != config['seed']
            or [config['training'][k] for k in ('old_steps', 'update_steps')] != [20000, 10000]):
        raise GateError('Distillation seed or training budget differs from the fixed batch')


def binary_distillation_loss(student: torch.Tensor, teacher: torch.Tensor,
                             temperature: float) -> torch.Tensor:
    if (not math.isfinite(temperature) or temperature <= 0
            or student.shape != teacher.shape or student.ndim < 3 or student.shape[1] != 1
            or student.numel() == 0 or not torch.is_floating_point(student)
            or not torch.is_floating_point(teacher)
            or not torch.isfinite(student).all() or not torch.isfinite(teacher).all()):
        raise GateError('Invalid binary distillation logits or temperature')
    current = student.float() / temperature
    previous = teacher.detach().float() / temperature
    log_p, log_not_p = F.logsigmoid(previous), F.logsigmoid(-previous)
    divergence = (log_p.exp() * (log_p - F.logsigmoid(current))
                  + log_not_p.exp() * (log_not_p - F.logsigmoid(-current)))
    return divergence.mean() * temperature**2


def frozen_teacher(student: torch.nn.Module) -> torch.nn.Module:
    return deepcopy(student).eval().requires_grad_(False)


def distillation_step(student: torch.nn.Module, teacher: torch.nn.Module,
                      image: torch.Tensor, target: torch.Tensor, criterion,
                      optimizer, coefficient: float, temperature: float) -> dict[str, float]:
    if (not math.isfinite(coefficient) or coefficient < 0 or teacher.training
            or any(p.requires_grad for p in teacher.parameters())):
        raise GateError('Distillation requires a frozen evaluation-mode teacher and nonnegative weight')
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(image.device.type, dtype=torch.bfloat16, enabled=image.is_cuda):
        prediction = student(image)
        with torch.no_grad():
            teacher_prediction = teacher(image)
        supervised = criterion(prediction, target)
    distillation = binary_distillation_loss(prediction, teacher_prediction, temperature)
    total = supervised + coefficient * distillation
    if not torch.isfinite(total):
        raise GateError('Nonfinite distillation training loss')
    total.backward()
    norm = torch.nn.utils.clip_grad_norm_(student.parameters(), float('inf'), error_if_nonfinite=True)
    optimizer.step()
    return {'supervised_loss': float(supervised.detach()),
            'distillation_loss': float(distillation.detach()),
            'total_loss': float(total.detach()), 'gradient_norm': float(norm)}


def load_distilled_checkpoint(directory: Path, name: str, config_sha: str, teacher_sha: str,
                              seed: int, policy_sha: str, coefficient: float,
                              temperature: float) -> tuple[dict, dict]:
    state, receipt = load_checkpoint(directory, name, config_sha, teacher_sha, seed, 10000)
    stored = torch.load(directory / (name + '.pt'), map_location='cpu', weights_only=True)
    expected = dict(teacher_sha256=teacher_sha, distillation_policy_sha256=policy_sha,
                    coefficient=coefficient, temperature=temperature)
    if (receipt.get('teacher_unchanged') is not True
            or any(stored.get(k) != v or receipt.get(k) != v for k, v in expected.items())):
        raise GateError('Distilled checkpoint teacher or policy differs from its receipt')
    return state, receipt


def train_distilled(name: str, coefficient: float, policy: dict, policy_sha: str,
                    config: dict, old_state: dict, old_sha: str, cache: Path,
                    records: list[dict], output: Path) -> tuple[dict, dict]:
    seed_runtime(config['seed'])
    student = make_model(config)
    student.load_state_dict(old_state, strict=True)
    student = student.cuda().train()
    teacher = frozen_teacher(student)
    teacher_hash = state_fingerprint(teacher.state_dict())
    optimizer = make_optimizer(student, config['training'])
    sampler = PatchSampler(cache, records, False, config['seed'])
    criterion = make_loss()
    steps = policy['student_steps']
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    totals = {key: 0. for key in ('supervised_loss', 'distillation_loss', 'total_loss')}
    count = 0
    with (output / (name + '_progress.jsonl')).open('x', encoding='utf-8') as stream:
        for step in range(1, steps + 1):
            image, target = sampler.batch(config['training']['batch_size'])
            values = distillation_step(student, teacher, image.cuda(), target.cuda(), criterion,
                                       optimizer, coefficient, policy['temperature'])
            for key in totals:
                totals[key] += values[key]
            count += 1
            if step == 1 or step % 100 == 0:
                row = {'phase': name, 'step': step, 'steps': steps,
                       'elapsed_seconds': time.monotonic() - started,
                       'gradient_norm': values['gradient_norm'],
                       **{'mean_' + key: value/count for key, value in totals.items()}}
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                print(json.dumps(row), flush=True)
                totals = dict.fromkeys(totals, 0.)
                count = 0
    if state_fingerprint(teacher.state_dict()) != teacher_hash:
        raise GateError('Teacher changed during distillation')
    fields = dict(phase=name, steps=steps, seed=config['seed'], parent_sha256=old_sha,
                  config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                  teacher_sha256=old_sha, distillation_policy_sha256=policy_sha,
                  coefficient=coefficient, temperature=policy['temperature'])
    path = output / (name + '.pt')
    with path.open('xb') as stream:
        torch.save({**fields, 'state_dict': checkpoint_state(student)}, stream)
    receipt = {**fields, 'status': 'TRAINING_COMPLETE',
               'execution_id': config['execution_policy']['execution_id'],
               'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'checkpoint_bytes': path.stat().st_size, 'teacher_unchanged': True,
               'teacher_state_sha256': teacher_hash,
               'eligible_training_patients': len(sampler.groups),
               'elapsed_seconds': time.monotonic() - started,
               'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    (output / (name + '_receipt.json')).write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)
    del student, teacher, optimizer, image, target
    torch.cuda.empty_cache()
    return load_distilled_checkpoint(output, name, fields['config_sha256'], old_sha,
                                     config['seed'], policy_sha, coefficient, policy['temperature'])


def distillation_runtime_smoke(config: dict, output: Path) -> dict:
    seed_runtime(config['seed'])
    student = make_model(config).cuda().train()
    teacher = frozen_teacher(student)
    before = state_fingerprint(teacher.state_dict())
    optimizer = make_optimizer(student, config['training'])
    image = torch.zeros((config['training']['batch_size'], 1, *config['patch']['shape']), device='cuda')
    target = torch.zeros_like(image)
    target[:, :, 32:64, 32:64, 32:64] = 1
    image.copy_(target)
    observations = [distillation_step(student, teacher, image, target, make_loss(), optimizer, weight, 2.)
                    for weight in (1., 10.)]
    if (state_fingerprint(teacher.state_dict()) != before
            or state_fingerprint(student.state_dict()) == before
            or any(p.grad is not None for p in teacher.parameters())):
        raise GateError('Synthetic distillation teacher/student update failed')
    report = dict(status='DISTILLATION_GPU_PASS', patient_data_used=False, temperature=2.,
                  coefficients=[1., 10.], teacher_unchanged=True, student_updated=True,
                  synthetic_steps=observations)
    (output / 'distillation_gpu_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    del student, teacher, optimizer, image, target
    torch.cuda.empty_cache()
    return report
