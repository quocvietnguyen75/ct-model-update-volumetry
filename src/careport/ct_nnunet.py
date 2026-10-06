"""CT-UPDATE-017 adapter: nnU-Net v2 (ResEnc-M) on the frozen lesion-centred patches, with the study's update protocol.

The training cases are the patches of the frozen development cache (96^3 voxels at 1.5 mm, CT already clipped and
scaled), so nnU-Net runs without normalization or resampling and every model sees the inputs the U-Net saw.
nnU-Net reads its folder settings (nnUNet_raw, nnUNet_preprocessed, nnUNet_results) when first imported, so callers
set them in the environment first. Nothing here imports nnU-Net until it is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from careport.ct_update_gate import GateError
from careport.ct_update_metrics import training_groups

DATASETS = {'old': (901, 'CTUpdateOld'), 'expanded': (902, 'CTUpdateExpanded')}
PLANNER = 'nnUNetPlannerResEncM'
PLANS = 'nnUNetResEncUNetMPlans'
CONFIGURATION = '3d_fullres'
FOLD = 'all'
SPACING = (1.5, 1.5, 1.5)
# phase -> (training subset, parent phase, trainer class name); parents are loaded in full, segmentation head included
PHASES = {'old': ('old', None, 'nnUNetTrainerCTOld'),
          'new': ('expanded', 'old', 'nnUNetTrainerCTUpdate'),
          'continued': ('old', 'old', 'nnUNetTrainerCTContinue')}


def dataset_name(subset: str) -> str:
    if subset not in DATASETS:
        raise GateError('Unknown nnU-Net training subset')
    number, name = DATASETS[subset]
    return f'Dataset{number:03d}_{name}'


def dataset_json(count: int) -> dict:
    return dict(channel_names={'0': 'noNorm'}, labels={'background': 0, 'lesion': 1}, numTraining=count,
                file_ending='.nii.gz', overwrite_image_reader_writer='SimpleITKIO')


def training_records(records: list[dict], subset: str) -> list[dict]:
    """The U-Net's own membership: training split, and the old_training flag for the old subset."""
    if subset not in DATASETS:
        raise GateError('Unknown nnU-Net training subset')
    groups = training_groups(records, subset == 'old')
    return [record for patient in groups for pair in patient for record in pair]


def case_name(index: int) -> str:
    return f'ctpatch_{index:05d}'


def write_case(image: np.ndarray, target: np.ndarray, image_path: Path, label_path: Path) -> None:
    """SimpleITK keeps the array order, so nnU-Net reads back exactly the cached patch."""
    import SimpleITK as sitk
    if image.ndim != 3 or image.shape != target.shape:
        raise GateError('Patch image and target disagree')
    picture = sitk.GetImageFromArray(np.ascontiguousarray(image, dtype=np.float32))
    labels = sitk.GetImageFromArray(np.ascontiguousarray(target > 0, dtype=np.uint8))
    for item in (picture, labels):
        item.SetSpacing(SPACING)
    sitk.WriteImage(picture, str(image_path), useCompression=True)
    sitk.WriteImage(labels, str(label_path), useCompression=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def patch_file_for(record: dict) -> str:
    """The development cache names each patch by the SHA-256 of its patient:lesion:timepoint key."""
    key = f"{record['patient']}:{record['lesion_id']}:{record['timepoint']}"
    return 'patches/' + hashlib.sha256(key.encode()).hexdigest() + '.npz'


def write_dataset(cache: Path, records: list[dict], subset: str, raw_root: Path) -> dict:
    """Write one nnU-Net raw dataset from cached patches; case names carry no identifier.

    The cache index is bound by its hash elsewhere. Here each record must name its own patch, and the content hash
    of every patch read is recorded (privately per case, publicly as one digest) so later changes are detectable.
    """
    selected = training_records(records, subset)
    cache = Path(cache).resolve()
    folder = Path(raw_root)/dataset_name(subset)
    (folder/'imagesTr').mkdir(parents=True, exist_ok=False)
    (folder/'labelsTr').mkdir()
    mapping = []
    for index, record in enumerate(selected):
        if record['patch_file'] != patch_file_for(record):
            raise GateError('Index record does not name its own patch file')
        path = (cache/record['patch_file']).resolve()
        if not path.is_relative_to(cache):
            raise GateError('Patch path escapes the cache')
        content = path.read_bytes()
        with np.load(io.BytesIO(content), allow_pickle=False) as payload:
            image, target = payload['image'], payload['target']
        if not np.isin(target, (0, 1)).all():
            raise GateError('Cached target is not binary')
        write_case(image, target, folder/'imagesTr'/f'{case_name(index)}_0000.nii.gz',
                   folder/'labelsTr'/f'{case_name(index)}.nii.gz')
        mapping.append(dict(case=case_name(index), patch_file=record['patch_file'],
                            patch_sha256=hashlib.sha256(content).hexdigest()))
    (folder/'dataset.json').write_text(json.dumps(dataset_json(len(selected)), indent=2) + '\n')
    with (folder/'cases_private.jsonl').open('x') as stream:
        stream.writelines(json.dumps(m) + '\n' for m in mapping)
    digest = hashlib.sha256(''.join(f"{m['case']} {m['patch_sha256']}\n" for m in mapping).encode()).hexdigest()
    return dict(dataset=dataset_name(subset), cases=len(selected), patients=len({r['patient'] for r in selected}),
                patch_content_digest=digest, dataset_json_sha256=_sha256(folder/'dataset.json'),
                case_mapping_sha256=_sha256(folder/'cases_private.jsonl'))


def folder_digest(folder: Path) -> dict:
    """One SHA-256 over every file's relative path, size and content hash, in sorted order."""
    folder = Path(folder)
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    lines = ''.join(f'{p.relative_to(folder).as_posix()} {p.stat().st_size} {_sha256(p)}\n' for p in files)
    return dict(files=len(files), bytes=sum(p.stat().st_size for p in files),
                sha256=hashlib.sha256(lines.encode()).hexdigest())


def seed_all(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainer_class(phase: str, epochs: int):
    """An nnUNetTrainer that differs from the default only in its epoch count and its (phase-specific) name."""
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    if phase not in PHASES or type(epochs) is not int or epochs < 1:
        raise GateError('Invalid nnU-Net phase or epoch budget')

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        nnUNetTrainer.__init__(self, plans, configuration, fold, dataset_json, device)
        self.num_epochs = epochs

    return type(PHASES[phase][2], (nnUNetTrainer,), {'__init__': __init__, '__module__': __name__})


def make_trainer(phase: str, epochs: int, preprocessed_root: Path, device):
    folder = Path(preprocessed_root)/dataset_name(PHASES[phase][0])
    plans = json.loads((folder/f'{PLANS}.json').read_text())
    plans['continue_training'] = False  # nnU-Net 2.8 expects this key and removes it
    description = json.loads((folder/'dataset.json').read_text())
    return trainer_class(phase, epochs)(plans=plans, configuration=CONFIGURATION, fold=FOLD,
                                        dataset_json=description, device=device)


def base_module(network):
    from torch._dynamo import OptimizedModule
    from torch.nn.parallel import DistributedDataParallel
    if isinstance(network, DistributedDataParallel):
        network = network.module
    if isinstance(network, OptimizedModule):
        network = network._orig_mod
    return network


def load_full_weights(trainer, checkpoint: Path) -> None:
    """Strict load of every parameter; nnU-Net's own pretrained-weight loader would reset the segmentation head."""
    import torch
    if not trainer.was_initialized:
        trainer.initialize()
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['network_weights']
    base_module(trainer.network).load_state_dict(state, strict=True)


def train_phase(phase: str, epochs: int, preprocessed_root: Path, device, seed: int, parent: Path | None = None,
                iterations: tuple[int, int] | None = None) -> tuple[Path, dict]:
    if (parent is None) != (PHASES[phase][1] is None):
        raise GateError('Training phase has the wrong parent requirement')
    seed_all(seed)
    trainer = make_trainer(phase, epochs, preprocessed_root, device)
    if iterations is not None:  # synthetic checks only
        trainer.num_iterations_per_epoch, trainer.num_val_iterations_per_epoch = iterations
    if parent is not None:
        load_full_weights(trainer, parent)
    started = time.monotonic()
    trainer.run_training()
    final = Path(trainer.output_folder)/'checkpoint_final.pth'
    logged = trainer.logger.local_logger.my_fantastic_logging
    epoch_seconds = [end - start for start, end in zip(logged['epoch_start_timestamps'], logged['epoch_end_timestamps'])]
    return final, dict(phase=phase, epochs=epochs, seed=seed, iterations_per_epoch=trainer.num_iterations_per_epoch,
                       elapsed_seconds=time.monotonic() - started, epoch_seconds=epoch_seconds,
                       checkpoint_sha256=_sha256(final), parent_sha256=None if parent is None else _sha256(parent),
                       trainer=PHASES[phase][2])


def weights_match(network, state: dict) -> bool:
    """True only if the network holds exactly these weights: the same names and equal tensors."""
    import torch
    current = base_module(network).state_dict()
    return set(current) == set(state) and all(torch.equal(current[k].detach().cpu(), state[k].detach().cpu())
                                              for k in state)


def binary_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """Dice of two binary masks; the reference must be nonempty."""
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target) > 0
    if prediction.shape != target.shape or not target.any():
        raise GateError('Dice needs masks of one shape and a nonempty reference')
    return float(2 * np.logical_and(prediction, target).sum() / (prediction.sum() + target.sum()))


def sanity_patches(cache: Path, records: list[dict], count: int) -> list[tuple[dict, np.ndarray, np.ndarray]]:
    """The first `count` old-model training patches with a nonempty target, in membership order (training data only)."""
    cache = Path(cache).resolve()
    chosen = []
    for item in training_records(records, 'old'):
        if item['patch_file'] != patch_file_for(item):
            raise GateError('Index record does not name its own patch file')
        path = (cache/item['patch_file']).resolve()
        if not path.is_relative_to(cache):
            raise GateError('Patch path escapes the cache')
        with np.load(path, allow_pickle=False) as payload:
            image, target = payload['image'].astype(np.float32), payload['target']
        if target.any():
            chosen.append((item, image, target))
            if len(chosen) == count:
                return chosen
    raise GateError('Too few nonempty old-model training patches for the sanity check')


def make_predictor(checkpoint: Path, device):
    """nnU-Net's sliding-window predictor with default mirroring, built without a trainer-class lookup.

    manual_initialization only stores its parameter list; nnU-Net loads it in predict_logits_from_preprocessed_data,
    never in predict_sliding_window_return_logits. The trained weights are therefore loaded here and checked
    (CT-UPDATE-017 correction A: the first evaluation used untrained networks).
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    folder = Path(checkpoint).parent.parent
    plans_manager = PlansManager(json.loads((folder/'plans.json').read_text()))
    description = json.loads((folder/'dataset.json').read_text())
    configuration = plans_manager.get_configuration(CONFIGURATION)
    stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    network = nnUNetTrainer.build_network_architecture(
        plans_manager, configuration, determine_num_input_channels(plans_manager, configuration, description),
        plans_manager.get_label_manager(description).num_segmentation_heads, enable_deep_supervision=False)
    network.load_state_dict(stored['network_weights'])
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=device.type == 'cuda', device=device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(network, plans_manager, configuration, [stored['network_weights']], description,
                                    stored['trainer_name'], stored['inference_allowed_mirroring_axes'])
    if not weights_match(predictor.network, stored['network_weights']):
        raise GateError('The predictor does not hold the trained checkpoint weights')
    return predictor


def foreground_logit(predictor, image: np.ndarray) -> np.ndarray:
    """Lesion minus background logit: positive exactly where the two-class softmax exceeds one half."""
    import torch
    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)[None])
    logits = predictor.predict_sliding_window_return_logits(tensor)
    logits = logits.float().cpu().numpy() if isinstance(logits, torch.Tensor) else np.asarray(logits, np.float32)
    if logits.shape != (2, *image.shape) or not np.isfinite(logits).all():
        raise GateError('nnU-Net returned logits of an unexpected shape or nonfinite values')
    return logits[1] - logits[0]


def transfer_plans(raw: Path, preprocessed: Path) -> None:
    """The expanded dataset uses the old dataset's plans, so every model has the same architecture and inputs."""
    import shutil
    from nnunetv2.experiment_planning.plans_for_pretraining.move_plans_between_datasets import \
        move_plans_between_datasets
    move_plans_between_datasets(DATASETS['old'][0], DATASETS['expanded'][0], PLANS, PLANS)
    # nnU-Net's own planning step copies dataset.json; a plan transfer does not.
    target = dataset_name('expanded')
    shutil.copyfile(Path(raw)/target/'dataset.json', Path(preprocessed)/target/'dataset.json')


def _synthetic_cases(rng, folder: Path, count: int, size: int) -> None:
    (folder/'imagesTr').mkdir(parents=True)
    (folder/'labelsTr').mkdir()
    for index in range(count):
        target = np.zeros((size,) * 3, dtype=np.uint8)
        low = int(rng.integers(size // 4, size // 2))
        target[low:low + size // 4, low:low + size // 4, low:low + size // 4] = 1
        image = rng.normal(0, .1, target.shape) + target
        write_case(image, target, folder/'imagesTr'/f'{case_name(index)}_0000.nii.gz',
                   folder/'labelsTr'/f'{case_name(index)}.nii.gz')
    (folder/'dataset.json').write_text(json.dumps(dataset_json(count), indent=2) + '\n')


def synthetic_check(device_name: str = 'cpu', size: int = 32) -> dict:
    """Plan, preprocess, train each phase briefly, transfer weights and predict, on synthetic data only."""
    import torch
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
    raw, preprocessed = Path(os.environ['nnUNet_raw']), Path(os.environ['nnUNet_preprocessed'])
    rng = np.random.default_rng(0)
    for subset, count in (('old', 6), ('expanded', 8)):
        _synthetic_cases(rng, raw/dataset_name(subset), count, size)
    ids = [DATASETS['old'][0], DATASETS['expanded'][0]]
    workers = os.environ.get('nnUNet_n_proc_DA')
    os.environ['nnUNet_n_proc_DA'] = '1'  # the planner sets torch threads from this and rejects zero
    try:
        extract_fingerprints(ids, num_processes=1, check_dataset_integrity=True, verbose=False,
                             show_progress_bar=False)
        plan_experiments([ids[0]], experiment_planner_class_name=PLANNER)
        transfer_plans(raw, preprocessed)
        preprocess(ids, plans_identifier=PLANS, configurations=(CONFIGURATION,), num_processes=(1,), verbose=False,
                   show_progress_bar=False)
    finally:
        if workers is None:
            os.environ.pop('nnUNet_n_proc_DA')
        else:
            os.environ['nnUNet_n_proc_DA'] = workers
    device = torch.device(device_name)
    old, receipt = train_phase('old', 1, preprocessed, device, seed=1, iterations=(2, 1))
    old_state = torch.load(old, map_location='cpu', weights_only=False)['network_weights']
    probe = make_trainer('new', 1, preprocessed, device)
    load_full_weights(probe, old)
    loaded = base_module(probe.network).state_dict()
    transferred = set(loaded) == set(old_state) and all(torch.equal(loaded[k].cpu(), old_state[k]) for k in old_state)
    heads = [k for k in old_state if 'seg_layers' in k]
    heads_ok = bool(heads) and all(torch.equal(loaded[k].cpu(), old_state[k]) for k in heads)
    epochs = {'old': 1}
    for phase in ('new', 'continued'):
        _, phase_receipt = train_phase(phase, 1, preprocessed, device, seed=1, parent=old, iterations=(2, 1))
        epochs[phase] = phase_receipt['epochs']
    predictor = make_predictor(old, device)
    image = rng.normal(0, .1, (size,) * 3).astype(np.float32)
    z = foreground_logit(predictor, image)
    logits = predictor.predict_sliding_window_return_logits(torch.from_numpy(image[None]))
    logits = logits.float().cpu() if isinstance(logits, torch.Tensor) else torch.from_numpy(np.asarray(logits))
    softmax = torch.softmax(logits, dim=0)[1].numpy()
    return dict(status='NNUNET_SYNTHETIC_CHECK_PASS', epoch_seconds_logged=len(receipt['epoch_seconds']),
                all_weights_transferred=bool(transferred),
                predictor_weights_loaded=weights_match(predictor.network, old_state),
                segmentation_head_transferred=heads_ok, logit_shape=list(z.shape),
                threshold_matches_softmax=bool(np.array_equal(z > 0, softmax > .5)), epochs=epochs,
                old_training_seconds=receipt['elapsed_seconds'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['synthetic-check'])
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        if not Path(os.environ[key]).resolve().is_relative_to(args.workdir.resolve()):
            raise GateError('nnU-Net folders must lie inside the check directory')
    print(json.dumps(synthetic_check(args.device)), flush=True)


if __name__ == '__main__':
    main()
