"""Feature-buffer cosine monitoring on Disjoint Split-* tasks (DINO or random conv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

REPO_ROOT = Path(__file__).resolve().parents[1]
DISJOINT_ROOT = REPO_ROOT / "Disjoint"
DATA_PATH = REPO_ROOT / "data"
MODEL_ROOT = REPO_ROOT / "../Models"
DEFAULT_MODEL_TAG = "vitb16"


def model_path(tag: str = DEFAULT_MODEL_TAG) -> Path:
    return MODEL_ROOT / f"dinov3-{tag}-pretrain-lvd1689m"

# Mirrors Disjoint online_lora configs (Split-* + split_single_dataset).
DATASET_PRESETS: dict[str, dict] = {
    "cifar100": {
        "split_name": "CIFAR100",
        "num_tasks": 10,
        "input_size": 32,
        "batch_size": 1024,
    },
    "imagenetR": {
        "split_name": "Imagenet-R",
        "num_tasks": 10,
        "input_size": 224,  # imagenetR_online_lora default
        "batch_size": 512,
        "online_lora_transform": False,
    },
    "sketch": {
        "split_name": "Sketch",
        "num_tasks": 10,
        "input_size": 224,
        "batch_size": 512,
    },
    "cub200": {
        "split_name": "CUB200",
        "num_tasks": 10,
        "input_size": 224,
        "batch_size": 512,
    },
}


def _ensure_disjoint_importable() -> None:
    disjoint = str(DISJOINT_ROOT)
    if disjoint not in sys.path:
        sys.path.insert(0, disjoint)


def build_dino_transform(preset: str) -> transforms.Compose:
    """Build image transform for DINO feature extraction."""
    cfg = DATASET_PRESETS[preset]
    input_size = cfg["input_size"]

    if cfg.get("online_lora_transform"):
        _ensure_disjoint_importable()
        from datasets import build_transform

        # Train split + train loader in engine.py use build_transform(True, args).
        args = SimpleNamespace(input_size=input_size)
        return build_transform(is_train=True, args=args)

    if input_size <= 32:
        return transforms.Compose([transforms.ToTensor()])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])


def load_split_train_tasks(
    preset: str,
    num_tasks: int | None = None,
    shuffle: bool = False,
):
    """Load train subsets per task using Disjoint get_dataset + split_single_dataset."""
    _ensure_disjoint_importable()
    from datasets import get_dataset, split_single_dataset

    if preset not in DATASET_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {list(DATASET_PRESETS)}")

    cfg = DATASET_PRESETS[preset]
    n_tasks = num_tasks if num_tasks is not None else cfg["num_tasks"]
    input_size = cfg["input_size"]
    transform = build_dino_transform(preset)

    args = SimpleNamespace(
        data_path=str(DATA_PATH),
        num_tasks=n_tasks,
        shuffle=shuffle,
        input_size=input_size,
    )

    dataset_train, dataset_val = get_dataset(
        cfg["split_name"], transform, transform, args
    )
    split_datasets, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    tasks = [split_datasets[i][0] for i in range(n_tasks)]
    class_names = dataset_train.classes
    return tasks, class_names, class_mask


def task_label(
    task_id: int,
    class_names: list[str],
    class_mask: list[list[int]],
    num_tasks: int,
) -> str:
    scope = class_mask[task_id]
    lo, hi = min(scope), max(scope)
    names = ", ".join(class_names[i] for i in scope)
    return f"Task {task_id}: classes {lo}-{hi} ({names})"


def _dino_input_size(processor) -> int:
    size = processor.size
    if isinstance(size, dict):
        return int(size.get("height", size.get("shortest_edge", 224)))
    if isinstance(size, (list, tuple)):
        return int(size[0])
    return int(size)


def _dino_norm_tensors(processor, device: torch.device, dtype: torch.dtype):
    cached = getattr(processor, "_norm_tensors", None)
    if cached is None or cached[0].device != device or cached[0].dtype != dtype:
        mean = torch.as_tensor(processor.image_mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.as_tensor(processor.image_std, device=device, dtype=dtype).view(1, 3, 1, 1)
        cached = (mean, std)
        processor._norm_tensors = cached
    return cached


def prepare_dino_pixel_values(
    images: torch.Tensor, processor, device: torch.device
) -> torch.Tensor:
    """Match AutoImageProcessor: bilinear resize to model size + ImageNet normalize."""
    x = images.to(device, non_blocking=True).float()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    target = _dino_input_size(processor)
    if x.shape[-2] != target or x.shape[-1] != target:
        x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)

    mean, std = _dino_norm_tensors(processor, x.device, x.dtype)
    return (x - mean) / std


class RandomConvEncoder(nn.Module):
    """Fixed random conv features (Kaiming-init weights, not trained)."""

    def __init__(self, in_channels: int = 3, hidden: int = 16, out_channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, 1)
        self.conv2 = nn.Conv2d(hidden, out_channels, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.flatten(1)


@torch.inference_mode()
def extract_features_batch(model, processor, images, input_device, feat_device):
    pixel_values = prepare_dino_pixel_values(images, processor, input_device)
    return model(pixel_values=pixel_values).last_hidden_state[:, 0].to(feat_device)


@torch.inference_mode()
def extract_features_conv_batch(encoder: RandomConvEncoder, images: torch.Tensor, device: torch.device):
    return encoder(images.to(device))


def load_random_conv_encoder(
    in_channels: int = 3,
    *,
    seed: int | None = None,
    device: torch.device | None = None,
) -> tuple[RandomConvEncoder, torch.device]:
    if seed is not None:
        torch.manual_seed(seed)
    dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = RandomConvEncoder(in_channels=in_channels).to(dev)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder, dev


def compute_mean_cosine(batch_feat, buffer_feat):
    if buffer_feat is None or len(buffer_feat) == 0:
        return 0.0
    batch_feat = F.normalize(batch_feat.float(), dim=-1)
    buffer_feat = F.normalize(buffer_feat.float(), dim=-1)
    return (batch_feat @ buffer_feat.T).mean().item()


def load_dino_model(model_tag: str = DEFAULT_MODEL_TAG, multi_gpu: bool = True):
    path = model_path(model_tag)
    processor = AutoImageProcessor.from_pretrained(str(path))
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if n_gpu > 1 and multi_gpu:
        model = AutoModel.from_pretrained(str(path), device_map="auto")
        input_device = torch.device(next(model.parameters()).device)
        feat_device = torch.device(f"cuda:{n_gpu - 1}")
        model.eval()
        return model, processor, input_device, feat_device, n_gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(str(path)).to(device)
    model.eval()
    return model, processor, device, device, max(n_gpu, 1)


def run_dino_monitor(
    preset: str,
    *,
    num_tasks: int | None = None,
    shuffle: bool = False,
    batch_size: int | None = None,
    model_tag: str = DEFAULT_MODEL_TAG,
    buffer_size: int = 10000,
    num_workers: int = 4,
    output_path: str | Path | None = None,
    multi_gpu: bool = True,
) -> Path:
    cfg = DATASET_PRESETS[preset]
    n_tasks = num_tasks if num_tasks is not None else cfg["num_tasks"]
    batch_size = batch_size if batch_size is not None else cfg["batch_size"]
    path = model_path(model_tag)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tasks, class_names, class_mask = load_split_train_tasks(
        preset, num_tasks=n_tasks, shuffle=shuffle
    )
    task_names = [
        task_label(i, class_names, class_mask, n_tasks) for i in range(n_tasks)
    ]

    model, processor, input_device, feat_device, n_gpu = load_dino_model(
        model_tag=model_tag, multi_gpu=multi_gpu
    )
    print(
        f"preset={preset} data={DATA_PATH} tasks={n_tasks} "
        f"model={path.name} GPUs={n_gpu} input={input_device} buffer={feat_device}"
    )
    print(f"Train images (task 0): {len(tasks[0])}, classes: {len(class_names)}")

    feature_buffer = None
    records = []

    for task_id in range(n_tasks):
        # create random sampler
        sampler = torch.utils.data.RandomSampler(tasks[task_id])
        loader = DataLoader(tasks[task_id], batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
        print(f"\n=== {task_names[task_id]} ({len(tasks[task_id])} images) ===")

        for batch_idx, (images, _) in enumerate(loader):
            batch_feats = extract_features_batch(
                model, processor, images, input_device, feat_device
            )
            mean_sim = compute_mean_cosine(batch_feats, feature_buffer)
            print(f"Batch {batch_idx:3d} | Mean Cosine Sim = {mean_sim:.4f}")
            records.append({
                "dataset": preset,
                "task_id": task_id,
                "task": task_names[task_id],
                "batch_idx": batch_idx,
                "mean_cosine_sim": mean_sim,
            })

            if feature_buffer is None:
                feature_buffer = batch_feats
            else:
                feature_buffer = torch.cat([feature_buffer, batch_feats], dim=0)
            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]

    if output_path is None:
        output_path = (
            f"{preset}_dino_{model_tag}_buffer_{buffer_size}_num_tasks_{n_tasks}_shuffle_{shuffle}.csv"
        )
    out_path = Path(output_path)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return out_path


def run_random_conv_monitor(
    preset: str,
    *,
    num_tasks: int | None = None,
    shuffle: bool = False,
    batch_size: int | None = None,
    buffer_size: int = 10000,
    num_workers: int = 4,
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> Path:
    cfg = DATASET_PRESETS[preset]
    n_tasks = num_tasks if num_tasks is not None else cfg["num_tasks"]
    batch_size = batch_size if batch_size is not None else cfg["batch_size"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tasks, class_names, class_mask = load_split_train_tasks(
        preset, num_tasks=n_tasks, shuffle=shuffle
    )
    task_names = [
        task_label(i, class_names, class_mask, n_tasks) for i in range(n_tasks)
    ]

    encoder, device = load_random_conv_encoder(seed=seed)
    print(
        f"preset={preset} data={DATA_PATH} tasks={n_tasks} "
        f"encoder=RandomConvEncoder feat_dim={encoder.conv2.out_channels} "
        f"device={device} seed={seed}"
    )
    print(f"Train images (task 0): {len(tasks[0])}, classes: {len(class_names)}")

    feature_buffer = None
    records = []

    for task_id in range(n_tasks):
        sampler = torch.utils.data.RandomSampler(tasks[task_id])
        loader = DataLoader(
            tasks[task_id],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"\n=== {task_names[task_id]} ({len(tasks[task_id])} images) ===")

        for batch_idx, (images, _) in enumerate(loader):
            batch_feats = extract_features_conv_batch(encoder, images, device)
            mean_sim = compute_mean_cosine(batch_feats, feature_buffer)
            print(f"Batch {batch_idx:3d} | Mean Cosine Sim = {mean_sim:.4f}")
            records.append({
                "dataset": preset,
                "task_id": task_id,
                "task": task_names[task_id],
                "batch_idx": batch_idx,
                "mean_cosine_sim": mean_sim,
            })

            if feature_buffer is None:
                feature_buffer = batch_feats
            else:
                feature_buffer = torch.cat([feature_buffer, batch_feats], dim=0)
            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]

    if output_path is None:
        seed_tag = f"seed_{seed}" if seed is not None else "seed_none"
        output_path = (
            f"{preset}_random_conv_{seed_tag}_buffer_{buffer_size}_"
            f"num_tasks_{n_tasks}_shuffle_{shuffle}.csv"
        )
    out_path = Path(output_path)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature-buffer cosine on Disjoint Split-* (DINO or random conv)."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PRESETS),
        help="Dataset preset (matches Disjoint online_lora configs).",
    )
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--shuffle", type=int, default=0, help="Shuffle class order per task.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="DINOv3 tag, e.g. vitb16, vits16.",
    )
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
    parser.add_argument("--single-gpu", action="store_true", help="Disable device_map=auto.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for random conv weight init (random-conv only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.model_tag == "random":
        run_random_conv_monitor(
            args.dataset,
            num_tasks=args.num_tasks,
            shuffle=args.shuffle,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            num_workers=args.num_workers,
            output_path=args.output,
            seed=args.seed,
        )
        return
    run_dino_monitor(
        args.dataset,
        num_tasks=args.num_tasks,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        model_tag=args.model_tag,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        output_path=args.output,
        multi_gpu=not args.single_gpu,
    )


if __name__ == "__main__":
    main()
