"""DINO feature-buffer linear CKA on ImageNet-HS stream (T1–T10).

Loads train tasks from ``imagenet_hs`` / ``datasets.ImageNet_HS`` paths.
DINO utilities are imported from LVMonitor (``../LVMonitor`` on PYTHONPATH if needed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from imagenet_hs import (
    NUM_TASKS,
    TASK_SPECS,
    TaskSpec,
    build_task_entries,
)
from datasets.ImageNet_HS import resolve_hs_paths

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "cka"
LVMONITOR_ROOT = REPO_ROOT.parent / "LVMonitor"

STREAM_DEFAULTS = {
    "input_size": 224,
    "batch_size": 64,
    "num_tasks": NUM_TASKS,
}


def _ensure_lvmonitor() -> None:
    if str(LVMONITOR_ROOT) not in sys.path and LVMONITOR_ROOT.is_dir():
        sys.path.insert(0, str(LVMONITOR_ROOT))


_ensure_lvmonitor()
from lvmonitor.disjoint_dino import (  # noqa: E402
    DEFAULT_MODEL_TAG,
    extract_features_batch,
    load_dino_model,
    model_path,
)


class StreamTaskDataset(Dataset):
    """One HS stream step: images from a TaskSpec train split."""

    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        img = Image.open(entry.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, entry.label

    @property
    def targets(self) -> list[int]:
        return [e.label for e in self.entries]


def build_transform(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])


def stream_task_label(spec: TaskSpec) -> str:
    labs = ",".join(str(i) for i in spec.label_indices)
    return (
        f"T{spec.task_id} group={spec.group} labels=[{labs}] "
        f"source={spec.source} change={spec.change_type}"
    )


def _select_specs(
    *,
    task_ids: list[int] | None = None,
    num_tasks: int | None = None,
) -> list[TaskSpec]:
    specs = list(TASK_SPECS)
    if task_ids is not None:
        wanted = set(task_ids)
        specs = [s for s in specs if s.task_id in wanted]
    if num_tasks is not None:
        specs = specs[:num_tasks]
    return specs


def load_hs_train_loaders(
    data_dir: str | Path,
    *,
    batch_size: int = STREAM_DEFAULTS["batch_size"],
    num_workers: int = 4,
    input_size: int = STREAM_DEFAULTS["input_size"],
    task_ids: list[int] | None = None,
    num_tasks: int | None = None,
    pin_memory: bool = True,
) -> list[tuple[TaskSpec, DataLoader]]:
    source_paths = resolve_hs_paths(data_dir)
    transform = build_transform(input_size)
    specs = _select_specs(task_ids=task_ids, num_tasks=num_tasks)
    if not specs:
        raise ValueError("No HS tasks selected")

    pairs: list[tuple[TaskSpec, DataLoader]] = []
    for spec in specs:
        entries = build_task_entries(spec, source_paths=source_paths)["train"]
        ds = StreamTaskDataset(entries, transform=transform)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=torch.utils.data.RandomSampler(ds),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        pairs.append((spec, loader))
    return pairs


def compute_linear_cka(
    batch_feat: torch.Tensor, buffer_feat: torch.Tensor | None, eps: float = 1e-12
) -> float:
    """Linear CKA between batch and buffer representation matrices (rows = samples)."""
    if buffer_feat is None or len(buffer_feat) == 0:
        return 0.0

    x = batch_feat.float()
    y = buffer_feat.float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    x_gram = x.T @ x
    y_gram = y.T @ y
    cross = (x_gram * y_gram).sum()
    norm_x = x_gram.pow(2).sum().sqrt()
    norm_y = y_gram.pow(2).sum().sqrt()
    return (cross / (norm_x * norm_y + eps)).clamp(max=1.0).item()


def run_dino_cka_monitor(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    task_ids: list[int] | None = None,
    num_tasks: int | None = None,
    batch_size: int | None = None,
    model_tag: str = DEFAULT_MODEL_TAG,
    buffer_size: int = 10000,
    num_workers: int = 4,
    input_size: int = STREAM_DEFAULTS["input_size"],
    output_path: str | Path | None = None,
    multi_gpu: bool = True,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    batch_size = batch_size or STREAM_DEFAULTS["batch_size"]
    pairs = load_hs_train_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )
    n_tasks = len(pairs)
    path = model_path(model_tag)

    model, processor, input_device, feat_device, n_gpu = load_dino_model(
        model_tag=model_tag, multi_gpu=multi_gpu
    )
    print(
        f"dataset=imagenet-hs data={Path(data_dir).resolve()} tasks={n_tasks} "
        f"model={path.name} GPUs={n_gpu} input={input_device} buffer={feat_device} "
        f"batch_size={batch_size} buffer_size={buffer_size}"
    )

    feature_buffer: torch.Tensor | None = None
    records: list[dict] = []

    for spec, loader in pairs:
        name = stream_task_label(spec)
        n_batches = len(loader)
        print(
            f"\n=== {spec.description} | {name} | "
            f"{len(loader.dataset)} train images ==="
        )

        for batch_idx, (images, _labels) in enumerate(loader):
            batch_feats = extract_features_batch(
                model, processor, images, input_device, feat_device
            )
            cka = compute_linear_cka(batch_feats, feature_buffer)
            print(f"  Batch {batch_idx:3d}/{max(n_batches - 1, 0)} | Linear CKA = {cka:.4f}")
            records.append({
                "dataset": "imagenet-hs",
                "task_id": spec.task_id,
                "group": spec.group,
                "source": spec.source,
                "change_type": spec.change_type,
                "description": spec.description,
                "task": name,
                "batch_idx": batch_idx,
                "linear_cka": cka,
                "model_tag": model_tag,
            })

            if feature_buffer is None:
                feature_buffer = batch_feats
            else:
                feature_buffer = torch.cat([feature_buffer, batch_feats], dim=0)
            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]

    if output_path is None:
        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"imagenet_hs_dino_{model_tag}_cka_buf_{buffer_size}_"
            f"tasks_{n_tasks}_bs_{batch_size}.csv"
        )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {len(records)} rows to {out_path}")
    return out_path


def _parse_task_ids(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINO feature-buffer linear CKA on ImageNet-HS T1–T10 stream."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root with tiny-imagenet-200, imagenet-r, tiny-imagenet-200-corr",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task ids, e.g. 1,2,3 (default: all TASK_SPECS)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Use first N tasks in stream order (default: all 10)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="DINOv3 tag, e.g. vitb16, vits16 (under ../Models/)",
    )
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=STREAM_DEFAULTS["input_size"])
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--single-gpu", action="store_true", help="Disable device_map=auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dino_cka_monitor(
        data_dir=args.data_dir,
        task_ids=_parse_task_ids(args.tasks),
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        model_tag=args.model_tag,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        output_path=args.output,
        multi_gpu=not args.single_gpu,
    )


if __name__ == "__main__":
    main()
