"""DINO feature-buffer linear CKA monitoring on Disjoint Split-* tasks (online_lora splits)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lvmonitor.disjoint_dino import (
    DATASET_PRESETS,
    DATA_PATH,
    DEFAULT_MODEL_TAG,
    extract_features_batch,
    load_dino_model,
    load_split_train_tasks,
    model_path,
    task_label,
)


def compute_linear_cka(batch_feat: torch.Tensor, buffer_feat: torch.Tensor | None, eps: float = 1e-12) -> float:
    """Linear CKA between batch and buffer representation matrices (rows = samples).

    Uses ||X_c Y_c^T||_F^2 = <X_c^T X_c, Y_c^T Y_c>_F so batch and buffer can differ in size.
    """
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
    drop_last: bool = False,
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
        sampler = torch.utils.data.RandomSampler(tasks[task_id])
        loader = DataLoader(tasks[task_id], batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        print(f"\n=== {task_names[task_id]} ({len(tasks[task_id])} images) ===")

        for batch_idx, (images, _) in enumerate(loader):
            batch_feats = extract_features_batch(
                model, processor, images, input_device, feat_device
            )
            cka = compute_linear_cka(batch_feats, feature_buffer)
            print(f"Batch {batch_idx:3d} | Linear CKA = {cka:.4f}")
            records.append({
                "dataset": preset,
                "task_id": task_id,
                "task": task_names[task_id],
                "batch_idx": batch_idx,
                "linear_cka": cka,
            })

            if feature_buffer is None:
                feature_buffer = batch_feats
            else:
                feature_buffer = torch.cat([feature_buffer, batch_feats], dim=0)
            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]

    if output_path is None:
        output_path = (
            f"{preset}_dino_{model_tag}_cka_buffer_{buffer_size}_num_tasks_{n_tasks}.csv"
        )
    out_path = Path(output_path)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINO feature-buffer linear CKA on Disjoint Split-* (online_lora splits)."
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
    parser.add_argument("--drop-last", action="store_true", help="Drop last batch.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_dino_cka_monitor(
        args.dataset,
        num_tasks=args.num_tasks,
        shuffle=args.shuffle,
        batch_size=args.batch_size,
        model_tag=args.model_tag,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        output_path=args.output,
        multi_gpu=not args.single_gpu,
        drop_last=args.drop_last,
    )


if __name__ == "__main__":
    main()
