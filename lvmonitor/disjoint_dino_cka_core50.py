"""DINO feature-buffer linear CKA for CORe50.

Supports Disjoint layout ``train/<session>/`` or flat zip layout ``<session>/`` under core50_128x128.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms

from lvmonitor.disjoint_dino import (
    DATA_PATH,
    DEFAULT_MODEL_TAG,
    extract_features_batch,
    load_dino_model,
    model_path,
)
from lvmonitor.disjoint_dino_cka import compute_linear_cka

# Same session list as Disjoint/datasets.py split_core50_datasets().
TRAIN_SESSIONS = ("s1", "s2", "s4", "s5", "s6", "s8", "s9", "s11")

DEFAULT_BATCH_SIZE = 512


def build_core50_transform() -> transforms.Compose:
    """Match Disjoint split_core50_datasets: Resize(224) + ToTensor."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def _session_parent_dir(core50_root: Path) -> Path:
    """Disjoint preprocessed: ``core50_128x128/train/<session>``.
    Raw zip extract: ``core50_128x128/<session>`` (sessions at dataset root).
    """
    core50_root = core50_root.resolve()
    train_sub = core50_root / "train"
    first = TRAIN_SESSIONS[0]
    if (train_sub / first).is_dir():
        return train_sub
    if (core50_root / first).is_dir():
        return core50_root
    raise FileNotFoundError(
        f"No CORe50 session folders found under {core50_root}. Expected either\n"
        f"  - Disjoint layout: {train_sub}/{first}/\n"
        f"  - Flat zip layout: {core50_root}/{first}/\n"
        f"Pass --core50-root pointing at the core50_128x128 directory."
    )


def load_core50_train_tasks(
    core50_root: Path,
    *,
    mode: str = "per_session",
    transform: Callable[[Any], Any] | None = None,
) -> tuple[list, list[str]]:
    """Load CORe50 train sessions.

    ``core50_root`` is the ``core50_128x128`` directory. Supports:

    - **Disjoint layout**: ``train/s1``, ``train/s2``, … (after ``CORe50.split()``).
    - **Flat layout**: ``s1``, ``s2``, … directly under ``core50_128x128`` (plain zip extract).

    Args:
        core50_root: Path to ``.../core50_128x128``.
        mode:
            - ``per_session``: one dataset per training session (8 tasks), usual for CKA curves.
            - ``concat``: single ``ConcatDataset`` over all sessions (matches Disjoint train concat).
        transform: Optional torchvision transform (default: Resize(224)+ToTensor for DINO pipeline).
    """
    core50_root = core50_root.resolve()
    session_root = _session_parent_dir(core50_root)
    layout = "disjoint train/" if session_root.name == "train" else "flat"
    print(f"CORe50 layout: {layout} (sessions under {session_root})")

    if transform is None:
        transform = build_core50_transform()
    task_ds: list = []
    task_names: list[str] = []

    if mode == "per_session":
        for sess in TRAIN_SESSIONS:
            sess_path = session_root / sess
            if not sess_path.is_dir():
                raise FileNotFoundError(
                    f"Missing session folder {sess_path} (sessions expected under {session_root})."
                )
            ds = datasets.ImageFolder(root=str(sess_path), transform=transform)
            task_ds.append(ds)
            task_names.append(f"CORe50 train {sess} ({len(ds)} images, {len(ds.classes)} classes)")
    elif mode == "concat":
        train_sets = []
        for sess in TRAIN_SESSIONS:
            sess_path = session_root / sess
            if not sess_path.is_dir():
                raise FileNotFoundError(f"Missing session folder {sess_path}")
            train_sets.append(datasets.ImageFolder(root=str(sess_path), transform=transform))
        merged = ConcatDataset(train_sets)
        task_ds.append(merged)
        task_names.append(
            f"CORe50 train concat {TRAIN_SESSIONS} ({len(merged)} images)"
        )
    else:
        raise ValueError(f"mode must be 'per_session' or 'concat', got {mode!r}")

    return task_ds, task_names


def run_core50_cka_monitor(
    *,
    core50_root: Path | None = None,
    mode: str = "per_session",
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_tag: str = DEFAULT_MODEL_TAG,
    buffer_size: int = 10000,
    num_workers: int = 4,
    output_path: str | Path | None = None,
    multi_gpu: bool = True,
    drop_last: bool = False,
) -> Path:
    root = core50_root if core50_root is not None else DATA_PATH / "core50_128x128"
    path = model_path(model_tag)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tasks, task_names = load_core50_train_tasks(root, mode=mode)
    n_tasks = len(tasks)

    # Class names from first underlying dataset (all sessions share labels o1..o50).
    sample_ds = tasks[0]
    if isinstance(sample_ds, ConcatDataset):
        sample_ds = sample_ds.datasets[0]
    class_names = list(sample_ds.classes)

    model, processor, input_device, feat_device, n_gpu = load_dino_model(
        model_tag=model_tag, multi_gpu=multi_gpu
    )
    print(
        f"core50_root={root} mode={mode} tasks={n_tasks} "
        f"model={path.name} GPUs={n_gpu} input={input_device} buffer={feat_device}"
    )
    print(f"Train images (task 0): {len(tasks[0])}, num global classes: {len(class_names)}")

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
            drop_last=drop_last,
        )
        print(f"\n=== {task_names[task_id]} ===")

        for batch_idx, (images, _) in enumerate(loader):
            batch_feats = extract_features_batch(
                model, processor, images, input_device, feat_device
            )
            cka = compute_linear_cka(batch_feats, feature_buffer)
            print(f"Batch {batch_idx:3d} | Linear CKA = {cka:.4f}")
            records.append({
                "dataset": "core50",
                "mode": mode,
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
            f"core50_dino_{model_tag}_cka_{mode}_buffer_{buffer_size}_tasks_{n_tasks}.csv"
        )
    out_path = Path(output_path)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DINO buffer linear CKA on CORe50: Disjoint train/<session>/ or flat s1,s2,... at dataset root."
        )
    )
    parser.add_argument(
        "--core50-root",
        type=str,
        default=None,
        help=(
            f"Path to core50_128x128 (default: {DATA_PATH}/core50_128x128). "
            "Sessions may live under train/ or directly at root after zip extract."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("per_session", "concat"),
        default="per_session",
        help=(
            "per_session: 8 tasks (s1,s2,...); concat: one task, all sessions concatenated "
            "(matches Disjoint split_core50_datasets train side)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="DINOv3 tag, e.g. vitb16, vits16.",
    )
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--drop-last", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.core50_root) if args.core50_root else DATA_PATH / "core50_128x128"
    run_core50_cka_monitor(
        core50_root=root,
        mode=args.mode,
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
