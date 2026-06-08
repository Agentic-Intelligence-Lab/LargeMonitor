"""DINO feature-buffer linear CKA on Si-blurry blurry/disjoint task splits (OnlineSampler)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from lvmonitor.disjoint_dino import (
    DEFAULT_MODEL_TAG,
    extract_features_batch,
    load_dino_model,
    model_path,
)
from lvmonitor.disjoint_dino_cka import compute_linear_cka
from lvmonitor.sib_datasets import get_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SIB_ROOT = REPO_ROOT / "Si-blurry"
DATA_PATH = REPO_ROOT / "data"
DINO_IMAGE_SIZE = 224

# Mirrors Si-blurry main_blurry / blurry.sh dataset names and typical settings.
DATASET_PRESETS: dict[str, dict] = {
    "cifar10": {"num_tasks": 5, "input_size": 32, "batch_size": 1024},
    "cifar100": {"num_tasks": 10, "input_size": 32, "batch_size": 1024},
    "tinyimagenet": {"num_tasks": 10, "input_size": 224, "batch_size": 512},
    "imagenet-r": {"num_tasks": 5, "input_size": 224, "batch_size": 512},
    "cub200": {"num_tasks": 10, "input_size": 224, "batch_size": 512},
    "imagenet_sketch": {"num_tasks": 10, "input_size": 224, "batch_size": 512},
    "core50": {"num_tasks": 8, "input_size": 224, "batch_size": 512},
}

DEFAULT_N = 50
DEFAULT_M = 10
DEFAULT_RND_SEED = 1


def _ensure_sib_importable() -> None:
    """Si-blurry utils (OnlineSampler) only; datasets live in lvmonitor.sib_datasets."""
    sib = str(SIB_ROOT)
    if sib not in sys.path:
        sys.path.insert(0, sib)


def build_dino_transform(dataset_name: str) -> transforms.Compose:
    """Match disjoint_dino: 32px datasets use ToTensor; others resize to 224."""
    if DATASET_PRESETS[dataset_name]["input_size"] <= 32:
        return transforms.Compose([transforms.ToTensor()])
    return transforms.Compose([
        transforms.Resize((DINO_IMAGE_SIZE, DINO_IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def load_sib_train_tasks(
    dataset_name: str,
    data_dir: Path,
    *,
    num_tasks: int,
    n: int,
    m: int,
    rnd_seed: int,
    rnd_nm: bool = False,
):
    """Build per-task train subsets using Si-blurry OnlineSampler (n/m blurry split)."""
    _ensure_sib_importable()
    from utils.indexed_dataset import IndexedDataset
    from utils.online_sampler import OnlineSampler

    if dataset_name not in DATASET_PRESETS:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Choose from {list(DATASET_PRESETS)}"
        )

    dataset_cls, _mean, _std, _n_classes = get_dataset(dataset_name)
    transform = build_dino_transform(dataset_name)
    train_ds = dataset_cls(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    indexed = IndexedDataset(train_ds)
    sampler = OnlineSampler(
        indexed,
        num_tasks,
        m,
        n,
        rnd_seed,
        cur_iter=0,
        varing_NM=rnd_nm,
    )

    tasks = [Subset(train_ds, sampler.indices[i]) for i in range(num_tasks)]
    class_names = [_class_name(c) for c in train_ds.classes]
    return tasks, class_names, sampler


def _class_name(cls) -> str:
    return cls if isinstance(cls, str) else str(cls)


def task_label(
    task_id: int,
    class_names: list[str],
    sampler,
) -> str:
    disjoint = sampler.disjoint_classes[task_id]
    blurry = sampler.blurry_classes[task_id]
    d_names = ", ".join(class_names[c] for c in disjoint[:5])
    if len(disjoint) > 5:
        d_names += ", ..."
    b_names = ", ".join(class_names[c] for c in blurry[:5])
    if len(blurry) > 5:
        b_names += ", ..."
    return (
        f"Task {task_id}: disjoint {len(disjoint)} cls [{d_names}], "
        f"blurry {len(blurry)} cls [{b_names}]"
    )


def run_sib_dino_cka_monitor(
    dataset_name: str,
    *,
    data_dir: Path | None = None,
    num_tasks: int | None = None,
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    rnd_seed: int = DEFAULT_RND_SEED,
    rnd_nm: bool = False,
    batch_size: int | None = None,
    model_tag: str = DEFAULT_MODEL_TAG,
    buffer_size: int = 10000,
    num_workers: int = 4,
    output_path: str | Path | None = None,
    multi_gpu: bool = True,
    drop_last: bool = True,
) -> Path:
    cfg = DATASET_PRESETS[dataset_name]
    n_tasks = num_tasks if num_tasks is not None else cfg["num_tasks"]
    batch_size = batch_size if batch_size is not None else cfg["batch_size"]
    data_dir = data_dir if data_dir is not None else DATA_PATH
    path = model_path(model_tag)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    tasks, class_names, sampler = load_sib_train_tasks(
        dataset_name,
        data_dir,
        num_tasks=n_tasks,
        n=n,
        m=m,
        rnd_seed=rnd_seed,
        rnd_nm=rnd_nm,
    )
    task_names = [task_label(i, class_names, sampler) for i in range(n_tasks)]

    model, processor, input_device, feat_device, n_gpu = load_dino_model(
        model_tag=model_tag, multi_gpu=multi_gpu
    )
    print(
        f"dataset={dataset_name} data={data_dir} tasks={n_tasks} n={n} m={m} "
        f"seed={rnd_seed} rnd_nm={rnd_nm} model={path.name} GPUs={n_gpu} "
        f"input={input_device} buffer={feat_device} dino_size={DINO_IMAGE_SIZE}"
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
            drop_last=drop_last,
        )
        print(f"\n=== {task_names[task_id]} ({len(tasks[task_id])} images) ===")

        for batch_idx, batch in enumerate(loader):
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            batch_feats = extract_features_batch(
                model, processor, images, input_device, feat_device
            )
            cka = compute_linear_cka(batch_feats, feature_buffer)
            print(f"Batch {batch_idx:3d} | Linear CKA = {cka:.4f}")
            records.append({
                "dataset": dataset_name,
                "n": n,
                "m": m,
                "rnd_seed": rnd_seed,
                "rnd_nm": int(rnd_nm),
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
            f"{dataset_name}_sib_dino_{model_tag}_cka_n{n}_m{m}_seed{rnd_seed}"
            f"_buffer_{buffer_size}_tasks_{n_tasks}.csv"
        )
    out_path = Path(output_path)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINO buffer linear CKA with Si-blurry OnlineSampler (n/m blurry split)."
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_PRESETS),
        required=True,
        help="Si-blurry dataset name (e.g. cifar100, imagenet-r).",
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root path.")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help="Disjoint class percentage (100=fully disjoint per task).",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=DEFAULT_M,
        help="Blurry sample shuffle percentage within blurry classes.",
    )
    parser.add_argument("--rnd-seed", type=int, default=DEFAULT_RND_SEED)
    parser.add_argument(
        "--rnd-nm",
        action="store_true",
        help="Randomize N/M per task (Si-blurry rnd_NM).",
    )
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
    data_dir = Path(args.data_dir) if args.data_dir else DATA_PATH
    run_sib_dino_cka_monitor(
        args.dataset,
        data_dir=data_dir,
        num_tasks=args.num_tasks,
        n=args.n,
        m=args.m,
        rnd_seed=args.rnd_seed,
        rnd_nm=args.rnd_nm,
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
