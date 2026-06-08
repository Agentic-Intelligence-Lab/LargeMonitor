"""Export raw images per Disjoint task (get_dataset + split_single_dataset, no transforms)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
DISJOINT_ROOT = REPO_ROOT / "Disjoint"
DATA_PATH = REPO_ROOT / "data"

# batch_size aligned with lvmonitor/disjoint_dino.py (DataLoader, shuffle=False)
DATASET_PRESETS: dict[str, dict] = {
    "cifar100": {"split_name": "CIFAR100", "num_tasks": 10, "batch_size": 1024},
    "imagenetR": {"split_name": "Imagenet-R", "num_tasks": 10, "batch_size": 512},
    "sketch": {"split_name": "Sketch", "num_tasks": 10, "batch_size": 512},
    "cub200": {"split_name": "CUB200", "num_tasks": 10, "batch_size": 512},
}


def _ensure_disjoint_importable() -> None:
    disjoint = str(DISJOINT_ROOT)
    if disjoint not in sys.path:
        sys.path.insert(0, disjoint)


def load_disjoint_train_tasks(
    preset: str,
    *,
    data_path: Path,
    num_tasks: int | None = None,
    shuffle: bool = False,
) -> tuple[list[Dataset], list[str], list[list[int]]]:
    _ensure_disjoint_importable()
    from datasets import get_dataset, split_single_dataset

    if preset not in DATASET_PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from {list(DATASET_PRESETS)}")

    cfg = DATASET_PRESETS[preset]
    n_tasks = num_tasks if num_tasks is not None else cfg["num_tasks"]

    args = SimpleNamespace(
        data_path=str(data_path),
        num_tasks=n_tasks,
        shuffle=shuffle,
        input_size=224,
    )

    dataset_train, dataset_val = get_dataset(cfg["split_name"], None, None, args)
    split_datasets, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    tasks = [split_datasets[i][0] for i in range(n_tasks)]
    class_names = dataset_train.classes
    task_names = [
        _task_label(i, class_names, class_mask, n_tasks) for i in range(n_tasks)
    ]
    return tasks, task_names, class_mask


def _task_label(
    task_id: int,
    class_names: list[str],
    class_mask: list[list[int]],
    num_tasks: int,
) -> str:
    scope = class_mask[task_id]
    lo, hi = min(scope), max(scope)
    names = ", ".join(str(class_names[i]) for i in scope[:5])
    if len(scope) > 5:
        names += ", ..."
    return f"Task {task_id}: classes {lo}-{hi} ({names})"


def _resolve_index(dataset: Dataset, local_idx: int) -> tuple[Dataset, int]:
    ds = dataset
    idx = local_idx
    while isinstance(ds, Subset):
        idx = ds.indices[idx]
        ds = ds.dataset
    if isinstance(ds, ConcatDataset):
        raise NotImplementedError("ConcatDataset preview not supported (use Split-* presets).")
    return ds, idx


def _source_path(dataset: Dataset, local_idx: int) -> Path | None:
    """Filesystem path of the original file, if the dataset exposes it."""
    root, global_idx = _resolve_index(dataset, local_idx)

    if hasattr(root, "samples") and root.samples:
        return Path(root.samples[global_idx][0])

    if hasattr(root, "data") and hasattr(root.data, "samples"):
        return Path(root.data.samples[global_idx][0])

    return None


def _pil_from_sample(dataset: Dataset, local_idx: int) -> Image.Image:
    path = _source_path(dataset, local_idx)
    if path is not None:
        return Image.open(path)

    root, global_idx = _resolve_index(dataset, local_idx)
    img, _ = root[global_idx]
    return _to_pil(img)


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise TypeError(f"Unsupported image type: {type(img)}")


def _resize(img: Image.Image, size: int) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return img.resize((size, size), resample=resample)


def _unique_dest(dest_dir: Path, src: Path) -> Path:
    dest = dest_dir / src.name
    if not dest.exists():
        return dest
    stem, suffix = src.stem, src.suffix
    k = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_{k}{suffix}"
        k += 1
    return dest


def iter_batch_indices(dataset: Dataset, batch_size: int):
    """Yield local indices per batch (same order as DataLoader shuffle=False)."""
    n = len(dataset)
    for start in range(0, n, batch_size):
        yield list(range(start, min(start + batch_size, n)))


def nth_batch_indices(dataset: Dataset, batch_size: int, batch_idx: int) -> list[int]:
    """Local indices for one training batch (DataLoader shuffle=False order)."""
    for i, indices in enumerate(iter_batch_indices(dataset, batch_size)):
        if i == batch_idx:
            return indices
    n_batches = (len(dataset) + batch_size - 1) // batch_size if len(dataset) else 0
    raise IndexError(
        f"batch_idx={batch_idx} out of range (dataset len={len(dataset)}, "
        f"batch_size={batch_size}, n_batches={n_batches})"
    )


def sample_from_batch(
    task_ds: Dataset,
    batch_indices: list[int],
    n: int,
    rng: random.Random,
    *,
    image_size: int = 256,
) -> list[Image.Image]:
    """Sample up to n images from one training batch."""
    if not batch_indices:
        return []
    k = min(n, len(batch_indices))
    chosen = rng.sample(batch_indices, k)
    return [_resize(_pil_from_sample(task_ds, i), image_size) for i in chosen]


def sample_task_images(
    task_ds: Dataset,
    n: int,
    rng: random.Random,
    *,
    image_size: int = 256,
) -> list[Image.Image]:
    """Randomly sample up to n PIL images (resized) from entire task subset."""
    size = len(task_ds)
    if size == 0:
        return []
    k = min(n, size)
    indices = rng.sample(range(size), k)
    return [_resize(_pil_from_sample(task_ds, i), image_size) for i in indices]


def export_task_images(
    task_ds: Dataset,
    n: int,
    rng: random.Random,
    dest_dir: Path,
    *,
    image_size: int = 256,
    show: bool = False,
) -> list[Path]:
    """Write n images into dest_dir, each resized to image_size x image_size."""
    size = len(task_ds)
    if size == 0:
        return []

    dest_dir.mkdir(parents=True, exist_ok=True)
    k = min(n, size)
    indices = rng.sample(range(size), k)
    written: list[Path] = []

    for rank, local_idx in enumerate(indices):
        _, label = task_ds[local_idx]
        src = _source_path(task_ds, local_idx)
        img = _resize(_pil_from_sample(task_ds, local_idx), image_size)

        if src is not None:
            dest = _unique_dest(dest_dir, src.with_suffix(".png"))
        else:
            dest = dest_dir / f"idx{local_idx:05d}_label{label}.png"
            if dest.exists():
                dest = dest_dir / f"idx{local_idx:05d}_label{label}_{rank}.png"

        img.save(dest)
        src_note = str(src) if src is not None else "packed dataset"
        print(f"  [{rank}] {src_note} -> {dest}  ({image_size}x{image_size}, label={label})")
        written.append(dest)
        if show:
            Image.open(dest).show()

    return written


def run_preview(
    preset: str,
    *,
    data_path: Path,
    num_tasks: int | None,
    shuffle: bool,
    n_images: int,
    seed: int,
    output_dir: Path | None,
    image_size: int,
    show: bool,
) -> None:
    tasks, task_names, _ = load_disjoint_train_tasks(
        preset,
        data_path=data_path,
        num_tasks=num_tasks,
        shuffle=shuffle,
    )
    rng = random.Random(seed)

    print(
        f"preset={preset} data={data_path} tasks={len(tasks)} "
        f"n_per_task={n_images} seed={seed} size={image_size}"
    )
    if output_dir is None and not show:
        raise ValueError("Set --output-dir or pass --show to export/view images.")

    for task_id, (task_ds, name) in enumerate(zip(tasks, task_names)):
        print(f"\n=== {name} ({len(task_ds)} train images) ===")
        task_dir = None
        if output_dir is not None:
            task_dir = output_dir / preset / f"seed{seed}" / f"task{task_id:02d}"
        elif show:
            task_dir = Path("/tmp/disjoint_preview") / preset / f"task{task_id:02d}"
        export_task_images(
            task_ds, n_images, rng, task_dir, image_size=image_size, show=show
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample raw images per Disjoint task and save each file separately "
            "(one file per sample, resized; no montage or on-image labels)."
        )
    )
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS), required=True)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("-n", "--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Export square side length (default: 256).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/disjoint_preview",
        help="Root dir; writes {output_dir}/{preset}/seed{N}/task{TT}/<original filenames>.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open each saved image with the system viewer (PIL).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_path = Path(args.data_path) if args.data_path else DATA_PATH
    out_dir = Path(args.output_dir) if args.output_dir else None

    run_preview(
        args.dataset,
        data_path=data_path,
        num_tasks=args.num_tasks,
        shuffle=bool(args.shuffle),
        n_images=args.num_images,
        seed=args.seed,
        output_dir=out_dir,
        image_size=args.size,
        show=args.show,
    )


if __name__ == "__main__":
    main()
