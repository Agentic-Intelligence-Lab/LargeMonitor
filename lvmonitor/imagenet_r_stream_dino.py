"""Feature-buffer cosine monitoring on ImageNet-R TASK_SPECS stream (DINO or random conv).

Mirrors lvmonitor/disjoint_dino.py but loads tasks from imagenet_r_stream_config
/ imagenet_r_stream instead of Disjoint Split-* presets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lvmonitor.disjoint_dino import (
    DEFAULT_MODEL_TAG,
    compute_mean_cosine,
    extract_features_batch,
    extract_features_conv_batch,
    load_dino_model,
    load_random_conv_encoder,
    model_path,
)
from lvmonitor.imagenet_r_stream import build_continual_loaders
from lvmonitor.imagenet_r_stream_config import (
    DEFAULT_PATHS,
    NUM_TASKS,
    TASK_SPECS,
    TaskSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"

STREAM_DEFAULTS = {
    "input_size": 224,
    "batch_size": 64,
    "num_tasks": NUM_TASKS,
}


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


def load_stream_train_loaders(
    *,
    source_paths: dict[str, Path] | None = None,
    batch_size: int = STREAM_DEFAULTS["batch_size"],
    num_workers: int = 4,
    input_size: int = STREAM_DEFAULTS["input_size"],
    task_ids: list[int] | None = None,
    num_tasks: int | None = None,
    pin_memory: bool = True,
) -> tuple[list[tuple[TaskSpec, DataLoader]], list[list[int]]]:
    """Return (spec, train_loader) pairs aligned with TASK_SPECS order."""
    specs = _select_specs(task_ids=task_ids, num_tasks=num_tasks)
    if not specs:
        raise ValueError("No tasks selected")

    all_loaders, all_masks = build_continual_loaders(
        source_paths=source_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        pin_memory=pin_memory,
    )
    id_to_idx = {s.task_id: i for i, s in enumerate(TASK_SPECS)}
    pairs: list[tuple[TaskSpec, DataLoader]] = []
    masks: list[list[int]] = []
    for spec in specs:
        idx = id_to_idx[spec.task_id]
        pairs.append((spec, all_loaders[idx]["train"]))
        masks.append(all_masks[idx])
    return pairs, masks


def _default_output_name(
    encoder: str,
    *,
    model_tag: str,
    buffer_size: int,
    n_tasks: int,
    batch_size: int,
    seed: int | None,
) -> str:
    if encoder == "random-conv":
        seed_tag = f"seed_{seed}" if seed is not None else "seed_none"
        return (
            f"imagenet_r_stream_random_conv_{seed_tag}_"
            f"buf_{buffer_size}_tasks_{n_tasks}_bs_{batch_size}.csv"
        )
    return (
        f"imagenet_r_stream_dino_{model_tag}_"
        f"buf_{buffer_size}_tasks_{n_tasks}_bs_{batch_size}.csv"
    )


def run_dino_monitor(
    *,
    source_paths: dict[str, Path] | None = None,
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
        raise RuntimeError("CUDA is required for DINO monitoring")

    batch_size = batch_size or STREAM_DEFAULTS["batch_size"]
    pairs, _ = load_stream_train_loaders(
        source_paths=source_paths,
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
        f"stream=ImageNet-R TASK_SPECS tasks={n_tasks} "
        f"model={path.name} GPUs={n_gpu} input={input_device} buffer={feat_device} "
        f"batch_size={batch_size} buffer_size={buffer_size}"
    )

    feature_buffer: torch.Tensor | None = None
    records: list[dict] = []

    for spec, loader in pairs:
        name = stream_task_label(spec)
        n_batches = len(loader)
        print(f"\n=== {spec.description} | {name} | {len(loader.dataset)} train images ===")

        for batch_idx, (images, _labels) in enumerate(loader):
            batch_feats = extract_features_batch(
                model, processor, images, input_device, feat_device
            )
            mean_sim = compute_mean_cosine(batch_feats, feature_buffer)
            print(f"  Batch {batch_idx:3d}/{n_batches - 1} | Mean Cosine Sim = {mean_sim:.4f}")
            records.append({
                "task_id": spec.task_id,
                "group": spec.group,
                "source": spec.source,
                "change_type": spec.change_type,
                "description": spec.description,
                "task": name,
                "batch_idx": batch_idx,
                "mean_cosine_sim": mean_sim,
                "encoder": "dino",
                "model_tag": model_tag,
            })

            if feature_buffer is None:
                feature_buffer = batch_feats
            else:
                feature_buffer = torch.cat([feature_buffer, batch_feats], dim=0)
            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / _default_output_name(
            "dino",
            model_tag=model_tag,
            buffer_size=buffer_size,
            n_tasks=n_tasks,
            batch_size=batch_size,
            seed=None,
        )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {len(records)} rows to {out_path}")
    return out_path


def run_random_conv_monitor(
    *,
    source_paths: dict[str, Path] | None = None,
    task_ids: list[int] | None = None,
    num_tasks: int | None = None,
    batch_size: int | None = None,
    buffer_size: int = 10000,
    num_workers: int = 4,
    input_size: int = STREAM_DEFAULTS["input_size"],
    output_path: str | Path | None = None,
    seed: int | None = None,
) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    batch_size = batch_size or STREAM_DEFAULTS["batch_size"]
    pairs, _ = load_stream_train_loaders(
        source_paths=source_paths,
        batch_size=batch_size,
        num_workers=num_workers,
        input_size=input_size,
        task_ids=task_ids,
        num_tasks=num_tasks,
    )
    n_tasks = len(pairs)

    encoder, device = load_random_conv_encoder(seed=seed)
    print(
        f"stream=ImageNet-R TASK_SPECS tasks={n_tasks} "
        f"encoder=RandomConvEncoder feat_dim={encoder.conv2.out_channels} "
        f"device={device} batch_size={batch_size} buffer_size={buffer_size}"
    )

    feature_buffer: torch.Tensor | None = None
    records: list[dict] = []

    for spec, loader in pairs:
        name = stream_task_label(spec)
        n_batches = len(loader)
        print(
            f"\n=== {spec.description} | {name} | {len(loader.dataset)} train images ==="
        )

        for batch_idx, (images, _labels) in enumerate(loader):
            batch_feats = extract_features_conv_batch(encoder, images, device)
            mean_sim = compute_mean_cosine(batch_feats, feature_buffer)
            print(f"  Batch {batch_idx:3d}/{n_batches - 1} | Mean Cosine Sim = {mean_sim:.4f}")
            records.append({
                "task_id": spec.task_id,
                "group": spec.group,
                "source": spec.source,
                "change_type": spec.change_type,
                "description": spec.description,
                "task": name,
                "batch_idx": batch_idx,
                "mean_cosine_sim": mean_sim,
                "encoder": "random-conv",
                "model_tag": "",
            })

            if feature_buffer is None:
                feature_buffer = batch_feats
            else:
                feature_buffer = torch.cat([feature_buffer, batch_feats], dim=0)
            if len(feature_buffer) > buffer_size:
                feature_buffer = feature_buffer[-buffer_size:]

    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / _default_output_name(
            "random-conv",
            model_tag="",
            buffer_size=buffer_size,
            n_tasks=n_tasks,
            batch_size=batch_size,
            seed=seed,
        )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved {len(records)} rows to {out_path}")
    return out_path


def parse_source_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = dict(DEFAULT_PATHS)
    if args.r_root:
        paths["r"] = Path(args.r_root).resolve()
    if args.sketch_root:
        paths["sketch"] = Path(args.sketch_root).resolve()
    if args.corr_root:
        paths["corr"] = Path(args.corr_root).resolve()
    return paths


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "DINO / random-conv feature-buffer cosine on ImageNet-R TASK_SPECS "
            "(T1–T10 ImageNet-CU stream from largemonitor.imagenet_cu)."
        )
    )
    p.add_argument(
        "--encoder",
        choices=("dino", "random-conv"),
        default="dino",
    )
    p.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task ids (default: all TASK_SPECS).",
    )
    p.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Use first N tasks in stream order (after --tasks filter).",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=10000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-size", type=int, default=STREAM_DEFAULTS["input_size"])
    p.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--single-gpu", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="Random-conv weight init seed.")
    p.add_argument("--r-root", type=Path, default=None)
    p.add_argument("--sketch-root", type=Path, default=None)
    p.add_argument("--corr-root", type=Path, default=None)
    return p.parse_args(argv)


def _parse_task_ids(s: str | None) -> list[int] | None:
    if not s:
        return None
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_paths = parse_source_paths(args)
    task_ids = _parse_task_ids(args.tasks)

    common = dict(
        source_paths=source_paths,
        task_ids=task_ids,
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        output_path=args.output,
    )

    if args.encoder == "random-conv":
        run_random_conv_monitor(**common, seed=args.seed)
        return
    run_dino_monitor(**common, model_tag=args.model_tag, multi_gpu=not args.single_gpu)


if __name__ == "__main__":
    main()
