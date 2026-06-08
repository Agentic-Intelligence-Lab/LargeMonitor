"""CORe50 linear CKA with EUPE features (torch.hub local ViT)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from lvmonitor.disjoint_dino import DATA_PATH
from lvmonitor.disjoint_dino_cka import compute_linear_cka
from lvmonitor.disjoint_dino_cka_core50 import load_core50_train_tasks

DEFAULT_BATCH_SIZE = 256
DEFAULT_IMG_SIZE = 256


def build_eupe_transform(resize_size: int = DEFAULT_IMG_SIZE) -> v2.Compose:
    """Match EUPE README: ImageNet norm, square resize."""
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def _state_dict_from_checkpoint(path: Path, checkpoint_key: str | None) -> dict:
    """Load ``.pth`` / ``.pt`` from disk; optional nested key (e.g. ``model`` or ``state_dict``)."""
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")

    if isinstance(blob, dict):
        if checkpoint_key:
            if checkpoint_key not in blob:
                raise KeyError(
                    f"Checkpoint key {checkpoint_key!r} not in {list(blob.keys())[:20]}..."
                )
            inner = blob[checkpoint_key]
        elif "state_dict" in blob:
            inner = blob["state_dict"]
        elif "model" in blob and isinstance(blob["model"], dict):
            inner = blob["model"]
        elif "model_state_dict" in blob:
            inner = blob["model_state_dict"]
        else:
            # whole dict might be state_dict
            inner = blob
        state = inner if isinstance(inner, dict) else inner
    else:
        state = blob

    if not isinstance(state, dict):
        raise TypeError(f"Expected state dict from {path}, got {type(state)}")

    # strip DDP prefix
    if state and all(isinstance(k, str) for k in state):
        sample = next(iter(state))
        if sample.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _load_weights_into_model(
    model: torch.nn.Module,
    weights_path: Path,
    checkpoint_key: str | None,
) -> None:
    state = _state_dict_from_checkpoint(weights_path, checkpoint_key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"load_state_dict: missing {len(missing)} keys (e.g. {list(missing)[:3]})")
    if unexpected:
        print(f"load_state_dict: unexpected {len(unexpected)} keys (e.g. {list(unexpected)[:3]})")


def load_eupe_model(
    repo_dir: Path | str,
    weights: Path | str,
    hub_entrypoint: str = "eupe_vitb16",
    device: torch.device | None = None,
    checkpoint_key: str | None = None,
) -> torch.nn.Module:
    """Load EUPE: local torch.hub repo + **local weights file** on disk.

    Tries ``torch.hub.load(..., weights=...)`` first (EUPE README). If the hub entrypoint
    uses another keyword or no keyword, falls back to building the model then
    ``load_state_dict`` from the checkpoint.
    """
    repo_dir = Path(repo_dir).expanduser().resolve()
    weights = Path(weights).expanduser().resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"EUPE repo not found: {repo_dir}")
    if not weights.is_file():
        raise FileNotFoundError(
            f"Local weights file not found: {weights}\n"
            "Pass an absolute path to your .pth/.pt checkpoint via --eupe-weights."
        )

    dev = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise RuntimeError("EUPE script expects CUDA (matches reference autocast usage).")

    hub_kw_variants = (
        {"weights": str(weights)},
        {"checkpoint": str(weights)},
        {"checkpoint_path": str(weights)},
        {"pretrained_path": str(weights)},
    )
    model: torch.nn.Module | None = None

    for kw in hub_kw_variants:
        try:
            model = torch.hub.load(
                str(repo_dir),
                hub_entrypoint,
                source="local",
                trust_repo=True,
                **kw,
            )
            break
        except TypeError:
            continue

    if model is None:
        try:
            model = torch.hub.load(
                str(repo_dir),
                hub_entrypoint,
                source="local",
                trust_repo=True,
            )
        except TypeError:
            model = torch.hub.load(str(repo_dir), hub_entrypoint, source="local")
        _load_weights_into_model(model, weights, checkpoint_key)

    model = model.to(dev)
    model.eval()
    return model


@torch.inference_mode()
def extract_eupe_cls_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: torch.device,
    *,
    use_bf16: bool = True,
) -> torch.Tensor:
    """``images``: normalized batch [B,3,H,W] on CPU. Returns float32 [B, D] cls token."""
    x = images.to(device, non_blocking=True)
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        out = model.forward_features(x)
    feat = out["x_norm_clstoken"]
    return feat.float()


def run_core50_eupe_cka_monitor(
    *,
    core50_root: Path | None = None,
    eupe_repo: Path | str | None = None,
    eupe_weights: Path | str | None = None,
    hub_entrypoint: str = "eupe_vitb16",
    img_size: int = DEFAULT_IMG_SIZE,
    mode: str = "per_session",
    batch_size: int = DEFAULT_BATCH_SIZE,
    buffer_size: int = 10000,
    num_workers: int = 4,
    output_path: str | Path | None = None,
    use_bf16: bool = True,
    checkpoint_key: str | None = None,
) -> Path:
    if eupe_repo is None or eupe_weights is None:
        raise ValueError(
            "Pass --eupe-repo (local clone) and --eupe-weights (path to your local .pth/.pt)."
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    root = core50_root if core50_root is not None else DATA_PATH / "core50_128x128"
    transform = build_eupe_transform(img_size)
    tasks, task_names = load_core50_train_tasks(root, mode=mode, transform=transform)
    n_tasks = len(tasks)

    device = torch.device("cuda:0")
    model = load_eupe_model(
        eupe_repo,
        eupe_weights,
        hub_entrypoint=hub_entrypoint,
        device=device,
        checkpoint_key=checkpoint_key,
    )

    wpath = Path(eupe_weights).expanduser().resolve()
    print(
        f"core50_root={root} mode={mode} tasks={n_tasks} "
        f"eupe={hub_entrypoint} local_weights={wpath} img_size={img_size} "
        f"device={device} bf16={use_bf16}"
    )
    print(f"Train images (task 0): {len(tasks[0])}")

    feature_buffer = None
    records = []

    for task_id in range(n_tasks):
        loader = DataLoader(
            tasks[task_id],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"\n=== {task_names[task_id]} ===")

        for batch_idx, (images, _) in enumerate(loader):
            batch_feats = extract_eupe_cls_batch(
                model, images, device, use_bf16=use_bf16
            )
            cka = compute_linear_cka(batch_feats, feature_buffer)
            print(f"Batch {batch_idx:3d} | Linear CKA = {cka:.4f}")
            records.append({
                "dataset": "core50",
                "backbone": "eupe",
                "hub": hub_entrypoint,
                "img_size": img_size,
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
            f"core50_eupe_{hub_entrypoint}_cka_{mode}_im{img_size}_"
            f"buffer_{buffer_size}_tasks_{n_tasks}.csv"
        )
    out_path = Path(output_path)
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EUPE feature-buffer linear CKA on CORe50 (same layout as disjoint_dino_cka_core50)."
    )
    parser.add_argument(
        "--core50-root",
        type=str,
        default=None,
        help=f"core50_128x128 dir (default: {DATA_PATH}/core50_128x128).",
    )
    parser.add_argument(
        "--eupe-repo",
        type=str,
        required=True,
        help="Path to local EUPE repo clone (passed to torch.hub.load).",
    )
    parser.add_argument(
        "--eupe-weights",
        type=str,
        required=True,
        help="Local checkpoint path (.pth / .pt), absolute or relative. Never downloads from the web.",
    )
    parser.add_argument(
        "--checkpoint-key",
        type=str,
        default=None,
        help=(
            "If the file is a dict, load this key (e.g. model, state_dict). "
            "Default: auto (state_dict, model, model_state_dict, or whole dict)."
        ),
    )
    parser.add_argument(
        "--hub-entrypoint",
        type=str,
        default="eupe_vitb16",
        help="torch.hub entrypoint name inside the EUPE repo.",
    )
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument(
        "--mode",
        choices=("per_session", "concat"),
        default="per_session",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 autocast (use float32 forward).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.core50_root) if args.core50_root else DATA_PATH / "core50_128x128"
    run_core50_eupe_cka_monitor(
        core50_root=root,
        eupe_repo=args.eupe_repo,
        eupe_weights=args.eupe_weights,
        hub_entrypoint=args.hub_entrypoint,
        img_size=args.img_size,
        mode=args.mode,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        num_workers=args.num_workers,
        output_path=args.output,
        use_bf16=not args.no_bf16,
        checkpoint_key=args.checkpoint_key,
    )


if __name__ == "__main__":
    main()
