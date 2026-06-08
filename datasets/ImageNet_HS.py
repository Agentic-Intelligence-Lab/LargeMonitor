"""ImageNet-HS: Tiny ImageNet / ImageNet-R / noisy-Tiny stream for MVP class-incremental learning.

Builds train & test pools from ``imagenet_hs`` task specs (50 shared classes, 10 stream steps).
Use with ``ImageNetHSDisjointSampler`` and ``--n 100 --m 0 --n_tasks 10`` (default stream:
T1–T10 from ``imagenet_hs``). Optional ``--n_tasks 5`` merges two stream steps per group.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from imagenet_hs import (
    CLASS40,
    GROUPS,
    NUM_CLASSES,
    SOURCE_CORR,
    SOURCE_INR,
    SOURCE_TINY,
    TASK_SPECS,
    build_task_entries,
)

GROUP_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "E")


def resolve_hs_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root).expanduser().resolve()
    return {
        SOURCE_TINY: root / "tiny-imagenet-200",
        SOURCE_INR: root / "imagenet-r",
        SOURCE_CORR: root / "tiny-imagenet-200-corr",
    }


def _check_paths(paths: dict[str, Path]) -> None:
    missing = [p for p in paths.values() if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "ImageNet-HS requires data directories under --data_dir:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + "\nRun: python build_tiny_imagenet_corr.py"
        )


class ImageNet_HS(Dataset):
    """All stream-step images with global labels 0..49 (synset order in ``CLASS40``)."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        del download  # offline only; use imagenet_hs / build_tiny_imagenet_corr to prepare data

        self.root = os.path.expanduser(root)
        self.path = self.root
        # Match Imagenet_R behavior: force a fixed image size before DataLoader collation.
        if transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize(256), transforms.RandomCrop(224)]
            )
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(256), transforms.RandomCrop(224), transform]
            )
        self.target_transform = target_transform
        self.train = train

        source_paths = resolve_hs_paths(self.root)
        _check_paths(source_paths)

        split = "train" if train else "test"
        self.samples: list[tuple[str, int]] = []
        self.stream_task_ids: list[int] = []

        for spec in TASK_SPECS:
            entries = build_task_entries(spec, source_paths=source_paths)[split]
            for entry in entries:
                self.samples.append((str(entry.path), entry.label))
                self.stream_task_ids.append(spec.task_id - 1)

        self.classes = list(range(NUM_CLASSES))
        # Keep ImageFolder-like shape for legacy trainers.
        self.class_to_idx = list(range(NUM_CLASSES))
        self.targets = [label for _, label in self.samples]
        self.imgs = self.samples
        self.class_names = list(CLASS40)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if hasattr(index, "item"):
            index = int(index.item())
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def indices_for_group(self, group: str) -> list[int]:
        labels = set(GROUPS[group])
        return [i for i, target in enumerate(self.targets) if target in labels]

    @staticmethod
    def disjoint_group_for_task(task_id: int) -> str:
        if task_id < 0 or task_id >= len(GROUP_ORDER):
            raise ValueError(f"task_id must be in [0, {len(GROUP_ORDER)}), got {task_id}")
        return GROUP_ORDER[task_id]
