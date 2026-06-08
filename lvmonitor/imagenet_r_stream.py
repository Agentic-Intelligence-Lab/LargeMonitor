"""ImageNet-R 50-class continual stream (PyTorch loaders)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lvmonitor.imagenet_r_stream_config import (
    CLASS40,
    DEFAULT_PATHS,
    GROUPS,
    LABEL_TO_SYNSET,
    NUM_TASKS,
    SOURCE_CORR,
    SOURCE_R,
    SOURCE_SKETCH,
    SYNSET_TO_LABEL,
    TASK_SPECS,
    SampleEntry,
    TaskSpec,
    build_task_entries,
    class_mask_for_task,
    group_synsets,
    print_task_table,
    save_metadata,
    stream_metadata,
)

__all__ = [
    "CLASS40",
    "DEFAULT_PATHS",
    "GROUPS",
    "LABEL_TO_SYNSET",
    "NUM_TASKS",
    "SOURCE_CORR",
    "SOURCE_R",
    "SOURCE_SKETCH",
    "SYNSET_TO_LABEL",
    "TASK_SPECS",
    "SampleEntry",
    "StreamTaskDataset",
    "TaskSpec",
    "build_continual_loaders",
    "build_task_entries",
    "build_transform",
    "class_mask_for_task",
    "group_synsets",
    "print_task_table",
    "save_metadata",
    "stream_metadata",
]


class StreamTaskDataset(Dataset):
    """One continual task: images from a group + source, labels remapped 0..49."""

    def __init__(
        self,
        entries: list[SampleEntry],
        transform: transforms.Compose | None = None,
    ):
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


def build_transform(input_size: int = 224, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])


def build_continual_loaders(
    *,
    source_paths: dict[str, Path] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    input_size: int = 224,
    pin_memory: bool = True,
) -> tuple[list[dict[str, DataLoader]], list[list[int]]]:
    """Same structure as Disjoint build_continual_dataloader return value."""
    dataloaders: list[dict[str, DataLoader]] = []
    masks: list[list[int]] = []

    for spec in TASK_SPECS:
        split_entries = build_task_entries(spec, source_paths=source_paths)
        mask = class_mask_for_task(spec)
        masks.append(mask)

        loaders: dict[str, DataLoader] = {}
        for split, train_flag in (("train", True), ("test", False)):
            ds = StreamTaskDataset(
                split_entries[split],
                transform=build_transform(input_size, train=train_flag),
            )
            sampler = torch.utils.data.RandomSampler(ds) if train_flag else None
            loaders[split] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        dataloaders.append(loaders)

    return dataloaders, masks


if __name__ == "__main__":
    print_task_table()
    for spec in TASK_SPECS:
        n_train = len(build_task_entries(spec, splits=("train",))["train"])
        n_test = len(build_task_entries(spec, splits=("test",))["test"])
        print(f"  T{spec.task_id}: train={n_train} test={n_test}")
