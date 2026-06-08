"""ImageNet-R continual stream config (re-exported from largemonitor.imagenet_cu)."""

from __future__ import annotations

import json
from pathlib import Path

from largemonitor.imagenet_cu import (
    CLASS40,
    DEFAULT_PATHS,
    GROUPS,
    IMAGE_SUFFIXES,
    LABEL_TO_SYNSET,
    NUM_CLASSES,
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
    collect_split_entries,
    group_synsets,
    print_task_table,
    stream_metadata,
)

__all__ = [
    "CLASS40",
    "DEFAULT_PATHS",
    "GROUPS",
    "IMAGE_SUFFIXES",
    "LABEL_TO_SYNSET",
    "NUM_CLASSES",
    "NUM_TASKS",
    "SOURCE_CORR",
    "SOURCE_R",
    "SOURCE_SKETCH",
    "SYNSET_TO_LABEL",
    "TASK_SPECS",
    "SampleEntry",
    "TaskSpec",
    "build_task_entries",
    "class_mask_for_task",
    "collect_split_entries",
    "group_synsets",
    "print_task_table",
    "save_metadata",
    "stream_metadata",
]


def save_metadata(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stream_metadata(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    print_task_table()
    for spec in TASK_SPECS:
        n_train = len(build_task_entries(spec, splits=("train",))["train"])
        n_test = len(build_task_entries(spec, splits=("test",))["test"])
        print(f"  T{spec.task_id}: train={n_train} test={n_test}")
