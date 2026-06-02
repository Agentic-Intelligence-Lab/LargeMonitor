"""Tiny ImageNet → ImageNet-R 50-class continual stream (config + index, no PyTorch).

Classes are the intersection of Tiny ImageNet-200 and ImageNet-R (50 shared WordNet
synsets). Source domain: Tiny ImageNet; domain shift: ImageNet-R; corruption: noisy Tiny.

Task sequence (10 steps, groups A–E): each group at most twice in a row —
``tiny`` (source), then ``inr`` (domain shift) or ``corr`` (corruption).

For DataLoaders see ``imagenet_hs_loaders``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"

# 50 synsets present in both Tiny ImageNet-200 and ImageNet-R.
CLASS40: tuple[str, ...] = (
    "n01443537",
    "n01770393",
    "n01774750",
    "n01784675",
    "n01855672",
    "n01882714",
    "n01910747",
    "n01944390",
    "n01983481",
    "n02056570",
    "n02085620",
    "n02094433",
    "n02099601",
    "n02099712",
    "n02106662",
    "n02113799",
    "n02123045",
    "n02129165",
    "n02165456",
    "n02190166",
    "n02206856",
    "n02226429",
    "n02233338",
    "n02236044",
    "n02268443",
    "n02279972",
    "n02364673",
    "n02395406",
    "n02410509",
    "n02423022",
    "n02480495",
    "n02481823",
    "n02486410",
    "n02769748",
    "n02793495",
    "n02802426",
    "n02808440",
    "n02814860",
    "n02841315",
    "n02843684",
    "n02883205",
    "n02906734",
    "n02909870",
    "n02948072",
    "n02950826",
    "n03424325",
    "n03649909",
    "n04118538",
    "n04133789",
    "n04146614",
)

GROUPS: dict[str, tuple[int, ...]] = {
    "A": tuple(range(0, 10)),
    "B": tuple(range(10, 20)),
    "C": tuple(range(20, 30)),
    "D": tuple(range(30, 40)),
    "E": tuple(range(40, 50)),
}

NUM_CLASSES = len(CLASS40)
SYNSET_TO_LABEL: dict[str, int] = {s: i for i, s in enumerate(CLASS40)}
LABEL_TO_SYNSET: dict[int, str] = {i: s for i, s in enumerate(CLASS40)}

SOURCE_TINY = "tiny"
SOURCE_INR = "inr"
SOURCE_CORR = "corr"

DEFAULT_PATHS: dict[str, Path] = {
    SOURCE_TINY: DATA_ROOT / "tiny-imagenet-200",
    SOURCE_INR: DATA_ROOT / "imagenet-r",
    SOURCE_CORR: DATA_ROOT / "tiny-imagenet-200-corr",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    group: str
    source: str
    change_type: str
    description: str

    @property
    def label_indices(self) -> tuple[int, ...]:
        return GROUPS[self.group]


TASK_SPECS = (
    TaskSpec(1, "A", SOURCE_TINY, "initial", "A · Tiny ImageNet (source)"),
    TaskSpec(2, "A", SOURCE_INR, "domain_shift", "A · ImageNet-R (domain shift)"),
    TaskSpec(3, "B", SOURCE_TINY, "new_class", "B · Tiny ImageNet (source)"),
    TaskSpec(4, "B", SOURCE_CORR, "corruption", "B · Tiny + Gaussian noise"),
    TaskSpec(5, "C", SOURCE_TINY, "new_class", "C · Tiny ImageNet (source)"),
    TaskSpec(6, "C", SOURCE_INR, "domain_shift", "C · ImageNet-R (domain shift)"),
    TaskSpec(7, "D", SOURCE_TINY, "new_class", "D · Tiny ImageNet (source)"),
    TaskSpec(8, "D", SOURCE_CORR, "corruption", "D · Tiny + Gaussian noise"),
    TaskSpec(9, "E", SOURCE_TINY, "new_class", "E · Tiny ImageNet (source)"),
    TaskSpec(10, "E", SOURCE_INR, "domain_shift", "E · ImageNet-R (domain shift)"),
)
NUM_TASKS = len(TASK_SPECS)


@dataclass
class SampleEntry:
    path: Path
    label: int


def resolve_source_paths(args) -> dict[str, Path]:
    """Merge DEFAULT_PATHS with optional paths on ``args``."""
    paths = dict(DEFAULT_PATHS)
    overrides = (
        (SOURCE_TINY, ("tiny_root", "r_root")),
        (SOURCE_INR, ("inr_root", "sketch_root")),
        (SOURCE_CORR, ("corr_root",)),
    )
    for key, attrs in overrides:
        for attr in attrs:
            p = getattr(args, attr, None)
            if p:
                paths[key] = Path(p).resolve()
                break
    data_path = getattr(args, "data_path", None)
    if data_path and not any(getattr(args, a, None) for a in ("tiny_root", "r_root")):
        paths[SOURCE_TINY] = Path(data_path).resolve() / "tiny-imagenet-200"
    return paths


def group_synsets(group: str) -> list[str]:
    return [LABEL_TO_SYNSET[i] for i in GROUPS[group]]


def class_mask_for_task(spec: TaskSpec) -> list[int]:
    return list(spec.label_indices)


def _iter_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing directory: {directory}")
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_tiny_split_entries(
    source_root: Path,
    split: str,
    label_indices: tuple[int, ...],
) -> list[SampleEntry]:
    """Tiny ImageNet: ``train/{syn}/images/``; eval split uses ``val/`` (mapped from ``test``)."""
    wanted = {LABEL_TO_SYNSET[i] for i in label_indices}
    entries: list[SampleEntry] = []

    if split == "train":
        for label_idx in label_indices:
            syn = LABEL_TO_SYNSET[label_idx]
            class_dir = source_root / "train" / syn / "images"
            for path in _iter_image_files(class_dir):
                entries.append(SampleEntry(path=path, label=label_idx))
        return entries

    if split != "test":
        raise ValueError(f"Tiny ImageNet supports split 'train' or 'test' (val), got {split!r}")

    ann_path = source_root / "val" / "val_annotations.txt"
    val_dir = source_root / "val" / "images"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing Tiny val annotations: {ann_path}")
    for line in ann_path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        fname, syn = parts[0], parts[1]
        if syn not in wanted:
            continue
        path = val_dir / fname
        if path.is_file():
            entries.append(SampleEntry(path=path, label=SYNSET_TO_LABEL[syn]))
    return entries


def _inr_synset_dir(source_root: Path, split: str, syn: str) -> Path | None:
    """Resolve class image dir: ``{split}/{syn}`` (corr) or flat ``{syn}`` (ImageNet-R)."""
    split_dir = source_root / split / syn
    if split_dir.is_dir():
        return split_dir
    flat_dir = source_root / syn
    if flat_dir.is_dir() and split == "train":
        return flat_dir
    return None


def collect_inr_split_entries(
    source_root: Path,
    split: str,
    label_indices: tuple[int, ...],
) -> list[SampleEntry]:
    """ImageNet-R / corr: ``{train,test}/{syn}/*`` or flat ImageNet-R ``{syn}/*`` (train only)."""
    entries: list[SampleEntry] = []
    for label_idx in label_indices:
        syn = LABEL_TO_SYNSET[label_idx]
        class_dir = _inr_synset_dir(source_root, split, syn)
        if class_dir is None:
            if split == "test" and (source_root / syn).is_dir():
                continue
            raise FileNotFoundError(
                f"Missing class directory for {syn!r} under {source_root} "
                f"(tried {split}/{syn} and flat {syn})"
            )
        for path in _iter_image_files(class_dir):
            entries.append(SampleEntry(path=path, label=label_idx))
    return entries


def collect_split_entries(
    source: str,
    source_root: Path,
    split: str,
    label_indices: tuple[int, ...],
) -> list[SampleEntry]:
    if source == SOURCE_TINY:
        return collect_tiny_split_entries(source_root, split, label_indices)
    if source in (SOURCE_INR, SOURCE_CORR):
        return collect_inr_split_entries(source_root, split, label_indices)
    raise ValueError(f"Unknown source: {source!r}")


def build_task_entries(
    spec: TaskSpec,
    *,
    source_paths: dict[str, Path] | None = None,
    splits: tuple[str, ...] = ("train", "test"),
) -> dict[str, list[SampleEntry]]:
    paths = {**DEFAULT_PATHS, **(source_paths or {})}
    root = paths[spec.source]
    return {
        split: collect_split_entries(spec.source, root, split, spec.label_indices)
        for split in splits
    }


def stream_metadata() -> dict:
    return {
        "num_classes": NUM_CLASSES,
        "class40": list(CLASS40),
        "synset_to_label": SYNSET_TO_LABEL,
        "groups": {k: list(v) for k, v in GROUPS.items()},
        "tasks": [
            {**asdict(spec), "synsets": group_synsets(spec.group), "labels": list(spec.label_indices)}
            for spec in TASK_SPECS
        ],
    }


def print_task_table() -> None:
    print("task_id\tgroup\tlabels\tsource\tchange_type")
    for spec in TASK_SPECS:
        labs = ",".join(str(i) for i in spec.label_indices)
        print(f"T{spec.task_id}\t{spec.group}\t[{labs}]\t{spec.source}\t{spec.change_type}")


if __name__ == "__main__":
    print_task_table()
    paths = dict(DEFAULT_PATHS)
    for spec in TASK_SPECS:
        ent = build_task_entries(spec, source_paths=paths)
        print(f"  T{spec.task_id}: train={len(ent['train'])} test={len(ent['test'])}")
