"""Qwen stream agent: pairwise vision comparison (no text memory)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]

from imagenet_hs import (
    LABEL_TO_SYNSET,
    SOURCE_CORR,
    SOURCE_INR,
    SOURCE_TINY,
    TASK_SPECS,
    SampleEntry,
    TaskSpec,
    build_task_entries,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT_PATH = REPO_ROOT / "results" / "qwen_stream_verify.jsonl"
DEFAULT_DRYRUN_BATCH_DIR = REPO_ROOT / "results" / "qwen_dryrun_batches"

# Dataset roots under repo data/
DATA_ROOT = REPO_ROOT / "data"
SOURCE_PATHS: dict[str, Path] = {
    SOURCE_TINY: DATA_ROOT / "tiny-imagenet-200",
    SOURCE_INR: DATA_ROOT / "imagenet-r",
    SOURCE_CORR: DATA_ROOT / "tiny-imagenet-200-corr",
}

SPLIT = "train"
IMAGE_SIZE = 256
QWEN_MODEL = "qwen3.6-flash"
SEED = 0

MIN_IMAGES = 4
MAX_IMAGES = 10

CHANGE_TYPE_TO_OPTION: dict[str, str | None] = {
    "initial": None,
    "new_class": "A",
    "domain_shift": "B",
    "corruption": "C",
}

OPTION_ALIASES: dict[str, str] = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "NEW CLASS": "A",
    "NEW CATEGORIES": "A",
    "NEW_CATEGORY": "A",
    "DOMAIN SHIFT": "B",
    "DOMAIN_SHIFT": "B",
    "CORRUPTION": "C",
    "CORRUPTION / NOISE": "C",
    "NOISE": "C",
    "NO SIGNIFICANT CHANGE": "D",
    "NO CHANGE": "D",
    "FALSE ALARM": "D",
}

PAIR_DECISION_GUIDE = """
The stream uses **pairs of consecutive tasks**. Compare the **previous** batch to the
**current** batch (images only — no external labels).

**Decision order** (first match wins):
1. **C — Corruption** — Same object types as the previous batch, but the current batch
   is dominated by heavy noise, grain, blur, or compression artifacts.
2. **B — Domain shift** — Same object types / group as the previous batch, but the visual
   medium changed strongly (e.g. photo → sketch, tattoo, line art).
3. **A — New categories** — The current batch introduces object types not present in the
   previous batch (group changed or clearly new taxa).
4. **D — No significant change** — Current matches previous; plateau may be a false alarm.

Output format:
Change cause: <A, B, C, or D>
Reasoning: <one short sentence>
New categories: <comma-separated English names new in the current batch vs previous, or **none**>
"""


@dataclass
class TaskBatchSample:
    task_id: int
    name: str
    images: list[Image.Image]
    categories: list[str]
    labels: list[int] = field(default_factory=list)


@dataclass
class TransitionResult:
    from_task: int
    to_task: int
    gt_change: str
    expected: str | None
    predicted: str | None
    raw_response: str
    correct: bool | None
    prev_batch_categories: list[str]
    curr_batch_categories: list[str]
    new_categories: list[str] = field(default_factory=list)
    reasoning: str = ""


def label_to_category(label: int) -> str:
    return LABEL_TO_SYNSET[label]


def resize_pil(img: Image.Image, size: int) -> Image.Image:
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    return img.resize((size, size), resample=resample)


def _pick_entries_stratified(
    entries: list[SampleEntry],
    k: int,
    rng: random.Random,
) -> list[SampleEntry]:
    by_label: dict[int, list[SampleEntry]] = {}
    for e in entries:
        by_label.setdefault(e.label, []).append(e)

    label_order = list(by_label.keys())
    rng.shuffle(label_order)

    picked: list[SampleEntry] = []
    used: set[Path] = set()

    for lab in label_order:
        if len(picked) >= k:
            break
        e = rng.choice(by_label[lab])
        picked.append(e)
        used.add(e.path)

    if len(picked) < k:
        remaining = [e for e in entries if e.path not in used]
        rng.shuffle(remaining)
        for e in remaining:
            if len(picked) >= k:
                break
            picked.append(e)
            used.add(e.path)

    rng.shuffle(picked)
    return picked


def sample_task_batch(
    spec: TaskSpec,
    *,
    source_paths: dict[str, Path] | None,
    n_per_batch: int,
    seed: int,
    image_size: int,
    split: str = "train",
) -> TaskBatchSample:
    entries = build_task_entries(
        spec, source_paths=source_paths, splits=(split,)
    )[split]
    if not entries:
        raise RuntimeError(f"T{spec.task_id}: no images in {split}")

    rng = random.Random(seed + spec.task_id * 1000)
    k = min(n_per_batch, len(entries))
    picked = _pick_entries_stratified(entries, k, rng)

    images = [resize_pil(Image.open(e.path).convert("RGB"), image_size) for e in picked]
    labels = [e.label for e in picked]
    cats = [label_to_category(l) for l in labels]
    name = (
        f"T{spec.task_id} group={spec.group} source={spec.source} "
        f"change={spec.change_type} n={len(images)}"
    )
    return TaskBatchSample(
        task_id=spec.task_id,
        name=name,
        images=images,
        categories=cats,
        labels=labels,
    )


def build_client():
    if OpenAI is None:
        raise ImportError("Install openai: pip install openai")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise EnvironmentError("Set DASHSCOPE_API_KEY before running.")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def pil_to_base64(img: Image.Image, *, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def images_to_content(images: list[Image.Image]) -> list[dict]:
    return [
        {"type": "image_url", "image_url": {"url": pil_to_base64(img)}}
        for img in images
    ]


def _parse_category_list(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    lower = text.lower()
    if lower in ("none", "n/a", "na", "(none)", "-", "no new categories", "no new category"):
        return []
    parts = re.split(r"[,;]", text)
    out: list[str] = []
    for p in parts:
        p = p.strip().strip(".")
        if p and p.lower() not in ("none", "n/a"):
            out.append(p)
    return out


def parse_detection_response(raw: str) -> tuple[str | None, str, list[str]]:
    text = raw.strip()
    cause = None
    m = re.search(r"CHANGE\s*CAUSE\s*:\s*([ABCD])\b", text, re.I)
    if m:
        cause = m.group(1).upper()
    else:
        m2 = re.search(r"\b([ABCD])\b", text.upper())
        if m2:
            cause = m2.group(1).upper()

    reasoning = ""
    rm = re.search(
        r"REASONING\s*:\s*(.+?)(?=\n\s*NEW\s*CATEGORIES\s*:|$)",
        text,
        re.I | re.S,
    )
    if rm:
        reasoning = rm.group(1).strip()

    new_categories: list[str] = []
    nm = re.search(r"NEW\s*CATEGORIES\s*:\s*(.+?)(?:\n\n|\Z)", text, re.I | re.S)
    if nm:
        new_categories = _parse_category_list(nm.group(1))

    return cause, reasoning, new_categories


def parse_detection(raw: str) -> str | None:
    text = raw.strip().upper()
    m = re.search(r"CHANGE\s*CAUSE\s*:\s*([ABCD])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1).upper()
    for phrase, letter in OPTION_ALIASES.items():
        if phrase in text and len(phrase) > 1:
            return letter
    return None


def build_pair_message(
    prev: TaskBatchSample,
    curr: TaskBatchSample,
    *,
    prev_spec: TaskSpec,
    curr_spec: TaskSpec,
) -> list[dict]:
    text_tail = (
        f"Previous task: group {prev_spec.group}, source `{prev_spec.source}` "
        f"({len(prev.images)} images).\n"
        f"Current task: group {curr_spec.group}, source `{curr_spec.source}` "
        f"({len(curr.images)} images).\n\n"
        f"{PAIR_DECISION_GUIDE}"
    )
    intro = (
        "You compare **two consecutive data batches** in a continual learning stream. "
        "No text memory — use only the images below.\n\n"
        f"**Previous batch** (T{prev.task_id}, {prev_spec.description}):\n"
    )
    content: list[dict] = [
        {"type": "text", "text": intro},
        *images_to_content(prev.images),
        {
            "type": "text",
            "text": (
                f"\n**Current batch** (T{curr.task_id}, {curr_spec.description}):\n"
            ),
        },
        *images_to_content(curr.images),
        {"type": "text", "text": "\n" + text_tail},
    ]
    return [{"role": "user", "content": content}]


def detect_pair(
    client,
    prev: TaskBatchSample,
    curr: TaskBatchSample,
    *,
    prev_spec: TaskSpec,
    curr_spec: TaskSpec,
    model: str,
    temperature: float,
) -> str:
    messages = build_pair_message(prev, curr, prev_spec=prev_spec, curr_spec=curr_spec)
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content or ""


def expected_option(spec: TaskSpec) -> str | None:
    return CHANGE_TYPE_TO_OPTION.get(spec.change_type)


@dataclass
class StreamAgent:
    """Pairwise Qwen agent: each detection sends previous + current task images."""

    n_per_batch: int = 8
    seed: int = SEED
    client: object | None = None

    def _client(self):
        if self.client is None:
            self.client = build_client()
        return self.client

    def run_pair(
        self,
        prev_spec: TaskSpec,
        curr_spec: TaskSpec,
        prev_batch: TaskBatchSample,
        curr_batch: TaskBatchSample,
        *,
        call_api: bool = True,
    ) -> TransitionResult:
        expected = expected_option(curr_spec)
        raw = ""
        predicted: str | None = None
        reasoning = ""
        new_categories: list[str] = []

        if call_api:
            raw = detect_pair(
                self._client(),
                prev_batch,
                curr_batch,
                prev_spec=prev_spec,
                curr_spec=curr_spec,
                model=QWEN_MODEL,
                temperature=0.0,
            )
            predicted, reasoning, new_categories = parse_detection_response(raw)
            if predicted is None:
                predicted = parse_detection(raw)

        correct: bool | None = None
        if expected is not None and predicted is not None:
            correct = predicted == expected

        return TransitionResult(
            from_task=prev_spec.task_id,
            to_task=curr_spec.task_id,
            gt_change=curr_spec.change_type,
            expected=expected,
            predicted=predicted,
            raw_response=raw,
            correct=correct,
            prev_batch_categories=prev_batch.categories,
            curr_batch_categories=curr_batch.categories,
            new_categories=new_categories,
            reasoning=reasoning,
        )

    def run_stream(
        self,
        *,
        call_api: bool = True,
        report_path: Path | None = None,
        dryrun_batch_dir: Path | None = None,
    ) -> list[TransitionResult]:
        specs = list(TASK_SPECS)
        results: list[TransitionResult] = []
        prev_spec: TaskSpec | None = None
        prev_batch: TaskBatchSample | None = None

        n_img = self.n_per_batch
        print(
            f"Stream agent: {len(specs)} tasks, n={n_img} per task "
            f"({n_img * 2} images per pairwise call), no text memory"
        )

        for spec in specs:
            batch = sample_task_batch(
                spec,
                source_paths=SOURCE_PATHS,
                n_per_batch=self.n_per_batch,
                seed=self.seed,
                image_size=IMAGE_SIZE,
                split=SPLIT,
            )
            if dryrun_batch_dir is not None:
                _save_task_batch_images(spec=spec, batch=batch, output_root=dryrun_batch_dir)
            print(f"\n--- T{spec.task_id} {spec.change_type} ({spec.description}) ---")
            print(
                f"  sampled labels: {batch.labels} "
                f"synsets: {batch.categories[:3]}..."
            )

            if spec.change_type == "initial":
                print("  (initial — no pairwise detection; starts at T2)")
            elif prev_spec is not None and prev_batch is not None:
                tr = self.run_pair(
                    prev_spec,
                    spec,
                    prev_batch,
                    batch,
                    call_api=call_api,
                )
                results.append(tr)
                mark = "✓" if tr.correct else ("✗" if tr.correct is False else "?")
                print(
                    f"  pair T{tr.from_task}→T{tr.to_task} "
                    f"gt={tr.gt_change} expected={tr.expected} "
                    f"predicted={tr.predicted} {mark}"
                )
                if tr.new_categories:
                    print(f"  new_categories: {tr.new_categories}")
                if tr.reasoning:
                    print(f"  reasoning: {tr.reasoning}")
                self._append_report(tr, report_path)

            prev_spec = spec
            prev_batch = batch

        if results:
            n_ok = sum(1 for r in results if r.correct is True)
            n_bad = sum(1 for r in results if r.correct is False)
            n_unk = sum(1 for r in results if r.correct is None)
            print(
                f"\nVerification: {n_ok} correct, {n_bad} wrong, {n_unk} unparsed "
                f"(of {len(results)} pairs)"
            )
        return results

    def _append_report(self, tr: TransitionResult, report_path: Path | None) -> None:
        if report_path is None:
            return
        report_path.parent.mkdir(parents=True, exist_ok=True)
        row = {**asdict(tr), "ts": datetime.now(timezone.utc).isoformat()}
        with report_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sources_for_specs(specs: list[TaskSpec]) -> set[str]:
    return {s.source for s in specs}


def check_source_paths(
    paths: dict[str, Path],
    *,
    required_sources: set[str] | None = None,
) -> None:
    """Ensure dataset roots exist (all three by default)."""
    needed = required_sources or {SOURCE_TINY, SOURCE_INR, SOURCE_CORR}
    missing = [paths[src] for src in sorted(needed) if not paths[src].is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing ImageNet-HS data directories:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\nExpected under DATA_ROOT={DATA_ROOT}"
        )


def _save_task_batch_images(
    *,
    spec: TaskSpec,
    batch: TaskBatchSample,
    output_root: Path,
) -> None:
    """Save sampled task images and metadata for plotting."""
    task_dir = output_root / f"T{spec.task_id:02d}_{spec.group}_{spec.source}_{spec.change_type}"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int]] = []
    for idx, (img, label, synset) in enumerate(zip(batch.images, batch.labels, batch.categories)):
        file_name = f"{idx:02d}_label{label:02d}_{synset}.png"
        img.save(task_dir / file_name, format="PNG")
        rows.append({
            "index": idx,
            "file": file_name,
            "label": label,
            "synset": synset,
        })

    meta = {
        "task_id": spec.task_id,
        "group": spec.group,
        "source": spec.source,
        "change_type": spec.change_type,
        "description": spec.description,
        "num_images": len(batch.images),
        "rows": rows,
    }
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Qwen pairwise stream: each call sends previous + current task images "
            "(no text memory)."
        )
    )
    p.add_argument(
        "-n",
        "--num-images",
        dest="n_per_batch",
        type=int,
        default=8,
        help=f"Images per task; pairwise call sends 2×n ({MIN_IMAGES}–{MAX_IMAGES} each).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample batches only; skip Qwen API.",
    )
    p.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Run one pairwise step at task id N (uses T(N-1) and TN), e.g. 4 or 3-4.",
    )
    return p.parse_args()


def _spec_by_id(task_id: int) -> TaskSpec:
    for spec in TASK_SPECS:
        if spec.task_id == task_id:
            return spec
    raise ValueError(f"Unknown task id {task_id}")


def main() -> None:
    args = parse_args()
    if not MIN_IMAGES <= args.n_per_batch <= MAX_IMAGES:
        raise SystemExit(f"-n must be in [{MIN_IMAGES}, {MAX_IMAGES}]")

    agent = StreamAgent(n_per_batch=args.n_per_batch)

    if args.pair:
        m = re.match(r"(\d+)\s*[-–]\s*(\d+)", args.pair.strip())
        if m:
            to_id = int(m.group(2))
        elif args.pair.strip().isdigit():
            to_id = int(args.pair.strip())
        else:
            raise SystemExit("--pair format: 4 or 3-4 (detect at arrival of T4)")
        if to_id <= 1:
            raise SystemExit("T1 has no previous task; use --pair with id >= 2")
        to_spec = _spec_by_id(to_id)
        prev_spec = _spec_by_id(to_id - 1)
        check_source_paths(
            SOURCE_PATHS,
            required_sources=_sources_for_specs([prev_spec, to_spec]),
        )
        prev_batch = sample_task_batch(
            prev_spec,
            source_paths=SOURCE_PATHS,
            n_per_batch=agent.n_per_batch,
            seed=agent.seed,
            image_size=IMAGE_SIZE,
            split=SPLIT,
        )
        curr_batch = sample_task_batch(
            to_spec,
            source_paths=SOURCE_PATHS,
            n_per_batch=agent.n_per_batch,
            seed=agent.seed,
            image_size=IMAGE_SIZE,
            split=SPLIT,
        )
        tr = agent.run_pair(
            prev_spec,
            to_spec,
            prev_batch,
            curr_batch,
            call_api=not args.dry_run,
        )
        if args.dry_run:
            DEFAULT_DRYRUN_BATCH_DIR.mkdir(parents=True, exist_ok=True)
            _save_task_batch_images(spec=prev_spec, batch=prev_batch, output_root=DEFAULT_DRYRUN_BATCH_DIR)
            _save_task_batch_images(spec=to_spec, batch=curr_batch, output_root=DEFAULT_DRYRUN_BATCH_DIR)
            print(f"Saved dry-run batches to: {DEFAULT_DRYRUN_BATCH_DIR}")
        print(json.dumps(asdict(tr), indent=2, ensure_ascii=False))
        agent._append_report(tr, DEFAULT_REPORT_PATH)
        return

    check_source_paths(SOURCE_PATHS, required_sources=_sources_for_specs(list(TASK_SPECS)))

    dryrun_batch_dir = DEFAULT_DRYRUN_BATCH_DIR if args.dry_run else None
    if dryrun_batch_dir is not None:
        dryrun_batch_dir.mkdir(parents=True, exist_ok=True)

    agent.run_stream(
        call_api=not args.dry_run,
        report_path=None if args.dry_run else DEFAULT_REPORT_PATH,
        dryrun_batch_dir=dryrun_batch_dir,
    )
    if args.dry_run:
        print(f"Saved dry-run batches to: {DEFAULT_DRYRUN_BATCH_DIR}")


if __name__ == "__main__":
    main()
