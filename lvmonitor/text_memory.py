"""Task-free text memory: agent summary + agent-reported category lists only."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MEMORY_VERSION = 4
INDEX_NAME = "text_memory.json"
INITIAL_CAUSE = "init"


def normalize_category(name: str) -> str:
    return name.strip().lower()


def categories_not_in_known(names: list[str], known: list[str]) -> list[str]:
    """Names from the agent that are not already in stream memory."""
    known_set = {normalize_category(k) for k in known}
    out: list[str] = []
    for n in names:
        n = n.strip()
        if n and normalize_category(n) not in known_set:
            out.append(n)
    return out


@dataclass
class DetectionRecord:
    step: int
    cause: str | None
    summary: str
    new_categories: list[str]
    raw_response: str
    ts: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_category_list(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    lower = text.lower()
    if lower in ("none", "n/a", "na", "(none)", "-", "no new categories", "no new category"):
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass
    parts = re.split(r"[,;]", text)
    out: list[str] = []
    for p in parts:
        p = p.strip().strip(".")
        if p and p.lower() not in ("none", "n/a"):
            out.append(p)
    return out


def parse_detection_response(raw: str) -> tuple[str | None, str, str, list[str]]:
    """Parse cause, reasoning, memory summary, and agent new-category list."""
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
        r"REASONING\s*:\s*(.+?)(?=\n\s*(?:MEMORY|NEW\s*CATEGORIES)\s*:|$)",
        text,
        re.I | re.S,
    )
    if rm:
        reasoning = rm.group(1).strip()

    new_categories: list[str] = []
    nm = re.search(
        r"NEW\s*CATEGORIES\s*:\s*(.+?)(?=\n\s*MEMORY\s*:|\n\n|\Z)",
        text,
        re.I | re.S,
    )
    if nm:
        new_categories = _parse_category_list(nm.group(1))

    summary = ""
    mm = re.search(r"MEMORY\s*:\s*(.+?)(?:\n\n|\Z)", text, re.I | re.S)
    if mm:
        summary = mm.group(1).strip()
    elif reasoning:
        summary = reasoning
    elif text:
        summary = text[:500]

    return cause, reasoning, summary, new_categories


def parse_initial_response(raw: str) -> tuple[str, list[str]]:
    """Parse first-batch bootstrap (Memory + New categories only)."""
    _, _, summary, new_categories = parse_detection_response(raw)
    return summary, new_categories


@dataclass
class TextMemoryStore:
    memory_dir: Path
    detections: list[DetectionRecord] = field(default_factory=list)

    @property
    def index_path(self) -> Path:
        return self.memory_dir / INDEX_NAME

    @property
    def step_count(self) -> int:
        return len(self.detections)

    def known_categories(self) -> list[str]:
        """Categories recorded on the immediately previous step only."""
        if not self.detections:
            return []
        return list(self.detections[-1].new_categories)

    def save(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MEMORY_VERSION,
            "detections": [d.to_dict() for d in self.detections],
        }
        self.index_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def _record_from_dict(cls, d: dict) -> DetectionRecord:
        return DetectionRecord(
            step=d["step"],
            cause=d.get("cause"),
            summary=d.get("summary", ""),
            new_categories=list(d.get("new_categories") or []),
            raw_response=d.get("raw_response", ""),
            ts=d.get("ts", ""),
        )

    @classmethod
    def load(cls, memory_dir: Path) -> TextMemoryStore:
        memory_dir = memory_dir.resolve()
        store = cls(memory_dir=memory_dir)
        for legacy_name in (INDEX_NAME, "structured_memory.json"):
            path = memory_dir / legacy_name
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            ver = data.get("version", 1)
            if ver >= 3 and data.get("detections"):
                store.detections = [cls._record_from_dict(d) for d in data["detections"]]
                store.save()
                return store
            if ver >= 3:
                store.save()
                return store
            store._import_legacy(data)
            store.save()
            return store
        return store

    def _import_legacy(self, data: dict) -> None:
        for note in data.get("criteria_notes", []):
            s = note.get("summary") or note.get("reasoning", "")
            if s:
                self._append_parsed(
                    cause=note.get("change_cause"),
                    summary=s[:400],
                    new_categories=[],
                    raw_response="",
                    ts=note.get("ts", ""),
                )
        for tr in data.get("transitions", []):
            s = tr.get("reasoning") or ""
            if s:
                self._append_parsed(
                    cause=tr.get("change_cause") or tr.get("predicted"),
                    summary=s[:400],
                    new_categories=[],
                    raw_response=tr.get("raw_response", "")[:200],
                    ts=tr.get("ts", ""),
                )

    def append_detection(
        self,
        *,
        cause: str | None,
        summary: str,
        new_categories: list[str] | None,
        raw_response: str,
    ) -> DetectionRecord:
        agent_cats = list(new_categories or [])
        if cause == INITIAL_CAUSE:
            cats = agent_cats
        elif cause == "A":
            cats = categories_not_in_known(agent_cats, self.known_categories())
        else:
            cats = []

        return self._append_parsed(
            cause=cause,
            summary=summary,
            new_categories=cats,
            raw_response=raw_response,
            ts=datetime.now(timezone.utc).isoformat(),
        )

    def _append_parsed(
        self,
        *,
        cause: str | None,
        summary: str,
        new_categories: list[str],
        raw_response: str,
        ts: str,
    ) -> DetectionRecord:
        rec = DetectionRecord(
            step=len(self.detections) + 1,
            cause=cause,
            summary=summary.strip(),
            new_categories=new_categories,
            raw_response=raw_response,
            ts=ts,
        )
        self.detections.append(rec)
        self.save()
        return rec

    def build_prompt_context(self) -> str:
        if not self.detections:
            return ""
        d = self.detections[-1]
        cause = d.cause or "?"
        if cause == INITIAL_CAUSE:
            cause = "init"
        lines = [
            "Memory from the **immediately previous** stream step only "
            "(compare the current batch to this; ignore earlier history):\n",
        ]
        known = self.known_categories()
        if known:
            lines.append(
                "Categories from that step: " + ", ".join(known) + ".\n"
            )
        extra = ""
        if d.new_categories:
            extra = f" [categories: {', '.join(d.new_categories)}]"
        lines.append(f"- [{cause}] {d.summary}{extra}\n")
        lines.append(
            "If the **same group** continues (same labels, new rendering/noise), "
            "choose B or C, not A. If the **group changed**, choose A for new classes.\n"
        )
        return "\n".join(lines)

    def query(self, topic: str, *, arg: str | None = None) -> str:
        topic = topic.lower().strip()
        if topic in ("summary", "all"):
            known = self.known_categories()
            return (
                f"{self.index_path}\n"
                f"detections={len(self.detections)} "
                f"known_categories={len(known)}"
            )
        if topic in ("categories", "known", "seen"):
            known = self.known_categories()
            return ", ".join(known) if known else "(none)"
        if topic in ("detection", "detections", "history"):
            if arg and arg.isdigit():
                return self._one_detection(int(arg))
            return self._all_detections()
        if topic in ("last", "latest"):
            return self._one_detection(len(self.detections)) if self.detections else "(none)"
        return "Queries: summary, categories, detections, last, detection <n>"

    def _all_detections(self) -> str:
        if not self.detections:
            return "(no detections)"
        parts = []
        for d in self.detections:
            nc = f" new_categories={d.new_categories}" if d.new_categories else ""
            parts.append(f"[{d.step}] cause={d.cause}{nc}\n{d.summary}")
        return "\n\n".join(parts)

    def _one_detection(self, step: int) -> str:
        for d in self.detections:
            if d.step == step:
                return json.dumps(d.to_dict(), indent=2, ensure_ascii=False)
        return f"No detection step {step}"
