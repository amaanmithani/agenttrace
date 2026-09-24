"""The run format: one JSON object per line, a header line then one line per step."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Kind = Literal["llm", "tool", "node", "mcp", "custom"]
FORMAT = "agenttrace/1"


def canonical(value: Any) -> str:
    """Stable JSON: sorted keys, no whitespace, non-JSON values stringified."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()[:16]


@dataclass
class Step:
    id: int
    kind: Kind
    name: str
    input: Any = None
    output: Any = None
    error: str | None = None
    parent: int | None = None
    start_ns: int = 0
    end_ns: int = 0
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6

    def signature(self) -> str:
        """What the step was asked to do: kind, name and input, not its result.
        Two runs that agree on signatures up to step k made the same requests."""
        return f"{self.kind}:{self.name}:{digest(self.input)}"

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, {}) or k in ("input", "output")}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Step:
        return cls(
            id=int(d["id"]),
            kind=d["kind"],
            name=str(d["name"]),
            input=d.get("input"),
            output=d.get("output"),
            error=d.get("error"),
            parent=d.get("parent"),
            start_ns=int(d.get("start_ns", 0)),
            end_ns=int(d.get("end_ns", 0)),
            attrs=dict(d.get("attrs") or {}),
        )


@dataclass
class Run:
    steps: list[Step] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def of_kind(self, *kinds: Kind) -> list[Step]:
        return [s for s in self.steps if s.kind in kinds]

    def children(self, parent: int | None) -> list[Step]:
        return [s for s in self.steps if s.parent == parent]

    def lines(self) -> Iterable[str]:
        yield canonical({"format": FORMAT, **self.meta})
        for s in self.steps:
            yield canonical(s.to_json())

    def save(self, path: str | Path) -> None:
        Path(path).write_text("\n".join(self.lines()) + "\n", encoding="utf-8")

    @classmethod
    def parse(cls, text: str) -> Run:
        run = cls()
        for n, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {n}: not JSON ({e.msg})") from e
            if not isinstance(obj, dict):
                raise ValueError(f"line {n}: expected an object")
            if "format" in obj:
                if obj["format"] != FORMAT:
                    raise ValueError(f"line {n}: unsupported format {obj['format']!r}")
                run.meta = {k: v for k, v in obj.items() if k != "format"}
            else:
                try:
                    run.steps.append(Step.from_json(obj))
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(f"line {n}: bad step ({e})") from e
        run.steps.sort(key=lambda s: s.id)
        return run

    @classmethod
    def load(cls, path: str | Path) -> Run:
        return cls.parse(Path(path).read_text(encoding="utf-8"))
