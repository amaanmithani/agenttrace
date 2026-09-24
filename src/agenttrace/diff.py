"""Aligning two runs and finding where they part ways."""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agenttrace.trace import Run, Step, canonical

Op = Literal["equal", "changed", "removed", "added"]


def myers(a: Sequence[str], b: Sequence[str]) -> list[tuple[Literal["equal", "removed", "added"], int, int]]:
    """Shortest edit script between `a` and `b` (Myers 1986), O((n+m)·d) time.
    Returns (op, i, j) triples in order; i/j index a/b (-1 where absent)."""
    n, m = len(a), len(b)
    maxd = n + m
    v = {1: 0}
    trace: list[dict[int, int]] = []
    for d in range(maxd + 1):
        trace.append(dict(v))
        for k in range(-d, d + 1, 2):
            if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
                x = v.get(k + 1, 0)
            else:
                x = v.get(k - 1, 0) + 1
            y = x - k
            while x < n and y < m and a[x] == b[y]:
                x, y = x + 1, y + 1
            v[k] = x
            if x >= n and y >= m:
                return _backtrack(trace, a, b, n, m)
    raise AssertionError("unreachable")


def _backtrack(
    trace: list[dict[int, int]], a: Sequence[str], b: Sequence[str], n: int, m: int
) -> list[tuple[Literal["equal", "removed", "added"], int, int]]:
    out: list[tuple[Literal["equal", "removed", "added"], int, int]] = []
    x, y = n, m
    for d in range(len(trace) - 1, -1, -1):
        v = trace[d]
        k = x - y
        pk = k + 1 if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)) else k - 1
        px = v.get(pk, 0)
        py = px - pk
        while x > px and y > py:
            x, y = x - 1, y - 1
            out.append(("equal", x, y))
        if d > 0:
            if x == px:
                y -= 1
                out.append(("added", -1, y))
            else:
                x -= 1
                out.append(("removed", x, -1))
    out.reverse()
    return out


@dataclass
class Entry:
    op: Op
    a: Step | None
    b: Step | None
    output_differs: bool = False

    @property
    def diverges(self) -> bool:
        return self.op != "equal" or self.output_differs


@dataclass
class RunDiff:
    entries: list[Entry]
    first: Entry | None
    stats: dict[str, int] = field(default_factory=dict)

    @property
    def identical(self) -> bool:
        return self.first is None

    def to_json(self) -> dict[str, Any]:
        def ref(s: Step | None) -> dict[str, Any] | None:
            return None if s is None else {"id": s.id, "kind": s.kind, "name": s.name}

        first = None
        if self.first:
            first = {
                "op": self.first.op,
                "a": ref(self.first.a),
                "b": ref(self.first.b),
                "output_differs": self.first.output_differs,
                "input_delta": delta(self.first.a, self.first.b, "input"),
                "output_delta": delta(self.first.a, self.first.b, "output"),
            }
        return {
            "identical": self.identical,
            "stats": self.stats,
            "first_divergence": first,
            "entries": [
                {"op": e.op, "a": ref(e.a), "b": ref(e.b), "output_differs": e.output_differs}
                for e in self.entries
            ],
        }


def _text(value: Any) -> list[str]:
    """A readable multi-line form for text diffs: chat messages as role: content."""
    if isinstance(value, str):
        return value.splitlines()
    if isinstance(value, dict) and isinstance(value.get("messages"), list):
        lines = []
        for m in value["messages"]:
            if isinstance(m, dict):
                lines.append(f"[{m.get('role', '?')}]")
                content = m.get("content")
                lines.extend(content.splitlines() if isinstance(content, str) else [canonical(content)])
                if m.get("tool_calls"):
                    lines.append("tool_calls: " + canonical(m["tool_calls"]))
            else:
                lines.append(canonical(m))
        rest = {k: v for k, v in value.items() if k != "messages"}
        if rest:
            lines.extend(json.dumps(rest, indent=2, sort_keys=True, default=str).splitlines())
        return lines
    return json.dumps(value, indent=2, sort_keys=True, default=str, ensure_ascii=False).splitlines()


def delta(a: Step | None, b: Step | None, fld: Literal["input", "output"]) -> list[str]:
    """Unified diff of one field of two steps (empty if equal)."""
    va = getattr(a, fld) if a else None
    vb = getattr(b, fld) if b else None
    if a and b and canonical(va) == canonical(vb):
        return []
    return list(
        difflib.unified_diff(_text(va) if a else [], _text(vb) if b else [], "a", "b", lineterm="", n=2)
    )


def diff_runs(a: Run, b: Run) -> RunDiff:
    """Align two runs by step signature (kind, name, input) and find the first divergence.

    Steps with the same signature at aligned positions are `equal` (their outputs
    may still differ: a non-deterministic model or a changed tool). A removed step
    followed by an added one of the same kind and name is a `changed` step: the
    same call with different input. The first divergence is the earliest entry
    that is not equal or whose output differs, skipping container steps (graph
    nodes) whose output differs only because something inside them did.
    """
    sa, sb = [s.signature() for s in a.steps], [s.signature() for s in b.steps]
    raw = myers(sa, sb)

    entries: list[Entry] = []
    i = 0
    while i < len(raw):
        op, x, y = raw[i]
        if op == "equal":
            ea, eb = a.steps[x], b.steps[y]
            entries.append(
                Entry("equal", ea, eb, canonical(ea.output) != canonical(eb.output) or ea.error != eb.error)
            )
            i += 1
            continue
        # A run of removals and additions: pair same-kind, same-name steps in order
        # (monotone in both runs), and keep every entry in its place in both runs.
        rem: list[Step] = []
        add: list[Step] = []
        while i < len(raw) and raw[i][0] != "equal":
            o, x, y = raw[i]
            if o == "removed":
                rem.append(a.steps[x])
            else:
                add.append(b.steps[y])
            i += 1
        pairs: list[tuple[int, int]] = []
        j0 = 0
        for ri, r in enumerate(rem):
            j = next((j for j in range(j0, len(add)) if (add[j].kind, add[j].name) == (r.kind, r.name)), None)
            if j is not None:
                pairs.append((ri, j))
                j0 = j + 1
        ri = aj = 0
        for pr, pa in [*pairs, (len(rem), len(add))]:
            # Unpaired steps before the next pair: A's, then B's (timestamps from two
            # runs aren't comparable, so there is no finer order to use).
            entries.extend(Entry("removed", r, None) for r in rem[ri:pr])
            entries.extend(Entry("added", None, s_) for s_ in add[aj:pa])
            if pr < len(rem) and pa < len(add):
                entries.append(Entry("changed", rem[pr], add[pa]))
            ri, aj = pr + 1, pa + 1

    parents_a = {s.parent for s in a.steps if s.parent is not None}
    parents_b = {s.parent for s in b.steps if s.parent is not None}

    def container(e: Entry) -> bool:
        return (e.a is not None and e.a.id in parents_a) or (e.b is not None and e.b.id in parents_b)

    first = next((e for e in entries if e.diverges and not (e.op == "equal" and container(e))), None)
    if first is None:
        first = next((e for e in entries if e.diverges), None)
    stats = {op: sum(1 for e in entries if e.op == op) for op in ("equal", "changed", "removed", "added")}
    stats["output_differs"] = sum(1 for e in entries if e.output_differs)
    return RunDiff(entries, first, stats)
