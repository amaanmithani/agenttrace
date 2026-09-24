"""Render README results sections from committed results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def splice(text: str, name: str, body: str) -> str:
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    i, j = text.index(start), text.index(end)
    return text[: i + len(start)] + "\n" + body.strip() + "\n" + text[j:]


def overhead() -> str:
    o = json.loads((ROOT / "results" / "overhead.json").read_text())
    r, c, env = o["recorder"], o["langgraph_callback"], o["env"]
    return f"""
| What | Per step |
|---|---|
| `Recorder.step()` around a no-op (median of {r["reps"]}×{r["calls"]:,} calls) | {r["median_us"]:.1f} µs |
| LangGraph callback, support agent with a scripted model ({c["steps_per_run"]} steps/run, {c["reps"]}×{c["runs"]} runs) | {c["overhead_per_step_us"]:.1f} µs |

The callback adds {c["overhead_per_run_ms"]:.2f} ms to a run that takes {c["run_without_ms"]:.1f} ms with a model that
answers instantly; against a real model call (hundreds of milliseconds) it is noise.
Python {env["python"]}, {env["system"]} {env["machine"]}. Reproduce: `uv run python bench/overhead.py`.
"""


def case_study() -> str:
    c = json.loads((ROOT / "results" / "case-study.json").read_text())
    f, ctl = c["first_divergence"], c["controls"]
    yes = lambda b: "yes" if b else "**no**"  # noqa: E731
    return f"""
Model: `{c["model"]}` via Ollama, temperature 0. Question: *"{c["question"]}"*

| | Final answer |
|---|---|
| Before (order service v1) | {c["answers"]["before"]} |
| After (v2 drops `opened`) | {c["answers"]["after"]} |

`agenttrace diff examples/runs/before.jsonl examples/runs/after.jsonl`:

```
{c["diff_text"]}
```

Both answers are from an 8B local model and neither is fully right (the "before" answer misstates the delivery
window; the "after" answer invents a partial refund). The point is not answer quality but where the runs first differ.

The first divergence is step {f["step_index_in_before"]} of {c["steps"]["before"]}: `{f["kind"]} {f["name"]}`, not the
final message where the symptom shows.

Controls:

| Check | Result |
|---|---|
| Full replay of "before" (recorded model + recorded tools, no network) is step-for-step identical | {yes(ctl["full_replay_identical"])} |
| ...and gives the same final answer | {yes(ctl["full_replay_answer_matches"])} |
| Live model again with "before"'s tool responses pinned: identical run | {yes(ctl["tools_only_replay_identical"])} |
| ...same final answer | {yes(ctl["tools_only_answer_matches"])} |

The last two rows are the control for sampling noise: with the tool responses held fixed, the live model reproduces the
"before" run, so the change in behaviour comes from the tool.
Reproduce: `uv run python examples/case_study.py` (needs Ollama with `{c["model"]}`); the recorded runs are committed in
`examples/runs/`, and CI re-diffs them on every push.
"""


if __name__ == "__main__":
    p = ROOT / "README.md"
    t = p.read_text()
    t = splice(t, "overhead", overhead())
    t = splice(t, "case", case_study())
    p.write_text(t)
