"""Recording overhead, per step.

1. Recorder alone: `with rec.step(...)` around a no-op, vs the bare no-op.
2. LangGraph callback: the support agent (scripted model, real tools) invoked with and
   without AgentTraceCallback; the difference divided by the number of recorded steps.

Writes results/overhead.json. Run: uv run python bench/overhead.py
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "tests")]

import support_agent as sa  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from agenttrace import Recorder  # noqa: E402
from agenttrace.langchain import AgentTraceCallback  # noqa: E402
from conftest import GOOD, ScriptedModel  # noqa: E402


def per_call_ns(fn, n: int) -> float:  # type: ignore[no-untyped-def]
    t = time.perf_counter_ns()
    for _ in range(n):
        fn()
    return (time.perf_counter_ns() - t) / n


def recorder_overhead(n: int = 20000, reps: int = 7) -> dict[str, float]:
    payload = {"messages": [{"role": "user", "content": "x" * 200}], "tools": ["a", "b"]}

    def bare() -> None:
        out = {"content": "ok"}
        del out

    rec = Recorder()

    def recorded() -> None:
        with rec.step("llm", "m", payload) as s:
            s.set_output({"content": "ok"})

    samples = []
    for _ in range(reps):
        rec.run.steps.clear()
        samples.append(per_call_ns(recorded, n) - per_call_ns(bare, n))
    return {
        "median_us": statistics.median(samples) / 1e3,
        "min_us": min(samples) / 1e3,
        "calls": n,
        "reps": reps,
    }


def callback_overhead(runs: int = 300, reps: int = 5) -> dict[str, float]:
    q = {"messages": [HumanMessage("Can I get a refund for order 42?")]}

    def once(record: bool) -> int:
        g = sa.build_graph(ScriptedModel(script=list(GOOD)))
        if not record:
            g.invoke(q)
            return 0
        rec = Recorder()
        g.invoke(q, config={"callbacks": [AgentTraceCallback(rec)]})
        return len(rec.run)

    steps = once(True)
    diffs = []
    base = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(runs):
            once(False)
        t1 = time.perf_counter_ns()
        for _ in range(runs):
            once(True)
        t2 = time.perf_counter_ns()
        base.append((t1 - t0) / runs)
        diffs.append(((t2 - t1) - (t1 - t0)) / runs)
    return {
        "steps_per_run": steps,
        "run_without_ms": statistics.median(base) / 1e6,
        "overhead_per_run_ms": statistics.median(diffs) / 1e6,
        "overhead_per_step_us": statistics.median(diffs) / steps / 1e3,
        "runs": runs,
        "reps": reps,
    }


if __name__ == "__main__":
    out = {
        "env": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "recorder": recorder_overhead(),
        "langgraph_callback": callback_overhead(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "overhead.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
