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


def callback_overhead(runs: int = 200, reps: int = 15) -> dict[str, float]:
    """Paired design: the graph is built once; each rep times `runs` invocations without
    and with the callback, in alternating order, and the per-rep difference is kept.
    Reports the median and the spread of those differences."""
    q = {"messages": [HumanMessage("Can I get a refund for order 42?")]}
    model = ScriptedModel(script=list(GOOD) * (runs * 2 + 2))
    graph = sa.build_graph(model)

    def batch(record: bool) -> float:
        t = time.perf_counter_ns()
        for _ in range(runs):
            model.calls = 0
            if record:
                rec = Recorder()
                graph.invoke(q, config={"callbacks": [AgentTraceCallback(rec)]})
            else:
                graph.invoke(q)
        return (time.perf_counter_ns() - t) / runs

    rec = Recorder()
    model.calls = 0
    graph.invoke(q, config={"callbacks": [AgentTraceCallback(rec)]})
    steps = len(rec.run)
    batch(False)
    batch(True)  # warm-up
    base, diffs = [], []
    for i in range(reps):
        order = (False, True) if i % 2 == 0 else (True, False)
        t = {rec_: batch(rec_) for rec_ in order}
        base.append(t[False])
        diffs.append(t[True] - t[False])
    per_step = sorted(d / steps / 1e3 for d in diffs)
    return {
        "steps_per_run": steps,
        "run_without_ms": statistics.median(base) / 1e6,
        "overhead_per_run_ms": statistics.median(diffs) / 1e6,
        "overhead_per_step_us": statistics.median(per_step),
        "overhead_per_step_us_p25": per_step[len(per_step) // 4],
        "overhead_per_step_us_p75": per_step[(3 * len(per_step)) // 4],
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
