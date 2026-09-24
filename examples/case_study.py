"""Case study: a backend change that silently changed a support agent's answer.

The order service's v2 response drops the `opened` field (a serializer refactor).
Nothing errors. The agent's final answer changes from "store credit" to "full refund".
This script records the agent before and after the change with a live local model,
then uses agenttrace to find where the runs part ways, and runs two controls:

  1. full replay of the "before" run with no model and no tools: must be identical;
  2. tools-only replay: the live model again, with the "before" tool responses pinned.
     If this is identical to "before", the model is deterministic here and the
     divergence is caused by the tool, not by sampling noise.

Writes examples/runs/*.jsonl, results/case-study.json and results/case-study-diff.json.
Run: uv run python examples/case_study.py [--model llama3.1:8b]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import support_agent as sa  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from agenttrace import Recorder, diff_runs  # noqa: E402
from agenttrace.cli import render_diff  # noqa: E402
from agenttrace.langchain import AgentTraceCallback  # noqa: E402
from agenttrace.replay import ReplayChatModel, pin_tools  # noqa: E402
from agenttrace.trace import Run  # noqa: E402

QUESTION = "Hi, I'd like a refund for order 42, the headphones. Can I get my money back?"


@tool
def lookup_order_v2(order_id: int) -> str:
    """Look up an order by its number."""
    o = sa.ORDERS.get(order_id)
    if o is None:
        return "no such order"
    # v2 serializer: `opened` was dropped by mistake.
    return ", ".join(f"{k}={v}" for k, v in o.items() if k != "opened")


lookup_order_v2.name = "lookup_order"
V2_TOOLS = [lookup_order_v2, sa.refund_policy]


def record(graph: Any, path: Path, meta: dict[str, Any]) -> tuple[Run, str]:
    rec = Recorder(path, meta=meta)
    with rec:
        out = graph.invoke(
            {"messages": [HumanMessage(QUESTION)]}, config={"callbacks": [AgentTraceCallback(rec)]}
        )
    return rec.run, str(out["messages"][-1].content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1:8b")
    a = ap.parse_args()
    from langchain_ollama import ChatOllama

    def model() -> Any:
        return ChatOllama(model=a.model, temperature=0, seed=7)

    runs = ROOT / "examples" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    before, ans_before = record(
        sa.build_graph(model()), runs / "before.jsonl", {"agent": "support", "tools": "v1"}
    )
    after, ans_after = record(
        sa.build_graph(model(), V2_TOOLS), runs / "after.jsonl", {"agent": "support", "tools": "v2"}
    )

    d = diff_runs(before, after)
    text, _ = render_diff(before, after)
    print(text)

    # Control 1: full replay, offline.
    replay, ans_replay = record(
        sa.build_graph(ReplayChatModel(before), pin_tools(sa.TOOLS, before)),
        runs / "before-replayed.jsonl",
        {"agent": "support", "tools": "v1", "replay": "full"},
    )
    full = diff_runs(before, replay)
    # Control 2: live model, tools pinned to the "before" run.
    tonly, ans_tonly = record(
        sa.build_graph(model(), pin_tools(sa.TOOLS, before)),
        runs / "before-tools-pinned.jsonl",
        {"agent": "support", "tools": "v1 (pinned)", "replay": "tools-only"},
    )
    tools_only = diff_runs(before, tonly)

    first = d.first
    assert first is not None
    step = first.a or first.b
    assert step is not None
    result = {
        "model": a.model,
        "question": QUESTION,
        "answers": {"before": ans_before, "after": ans_after},
        "steps": {"before": len(before), "after": len(after)},
        "first_divergence": {
            "kind": step.kind,
            "name": step.name,
            "op": first.op,
            "output_differs": first.output_differs,
            "step_index_in_before": [s.id for s in before].index(first.a.id) + 1 if first.a else None,
        },
        "stats": d.stats,
        "controls": {
            "full_replay_identical": full.identical,
            "full_replay_answer_matches": ans_replay == ans_before,
            "tools_only_replay_identical": tools_only.identical,
            "tools_only_answer_matches": ans_tonly == ans_before,
        },
        "diff_text": text,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "case-study.json").write_text(json.dumps(result, indent=2) + "\n")
    (ROOT / "results" / "case-study-diff.json").write_text(
        json.dumps(d.to_json(), indent=2, default=str) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "diff_text"}, indent=2))


if __name__ == "__main__":
    main()
