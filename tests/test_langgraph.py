from typing import Any

import pytest
import support_agent as sa
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from agenttrace import Recorder, diff_runs
from agenttrace.langchain import AgentTraceCallback
from agenttrace.replay import ReplayChatModel, ReplayDivergence, pin_tools
from agenttrace.trace import Run
from conftest import GOOD, ScriptedModel, call

Q = {"messages": [HumanMessage("Can I get a refund for order 42?")]}


def record(graph: Any, path: Any = None) -> tuple[Run, Any]:
    rec = Recorder(path, meta={"agent": "support"})
    with rec:
        out = graph.invoke(Q, config={"callbacks": [AgentTraceCallback(rec)]})
    return rec.run, out


def exploding_tools() -> list[Any]:
    @tool
    def lookup_order(order_id: int) -> str:
        """Look up an order by its number."""
        raise AssertionError("live tool called during replay")

    @tool
    def refund_policy() -> str:
        """The store's refund policy."""
        raise AssertionError("live tool called during replay")

    return [lookup_order, refund_policy]


def test_records_nodes_models_and_tools(tmp_path: Any) -> None:
    run, out = record(sa.build_graph(ScriptedModel(script=GOOD)), tmp_path / "r.jsonl")
    kinds = [(s.kind, s.name) for s in run]
    assert kinds[0] == ("node", "LangGraph")
    assert [k for k in kinds if k[0] != "node"] == [
        ("llm", "scripted"),
        ("tool", "lookup_order"),
        ("llm", "scripted"),
        ("tool", "refund_policy"),
        ("llm", "scripted"),
    ]
    by = {s.id: s for s in run}
    for s in run.of_kind("llm", "tool"):
        assert s.parent is not None and by[s.parent].kind == "node"
        assert by[s.parent].name in ("agent", "tools")
        assert s.end_ns >= s.start_ns
    tool_step = run.of_kind("tool")[0]
    assert tool_step.input == {"order_id": 42}
    assert "headphones" in tool_step.output
    first_llm = run.of_kind("llm")[0]
    assert first_llm.input["messages"][0]["role"] == "system"
    assert first_llm.input["tools"] == ["lookup_order", "refund_policy"]
    assert run.of_kind("llm")[1].output == {
        "content": "",
        "tool_calls": [{"name": "refund_policy", "args": {}}],
    }
    # Round-trips through the file.
    again = Run.load(tmp_path / "r.jsonl")
    assert [s.signature() for s in again] == [s.signature() for s in run]
    assert again.meta == {"agent": "support"}
    assert out["messages"][-1].content.startswith("Order 42")


def test_full_replay_is_deterministic_and_offline() -> None:
    run, out = record(sa.build_graph(ScriptedModel(script=GOOD)))
    replayed, out2 = record(sa.build_graph(ReplayChatModel(run), pin_tools(exploding_tools(), run)))
    assert out2["messages"][-1].content == out["messages"][-1].content
    assert diff_runs(run, replayed).identical


def test_strict_replay_catches_a_prompt_change() -> None:
    run, _ = record(sa.build_graph(ScriptedModel(script=GOOD)))
    g = sa.build_graph(ReplayChatModel(run), pin_tools(exploding_tools(), run), system="Be brief.")
    with pytest.raises(ReplayDivergence) as e:
        g.invoke(Q)
    assert e.value.step is not None and e.value.step.id == run.of_kind("llm")[0].id
    assert any(line.startswith("+Be brief.") for line in e.value.delta)


def test_tools_only_replay_localises_a_regression() -> None:
    good, _ = record(sa.build_graph(ScriptedModel(script=GOOD)))
    # A regressed model skips the policy and answers wrongly.
    bad_script = [call("lookup_order", {"order_id": 42}), AIMessage(content="Yes, full refund for order 42.")]
    bad, _ = record(sa.build_graph(ScriptedModel(script=bad_script), pin_tools(exploding_tools(), good)))
    d = diff_runs(good, bad)
    assert d.first is not None and d.first.a is not None
    # The second model call got the same input but answered differently.
    assert d.first.a.kind == "llm" and d.first.output_differs
    assert d.first.a.id == good.of_kind("llm")[1].id


def test_pinned_tool_miss() -> None:
    good, _ = record(sa.build_graph(ScriptedModel(script=GOOD)))
    other = [call("lookup_order", {"order_id": 77}), AIMessage(content="done")]
    with pytest.raises(ReplayDivergence, match="order_id"):
        sa.build_graph(ScriptedModel(script=other), pin_tools(exploding_tools(), good)).invoke(Q)
    live = sa.build_graph(ScriptedModel(script=other), pin_tools(sa.TOOLS, good, on_miss="live"))
    run, _ = record(live)
    assert "keyboard" in run.of_kind("tool")[0].output


def test_replay_model_runs_out() -> None:
    run, _ = record(sa.build_graph(ScriptedModel(script=[AIMessage(content="hi")])))
    m = ReplayChatModel(run, strict=False)
    m.invoke([HumanMessage("a")])
    with pytest.raises(ReplayDivergence, match="made 1"):
        m.invoke([HumanMessage("b")])


def test_errors_are_recorded() -> None:
    @tool
    def broken(x: int) -> str:
        """Always fails."""
        raise ValueError("backend down")

    g = sa.build_graph(ScriptedModel(script=[call("broken", {"x": 1}), AIMessage(content="sorry")]), [broken])
    rec = Recorder()
    with pytest.raises(ValueError), rec:
        g.invoke(Q, config={"callbacks": [AgentTraceCallback(rec)]})
    t = rec.run.of_kind("tool")[0]
    assert rec.run.steps[0].error is not None  # the graph itself failed too
    assert t.error is not None and "backend down" in t.error
