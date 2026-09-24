"""Regression tests for the adversarial review findings."""

import asyncio
import json
import threading
from pathlib import Path
from typing import Any

import pytest
import support_agent as sa
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from agenttrace import Recorder, diff_runs
from agenttrace.langchain import AgentTraceCallback
from agenttrace.mcp_proxy import McpReplayer, McpTap
from agenttrace.messages import messages_to_dicts
from agenttrace.otel import to_otlp
from agenttrace.recorder import traced
from agenttrace.replay import pin_tools
from agenttrace.trace import Run, Step
from conftest import ScriptedModel, call


def test_redaction_never_touches_live_objects_or_leaks_open_steps(tmp_path: Path) -> None:
    def redact(step: Step) -> None:
        for v in (step.input, step.output):
            if isinstance(v, dict) and "key" in v:
                v["key"] = "***"

    payload = {"key": "sk-SECRET"}
    rec = Recorder(tmp_path / "r.jsonl", redact=redact)
    with rec.step("tool", "t", payload) as s:
        s.set_output({"key": "sk-OUT"})
        open_step = rec.begin("tool", "inflight", {"key": "sk-INFLIGHT"})
        rec.save()
    assert payload == {"key": "sk-SECRET"}
    rec.end(open_step)
    rec.save()
    text = (tmp_path / "r.jsonl").read_text()
    assert "sk-" not in text and text.count("***") == 3


def test_mcp_error_output_is_redacted() -> None:
    def redact(step: Step) -> None:
        step.output = "REDACTED"

    rec = Recorder(redact=redact)
    tap = McpTap(rec)
    tap.client_message({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "q"}})
    tap.server_message({"jsonrpc": "2.0", "id": 1, "error": {"code": 1, "message": "db password=hunter2"}})
    assert rec.run.steps[0].output == "REDACTED"


def test_traced_async_methods_and_metadata() -> None:
    class Svc:
        @traced("tool")
        def get(self, key: str) -> str:
            """Doc."""
            return key.upper()

        @traced("tool", "aget")
        async def aget(self, key: str) -> str:
            await asyncio.sleep(0)
            if key == "boom":
                raise ValueError("boom")
            return key

    assert Svc.get.__doc__ == "Doc." and Svc.get.__name__ == "get"
    with Recorder() as rec:
        Svc().get("a")
        Svc().get("a")
        assert asyncio.run(Svc().aget("x")) == "x"
        with pytest.raises(ValueError):
            asyncio.run(Svc().aget("boom"))
    get1, get2, ax, boom = rec.run.steps
    assert get1.input == {"key": "a"} and get1.signature() == get2.signature()
    assert ax.output == "x" and ax.name == "aget"
    assert boom.error is not None and "boom" in boom.error


def test_bind_carries_recorder_and_parent_into_threads() -> None:
    @traced("tool", "work")
    def work() -> int:
        return 1

    with Recorder() as rec, rec.step("node", "outer") as outer:
        t = threading.Thread(target=rec.bind(work))
        t.start()
        t.join()
    assert [(s.name, s.parent) for s in rec.run] == [("outer", None), ("work", outer.id)]


def test_pinned_tools_accept_defaults_and_coercion() -> None:
    @tool
    def lookup(order_id: int, verbose: bool = False) -> str:
        """Look up."""
        raise AssertionError("live")

    run = Run([Step(1, "tool", "lookup", {"order_id": 42}, "found")])
    (pinned,) = pin_tools([lookup], run)
    assert pinned.invoke({"order_id": 42}) == "found"
    assert pinned.invoke({"order_id": "42", "verbose": False}) == "found"


def test_messages_keep_what_distinguishes_runs() -> None:
    ai = AIMessage(
        content="",
        tool_calls=[{"name": "t", "args": {"i": 1}, "id": "r1"}, {"name": "t", "args": {"i": 2}, "id": "r2"}],
    )
    a = messages_to_dicts([ai, ToolMessage("one", tool_call_id="r1"), ToolMessage("two", tool_call_id="r2")])
    b = messages_to_dicts([ai, ToolMessage("one", tool_call_id="r2"), ToolMessage("two", tool_call_id="r1")])
    assert a != b
    ok = messages_to_dicts([ToolMessage("x", tool_call_id="1")])
    err = messages_to_dicts([ToolMessage("x", tool_call_id="1", status="error")])
    assert ok != err
    assert messages_to_dicts([HumanMessage("hi", name="alice")]) != messages_to_dicts(
        [HumanMessage("hi", name="bob")]
    )
    bad = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "t", "args": "{", "id": "z", "error": "bad json", "type": "invalid_tool_call"}
        ],
    )
    assert messages_to_dicts([bad])[0]["invalid_tool_calls"][0]["error"] == "bad json"
    # Random provider ids still don't matter.
    other = AIMessage(content="", tool_calls=[{"name": "t", "args": {"i": 1}, "id": "zz"}])
    same = AIMessage(content="", tool_calls=[{"name": "t", "args": {"i": 1}, "id": "yy"}])
    assert messages_to_dicts([other]) == messages_to_dicts([same])


def test_changed_tool_schema_changes_the_signature() -> None:
    @tool
    def refund_policy() -> str:
        """A different description."""
        return sa.POLICY

    def record(tools: list[Any]) -> Run:
        rec = Recorder()
        script = [AIMessage(content="ok")]
        with rec:
            sa.build_graph(ScriptedModel(script=script), tools).invoke(
                {"messages": [HumanMessage("q")]}, config={"callbacks": [AgentTraceCallback(rec)]}
            )
        return rec.run

    d = diff_runs(record(sa.TOOLS), record([sa.lookup_order, refund_policy]))
    assert not d.identical and d.first is not None and d.first.op == "changed"


def test_mcp_replay_falls_back_only_for_discovery() -> None:
    run = Run(
        [
            Step(
                1,
                "mcp",
                "resources/read",
                {"uri": "file:///a.txt"},
                {"text": "A"},
                attrs={"method": "resources/read"},
            ),
            Step(2, "mcp", "tools/list", None, {"tools": []}, attrs={"method": "tools/list"}),
        ]
    )
    rep = McpReplayer(run)
    rd = {"jsonrpc": "2.0", "id": 1, "method": "resources/read", "params": {"uri": "file:///b.txt"}}
    assert "error" in rep.handle(rd)
    meta = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/read",
        "params": {"uri": "file:///a.txt", "_meta": {"progressToken": 9}},
    }
    assert rep.handle(meta)["result"] == {"text": "A"}
    assert rep.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {"cursor": "x"}})[
        "result"
    ] == {"tools": []}
    assert rep.fallbacks == ["tools/list"] and len(rep.misses) == 1


def test_http_tap_scopes_ids_per_request() -> None:
    rec = Recorder()
    tap = McpTap(rec)
    x, y = object(), object()
    tap.client_message({"jsonrpc": "2.0", "id": 1, "method": "a"}, x)
    tap.client_message({"jsonrpc": "2.0", "id": 1, "method": "b"}, y)
    tap.server_message({"jsonrpc": "2.0", "id": 1, "result": "for-b"}, y)
    tap.server_message({"jsonrpc": "2.0", "id": 1, "result": "for-a"}, x)
    assert [(s.name, s.output) for s in rec.run] == [("a", "for-a"), ("b", "for-b")]


def test_diff_keeps_order_inside_edit_blocks() -> None:
    def run(*steps: tuple[str, str, object]) -> Run:
        return Run([Step(i + 1, k, n, inp, "o") for i, (k, n, inp) in enumerate(steps)])  # type: ignore[arg-type]

    d = diff_runs(run(("tool", "search", "a")), run(("tool", "lookup", 1), ("tool", "search", "b")))
    assert [e.op for e in d.entries] == ["added", "changed"]
    assert d.first is not None and d.first.b is not None and d.first.b.name == "lookup"
    d2 = diff_runs(run(("llm", "m", 1), ("tool", "t", 1)), run(("tool", "t", 2), ("llm", "m", 2)))
    assert d2.first is not None and d2.first.b is not None and d2.first.b.id == 1


def test_otlp_trace_ids_differ_and_semconv_shapes() -> None:
    a = Run(
        [
            Step(
                1,
                "llm",
                "m",
                {"messages": [{"role": "user", "content": "hi"}], "tools": ["t"]},
                {"content": "x"},
            )
        ]
    )
    b = Run(
        [
            Step(
                1,
                "llm",
                "m",
                {"messages": [{"role": "user", "content": "hi"}], "tools": ["t"]},
                {"content": "y"},
            )
        ]
    )
    sa_, sb = (to_otlp(r)["resourceSpans"][0]["scopeSpans"][0]["spans"][0] for r in (a, b))
    assert sa_["traceId"] != sb["traceId"] and sa_["kind"] == 3
    attrs = {x["key"]: x["value"]["stringValue"] for x in sa_["attributes"] if "stringValue" in x["value"]}
    assert json.loads(attrs["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "hi"}]}
    ]
    assert json.loads(attrs["gen_ai.output.messages"])[0]["parts"][0]["content"] == "x"
    custom = to_otlp(Run([Step(1, "custom", "c")]))["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert custom["name"] == "c" and all(x["key"] != "gen_ai.operation.name" for x in custom["attributes"])


def test_replay_model_reset() -> None:
    from agenttrace.replay import ReplayChatModel

    run = Run([Step(1, "llm", "m", {"messages": [{"role": "user", "content": "a"}]}, {"content": "x"})])
    m = ReplayChatModel(run)
    assert m.invoke([HumanMessage("a")]).content == "x"
    m.reset()
    assert m.invoke([HumanMessage("a")]).content == "x"
    _ = call
