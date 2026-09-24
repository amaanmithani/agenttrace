import asyncio
import json
import threading
from pathlib import Path

import pytest

from agenttrace import Recorder, current_recorder
from agenttrace.cli import main, show
from agenttrace.otel import to_otlp
from agenttrace.recorder import traced
from agenttrace.replay import Pins, ReplayDivergence
from agenttrace.trace import Run, Step


def sample(path: Path | None = None, answer: str = "4") -> Run:
    with Recorder(path, meta={"agent": "t"}) as rec, rec.step("node", "agent", {"q": "2+2"}) as n:
        with rec.step("llm", "m", {"messages": [{"role": "user", "content": "2+2"}]}) as s:
            s.set_output({"content": "", "tool_calls": [{"name": "add", "args": {"a": 2, "b": 2}}]})
            s.set_attr("usage", {"input_tokens": 5, "output_tokens": 3})
        with rec.step("tool", "add", {"a": 2, "b": 2}) as s:
            s.set_output(answer)
        n.set_output(answer)
    return rec.run


def test_nesting_ids_and_file_roundtrip(tmp_path: Path) -> None:
    run = sample(tmp_path / "r.jsonl")
    assert [s.parent for s in run] == [None, 1, 1]
    again = Run.load(tmp_path / "r.jsonl")
    assert [s.to_json() for s in again] == [s.to_json() for s in run]
    assert current_recorder() is None
    assert len(again) == 3 and again.children(1)[1].name == "add"


def test_errors_propagate_and_are_recorded() -> None:
    rec = Recorder()
    with pytest.raises(ZeroDivisionError), rec, rec.step("tool", "div", {"x": 0}):
        _ = 1 / 0
    assert rec.run.steps[0].error is not None and "ZeroDivisionError" in rec.run.steps[0].error


def test_parents_follow_threads_and_tasks() -> None:
    rec = Recorder()

    def worker(i: int) -> None:
        with rec.step("custom", f"w{i}") as w, rec.step("tool", "t", {"i": i}):
            assert w.id > 0

    async def task(i: int) -> None:
        with rec.step("custom", f"a{i}"):
            await asyncio.sleep(0)
            with rec.step("tool", "t", {"a": i}):
                await asyncio.sleep(0)

    with rec:
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        async def all_tasks() -> None:
            await asyncio.gather(*(task(i) for i in range(8)))

        asyncio.run(all_tasks())
    by = {s.id: s for s in rec.run}
    for s in rec.run.of_kind("tool"):
        parent = by[s.parent]  # type: ignore[index]
        key = "i" if "i" in s.input else "a"
        assert parent.name == ("w" if key == "i" else "a") + str(s.input[key])


def test_traced_decorator_and_redaction() -> None:
    @traced("tool", "lookup")
    def lookup(key: str, secret: str = "") -> str:
        return key.upper()

    assert lookup("x") == "X"  # no recorder: plain call

    def redact(step: Step) -> None:
        if isinstance(step.input, dict):
            step.input.pop("secret", None)

    with Recorder(redact=redact) as rec:
        assert lookup("y", secret="hunter2") == "Y"
    s = rec.run.steps[0]
    assert s.name == "lookup" and s.output == "Y" and "hunter2" not in json.dumps(s.to_json())


def test_parse_errors() -> None:
    with pytest.raises(ValueError, match="line 1: not JSON"):
        Run.parse("{")
    with pytest.raises(ValueError, match="unsupported format"):
        Run.parse('{"format": "other/9"}')
    with pytest.raises(ValueError, match="expected an object"):
        Run.parse("[1]")
    with pytest.raises(ValueError, match="line 2: bad step"):
        Run.parse('{"format": "agenttrace/1"}\n{"kind": "tool"}')
    assert len(Run.parse("\n\n")) == 0


def test_pins() -> None:
    run = sample()
    pins = Pins(run)
    assert pins.respond("add", {"b": 2, "a": 2}) == "4"
    assert pins.respond("add", {"a": 2, "b": 2}) == "4"  # exhausted: repeats
    with pytest.raises(ReplayDivergence):
        pins.respond("add", {"a": 1, "b": 2})
    assert (pins.hits, pins.misses) == (2, 1)
    bad = Run([Step(1, "tool", "t", {}, None, error="boom")])
    with pytest.raises(RuntimeError, match="boom"):
        Pins(bad).respond("t", {})


def test_otlp_export() -> None:
    run = sample()
    run.steps[2].error = "ValueError: x"
    doc = to_otlp(run, "svc", epoch_ns=10**18)
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert [s["name"] for s in spans] == ["invoke_agent agent", "chat m", "execute_tool add"]
    assert spans[1]["parentSpanId"] == spans[0]["spanId"] and spans[0]["parentSpanId"] == ""
    attrs = {a["key"]: a["value"] for a in spans[1]["attributes"]}
    assert attrs["gen_ai.request.model"] == {"stringValue": "m"}
    assert attrs["gen_ai.usage.input_tokens"] == {"intValue": "5"}
    assert max(int(s["endTimeUnixNano"]) for s in spans) == 10**18
    assert spans[2]["status"]["code"] == 2
    tool_attrs = {a["key"]: a["value"] for a in spans[2]["attributes"]}
    assert tool_attrs["error.type"] == {"stringValue": "ValueError"}
    assert to_otlp(Run())["resourceSpans"][0]["scopeSpans"][0]["spans"] == []


def test_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    sample(a)
    sample(b, answer="5")
    assert main(["show", str(a)]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[1].startswith("  #2 llm m")
    assert main(["diff", str(a), str(a)]) == 0
    assert "identical" in capsys.readouterr().out
    assert main(["diff", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "first divergence: tool add (A #3 / B #3): same input, different result" in out
    assert "    -4\n    +5" in out
    assert main(["diff", str(a), str(b), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["first_divergence"]["a"]["name"] == "add"
    assert main(["diff", str(a), str(a), "--json"]) == 0
    capsys.readouterr()
    assert main(["export-otlp", str(a), "-o", str(tmp_path / "o.json")]) == 0
    assert "resourceSpans" in (tmp_path / "o.json").read_text()
    assert main(["export-otlp", str(a)]) == 0
    assert "resourceSpans" in capsys.readouterr().out
    assert main(["show", str(tmp_path / "missing.jsonl")]) == 2
    (tmp_path / "bad.jsonl").write_text("nope")
    assert main(["diff", str(a), str(tmp_path / "bad.jsonl")]) == 2
    with pytest.raises(SystemExit):
        main(["mcp-proxy", "--record", str(tmp_path / "x.jsonl")])
    with pytest.raises(SystemExit):
        main(["mcp-proxy", "--record", str(tmp_path / "x.jsonl"), "--listen", "1"])


def test_show_orphans() -> None:
    run = Run([Step(1, "tool", "t", parent=99), Step(2, "llm", "m", error="x")])
    text = show(run)
    assert "#1 tool t" in text and "ERROR x" in text
