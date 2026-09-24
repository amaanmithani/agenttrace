"""Export a run as OTLP/JSON spans with OpenTelemetry GenAI semantic-convention attributes,
so it can be loaded into any OTLP backend (Jaeger, Tempo, Honeycomb, ...)."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from agenttrace.trace import Run, Step, canonical

OPERATION = {"llm": "chat", "tool": "execute_tool", "mcp": "execute_tool", "node": "invoke_agent"}
SPAN_KIND = {"llm": 3}  # CLIENT for model calls; everything else INTERNAL (1)


def _parts(m: Any) -> dict[str, Any]:
    """One message in the GenAI semconv shape: {role, parts: [...]}."""
    if not isinstance(m, dict):
        return {"role": "assistant", "parts": [{"type": "text", "content": m}]}
    parts: list[dict[str, Any]] = []
    if m.get("role") == "tool":
        parts.append(
            {"type": "tool_call_response", "id": m.get("tool_call_id"), "response": m.get("content")}
        )
    elif m.get("content") not in (None, ""):
        parts.append({"type": "text", "content": m.get("content")})
    for c in m.get("tool_calls") or []:
        parts.append(
            {"type": "tool_call", "id": c.get("id"), "name": c.get("name"), "arguments": c.get("args")}
        )
    return {"role": m.get("role", "assistant"), "parts": parts}


def input_messages(inp: Any) -> Any:
    if isinstance(inp, dict) and isinstance(inp.get("messages"), list):
        return [_parts(m) for m in inp["messages"]]
    return None


def output_messages(out: Any) -> Any:
    if out is None:
        return None
    return [_parts({"role": "assistant", **out} if isinstance(out, dict) else out)]


def _id(seed: str, n: int) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[: n * 2]


def _attr(key: str, value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": value if isinstance(value, str) else canonical(value)}}


def _attrs(s: Step) -> list[dict[str, Any]]:
    a: dict[str, Any] = {"agenttrace.step.id": s.id, "agenttrace.kind": s.kind}
    if s.kind in OPERATION:
        a["gen_ai.operation.name"] = OPERATION[s.kind]
    if s.attrs.get("provider"):
        a["gen_ai.provider.name"] = s.attrs["provider"]
    if s.kind == "llm":
        a["gen_ai.request.model"] = s.name
        usage = s.attrs.get("usage") or {}
        if usage.get("input_tokens") is not None:
            a["gen_ai.usage.input_tokens"] = int(usage["input_tokens"])
        if usage.get("output_tokens") is not None:
            a["gen_ai.usage.output_tokens"] = int(usage["output_tokens"])
        a["gen_ai.input.messages"] = input_messages(s.input)
        a["gen_ai.output.messages"] = output_messages(s.output)
        tools = s.input.get("tools") if isinstance(s.input, dict) else None
        if tools:
            a["gen_ai.tool.definitions"] = [{"type": "function", "name": t} for t in tools]
    elif s.kind in ("tool", "mcp"):
        a["gen_ai.tool.name"] = s.name
        a["gen_ai.tool.call.arguments"] = s.input
        a["gen_ai.tool.call.result"] = s.output
    elif s.kind == "node":
        a["gen_ai.agent.name"] = s.name
    if s.error:
        a["error.type"] = s.error.split(":")[0]
    return [_attr(k, v) for k, v in a.items() if v is not None]


def span_name(s: Step) -> str:
    op = OPERATION.get(s.kind)
    return f"{op} {s.name}" if op else s.name


def to_otlp(run: Run, service: str = "agent", epoch_ns: int | None = None) -> dict[str, Any]:
    """Step timestamps are monotonic (perf_counter); they are shifted so the run
    ends at `epoch_ns` (default: now)."""
    # The trace id is a hash of the whole run, so two runs never share one.
    trace_id = _id("\n".join(run.lines()), 16)
    last = max((s.end_ns for s in run.steps), default=0)
    shift = (epoch_ns if epoch_ns is not None else time.time_ns()) - last
    spans = [
        {
            "traceId": trace_id,
            "spanId": _id(f"{trace_id}/{s.id}", 8),
            "parentSpanId": _id(f"{trace_id}/{s.parent}", 8) if s.parent is not None else "",
            "name": span_name(s),
            "kind": SPAN_KIND.get(s.kind, 1),
            "startTimeUnixNano": str(s.start_ns + shift),
            "endTimeUnixNano": str(max(s.end_ns, s.start_ns) + shift),
            "attributes": _attrs(s),
            "status": {"code": 2, "message": s.error} if s.error else {"code": 0},
        }
        for s in run.steps
    ]
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": [_attr("service.name", service)]},
                "scopeSpans": [{"scope": {"name": "agenttrace"}, "spans": spans}],
            }
        ]
    }
