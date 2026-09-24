"""Export a run as OTLP/JSON spans with OpenTelemetry GenAI semantic-convention attributes,
so it can be loaded into any OTLP backend (Jaeger, Tempo, Honeycomb, ...)."""

from __future__ import annotations

import hashlib
import time
from typing import Any

from agenttrace.trace import Run, Step, canonical

OPERATION = {
    "llm": "chat",
    "tool": "execute_tool",
    "mcp": "execute_tool",
    "node": "invoke_agent",
    "custom": "custom",
}


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
    op = OPERATION.get(s.kind, "custom")
    a: dict[str, Any] = {"gen_ai.operation.name": op, "agenttrace.step.id": s.id, "agenttrace.kind": s.kind}
    if s.kind == "llm":
        a["gen_ai.request.model"] = s.name
        usage = s.attrs.get("usage") or {}
        if usage.get("input_tokens") is not None:
            a["gen_ai.usage.input_tokens"] = int(usage["input_tokens"])
        if usage.get("output_tokens") is not None:
            a["gen_ai.usage.output_tokens"] = int(usage["output_tokens"])
        a["gen_ai.input.messages"] = s.input
        a["gen_ai.output.messages"] = s.output
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
    op = OPERATION.get(s.kind, "custom")
    return f"{op} {s.name}"


def to_otlp(run: Run, service: str = "agent", epoch_ns: int | None = None) -> dict[str, Any]:
    """Step timestamps are monotonic (perf_counter); they are shifted so the run
    ends at `epoch_ns` (default: now)."""
    seed = canonical(run.meta) + str(len(run.steps)) + (run.steps[0].signature() if run.steps else "")
    trace_id = _id(seed, 16)
    last = max((s.end_ns for s in run.steps), default=0)
    shift = (epoch_ns if epoch_ns is not None else time.time_ns()) - last
    spans = [
        {
            "traceId": trace_id,
            "spanId": _id(f"{trace_id}/{s.id}", 8),
            "parentSpanId": _id(f"{trace_id}/{s.parent}", 8) if s.parent is not None else "",
            "name": span_name(s),
            "kind": 1,
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
