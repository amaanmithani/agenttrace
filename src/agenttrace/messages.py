"""Serialising LangChain messages to plain dicts, without provider-generated ids
(tool-call ids are random per run and would make identical runs look different)."""

from __future__ import annotations

from typing import Any

ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool", "function": "tool"}


def message_to_dict(m: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"role": ROLES.get(getattr(m, "type", ""), getattr(m, "type", "unknown"))}
    d["content"] = getattr(m, "content", "")
    calls = getattr(m, "tool_calls", None)
    if calls:
        d["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in calls]
    name = getattr(m, "name", None)
    if name and d["role"] == "tool":
        d["name"] = name
    return d


def dict_to_ai_message(d: Any, tag: str) -> Any:
    """Rebuild an AIMessage from a recorded output, with deterministic tool-call ids."""
    from langchain_core.messages import AIMessage

    if isinstance(d, str):
        return AIMessage(content=d)
    calls = [
        {"name": c["name"], "args": c["args"], "id": f"call_{tag}_{i}", "type": "tool_call"}
        for i, c in enumerate(d.get("tool_calls") or [])
    ]
    return AIMessage(content=d.get("content", ""), tool_calls=calls)
