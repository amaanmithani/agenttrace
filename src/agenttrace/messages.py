"""Serialising LangChain messages to plain dicts.

Provider tool-call ids are random per run, so recording them verbatim would make
identical runs look different. They are replaced by their position in the
conversation (`#0`, `#1`, ...), which keeps the pairing between a call and its
result (swapped results still show up) without the randomness."""

from __future__ import annotations

from typing import Any

ROLES = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool", "function": "tool"}


class IdMap:
    """Maps provider tool-call ids to stable positional labels within one conversation."""

    def __init__(self) -> None:
        self._ids: dict[str, str] = {}

    def __call__(self, raw: Any) -> str | None:
        if raw is None:
            return None
        key = str(raw)
        if key not in self._ids:
            self._ids[key] = f"#{len(self._ids)}"
        return self._ids[key]


def message_to_dict(m: Any, ids: IdMap | None = None) -> dict[str, Any]:
    ids = ids or IdMap()
    d: dict[str, Any] = {"role": ROLES.get(getattr(m, "type", ""), getattr(m, "type", "unknown"))}
    d["content"] = getattr(m, "content", "")
    calls = getattr(m, "tool_calls", None)
    if calls:
        d["tool_calls"] = [{"name": c["name"], "args": c["args"], "id": ids(c.get("id"))} for c in calls]
    invalid = getattr(m, "invalid_tool_calls", None)
    if invalid:
        d["invalid_tool_calls"] = [
            {"name": c.get("name"), "args": c.get("args"), "error": c.get("error")} for c in invalid
        ]
    if getattr(m, "tool_call_id", None) is not None:
        d["tool_call_id"] = ids(m.tool_call_id)
    status = getattr(m, "status", None)
    if status and status != "success":
        d["status"] = status
    name = getattr(m, "name", None)
    if name:
        d["name"] = name
    return d


def messages_to_dicts(messages: Any) -> list[dict[str, Any]]:
    ids = IdMap()
    return [message_to_dict(m, ids) for m in messages]


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
