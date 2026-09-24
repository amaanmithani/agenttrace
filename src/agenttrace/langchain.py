"""LangChain / LangGraph integration: a callback handler that records model calls,
tool calls and graph nodes into a Recorder.

    rec = Recorder("run.jsonl")
    with rec:
        graph.invoke(state, config={"callbacks": [AgentTraceCallback(rec)]})
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from agenttrace.messages import message_to_dict
from agenttrace.recorder import Recorder
from agenttrace.trace import Step


class AgentTraceCallback(BaseCallbackHandler):
    """Records LangGraph nodes as `node` steps, chat-model calls as `llm` steps and
    tool calls as `tool` steps, nested by LangChain's run tree. Chains that are not
    graph nodes (prompt templates, internal sequences) are not recorded; their
    children attach to the nearest recorded ancestor."""

    raise_error = False

    def __init__(self, recorder: Recorder, record_graph: bool = True) -> None:
        self.rec = recorder
        self.record_graph = record_graph
        self._steps: dict[UUID, Step] = {}
        self._done: dict[UUID, Step] = {}
        self._parents: dict[UUID, UUID | None] = {}

    def _parent(self, parent_run_id: UUID | None) -> int | None:
        seen = 0
        while parent_run_id is not None and seen < 1000:
            found = self._steps.get(parent_run_id) or self._done.get(parent_run_id)
            if found is not None:
                return found.id
            parent_run_id = self._parents.get(parent_run_id)
            seen += 1
        return None

    def _open(
        self, run_id: UUID, parent_run_id: UUID | None, kind: Any, name: str, inp: Any, **attrs: Any
    ) -> None:
        self._parents[run_id] = parent_run_id
        self._steps[run_id] = self.rec.begin(kind, name, inp, parent=self._parent(parent_run_id), **attrs)

    def _close(self, run_id: UUID, output: Any = None, error: BaseException | None = None) -> None:
        step = self._steps.pop(run_id, None)
        if step is not None:
            self.rec.end(step, output, error)
            # Keep it so children whose callbacks arrive late still resolve their parent.
            self._done[run_id] = step

    # graph nodes ---------------------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        node = (metadata or {}).get("langgraph_node")
        if parent_run_id is None and self.record_graph:
            self._open(run_id, None, "node", str(name), _plain(inputs), graph=True)
        elif node is not None and name == node:
            self._open(
                run_id,
                parent_run_id,
                "node",
                str(name),
                _plain(inputs),
                step=(metadata or {}).get("langgraph_step"),
            )
        else:
            self._parents[run_id] = parent_run_id

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, _plain(outputs))

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=error)

    # models --------------------------------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        invocation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params = invocation_params or kwargs.get("invocation_params") or {}
        inp: dict[str, Any] = {"messages": [message_to_dict(m) for m in (messages[0] if messages else [])]}
        tools = params.get("tools")
        if tools:
            inp["tools"] = sorted(_tool_name(t) for t in tools)
        model = params.get("model") or params.get("model_name") or (serialized or {}).get("name") or "chat"
        self._open(run_id, parent_run_id, "llm", str(model), inp)

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._open(
            run_id, parent_run_id, "llm", str((serialized or {}).get("name") or "llm"), {"prompts": prompts}
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        step = self._steps.get(run_id)
        out: Any = None
        gens = getattr(response, "generations", None) or [[]]
        first = gens[0][0] if gens and gens[0] else None
        if first is not None:
            msg = getattr(first, "message", None)
            if msg is not None:
                out = message_to_dict(msg)
                out.pop("role", None)
                usage = getattr(msg, "usage_metadata", None)
                if usage and step is not None:
                    step.attrs["usage"] = {k: usage.get(k) for k in ("input_tokens", "output_tokens")}
            else:
                out = getattr(first, "text", None)
        self._close(run_id, out)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=error)

    # tools ---------------------------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "tool"
        self._open(run_id, parent_run_id, "tool", str(name), inputs if inputs is not None else input_str)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, _plain(getattr(output, "content", output)))

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=error)


def _tool_name(t: Any) -> str:
    if isinstance(t, dict):
        fn = t.get("function")
        if isinstance(fn, dict) and "name" in fn:
            return str(fn["name"])
        return str(t.get("name", "?"))
    return str(getattr(t, "name", t))


def _plain(v: Any) -> Any:
    """Make graph state JSON-friendly: messages become dicts, other objects strings."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    if hasattr(v, "type") and hasattr(v, "content"):
        return message_to_dict(v)
    if v is None or isinstance(v, str | int | float | bool):
        return v
    if hasattr(v, "model_dump"):
        return _plain(v.model_dump())
    return str(v)
