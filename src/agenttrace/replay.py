"""Replaying a recorded run.

- `Pins`: framework-free lookup of recorded tool (and MCP) responses by name and arguments.
- `pin_tools`: wrap LangChain tools so they return recorded responses instead of running.
- `ReplayChatModel`: a chat model that returns the recorded model responses in order.

Tools-only replay (pinned tools, live model) tests a prompt or model change against
exactly the tool behaviour of the recorded run. Full replay (both pinned) re-runs the
agent deterministically with no network.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from typing import Any, Literal

from agenttrace.diff import delta
from agenttrace.messages import dict_to_ai_message, messages_to_dicts
from agenttrace.trace import Run, Step, canonical

OnMiss = Literal["error", "live"]


class ReplayDivergence(Exception):
    """The replayed agent asked for something the recorded run never did."""

    def __init__(self, message: str, step: Step | None = None, delta_lines: Sequence[str] = ()) -> None:
        super().__init__(message + ("\n" + "\n".join(delta_lines) if delta_lines else ""))
        self.step = step
        self.delta = list(delta_lines)


def _tool_key(name: str, args: Any) -> str:
    return f"{name}\0{canonical(args)}"


class Pins:
    """Recorded responses by (tool name, arguments). Repeated identical calls are
    answered in recorded order; once exhausted, the last response repeats."""

    def __init__(
        self,
        run: Run,
        kinds: tuple[str, ...] = ("tool", "mcp"),
        normalize: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self._q: dict[str, deque[Step]] = defaultdict(deque)
        self._last: dict[str, Step] = {}
        self.normalize = normalize or (lambda _name, args: args)
        self.hits = 0
        self.misses = 0
        for s in run.steps:
            if s.kind in kinds:
                self._q[_tool_key(s.name, self.normalize(s.name, s.input))].append(s)

    def lookup(self, name: str, args: Any) -> Step | None:
        k = _tool_key(name, self.normalize(name, args))
        q = self._q.get(k)
        s: Step | None
        if q:
            s = q.popleft()
            self._last[k] = s
        else:
            s = self._last.get(k)
        if s is None:
            self.misses += 1
        else:
            self.hits += 1
        return s

    def respond(self, name: str, args: Any) -> Any:
        s = self.lookup(name, args)
        if s is None:
            raise ReplayDivergence(f"tool call not in the recorded run: {name}({canonical(args)})")
        if s.error:
            raise RuntimeError(f"recorded error from {name}: {s.error}")
        return s.output


def pin_tools(tools: Sequence[Any], run: Run, on_miss: OnMiss = "error") -> list[Any]:
    """LangChain tools with the same names and schemas that answer from `run`."""
    from langchain_core.tools import StructuredTool

    schemas = {t.name: t.args_schema for t in tools}

    def normalize(name: str, args: Any) -> Any:
        # Recorded inputs are what the model sent; calls arrive with defaults filled
        # in and types coerced. Put both through the tool's schema to compare them.
        schema = schemas.get(name)
        if not isinstance(args, dict) or schema is None or not hasattr(schema, "model_validate"):
            return args
        try:
            return schema.model_validate(args).model_dump(mode="json")
        except Exception:
            return args

    pins = Pins(run, normalize=normalize)
    out = []
    for t in tools:

        def call(_t: Any = t, **kwargs: Any) -> Any:
            s = pins.lookup(_t.name, kwargs)
            if s is None:
                if on_miss == "live":
                    return _t.invoke(kwargs)
                raise ReplayDivergence(f"tool call not in the recorded run: {_t.name}({canonical(kwargs)})")
            if s.error:
                raise RuntimeError(f"recorded error from {_t.name}: {s.error}")
            return s.output

        out.append(
            StructuredTool.from_function(
                func=call,
                name=t.name,
                description=t.description,
                args_schema=t.args_schema,
                infer_schema=False,
            )
        )
    return out


def ReplayChatModel(run: Run, strict: bool = True) -> Any:
    """A LangChain chat model answering with `run`'s recorded model outputs, in order.
    With `strict`, each request must match the recorded one (else ReplayDivergence
    with a diff of the prompts)."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool

    steps = run.of_kind("llm")

    class _Replay(BaseChatModel):
        position: int = 0

        @property
        def _llm_type(self) -> str:
            return "agenttrace-replay"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            # Report the recorded model's name, so a replayed run diffs clean against the original.
            # ...and its sampling parameters, which are part of the recorded request.
            nxt = steps[min(self.position, len(steps) - 1)] if steps else None
            params = nxt.input.get("params", {}) if nxt and isinstance(nxt.input, dict) else {}
            return {**params, "model": nxt.name if nxt else "replay"}

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools])

        def reset(self) -> None:
            """Start again from the first recorded model call (for a second invoke)."""
            self.position = 0

        def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
            if self.position >= len(steps):
                raise ReplayDivergence(
                    f"model call #{self.position + 1} but the recorded run made {len(steps)}"
                )
            rec = steps[self.position]
            self.position += 1
            if strict:
                got = messages_to_dicts(messages)
                want = rec.input.get("messages") if isinstance(rec.input, dict) else None
                want_tools = rec.input.get("tools") if isinstance(rec.input, dict) else None
                got_tools = sorted(t["function"]["name"] for t in kw.get("tools") or []) or None
                if want_tools is not None and got_tools != want_tools:
                    raise ReplayDivergence(
                        f"model call #{self.position} offers tools {got_tools}, "
                        f"recorded step {rec.id} offered {want_tools}",
                        rec,
                    )
                if canonical(got) != canonical(want):
                    probe = Step(0, "llm", rec.name, input={"messages": got})
                    raise ReplayDivergence(
                        f"model call #{self.position} differs from recorded step {rec.id}",
                        rec,
                        delta(Step(rec.id, "llm", rec.name, input={"messages": want}), probe, "input"),
                    )
            if rec.error:
                raise RuntimeError(f"recorded model error: {rec.error}")
            msg = dict_to_ai_message(rec.output, str(rec.id))
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return _Replay()
