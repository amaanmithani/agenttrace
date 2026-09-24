"""Recording steps. Framework-free: integrations call `Recorder.step()`."""

from __future__ import annotations

import contextvars
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any

from agenttrace.trace import Kind, Run, Step

_current: contextvars.ContextVar[Recorder | None] = contextvars.ContextVar(
    "agenttrace_recorder", default=None
)
_parent: contextvars.ContextVar[int | None] = contextvars.ContextVar("agenttrace_parent", default=None)

Redactor = Callable[[Step], None]


def current_recorder() -> Recorder | None:
    return _current.get()


class StepHandle:
    """A step in progress; set `output` (and optionally attrs) before it closes."""

    def __init__(self, step: Step) -> None:
        self.step = step

    @property
    def id(self) -> int:
        return self.step.id

    def set_output(self, output: Any) -> None:
        self.step.output = output

    def set_attr(self, key: str, value: Any) -> None:
        self.step.attrs[key] = value


class Recorder:
    """Collects steps for one run. Thread- and asyncio-safe: the parent of a new
    step is the innermost open step in the current context.

        with Recorder("run.jsonl", meta={"agent": "support-bot"}) as rec:
            with rec.step("tool", "search", input={"q": "refund"}) as s:
                s.set_output(search("refund"))
    """

    def __init__(
        self,
        path: str | Path | None = None,
        meta: dict[str, Any] | None = None,
        redact: Redactor | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.run = Run(meta=dict(meta or {}))
        self.redact = redact
        self._lock = threading.Lock()
        self._next = 0
        self._token: contextvars.Token[Recorder | None] | None = None

    def _new_id(self) -> int:
        with self._lock:
            self._next += 1
            return self._next

    def begin(
        self, kind: Kind, name: str, input: Any = None, parent: int | None = None, **attrs: Any
    ) -> Step:
        """Open a step explicitly (for callback-style integrations); close it with `end`."""
        step = Step(
            id=self._new_id(),
            kind=kind,
            name=name,
            input=input,
            parent=parent if parent is not None else _parent.get(),
            start_ns=time.perf_counter_ns(),
            attrs=dict(attrs),
        )
        with self._lock:
            self.run.steps.append(step)
        return step

    def end(self, step: Step, output: Any = None, error: BaseException | str | None = None) -> None:
        step.end_ns = time.perf_counter_ns()
        if output is not None:
            step.output = output
        if error is not None:
            step.error = (
                error if isinstance(error, str) else "".join(traceback.format_exception_only(error)).strip()
            )
        if self.redact:
            self.redact(step)

    @contextmanager
    def step(self, kind: Kind, name: str, input: Any = None, **attrs: Any) -> Iterator[StepHandle]:
        s = self.begin(kind, name, input, **attrs)
        token = _parent.set(s.id)
        try:
            yield StepHandle(s)
        except BaseException as e:
            self.end(s, error=e)
            raise
        else:
            self.end(s)
        finally:
            _parent.reset(token)

    def save(self) -> None:
        if self.path:
            with self._lock:
                self.run.steps.sort(key=lambda s: s.id)
                self.run.save(self.path)

    def __enter__(self) -> Recorder:
        self._token = _current.set(self)
        return self

    def __exit__(
        self, et: type[BaseException] | None, e: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None
        self.save()


def traced(
    kind: Kind = "custom", name: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: record calls to a function as steps of the current recorder (no-op without one)."""

    def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        label = name or fn.__qualname__

        def inner(*args: Any, **kwargs: Any) -> Any:
            rec = current_recorder()
            if rec is None:
                return fn(*args, **kwargs)
            with rec.step(kind, label, {"args": list(args), "kwargs": kwargs}) as s:
                out = fn(*args, **kwargs)
                s.set_output(out)
                return out

        inner.__wrapped__ = fn  # type: ignore[attr-defined]
        inner.__name__ = fn.__name__
        return inner

    return wrap
