"""An MCP proxy that records JSON-RPC traffic between a client and a server, and
replays a recording without the server.

stdio:   agenttrace mcp-proxy --record run.jsonl -- python server.py
         agenttrace mcp-proxy --replay run.jsonl
HTTP:    agenttrace mcp-proxy --record run.jsonl --listen 8765 --upstream http://localhost:8000/mcp
         agenttrace mcp-proxy --replay run.jsonl --listen 8765

Point the MCP client at the proxy command (stdio) or http://127.0.0.1:8765/mcp (HTTP).
Each request/response pair becomes an `mcp` step: `name` is the tool name for
tools/call (so tool pins and diffs line up with LangChain tool steps) and the
method otherwise; `attrs.method` always holds the method.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import IO, Any

from agenttrace.recorder import Recorder
from agenttrace.trace import Run, Step, canonical

JsonObj = dict[str, Any]


def _name_and_input(method: str, params: Any) -> tuple[str, Any]:
    if isinstance(params, dict) and "_meta" in params:
        # Per-request metadata (progress tokens) differs every call and isn't part of the request.
        params = {k: v for k, v in params.items() if k != "_meta"}
    if method == "tools/call" and isinstance(params, dict):
        return str(params.get("name", "?")), params.get("arguments") or {}
    return method, params


class McpTap:
    """Pairs requests with responses by JSON-RPC id and records them as steps.

    `scope` separates id spaces: over HTTP each POST carries its own responses, so
    two clients (or sessions) that both use id 1 don't get each other's results.
    Writes to disk are throttled to one per `save_every` seconds; call `flush()` at exit.
    """

    def __init__(self, recorder: Recorder, save_every: float = 0.5) -> None:
        self.rec = recorder
        self.save_every = save_every
        self._open: dict[tuple[Any, str], Step] = {}
        self._lock = threading.Lock()
        self._saved = 0.0

    def client_message(self, msg: Any, scope: Any = None) -> None:
        for m in msg if isinstance(msg, list) else [msg]:
            if isinstance(m, dict) and "method" in m and "id" in m:
                name, inp = _name_and_input(str(m["method"]), m.get("params"))
                step = self.rec.begin("mcp", name, inp, method=m["method"])
                with self._lock:
                    self._open[(scope, canonical(m["id"]))] = step

    def server_message(self, msg: Any, scope: Any = None) -> None:
        for m in msg if isinstance(msg, list) else [msg]:
            if not isinstance(m, dict) or "id" not in m or "method" in m:
                continue
            with self._lock:
                step = self._open.pop((scope, canonical(m["id"])), None)
            if step is None:
                continue
            if "error" in m:
                err = m["error"]
                message = str(err.get("message", err)) if isinstance(err, dict) else str(err)
                self.rec.end(step, output=err, error=message)
            else:
                self.rec.end(step, m.get("result"))
        now = time.monotonic()
        if now - self._saved >= self.save_every:
            self._saved = now
            self.rec.save()

    def flush(self) -> None:
        self.rec.save()


class McpReplayer:
    """Answers JSON-RPC requests from a recording. Calls are matched on method, name and
    params (repeated identical calls answer in recorded order). Only discovery methods
    (`initialize`, `*/list`) fall back to the recorded answer for the method alone, so a
    newer client's `initialize` still works; anything else unrecorded is an error."""

    MISS = -32001
    FALLBACK = frozenset(
        {"initialize", "tools/list", "prompts/list", "resources/list", "resources/templates/list"}
    )

    def __init__(self, run: Run) -> None:
        self._exact: dict[str, deque[Step]] = defaultdict(deque)
        self._method: dict[str, Step] = {}
        self._last: dict[str, Step] = {}
        for s in run.steps:
            if s.kind != "mcp":
                continue
            method = str(s.attrs.get("method", s.name))
            self._exact[self._key(method, s.name, s.input)].append(s)
            if method in self.FALLBACK:
                self._method.setdefault(method, s)
        self.misses: list[str] = []
        self.fallbacks: list[str] = []

    @staticmethod
    def _key(method: str, name: str, inp: Any) -> str:
        return f"{method}\0{name}\0{canonical(inp)}"

    def handle(self, msg: Any) -> Any:
        if isinstance(msg, list):
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            return out or None
        if not isinstance(msg, dict) or "id" not in msg or "method" not in msg:
            return None  # notifications and responses need no answer
        method = str(msg["method"])
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        name, inp = _name_and_input(method, msg.get("params"))
        key = self._key(method, name, inp)
        q = self._exact.get(key)
        step = q.popleft() if q else self._last.get(key)
        if step is not None:
            self._last[key] = step
        elif method in self._method:
            step = self._method[method]
            self.fallbacks.append(method)
        if step is None:
            self.misses.append(f"{name} {canonical(inp)}")
            return {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {"code": self.MISS, "message": f"not in the recording: {name} {canonical(inp)}"},
            }
        if step.error is not None:
            err = step.output if isinstance(step.output, dict) else {"code": -32603, "message": step.error}
            return {"jsonrpc": "2.0", "id": msg["id"], "error": err}
        return {"jsonrpc": "2.0", "id": msg["id"], "result": step.output}


def _lines(stream: IO[bytes]) -> Any:
    for raw in iter(stream.readline, b""):
        line = raw.strip()
        if line:
            yield raw, line


def run_stdio_record(path: str | Path, command: list[str], stdin: IO[bytes], stdout: IO[bytes]) -> int:
    rec = Recorder(path, meta={"source": "mcp-proxy", "server": command})
    tap = McpTap(rec)
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin is not None and proc.stdout is not None
    out_lock = threading.Lock()

    def client_to_server() -> None:
        assert proc.stdin is not None
        try:
            for raw, line in _lines(stdin):
                with contextlib.suppress(json.JSONDecodeError):
                    tap.client_message(json.loads(line))
                proc.stdin.write(raw if raw.endswith(b"\n") else raw + b"\n")
                proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        finally:
            with contextlib.suppress(BrokenPipeError, OSError):
                proc.stdin.close()

    t = threading.Thread(target=client_to_server, daemon=True)
    t.start()
    for raw, line in _lines(proc.stdout):
        with contextlib.suppress(json.JSONDecodeError):
            tap.server_message(json.loads(line))
        with out_lock:
            stdout.write(raw if raw.endswith(b"\n") else raw + b"\n")
            stdout.flush()
    try:
        code = proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
        tap.flush()
    return code


def run_stdio_replay(path: str | Path, stdin: IO[bytes], stdout: IO[bytes]) -> int:
    rep = McpReplayer(Run.load(path))
    for _, line in _lines(stdin):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = rep.handle(msg)
        if resp is not None:
            stdout.write((json.dumps(resp) + "\n").encode())
            stdout.flush()
    if rep.misses:
        print(f"agenttrace: {len(rep.misses)} request(s) not in the recording", file=sys.stderr)
    return 0
