import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

from agenttrace.mcp_proxy import McpReplayer
from agenttrace.trace import Run

SERVER = str(Path(__file__).parent / "fixtures" / "calc_server.py")
PROXY = [sys.executable, "-m", "agenttrace", "mcp-proxy"]


def text(result: Any) -> str:
    return "".join(getattr(c, "text", "") for c in result.content)


@asynccontextmanager
async def stdio_session(args: list[str]) -> AsyncIterator[ClientSession]:
    params = StdioServerParameters(command=args[0], args=args[1:])
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        yield s


async def exercise(s: ClientSession) -> dict[str, Any]:
    tools = await s.list_tools()
    return {
        "tools": sorted(t.name for t in tools.tools),
        "add": text(await s.call_tool("add", {"a": 2, "b": 3})),
        "clock": text(await s.call_tool("clock", {})),
        "fail": (await s.call_tool("fail", {})).is_error,
    }


async def test_stdio_record_then_replay_without_server(tmp_path: Path) -> None:
    path = tmp_path / "mcp.jsonl"
    async with stdio_session([*PROXY, "--record", str(path), "--", sys.executable, SERVER]) as s:
        live = await exercise(s)
    assert live["tools"] == ["add", "clock", "fail"] and live["add"] == "5" and live["fail"] is True

    run = Run.load(path)
    names = [(st.name, st.attrs["method"]) for st in run]
    assert ("initialize", "initialize") in names and ("add", "tools/call") in names
    add = next(st for st in run if st.name == "add")
    assert add.input == {"a": 2, "b": 3}

    async with stdio_session([*PROXY, "--replay", str(path)]) as s:
        replayed = await exercise(s)
        # The clock is pinned to the recorded value: replay is deterministic.
        assert replayed == live
        with pytest.raises(Exception, match="not in the recording"):
            await s.call_tool("add", {"a": 1, "b": 1})


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@contextmanager
def background(args: list[str], port: int) -> Iterator[None]:
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    break
            time.sleep(0.1)
        else:
            raise TimeoutError(f"nothing listening on {port}")
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@asynccontextmanager
async def http_session(url: str) -> AsyncIterator[ClientSession]:
    async with streamable_http_client(url) as streams:
        r, w = streams[0], streams[1]
        async with ClientSession(r, w) as s:
            await s.initialize()
            yield s


async def test_http_record_then_replay(tmp_path: Path) -> None:
    path = tmp_path / "mcp-http.jsonl"
    up, px, rp = free_port(), free_port(), free_port()
    with (
        background([sys.executable, SERVER, "http", str(up)], up),
        background(
            [*PROXY, "--record", str(path), "--listen", str(px), "--upstream", f"http://127.0.0.1:{up}/mcp"],
            px,
        ),
    ):
        async with http_session(f"http://127.0.0.1:{px}/mcp") as s:
            live = await exercise(s)
    assert live["add"] == "5"
    run = Run.load(path)
    assert {st.name for st in run} >= {"initialize", "tools/list", "add", "clock", "fail"}

    with background([*PROXY, "--replay", str(path), "--listen", str(rp)], rp):
        async with http_session(f"http://127.0.0.1:{rp}/mcp") as s:
            assert await exercise(s) == live


def test_replayer_unit() -> None:
    from agenttrace.trace import Step

    run = Run(
        [
            Step(1, "mcp", "initialize", {"v": 1}, {"server": "x"}, attrs={"method": "initialize"}),
            Step(2, "mcp", "t", {"a": 1}, {"n": 1}, attrs={"method": "tools/call"}),
            Step(3, "mcp", "t", {"a": 1}, {"n": 2}, attrs={"method": "tools/call"}),
            Step(
                4,
                "mcp",
                "t",
                {"a": 2},
                {"code": 7, "message": "bad"},
                error="bad",
                attrs={"method": "tools/call"},
            ),
        ]
    )
    rep = McpReplayer(run)

    def call(i, a):
        return {
            "jsonrpc": "2.0",
            "id": i,
            "method": "tools/call",
            "params": {"name": "t", "arguments": a},
        }

    # A newer client's initialize params still get the recorded answer.
    assert rep.handle({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"v": 2}})["result"] == {
        "server": "x"
    }
    assert rep.handle(call(1, {"a": 1}))["result"] == {"n": 1}
    assert rep.handle(call(2, {"a": 1}))["result"] == {"n": 2}
    assert rep.handle(call(3, {"a": 1}))["result"] == {"n": 2}  # exhausted: last answer repeats
    assert rep.handle(call(4, {"a": 2}))["error"] == {"code": 7, "message": "bad"}
    assert rep.handle(call(5, {"a": 3}))["error"]["code"] == McpReplayer.MISS
    assert rep.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert rep.handle({"jsonrpc": "2.0", "id": 9, "method": "ping"})["result"] == {}
    batch = rep.handle([call(6, {"a": 1}), {"jsonrpc": "2.0", "method": "n"}])
    assert len(batch) == 1
    assert rep.misses == ['t {"a":3}']


def test_stdio_loops_in_process(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    import io
    import json as _json

    from agenttrace.mcp_proxy import run_stdio_record, run_stdio_replay

    msgs = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "add", "arguments": {"a": 4, "b": 5}},
        },
    ]
    import os
    import threading

    class Out(io.BytesIO):
        """Signals once the tools/call reply has been written."""

        got = threading.Event()

        def write(self, b: bytes) -> int:  # type: ignore[override]
            n = super().write(b)
            if b'"id":2' in b.replace(b" ", b""):
                self.got.set()
            return n

    r, w = os.pipe()
    out = Out()
    path = tmp_path / "s.jsonl"
    codes: list[int] = []
    t = threading.Thread(
        target=lambda: codes.append(run_stdio_record(path, [sys.executable, SERVER], os.fdopen(r, "rb"), out))
    )
    t.start()
    os.write(w, ("\n".join(_json.dumps(m) for m in msgs) + "\nnot json\n").encode())
    # Keep stdin open until the reply arrives, as a real client does; closing it stops the server.
    assert out.got.wait(30)
    os.close(w)
    t.join(30)
    assert codes == [0]
    replies = [_json.loads(x) for x in out.getvalue().splitlines() if x.strip()]
    assert any(r.get("id") == 2 for r in replies)
    run = Run.load(path)
    assert [s.name for s in run] == ["initialize", "add"]

    stdin = io.BytesIO(
        (
            _json.dumps(msgs[2])
            + "\n{bad\n"
            + _json.dumps({**msgs[2], "id": 3, "params": {"name": "add", "arguments": {"a": 0, "b": 0}}})
            + "\n"
        ).encode()
    )
    out = io.BytesIO()
    assert run_stdio_replay(path, stdin, out) == 0
    a, b = (_json.loads(x) for x in out.getvalue().splitlines())
    assert a["result"] == run.steps[1].output and "error" in b
    assert "1 request(s) not in the recording" in capsys.readouterr().err
