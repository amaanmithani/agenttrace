"""Streamable-HTTP side of the MCP proxy: a small ASGI app that forwards to an
upstream MCP endpoint (recording) or answers from a recording (replay)."""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from agenttrace.mcp_proxy import McpReplayer, McpTap
from agenttrace.recorder import Recorder
from agenttrace.trace import Run

FORWARD = (
    "content-type",
    "accept",
    "mcp-session-id",
    "mcp-protocol-version",
    "authorization",
    "last-event-id",
)


def _sse_messages(buffer: str) -> tuple[list[Any], str]:
    """Complete SSE events in `buffer` -> parsed JSON data, and the unparsed remainder."""
    out = []
    *events, rest = buffer.replace("\r\n", "\n").split("\n\n")
    for ev in events:
        data = "\n".join(line[5:].lstrip() for line in ev.split("\n") if line.startswith("data:"))
        if data:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(data))
    return out, rest


def record_app(path: str | Path, upstream: str, client: httpx.AsyncClient | None = None) -> Starlette:
    rec = Recorder(path, meta={"source": "mcp-proxy", "upstream": upstream})
    tap = McpTap(rec)
    client = client or httpx.AsyncClient(timeout=None)

    async def mcp(request: Request) -> Response:
        body = await request.body()
        if request.method == "POST" and body:
            with contextlib.suppress(json.JSONDecodeError):
                tap.client_message(json.loads(body))
        headers = {k: v for k, v in request.headers.items() if k.lower() in FORWARD}
        req = client.build_request(request.method, upstream, headers=headers, content=body)
        resp = await client.send(req, stream=True)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() in FORWARD}
        ctype = resp.headers.get("content-type", "")

        if ctype.startswith("text/event-stream"):

            async def stream() -> AsyncIterator[bytes]:
                buf = ""
                try:
                    async for chunk in resp.aiter_bytes():
                        buf += chunk.decode("utf-8", "replace")
                        msgs, buf = _sse_messages(buf)
                        for m in msgs:
                            tap.server_message(m)
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(stream(), status_code=resp.status_code, headers=out_headers)

        content = await resp.aread()
        await resp.aclose()
        if ctype.startswith("application/json") and content:
            with contextlib.suppress(json.JSONDecodeError):
                tap.server_message(json.loads(content))
        return Response(content, status_code=resp.status_code, headers=out_headers)

    return Starlette(routes=[Route("/mcp", mcp, methods=["GET", "POST", "DELETE"])])


def replay_app(path: str | Path) -> Starlette:
    rep = McpReplayer(Run.load(path))
    session = uuid.uuid4().hex

    async def mcp(request: Request) -> Response:
        if request.method != "POST":
            # No server-initiated stream and nothing to tear down.
            return Response(status_code=405 if request.method == "GET" else 200)
        try:
            msg = json.loads(await request.body())
        except json.JSONDecodeError:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}},
                status_code=400,
            )
        resp = rep.handle(msg)
        if resp is None:
            return Response(status_code=202, headers={"mcp-session-id": session})
        return JSONResponse(resp, headers={"mcp-session-id": session})

    return Starlette(routes=[Route("/mcp", mcp, methods=["GET", "POST", "DELETE"])])
