import json
from pathlib import Path

import httpx
from starlette.testclient import TestClient

from agenttrace.mcp_http import _sse_messages, record_app, replay_app
from agenttrace.trace import Run

INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"v": 1}}
CALL = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "add", "arguments": {"a": 1, "b": 2}},
}


def upstream(request: httpx.Request) -> httpx.Response:
    if request.method != "POST":
        return httpx.Response(405)
    try:
        msg = json.loads(request.content)
    except json.JSONDecodeError:
        return httpx.Response(202)
    if msg.get("method") == "initialize":
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": msg["id"], "result": {"server": "up"}},
            headers={"mcp-session-id": "s1", "x-internal": "drop"},
        )
    if msg.get("method") == "tools/call":
        body = (
            f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': {'n': 3}})}\n\n"
        )
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
    return httpx.Response(202)


def test_record_app_json_and_sse(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app = record_app(path, "http://up/mcp", client)
    with TestClient(app) as c:
        r = c.post("/mcp", json=INIT)
        assert r.json()["result"] == {"server": "up"}
        assert r.headers["mcp-session-id"] == "s1" and "x-internal" not in r.headers
        r = c.post("/mcp", json=CALL, headers={"accept": "text/event-stream"})
        assert '"n": 3' in r.text
        assert (
            c.post("/mcp", json={"jsonrpc": "2.0", "method": "notifications/initialized"}).status_code == 202
        )
        assert c.post("/mcp", content=b"not json").status_code == 202
        assert c.get("/mcp").status_code == 405
    run = Run.load(path)
    assert [(s.name, s.output) for s in run] == [("initialize", {"server": "up"}), ("add", {"n": 3})]

    with TestClient(replay_app(path)) as c:
        assert c.post("/mcp", json=CALL).json()["result"] == {"n": 3}
        assert c.post("/mcp", json={"jsonrpc": "2.0", "method": "x"}).status_code == 202
        assert c.post("/mcp", content=b"{").status_code == 400
        assert c.get("/mcp").status_code == 405
        assert c.delete("/mcp").status_code == 200


def test_sse_parsing() -> None:
    msgs, rest = _sse_messages('data: {"a":1}\r\n\r\nevent: x\ndata: not json\n\ndata: {"b"')
    assert msgs == [{"a": 1}] and rest == 'data: {"b"'
