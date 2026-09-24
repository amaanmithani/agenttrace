"""agenttrace command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from agenttrace.diff import delta, diff_runs
from agenttrace.otel import to_otlp
from agenttrace.trace import Run, Step, canonical


def _short(value: object, width: int = 70) -> str:
    s = value if isinstance(value, str) else canonical(value)
    s = s.replace("\n", " ")
    return s if len(s) <= width else s[: width - 1] + "…"


def _line(s: Step) -> str:
    err = f"  ERROR {s.error}" if s.error else ""
    head = f"#{s.id} {s.kind} {s.name} ({s.duration_ms:.1f} ms)"
    return f"{head}  in={_short(s.input, 50)}  out={_short(s.output, 50)}{err}"


def show(run: Run) -> str:
    out: list[str] = []
    ids = {s.id for s in run}

    def walk(parent: int | None, depth: int) -> None:
        for s in run.steps:
            if s.parent == parent or (
                depth == 0 and s.parent is not None and s.parent not in ids and parent is None
            ):
                out.append("  " * depth + _line(s))
                walk(s.id, depth + 1)

    walk(None, 0)
    return "\n".join(out)


def render_diff(a: Run, b: Run) -> tuple[str, bool]:
    d = diff_runs(a, b)
    if d.identical:
        return f"identical: {len(a)} steps, same requests and same results", True
    f = d.first
    assert f is not None
    st = d.stats
    lines = [
        f"{st['equal']} equal, {st['changed']} changed, {st['removed']} only in A, {st['added']} only in B, "
        f"{st['output_differs']} same request with a different result",
        "",
    ]
    step = f.a or f.b
    assert step is not None
    where = f"A #{f.a.id}" if f.a else ""
    where += (" / " if f.a and f.b else "") + (f"B #{f.b.id}" if f.b else "")
    what = {
        "changed": "same call, different input",
        "removed": "step only in A",
        "added": "step only in B",
        "equal": "same input, different result",
    }[f.op]
    lines.append(f"first divergence: {step.kind} {step.name} ({where}): {what}")
    for fld in ("input", "output"):
        dl = delta(f.a, f.b, fld)
        if dl:
            lines.append(f"  {fld}:")
            lines.extend("    " + x for x in dl[2:] if not x.startswith("@@"))
    return "\n".join(lines), False


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agenttrace", description="Record, replay and diff agent runs.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="print a run as a tree")
    s.add_argument("run")
    d = sub.add_parser("diff", help="align two runs and show where they diverge (exit 1 if they do)")
    d.add_argument("a")
    d.add_argument("b")
    d.add_argument("--json", action="store_true")
    e = sub.add_parser("export-otlp", help="write a run as OTLP/JSON with GenAI semantic conventions")
    e.add_argument("run")
    e.add_argument("-o", "--out", default="-")
    e.add_argument("--service", default="agent")
    m = sub.add_parser("mcp-proxy", help="record or replay MCP traffic (stdio or streamable HTTP)")
    g = m.add_mutually_exclusive_group(required=True)
    g.add_argument("--record", metavar="FILE")
    g.add_argument("--replay", metavar="FILE")
    m.add_argument("--listen", type=int, metavar="PORT", help="serve streamable HTTP on this port")
    m.add_argument("--upstream", metavar="URL", help="upstream MCP endpoint (HTTP recording)")
    m.add_argument("command", nargs=argparse.REMAINDER, help="-- server command (stdio recording)")
    a = p.parse_args(argv)

    try:
        if a.cmd == "show":
            print(show(Run.load(a.run)))
            return 0
        if a.cmd == "diff":
            ra, rb = Run.load(a.a), Run.load(a.b)
            if a.json:
                res = diff_runs(ra, rb)
                print(json.dumps(res.to_json(), indent=2, default=str))
                return 0 if res.identical else 1
            text, same = render_diff(ra, rb)
            print(text)
            return 0 if same else 1
        if a.cmd == "export-otlp":
            doc = json.dumps(to_otlp(Run.load(a.run), a.service))
            if a.out == "-":
                print(doc)
            else:
                Path(a.out).write_text(doc + "\n", encoding="utf-8")
            return 0
        return _mcp(a, p)
    except (OSError, ValueError) as err:
        print(f"agenttrace: {err}", file=sys.stderr)
        return 2


def _mcp(a: argparse.Namespace, p: argparse.ArgumentParser) -> int:
    import signal

    from agenttrace import mcp_proxy

    # MCP clients stop stdio servers with SIGTERM; exit normally so the recording is saved.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    command = [c for c in a.command if c != "--"] if a.command else []
    if a.listen:
        import uvicorn

        from agenttrace import mcp_http

        if a.record:
            if not a.upstream:
                p.error("--listen with --record needs --upstream")
            app = mcp_http.record_app(a.record, a.upstream)
        else:
            app = mcp_http.replay_app(a.replay)
        uvicorn.run(app, host="127.0.0.1", port=a.listen, log_level="warning")
        return 0
    if a.record:
        if not command:
            p.error("stdio recording needs the server command after --")
        return mcp_proxy.run_stdio_record(a.record, command, sys.stdin.buffer, sys.stdout.buffer)
    return mcp_proxy.run_stdio_replay(a.replay, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
