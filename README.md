# agenttrace

Record, replay and diff LLM agent runs. When an agent starts answering differently after a prompt, model or tool change,
`agenttrace diff` lines two runs up step by step and points at the first step where they part ways, which is usually
several steps before the answer that looks wrong.

- **Record** every model call, tool call and graph node: a LangChain/LangGraph callback, an MCP proxy (stdio and
  streamable HTTP) for any MCP client and server, or the framework-free `Recorder` API.
- **Replay** a run with its tool responses pinned (test a prompt or model change against identical tool behaviour) or
  with model responses pinned too (deterministic, offline). The MCP proxy replays a recorded server without starting it.
- **Diff** two runs: Myers alignment over step signatures (kind, name, input), first divergence, prompt / argument /
  output deltas.
- **Export** to OTLP/JSON with OpenTelemetry GenAI semantic-convention attributes, for Jaeger, Tempo or any OTLP backend.
- **View** one run as a tree or two runs as a forked rail in a static React page (`viewer/`).

Runs are JSON Lines (`agenttrace/1`): a header line, then one object per step.

## Use

```sh
uv add "agenttrace[langgraph]"      # or [mcp] for the proxy
```

Record a LangGraph agent:

```python
from agenttrace import Recorder
from agenttrace.langchain import AgentTraceCallback

with Recorder("runs/today.jsonl", meta={"agent": "support"}) as rec:
    graph.invoke(state, config={"callbacks": [AgentTraceCallback(rec)]})
```

Compare two runs:

```sh
agenttrace diff runs/yesterday.jsonl runs/today.jsonl      # exit 1 if they differ
agenttrace diff a.jsonl b.jsonl --json                    # machine-readable
agenttrace show runs/today.jsonl                          # tree with timings
agenttrace export-otlp runs/today.jsonl -o today.otlp.json
```

Replay:

```python
from agenttrace.replay import ReplayChatModel, pin_tools
from agenttrace.trace import Run

run = Run.load("runs/yesterday.jsonl")
# Tools-only: live model, yesterday's tool responses. A call yesterday never made raises ReplayDivergence
# (or runs the real tool with on_miss="live").
graph = build_graph(my_model, pin_tools(TOOLS, run))
# Full: no network at all; with strict=True a changed prompt raises ReplayDivergence with a diff of the prompts.
graph = build_graph(ReplayChatModel(run), pin_tools(TOOLS, run))
```

Record or replay any MCP server:

```sh
agenttrace mcp-proxy --record mcp.jsonl -- python my_server.py          # stdio: use this as the server command
agenttrace mcp-proxy --replay mcp.jsonl                                  # answers from the recording, no server
agenttrace mcp-proxy --record mcp.jsonl --listen 8765 --upstream http://localhost:8000/mcp   # streamable HTTP
```

Anything else can call the recorder directly:

```python
with rec.step("tool", "search", input={"q": "refund"}) as s:
    s.set_output(search("refund"))
```

Steps nest by context: asyncio tasks inherit the step they were started from. New threads don't inherit context in
Python, so wrap their target with `rec.bind(fn)` to carry the recorder and parent step across. Inputs and outputs are
copied when recorded, and a `redact` hook runs on the copy before a step is written; steps still in flight are never
written.

## Case study: the tool that changed the answer

A support agent (LangGraph, two tools) answers refund questions. A refactor of the order service's serializer drops
the `opened` field. Nothing errors.

<!-- case:start -->
Model: `llama3.1:8b` via Ollama, temperature 0. Question: *"Hi, I'd like a refund for order 42, the headphones. Can I get my money back?"*

| | Final answer |
|---|---|
| Before (order service v1) | Based on our refund policy, since the headphones have been delivered for over 30 days and you've opened them, unfortunately, we cannot offer a full refund, but we can provide a store credit of $20 towards your next purchase. |
| After (v2 drops `opened`) | Based on our refund policy, since it's been 12 days since the headphones were delivered and you haven't contacted us within 30 days of delivery, unfortunately, we can only offer a partial refund for any unopened or unused items. |

`agenttrace diff examples/runs/before.jsonl examples/runs/after.jsonl`:

```
5 equal, 2 changed, 0 only in A, 0 only in B, 3 same request with a different result

first divergence: tool lookup_order (A #5 / B #5): same input, different result
  output:
    -item=headphones, price=89.0, days_since_delivery=12, opened=True
    +item=headphones, price=89.0, days_since_delivery=12
```

Both answers are from an 8B local model and neither is fully right (the "before" answer misstates the delivery
window; the "after" answer invents a partial refund). The point is not answer quality but where the runs first differ.

The first divergence is step 5 of 7: `tool lookup_order`, not the
final message where the symptom shows.

Controls:

| Check | Result |
|---|---|
| Full replay of "before" (recorded model + recorded tools, no network) is step-for-step identical | yes |
| ...and gives the same final answer | yes |
| Live model again with "before"'s tool responses pinned: identical run | yes |
| ...same final answer | yes |
| Live model with "after"'s tool responses pinned: identical to "after" | yes |
| ...same final answer | yes |

With the tool responses pinned, the live model reproduces each run exactly, in both directions: at temperature 0 with a fixed seed it is deterministic on these inputs, so the tool output alone decides which run you get. This is one sample per condition; it rules out sampling noise for this model and these settings, not in general.

The model never called `refund_policy` in either run, although the system prompt tells it to; both answers state a refund policy the model never read. That's a real weakness of this 8B model as an agent, and exactly the kind of thing the recorded runs make visible.
Reproduce: `uv run python examples/case_study.py` (needs Ollama with `llama3.1:8b`); the recorded runs are committed in
`examples/runs/`, and CI re-diffs them on every push.
<!-- case:end -->

## Recording overhead

<!-- overhead:start -->
| What | Per step |
|---|---|
| `Recorder.step()` around a no-op (median of 7×20,000 calls) | 8.6 µs |
| LangGraph callback, support agent with a scripted model (11 steps/run, 15 paired reps of 200 runs; IQR 31.9–40.0) | 35.2 µs |

The recorder figure includes copying the input and output (so a redactor never touches live objects) but not writing
the file. The callback figure is a paired difference: the graph is built once and each rep times the same number of
invocations with and without the callback, alternating which goes first. The callback adds 0.39 ms to a run that takes 1.6 ms with a model that
answers instantly; against a real model call (hundreds of milliseconds) it is noise.
Python 3.13.14, Darwin arm64. Reproduce: `uv run python bench/overhead.py`.
<!-- overhead:end -->

## How the diff works

Each step's **signature** is its kind, name and a hash of its canonical input: what it was asked to do, not what it
returned. The two signature sequences are aligned with Myers' algorithm (O((n+m)·d), minimal edit script; the tests
check minimality against a brute-force LCS on random sequences). Then:

- aligned steps with equal signatures are **equal**, and flagged if their outputs differ (same request, different
  result: a changed tool, a non-deterministic model);
- a removed step followed by an added step of the same kind and name is **changed** (same call, different input);
- the **first divergence** is the earliest entry that isn't equal or whose output differs, skipping graph nodes whose
  output differs only because a step inside them did.

Providers generate tool-call ids at random, so they are recorded as their position in the conversation (`#0`, `#1`, ...):
identical runs get identical signatures, while a run whose tool results come back swapped between calls still differs.
A model step's input also carries a digest of the offered tool schemas and the sampling parameters, so changing a tool
description or the temperature changes the signature too. The viewer uses the same algorithm; a test checks it reproduces the Python diff on the case-study
runs.

## Limits

- One run per recorder; no storage server, search or auth. Runs are files.
- Integrations: LangChain/LangGraph callbacks and MCP. Other frameworks can use the `Recorder` API directly.
- Streaming tokens aren't captured individually; a streamed model call is recorded as one step with its final message.
  With n > 1 completions per call, only the first is recorded.
- The viewer compares values after `JSON.parse`, so it can't tell `1` from `1.0`; the Python diff can.
- Replay matches tool calls on exact arguments. An agent that puts timestamps or random ids in tool arguments needs a
  `redact` hook to normalise them before recording.
- The case study has one scenario and one local model. It shows the diff localising a real behaviour change, not a
  success rate.

## Development

```sh
uv sync && uv run pytest --cov        # coverage gate 85%
uv run ruff check . && uv run mypy
npm --prefix viewer ci && npm --prefix viewer run dev
```
