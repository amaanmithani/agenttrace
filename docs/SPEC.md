# AgentTrace spec

## Problem
An agent run is a sequence of model calls and tool calls whose outputs feed the next input. When a run
goes wrong after a prompt, model or tool change, the question is *where did it start to differ?* Logs
and tracing UIs show one run at a time; nobody shows the first step where two runs part ways, and
re-running the agent to check a fix calls live tools again (slow, costly, non-deterministic).

## In scope (v1)
- **Record**: a Python recorder that captures every model call, tool call and graph node as a step
  (kind, name, input, output, error, timing, parent), written as JSON Lines.
  Integrations: a LangChain/LangGraph callback handler; an MCP proxy (stdio and streamable HTTP)
  that records JSON-RPC traffic between any MCP client and server.
- **Replay**: re-run an agent with tool responses pinned to a recorded run (tools-only replay, to test a
  prompt or model change against identical tool behaviour) or with model responses pinned too (full replay,
  deterministic, no network). The MCP proxy replays a recorded server without starting it.
- **Diff**: align two runs step by step (Myers diff over step signatures, O((n+m)·d)), report the first
  divergent step and the prompt / tool-argument / output deltas.
- **Export**: OTLP/JSON spans with OpenTelemetry GenAI semantic-convention attributes.
- **Viewer**: static React + TypeScript page: open one run as a tree, or two runs as an aligned diff.

## Out of scope (v1)
Hosted backend, auth, multi-user storage; frameworks other than LangChain/LangGraph and MCP
(the recorder API is framework-free, so others can call it directly); streaming token capture.

## Success metrics
- Recording overhead per step, measured and committed (target: < 50 µs per step for the recorder itself).
- Case study: a prompt regression in a demo agent localized by `agenttrace diff` to the exact first
  divergent step.
- Full replay reproduces the recorded run's final output byte-for-byte with no model or tool calls.
- Coverage ≥ 85 %.
