# 0005: Provider-neutral LLM interface and an offline FakeLLM

**Status:** Accepted

## Context
Vendor SDKs differ in wire format (parallel tool results, fragmented tool-call arguments), and
tests against real models are slow, flaky, costly and non-deterministic.

## Decision
The agent loop, memory and tools speak only neutral types (`ChatMessage`, `ToolCall`, `TextDelta`).
Each adapter translates to its vendor's format. A deterministic `FakeLLM` with visible rules
(arithmetic calls the calculator, "time" calls the clock, "search ..." calls the knowledge base) is
used in tests and as the default when no API key is set.

## Consequences
- Switching vendors is a config change, and retries live in one place.
- Streaming, the tool loop, summarisation and concurrency are tested end to end, offline.
- The real adapters are tested against recorded wire shapes, not live APIs.
