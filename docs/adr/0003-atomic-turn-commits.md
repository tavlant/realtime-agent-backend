# 0003: Commit each turn atomically, shielded from disconnects

**Status:** Accepted

## Context
A turn produces a user message, possibly several tool calls and results, and a final answer.
Persisting them one by one leaves dangling tool calls or half-written replies after a crash, and
providers reject histories with a tool call but no result.

## Decision
Commit the whole turn in **one transaction** after the answer is complete. Run each turn as a
tracked task, shielded from the WebSocket handler, and drain in-flight turns on shutdown.

## Consequences
- History only ever contains whole turns, and a client that disconnects mid-stream still gets the
  answer saved.
- A turn that fails is not persisted; the client receives a `retryable` error and resends.
