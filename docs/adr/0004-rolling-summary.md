# 0004: Rolling summary to keep context under a token budget

**Status:** Accepted

## Context
Cost and latency grow with every token resent to the model, so context cannot grow forever.

## Decision
When the unsummarised context exceeds `CONTEXT_TOKEN_BUDGET`, fold the oldest turns into a summary
carried in the system prompt. Cut points sit only **before a user message**, so a tool call is never
separated from its result. Summarisation runs after `turn_completed` is sent (the user never waits
for it) but inside the conversation lock (it cannot race the next turn).

## Consequences
- Bounded prompt size with no extra latency for the user.
- If summarisation fails, the turn still succeeds and the full context is kept; Postgres always
  keeps every original message.
- Token counts are estimated (about 4 characters per token), not exact per provider.
