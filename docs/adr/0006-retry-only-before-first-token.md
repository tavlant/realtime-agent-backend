# 0006: Retry model streams only before the first token

**Status:** Accepted

## Context
Model providers fail transiently or stall. Retrying a stream after the user has already seen part
of the answer would repeat text.

## Decision
Retry with exponential backoff and jitter only while **no token has reached the client**. After the
first token, surface `{"type":"error","retryable":true}` instead. Cut stalled providers off with a
first-token timeout and an inter-token idle timeout.

## Consequences
- Users never see duplicated output; transient failures before streaming starts are invisible.
- A failure mid-stream needs a client-side resend.
