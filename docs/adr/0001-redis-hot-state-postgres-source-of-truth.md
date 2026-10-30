# 0001: Redis for hot state, Postgres as the source of truth

**Status:** Accepted

## Context
Every turn needs the recent conversation context immediately. Re-reading and re-sorting the full
history from Postgres on every turn adds latency and database load for no benefit, but the
history must never be lost.

## Decision
Keep the hot state (rolling summary plus the not-yet-summarised messages) in Redis with a sliding
TTL, and keep every message in Postgres. The write path is **Postgres first, then Redis**. If the
Redis write fails, the hot keys are deleted and rebuilt from Postgres on the next read.

## Consequences
- The cache can be missing but never wrong, and Redis can be flushed at any time without data loss.
- Idle conversations fall out of memory on their own through the TTL.
- Two stores to operate, and a cold read after expiry costs one Postgres query.
