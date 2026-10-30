# 0002: A distributed lock per conversation

**Status:** Accepted

## Context
A user can have one conversation open in two tabs, and the API runs as several replicas. Without
coordination, two turns can read the same context and write interleaved messages.

## Decision
Serialise turns per conversation with a Redis lock: `SET key token NX PX ttl`. Waiters poll with
capped exponential backoff and jitter; a watchdog renews the TTL during slow model calls. Release
and renew are Lua compare-and-act scripts on a random owner token. If the lock is lost mid-turn,
the turn is refused before commit, and a unique `(conversation_id, seq)` constraint in Postgres is
the final safety net.

## Consequences
- Turns are strictly ordered across tabs and replicas; a crashed worker cannot hold the lock forever.
- Not strictly correct under long GC pauses or clock skew; a fencing token passed to Postgres would
  close that gap.
