# Architecture Decision Records

Short records of the decisions that shape this service: the context, the decision, and what it
costs. New decisions get the next number; superseded ones stay, marked as such.

| # | Decision | Status |
|---|---|---|
| [0001](0001-redis-hot-state-postgres-source-of-truth.md) | Redis for hot state, Postgres as the source of truth | Accepted |
| [0002](0002-per-conversation-distributed-lock.md) | A distributed lock per conversation | Accepted |
| [0003](0003-atomic-turn-commits.md) | Commit each turn atomically, shielded from disconnects | Accepted |
| [0004](0004-rolling-summary.md) | Rolling summary to keep context under a token budget | Accepted |
| [0005](0005-provider-abstraction-and-fakellm.md) | Provider-neutral LLM interface and an offline FakeLLM | Accepted |
| [0006](0006-retry-only-before-first-token.md) | Retry model streams only before the first token | Accepted |
| [0007](0007-alembic-migrations.md) | Alembic migrations instead of `create_all` | Accepted |
