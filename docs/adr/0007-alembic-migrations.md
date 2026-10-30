# 0007: Alembic migrations instead of `create_all`

**Status:** Accepted (replaces creating tables on startup)

## Context
`Base.metadata.create_all` at startup is convenient but cannot evolve an existing schema, gives no
history of changes, and lets every replica race to create tables.

## Decision
Manage the schema with Alembic (`migrations/`). Deployments run `alembic upgrade head` before the
API starts (see `docker-compose.yml`) and set `DB_AUTO_CREATE=false`. Local dev and tests keep
`create_all` for zero-setup runs. A test checks that `upgrade head` produces exactly the schema the
ORM models describe, so models and migrations cannot drift.

## Consequences
- Schema changes are reviewed, versioned and reversible (`alembic downgrade`).
- New model changes need a migration (`make migration m="..."`); the drift test fails otherwise.
