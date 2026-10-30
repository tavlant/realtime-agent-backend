# Testing and load testing

## Running the checks

```bash
make lint        # ruff check + format check
make typecheck   # mypy --strict
make test        # pytest with coverage (fails under 85%)
```

Everything runs offline in about 20 seconds: SQLite (aiosqlite), fakeredis and the FakeLLM.
No Docker, no API keys.

## What the suites prove

| Suite | What it proves |
|---|---|
| `unit/test_calculator.py` | AST evaluator accepts arithmetic and rejects `__import__`, attribute access, names, `9**9**9`, complex results, oversized input |
| `unit/test_chunking.py` | Size limits, overlap, no lost content, Markdown heading paths, fenced code blocks |
| `unit/test_tool_registry.py` | Schema generation, argument validation, timeouts, crash isolation, concurrent execution |
| `unit/test_locks.py` | Mutual exclusion, owner-only release, TTL takeover, watchdog renewal, lost-lock detection |
| `unit/test_summarizer.py` | Cut points never split tool calls, summary persisted to Postgres and Redis, failure keeps full context, cache rebuild |
| `unit/test_retry.py` | Backoff, give-up, no retry after first token, stall timeouts |
| `unit/test_providers.py` | OpenAI and Anthropic wire translation, streamed tool-call reassembly, FakeLLM rules |
| `unit/test_rag.py` | Deterministic embeddings, ranking, ingestion, search tool output |
| `integration/test_websocket.py` | Token streaming, tool loop, tool errors, RAG tool, summarisation, disconnect mid-turn still persists |
| `integration/test_concurrency.py` | **50 simultaneous WebSocket conversations** with complete, ordered, unmixed histories; 15 concurrent turns on *one* conversation run strictly one at a time |
| `integration/test_rest.py` | Health and readiness, CRUD, pagination, validation, request IDs, document search |
| `integration/test_migrations.py` | `alembic upgrade head` produces exactly the schema the ORM models describe, and `downgrade base` removes it |

CI (GitHub Actions) runs lint and mypy, the test suite with coverage on Python 3.10 and 3.12, then
builds the Docker image and smoke-tests the full compose stack (Postgres + pgvector + Redis, with
migrations applied) using the load test.

## Load test

```bash
python scripts/load_test.py --url http://localhost:8000 --conversations 100 --turns 3
```

Sample run: Docker Compose stack, FakeLLM, **one** uvicorn worker, Docker Desktop on a
Windows laptop, load generator on the same machine:

```text
conversations     100 x 3 turns
completed turns   300 / 300
errors            0
throughput        55.4 turns/s, 610 tokens/s
turn latency      p50 656 ms | p95 1798 ms | p99 2549 ms
time to 1st token p50 288 ms | p95 1112 ms
(single user:     p50 15 ms per turn, 4 ms to first token)
```

With FakeLLM this measures the backend's own overhead (locking, Redis, Postgres, WebSocket
fan-out). With a real model, latency is dominated by the provider. Under load, time goes to
Redis and Postgres round trips, not CPU. Scaling out is horizontal: add uvicorn workers or
replicas, which the distributed lock already supports.
