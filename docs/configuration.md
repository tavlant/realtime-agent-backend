# Configuration

All settings come from environment variables (or `.env`). See [.env.example](.env.example).
The most important ones:

| Variable | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///./dev.db` | Postgres in production (`postgresql+asyncpg://...`) |
| `REDIS_URL` | `redis://localhost:6379/0` | `memory://` for an in-process fake (dev only) |
| `LLM_PROVIDER` | `auto` | `auto`, `fake`, `anthropic` or `openai` |
| `CONTEXT_TOKEN_BUDGET` | `3000` | Summarise older turns above this size |
| `MAX_TOOL_ROUNDS` | `5` | Upper bound on tool-call iterations per turn |
| `LLM_FIRST_TOKEN_TIMEOUT_S` / `LLM_IDLE_TIMEOUT_S` | `30` / `30` | Stream timeouts |
| `LLM_MAX_ATTEMPTS` | `3` | Retries before the first token |
| `LOCK_TTL_S` / `LOCK_WAIT_TIMEOUT_S` | `30` / `20` | Conversation lock tuning |
| `VECTOR_STORE` | `auto` | `pgvector` on Postgres, else `memory` |
| `EMBEDDING_PROVIDER` | `hashing` | `hashing` (offline) or `openai` |
| `DB_AUTO_CREATE` | `true` | Create tables on startup (dev/tests). Set `false` and run `alembic upgrade head` in production |
| `OTEL_ENABLED` | `false` | Export traces over OTLP (`pip install -e ".[otel]"`) |

## Real models

Set one key in `.env` and restart. With `LLM_PROVIDER=auto` (the default), Anthropic is used
when `ANTHROPIC_API_KEY` is set, otherwise OpenAI when `OPENAI_API_KEY` is set, otherwise
FakeLLM. Models are configurable (`ANTHROPIC_MODEL`, `OPENAI_MODEL`). For semantic rather
than lexical document search, set `EMBEDDING_PROVIDER=openai`.

## Running offline on Windows (no make)

```powershell
$env:DATABASE_URL="sqlite+aiosqlite:///./dev.db"; $env:REDIS_URL="memory://"; $env:LLM_PROVIDER="fake"; $env:FAKE_LLM_TOKEN_DELAY_MS="30"
uvicorn agent_backend.main:create_app --factory --reload
```
