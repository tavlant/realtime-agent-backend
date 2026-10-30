# Architecture

How a request flows through the service, and why it is built this way. The individual design
decisions are recorded as [ADRs](adr/README.md).

## Components

```mermaid
flowchart LR
    C["Client<br/>(browser / app)"] -- "WebSocket: tokens, tool events" --> WS
    C -- "REST: conversations, history, documents" --> API

    subgraph S["FastAPI service (async)"]
        WS["WS endpoint<br/>/ws/conversations/{id}"]
        API["REST routes"]
        AG["AgentRunner<br/>turn loop"]
        TR["ToolRegistry<br/>calculator, current_time, search_knowledge_base"]
        MEM["ConversationMemory<br/>hot state + rolling summary"]
        LK["ConversationLocks"]
        KB["KnowledgeBase<br/>chunk, embed, search"]
        LLM["LLMProvider<br/>Anthropic, OpenAI, FakeLLM"]
    end

    WS --> AG
    AG --> LK
    AG --> MEM
    AG --> LLM
    AG --> TR
    TR --> KB
    API --> MEM
    API --> KB

    LK -- "SET NX PX + Lua" --> R[("Redis<br/>locks, hot state (TTL)")]
    MEM -- "recent turns" --> R
    MEM -- "durable history, summary" --> PG[("PostgreSQL + pgvector<br/>source of truth")]
    KB -- "HNSW cosine search" --> PG
    LLM -- "streaming HTTPS" --> P["Model provider"]
```

## Life of a turn

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as AgentRunner
    participant R as Redis
    participant M as LLM
    participant T as Tools
    participant P as Postgres

    C->>A: {"type":"user_message"}
    A->>R: acquire lock:conversation:{id} (SET NX PX, watchdog renews)
    A->>R: load summary + recent turns (miss: rebuild from Postgres)
    loop until the model answers (at most MAX_TOOL_ROUNDS)
        A->>M: stream(messages, tools)
        M-->>C: token, token, token ...
        opt model requests tools
            A->>T: run tool calls concurrently (validated, timeout)
            T-->>C: tool_call / tool_result events
        end
    end
    A->>P: commit the whole turn in one transaction
    A->>R: append turn to hot state
    A-->>C: {"type":"turn_completed"}
    A->>M: (over budget?) fold oldest turns into the rolling summary
    A->>R: release lock (compare-and-delete)
```

## Write path and failure handling

- **Postgres first, then Redis.** A turn is committed to Postgres in one transaction, then appended
  to the Redis hot state. If the Redis write fails, the hot keys are dropped and rebuilt from
  Postgres on the next read. See [ADR 0001](adr/0001-redis-hot-state-postgres-source-of-truth.md).
- **One turn at a time per conversation**, across tabs and replicas, via a Redis lock with a
  watchdog. See [ADR 0002](adr/0002-per-conversation-distributed-lock.md).
- **Turns are atomic** and survive client disconnects. See [ADR 0003](adr/0003-atomic-turn-commits.md).
- **Schema changes go through Alembic.** See [ADR 0007](adr/0007-alembic-migrations.md).

## Code map

| Package | Responsibility |
|---|---|
| `api/` | REST routes, WebSocket endpoint, request-ID middleware, Pydantic schemas |
| `core/` | Agent turn loop, conversation memory, locks, summariser, retry policy, token estimates |
| `llm/` | Provider-neutral interface plus Anthropic, OpenAI and FakeLLM adapters |
| `tools/` | Typed tool registry and the built-in tools |
| `rag/` | Chunking, embeddings, vector stores, knowledge-base service |
| `db/` | ORM models, async engine, repository |
| `container.py` | Composition root: builds and owns every long-lived dependency |

Dependencies point inwards: `api` depends on `core`, `core` depends on the `llm`, `tools` and `db`
interfaces, and only `container.py` knows about concrete implementations.

## Trade-offs and limitations

- **No authentication.** Conversations are addressable by ID. Production needs auth (JWT or API keys) and per-user ownership checks.
- **Locks and fencing.** The Redis lock plus the lost-lock check before commit plus the unique sequence constraint is strong in practice. A strictly correct design under long GC pauses or clock skew would pass a fencing token to Postgres.
- **The hashing embedder** is lexical, not semantic. It exists so RAG runs offline. Use `EMBEDDING_PROVIDER=openai` for real semantic search.
- **Token counts** are estimated (about 4 characters per token) for budgeting. Exact per-provider counts would need the provider's tokenizer or count endpoint.
- **A failed turn is not persisted** (atomic turns). The client gets a `retryable` error and can resend.
- **Anthropic thinking blocks are not replayed.** History stays provider-neutral (text and tool calls only). The API accepts this, but replaying the blocks inside one turn's tool loop would give the model better reasoning continuity.
- **The Anthropic and OpenAI adapters** are unit-tested against recorded wire shapes, not live APIs, because CI runs without keys.
