# API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Readiness (Postgres + Redis reachable), 503 when degraded |
| `POST` | `/conversations` | Create a conversation `{"title": "optional"}` |
| `GET` | `/conversations?limit&offset` | List conversations, most recently active first |
| `GET` | `/conversations/{id}` | Conversation metadata, including the rolling summary |
| `GET` | `/conversations/{id}/messages?limit&before_seq` | Full history from Postgres, paginated |
| `POST` | `/documents` | Ingest a document `{"source": "faq.md", "content": "..."}` |
| `GET` | `/documents/search?q&k` | Semantic search over ingested chunks |
| `WS` | `/ws/conversations/{id}` | Streaming chat (protocol below) |

Every HTTP response carries an `X-Request-ID` header. A valid incoming one is reused, so a request can be traced across services.

### WebSocket protocol

```jsonc
// client -> server
{"type": "user_message", "content": "What is 12 * (3 + 4)?"}
{"type": "ping"}

// server -> client
{"type": "ready", "conversation_id": "...", "connection_id": "..."}
{"type": "turn_started", "turn_id": "..."}
{"type": "token", "turn_id": "...", "text": "The "}
{"type": "tool_call", "turn_id": "...", "id": "...", "name": "calculator", "arguments": {"expression": "12 * (3 + 4)"}}
{"type": "tool_result", "turn_id": "...", "id": "...", "name": "calculator", "content": "84", "is_error": false}
{"type": "turn_completed", "turn_id": "...", "content": "The answer is 84.", "tool_rounds": 1, "usage": {"input_tokens": 135, "output_tokens": 13}}
{"type": "error", "code": "conversation_busy | upstream_unavailable | invalid_message | not_found | internal_error", "message": "...", "retryable": true}
```

### Minimal client (Python)

```python
import asyncio, json
import httpx
from websockets.asyncio.client import connect


async def main() -> None:
    async with httpx.AsyncClient(base_url="http://localhost:8000") as http:
        conversation_id = (await http.post("/conversations")).json()["id"]

    async with connect(f"ws://localhost:8000/ws/conversations/{conversation_id}") as ws:
        await ws.recv()  # {"type": "ready", ...}
        await ws.send(json.dumps({"type": "user_message", "content": "What is 2 ** 10?"}))
        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] in ("turn_completed", "error"):
                break


asyncio.run(main())
```

### Minimal client (browser)

```js
const { id } = await (await fetch("/conversations", { method: "POST" })).json();
const ws = new WebSocket(`ws://${location.host}/ws/conversations/${id}`);
ws.onopen = () => ws.send(JSON.stringify({ type: "user_message", content: "Hi!" }));
ws.onmessage = ({ data }) => {
  const event = JSON.parse(data);
  if (event.type === "token") output.textContent += event.text;
};
```
