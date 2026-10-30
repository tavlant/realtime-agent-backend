"""End-to-end WebSocket flows with FakeLLM, SQLite and fakeredis."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.conftest import collect_turn


def new_conversation(client: TestClient) -> str:
    return client.post("/conversations", json={"title": "test"}).json()["id"]


def history(client: TestClient, conversation_id: str) -> list[dict]:
    return client.get(f"/conversations/{conversation_id}/messages").json()["items"]


def test_streams_tokens_and_persists_the_turn(client: TestClient) -> None:
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["conversation_id"] == conversation_id

        ws.send_json({"type": "user_message", "content": "Hello streaming world"})
        events = collect_turn(ws)

    types = [e["type"] for e in events]
    assert types[0] == "turn_started"
    assert types[-1] == "turn_completed"
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert len(tokens) > 1, "reply should arrive as several token frames"
    assert "".join(tokens) == "You said: Hello streaming world"
    assert events[-1]["content"] == "".join(tokens)
    assert {e["turn_id"] for e in events} == {events[0]["turn_id"]}

    stored = history(client, conversation_id)
    assert [(m["seq"], m["role"], m["content"]) for m in stored] == [
        (1, "user", "Hello streaming world"),
        (2, "assistant", "You said: Hello streaming world"),
    ]
    assert client.get(f"/conversations/{conversation_id}").json()["message_count"] == 2


def test_tool_calling_loop(client: TestClient) -> None:
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": "What is 12 * (3 + 4)?"})
        events = collect_turn(ws)

    call = next(e for e in events if e["type"] == "tool_call")
    result = next(e for e in events if e["type"] == "tool_result")
    assert call["name"] == "calculator"
    assert call["arguments"] == {"expression": "12 * (3 + 4)"}
    assert result == {
        "type": "tool_result",
        "turn_id": call["turn_id"],
        "id": call["id"],
        "name": "calculator",
        "content": "84",
        "is_error": False,
    }
    assert events[-1]["content"] == "The answer is 84."
    assert events[-1]["tool_rounds"] == 1

    stored = history(client, conversation_id)
    assert [m["role"] for m in stored] == ["user", "assistant", "tool", "assistant"]
    assert stored[1]["tool_calls"][0]["name"] == "calculator"
    assert stored[2]["tool_call_id"] == call["id"]


def test_tool_errors_are_fed_back_to_the_model(client: TestClient) -> None:
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": "compute 10 / 0"})
        events = collect_turn(ws)
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["is_error"] is True
    assert events[-1]["type"] == "turn_completed"
    assert "failed: division by zero" in events[-1]["content"]


def test_knowledge_base_tool(client: TestClient) -> None:
    client.post(
        "/documents",
        json={"source": "runbook.md", "content": "# Runbook\nRotate the API keys every 90 days."},
    )
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": "search rotate api keys"})
        events = collect_turn(ws)
    result = next(e for e in events if e["type"] == "tool_result")
    assert "source: runbook.md" in result["content"]
    assert "90 days" in events[-1]["content"]


def test_multiple_turns_share_context(client: TestClient) -> None:
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ws.receive_json()
        for text in ("first", "second", "third"):
            ws.send_json({"type": "user_message", "content": text})
            assert collect_turn(ws)[-1]["content"] == f"You said: {text}"
    contents = [m["content"] for m in history(client, conversation_id)]
    assert contents == [
        "first",
        "You said: first",
        "second",
        "You said: second",
        "third",
        "You said: third",
    ]


def test_ping_and_invalid_frames(client: TestClient) -> None:
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}

        ws.send_text("not json")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["code"] == "invalid_message"

        ws.send_json({"type": "user_message", "content": ""})
        assert ws.receive_json()["code"] == "invalid_message"

        # The socket is still usable after bad frames.
        ws.send_json({"type": "user_message", "content": "still alive"})
        assert collect_turn(ws)[-1]["content"] == "You said: still alive"


def test_unknown_conversation_closes_socket(client: TestClient) -> None:
    with client.websocket_connect("/ws/conversations/nope") as ws:
        error = ws.receive_json()
        assert error["code"] == "not_found"
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 4404


def test_long_conversations_are_summarised(client_factory: Callable[..., TestClient]) -> None:
    client = client_factory(context_token_budget=120, min_recent_messages=2)
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ws.receive_json()
        for i in range(8):
            ws.send_json({"type": "user_message", "content": f"fact number {i}: " + "x " * 30})
            collect_turn(ws)

    conversation = client.get(f"/conversations/{conversation_id}").json()
    assert conversation["summary"], "older turns should have been folded into a summary"
    assert "fact number 0" in conversation["summary"]
    # The durable history is complete regardless of summarisation.
    assert len(history(client, conversation_id)) == 16


def test_client_disconnect_mid_turn_still_persists(
    client_factory: Callable[..., TestClient],
) -> None:
    client = client_factory(fake_llm_token_delay_ms=20)
    conversation_id = new_conversation(client)
    with client.websocket_connect(f"/ws/conversations/{conversation_id}") as ws:
        ws.receive_json()
        ws.send_json({"type": "user_message", "content": "one two three four five six"})
        assert ws.receive_json()["type"] == "turn_started"
    # Socket closed before the reply finished streaming.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if len(history(client, conversation_id)) == 2:
            break
        time.sleep(0.05)
    assert [m["role"] for m in history(client, conversation_id)] == ["user", "assistant"]
