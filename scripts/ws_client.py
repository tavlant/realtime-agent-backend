"""Minimal WebSocket chat client.

Usage:
    python scripts/ws_client.py "What is 12 * (3 + 4)?" "what time is it?"
    python scripts/ws_client.py            # interactive mode

Creates a conversation, streams each reply to the terminal as it arrives, and shows tool
calls inline. Requires the dev extra (httpx, websockets).
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx
from websockets.asyncio.client import connect


async def send_and_stream(ws, text: str) -> None:  # type: ignore[no-untyped-def]
    await ws.send(json.dumps({"type": "user_message", "content": text}))
    print(f"\nyou > {text}\nbot > ", end="", flush=True)
    while True:
        event = json.loads(await ws.recv())
        kind = event["type"]
        if kind == "token":
            print(event["text"], end="", flush=True)
        elif kind == "tool_call":
            print(f"\n  [tool call] {event['name']}({json.dumps(event['arguments'])})")
        elif kind == "tool_result":
            flag = " (error)" if event["is_error"] else ""
            print(f"  [tool result{flag}] {event['content'][:200]}\nbot > ", end="", flush=True)
        elif kind == "turn_completed":
            usage = event["usage"]
            print(f"\n  [done] tokens in/out: {usage['input_tokens']}/{usage['output_tokens']}")
            return
        elif kind == "error":
            print(f"\n  [error] {event['code']}: {event['message']}")
            return


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("messages", nargs="*", help="messages to send (omit for interactive)")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--conversation", help="existing conversation id to resume")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=30) as http:
        conversation_id = args.conversation
        if conversation_id is None:
            response = await http.post("/conversations", json={"title": "ws_client"})
            response.raise_for_status()
            conversation_id = response.json()["id"]
    print(f"conversation {conversation_id}")

    ws_url = args.url.replace("http", "ws", 1) + f"/ws/conversations/{conversation_id}"
    async with connect(ws_url) as ws:
        ready = json.loads(await ws.recv())
        if ready["type"] != "ready":
            raise SystemExit(f"server refused the connection: {ready}")
        if args.messages:
            for text in args.messages:
                await send_and_stream(ws, text)
            return
        while True:
            try:
                text = await asyncio.to_thread(input, "\n> ")
            except (EOFError, KeyboardInterrupt):
                return
            if text.strip():
                await send_and_stream(ws, text.strip())


if __name__ == "__main__":
    asyncio.run(main())
