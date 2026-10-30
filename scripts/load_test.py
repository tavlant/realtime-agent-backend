"""Load test: open N concurrent WebSocket conversations against a running server.

Usage:
    python scripts/load_test.py --url http://localhost:8000 --conversations 100 --turns 3

Each simulated user creates a conversation over REST, opens a WebSocket and sends
`--turns` messages one after another (waiting for each reply, like a real chat user).
Prints throughput, time-to-first-token and full-turn latency percentiles.
Requires the dev extra (httpx, websockets).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx
from websockets.asyncio.client import connect


@dataclass
class Stats:
    turn_latencies: list[float] = field(default_factory=list)
    first_token: list[float] = field(default_factory=list)
    tokens: int = 0
    turns: int = 0
    errors: list[str] = field(default_factory=list)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


async def simulate_user(
    index: int, base_url: str, turns: int, message: str, stats: Stats, http: httpx.AsyncClient
) -> None:
    try:
        response = await http.post("/conversations", json={"title": f"load-{index}"})
        response.raise_for_status()
        conversation_id = response.json()["id"]
        ws_url = base_url.replace("http", "ws", 1) + f"/ws/conversations/{conversation_id}"
        async with connect(ws_url, open_timeout=30, max_queue=None) as ws:
            ready = json.loads(await ws.recv())
            assert ready["type"] == "ready", ready
            for turn in range(turns):
                sent = time.perf_counter()
                first: float | None = None
                await ws.send(
                    json.dumps(
                        {
                            "type": "user_message",
                            "content": f"{message} [user {index}, turn {turn}]",
                        }
                    )
                )
                while True:
                    event = json.loads(await ws.recv())
                    if event["type"] == "token":
                        stats.tokens += 1
                        if first is None:
                            first = time.perf_counter()
                    elif event["type"] == "turn_completed":
                        stats.turns += 1
                        stats.turn_latencies.append(time.perf_counter() - sent)
                        if first is not None:
                            stats.first_token.append(first - sent)
                        break
                    elif event["type"] == "error":
                        stats.errors.append(f"{event['code']}: {event['message']}")
                        break
    except Exception as exc:
        stats.errors.append(f"user {index}: {exc!r}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--conversations", "-n", type=int, default=50)
    parser.add_argument("--turns", "-t", type=int, default=3)
    parser.add_argument("--message", default="Hello from the load test")
    args = parser.parse_args()

    stats = Stats()
    limits = httpx.Limits(max_connections=args.conversations)
    async with httpx.AsyncClient(base_url=args.url, timeout=30, limits=limits) as http:
        (await http.get("/healthz")).raise_for_status()
        started = time.perf_counter()
        await asyncio.gather(
            *(
                simulate_user(i, args.url, args.turns, args.message, stats, http)
                for i in range(args.conversations)
            )
        )
        wall = time.perf_counter() - started

    ms = 1000
    print(f"conversations     {args.conversations} x {args.turns} turns")
    print(f"completed turns   {stats.turns} / {args.conversations * args.turns}")
    print(f"errors            {len(stats.errors)}")
    print(f"wall time         {wall:.2f} s")
    print(f"throughput        {stats.turns / wall:.1f} turns/s, {stats.tokens / wall:.0f} tokens/s")
    if stats.turn_latencies:
        lat = stats.turn_latencies
        print(
            "turn latency      "
            f"p50 {percentile(lat, 50) * ms:.0f} ms | p95 {percentile(lat, 95) * ms:.0f} ms | "
            f"p99 {percentile(lat, 99) * ms:.0f} ms | mean {statistics.mean(lat) * ms:.0f} ms"
        )
    if stats.first_token:
        ttft = stats.first_token
        print(
            "time to 1st token "
            f"p50 {percentile(ttft, 50) * ms:.0f} ms | p95 {percentile(ttft, 95) * ms:.0f} ms"
        )
    for error in stats.errors[:10]:
        print(f"  ! {error}")


if __name__ == "__main__":
    asyncio.run(main())
