"""Retries with exponential backoff + jitter, and timeouts for async streams."""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Raised by adapters for transient failures (rate limits, 5xx, network errors)."""


class StreamTimeoutError(RetryableError):
    """The model did not produce the next event in time."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_s: float = 0.25
    max_delay_s: float = 4.0
    retry_on: tuple[type[BaseException], ...] = (RetryableError, asyncio.TimeoutError)

    def delay_for(self, attempt: int) -> float:
        """Full-jitter exponential backoff (attempt is 1-based)."""
        cap = min(self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
        return random.uniform(0, cap)


async def retry_async(
    fn: Callable[[], Awaitable[T]], policy: RetryPolicy, *, op: str = "operation"
) -> T:
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await fn()
        except policy.retry_on as exc:
            if attempt == policy.max_attempts:
                raise
            delay = policy.delay_for(attempt)
            logger.warning(
                "retrying after transient failure",
                extra={
                    "op": op,
                    "attempt": attempt,
                    "delay_s": round(delay, 3),
                    "error": repr(exc),
                },
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


async def with_timeouts(
    stream: AsyncIterator[T], *, first_timeout_s: float, idle_timeout_s: float
) -> AsyncIterator[T]:
    """Wrap an async iterator so a stalled provider fails fast instead of hanging a socket."""
    timeout = first_timeout_s
    iterator = stream.__aiter__()
    while True:
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError as exc:
            raise StreamTimeoutError(f"no stream event within {timeout:.1f}s") from exc
        timeout = idle_timeout_s
        yield item


async def stream_with_retry(
    factory: Callable[[], AsyncIterator[T]],
    policy: RetryPolicy,
    *,
    first_timeout_s: float,
    idle_timeout_s: float,
    op: str = "llm.stream",
) -> AsyncIterator[T]:
    """Retry a streaming call, but only until the first event has been delivered.

    Once tokens have reached the client, a retry would duplicate output, so failures after
    the first event propagate to the caller instead.
    """
    for attempt in range(1, policy.max_attempts + 1):
        started = False
        try:
            async for item in with_timeouts(
                factory(), first_timeout_s=first_timeout_s, idle_timeout_s=idle_timeout_s
            ):
                started = True
                yield item
            return
        except policy.retry_on as exc:
            if started or attempt == policy.max_attempts:
                raise
            delay = policy.delay_for(attempt)
            logger.warning(
                "retrying stream before first token",
                extra={
                    "op": op,
                    "attempt": attempt,
                    "delay_s": round(delay, 3),
                    "error": repr(exc),
                },
            )
            await asyncio.sleep(delay)
