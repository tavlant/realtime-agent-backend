import asyncio
from collections.abc import AsyncIterator

import pytest

from agent_backend.core.retry import (
    RetryableError,
    RetryPolicy,
    StreamTimeoutError,
    retry_async,
    stream_with_retry,
    with_timeouts,
)

FAST = RetryPolicy(max_attempts=3, base_delay_s=0.001, max_delay_s=0.002)


async def test_retry_async_recovers_from_transient_errors() -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RetryableError("503")
        return "ok"

    assert await retry_async(flaky, FAST) == "ok"
    assert attempts == 3


async def test_retry_async_gives_up_after_max_attempts() -> None:
    attempts = 0

    async def down() -> str:
        nonlocal attempts
        attempts += 1
        raise RetryableError("still down")

    with pytest.raises(RetryableError):
        await retry_async(down, FAST)
    assert attempts == FAST.max_attempts


async def test_non_retryable_errors_propagate_immediately() -> None:
    attempts = 0

    async def bad_request() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("400")

    with pytest.raises(ValueError, match="400"):
        await retry_async(bad_request, FAST)
    assert attempts == 1


def test_backoff_is_capped() -> None:
    policy = RetryPolicy(base_delay_s=1, max_delay_s=2)
    assert all(0 <= policy.delay_for(n) <= 2 for n in range(1, 10))


async def test_stream_retries_only_before_first_event() -> None:
    calls = 0

    async def fails_then_streams() -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RetryableError("connect failed")
        for token in ("a", "b"):
            yield token

    out = [
        t
        async for t in stream_with_retry(
            fails_then_streams, FAST, first_timeout_s=1, idle_timeout_s=1
        )
    ]
    assert out == ["a", "b"]
    assert calls == 2


async def test_stream_failure_after_first_token_is_not_retried() -> None:
    calls = 0

    async def breaks_midway() -> AsyncIterator[str]:
        nonlocal calls
        calls += 1
        yield "partial"
        raise RetryableError("connection reset")

    received = []
    with pytest.raises(RetryableError):
        async for token in stream_with_retry(
            breaks_midway, FAST, first_timeout_s=1, idle_timeout_s=1
        ):
            received.append(token)
    assert received == ["partial"]
    assert calls == 1  # no duplicate output


async def test_stalled_stream_times_out() -> None:
    async def stalls() -> AsyncIterator[str]:
        yield "first"
        await asyncio.sleep(10)
        yield "never"  # pragma: no cover

    received = []
    with pytest.raises(StreamTimeoutError):
        async for token in with_timeouts(stalls(), first_timeout_s=1, idle_timeout_s=0.05):
            received.append(token)
    assert received == ["first"]
