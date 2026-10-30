"""Distributed per-conversation lock on Redis.

Guarantees that turns of one conversation never interleave, even when the same
conversation is open in several tabs or the API runs as several replicas.

* acquire: `SET key token NX PX ttl` - atomic, with a TTL so a crashed worker cannot hold
  the lock forever;
* wait: poll with capped exponential backoff + jitter until `wait_timeout_s`;
* hold: a watchdog task extends the TTL while a long turn (slow model, tool calls) runs;
* release / extend: Lua compare-and-act on the random token, so a worker can never
  release or extend a lock that has expired and been taken by someone else.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import secrets
import time
from types import TracebackType

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

_RELEASE = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""

_EXTEND = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""


class LockNotAcquiredError(TimeoutError):
    """Another turn held the lock for longer than the wait timeout."""


class LockLostError(RuntimeError):
    """The lock expired or was taken over while we were still working."""


class RedisLock:
    def __init__(
        self,
        redis: Redis,
        key: str,
        ttl_s: float = 30.0,
        wait_timeout_s: float = 20.0,
        poll_initial_s: float = 0.005,
        poll_max_s: float = 0.25,
    ) -> None:
        self.redis = redis
        self.key = key
        self.ttl_ms = int(ttl_s * 1000)
        self.wait_timeout_s = wait_timeout_s
        self.poll_initial_s = poll_initial_s
        self.poll_max_s = poll_max_s
        self.token = secrets.token_hex(16)
        self.lost = False
        self._watchdog: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def try_acquire(self) -> bool:
        return bool(await self.redis.set(self.key, self.token, nx=True, px=self.ttl_ms))

    async def acquire(self) -> bool:
        deadline = time.monotonic() + self.wait_timeout_s
        delay = self.poll_initial_s
        while True:
            if await self.try_acquire():
                self.lost = False
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, random.uniform(delay / 2, delay)))
            delay = min(delay * 2, self.poll_max_s)

    async def release(self) -> bool:
        return bool(await self.redis.eval(_RELEASE, 1, self.key, self.token))

    async def extend(self) -> bool:
        return bool(await self.redis.eval(_EXTEND, 1, self.key, self.token, self.ttl_ms))

    def ensure_held(self) -> None:
        if self.lost:
            raise LockLostError(f"lock {self.key} was lost")

    async def _keep_alive(self) -> None:
        interval = self.ttl_ms / 1000 / 3
        while True:
            try:
                # Wake up every `interval`, or immediately when the holder is done.
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                pass
            try:
                if not await self.extend():
                    self.lost = True
                    logger.error("conversation lock lost", extra={"lock": self.key})
                    return
            except Exception:
                logger.warning("lock extend failed; will retry", exc_info=True)

    async def __aenter__(self) -> RedisLock:
        if not await self.acquire():
            raise LockNotAcquiredError(f"could not acquire {self.key}")
        self._stop.clear()
        self._watchdog = asyncio.create_task(self._keep_alive())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._watchdog is not None:
            # Signal instead of cancelling, so an in-flight EXTEND is never interrupted
            # half-way through a Redis round trip.
            self._stop.set()
            try:
                await asyncio.wait_for(self._watchdog, timeout=self.ttl_ms / 1000)
            except asyncio.TimeoutError:
                self._watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._watchdog
            self._watchdog = None
        try:
            await self.release()
        except Exception:
            # The TTL guarantees eventual release even if Redis is briefly unreachable.
            logger.warning("lock release failed", extra={"lock": self.key}, exc_info=True)


class ConversationLocks:
    def __init__(self, redis: Redis, ttl_s: float, wait_timeout_s: float) -> None:
        self.redis = redis
        self.ttl_s = ttl_s
        self.wait_timeout_s = wait_timeout_s

    def for_conversation(self, conversation_id: str) -> RedisLock:
        return RedisLock(
            self.redis,
            f"lock:conversation:{conversation_id}",
            ttl_s=self.ttl_s,
            wait_timeout_s=self.wait_timeout_s,
        )
