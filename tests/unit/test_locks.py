import asyncio

import pytest
from fakeredis import FakeAsyncRedis

from agent_backend.core.locks import (
    ConversationLocks,
    LockLostError,
    LockNotAcquiredError,
    RedisLock,
)


@pytest.fixture
def redis() -> FakeAsyncRedis:
    return FakeAsyncRedis(decode_responses=True)


async def test_second_holder_waits_then_times_out(redis: FakeAsyncRedis) -> None:
    first = RedisLock(redis, "lock:a", ttl_s=5, wait_timeout_s=0.1)
    second = RedisLock(redis, "lock:a", ttl_s=5, wait_timeout_s=0.1)
    assert await first.acquire()
    assert not await second.acquire()
    assert await first.release()
    assert await second.acquire()


async def test_only_the_owner_can_release_or_extend(redis: FakeAsyncRedis) -> None:
    owner = RedisLock(redis, "lock:b", ttl_s=5)
    intruder = RedisLock(redis, "lock:b", ttl_s=5)
    assert await owner.acquire()
    assert not await intruder.release()
    assert not await intruder.extend()
    assert await redis.get("lock:b") == owner.token


async def test_expired_lock_can_be_taken_over_and_old_owner_notices(
    redis: FakeAsyncRedis,
) -> None:
    stale = RedisLock(redis, "lock:c", ttl_s=0.05)
    assert await stale.acquire()
    await asyncio.sleep(0.1)  # TTL elapses, e.g. the worker stalled
    fresh = RedisLock(redis, "lock:c", ttl_s=5, wait_timeout_s=0.1)
    assert await fresh.acquire()
    # The stale holder must not be able to release the new owner's lock.
    assert not await stale.release()
    assert await redis.get("lock:c") == fresh.token


async def test_context_manager_raises_when_busy(redis: FakeAsyncRedis) -> None:
    locks = ConversationLocks(redis, ttl_s=5, wait_timeout_s=0.05)
    async with locks.for_conversation("x"):
        with pytest.raises(LockNotAcquiredError):
            async with locks.for_conversation("x"):
                pass  # pragma: no cover
    # Released on exit: can be taken again.
    async with locks.for_conversation("x"):
        pass


async def test_watchdog_extends_ttl_during_long_turns(redis: FakeAsyncRedis) -> None:
    locks = ConversationLocks(redis, ttl_s=0.3, wait_timeout_s=0.05)
    async with locks.for_conversation("long") as held:
        await asyncio.sleep(0.8)  # well past the original TTL
        assert await redis.get("lock:conversation:long") == held.token
        with pytest.raises(LockNotAcquiredError):
            async with locks.for_conversation("long"):
                pass  # pragma: no cover
        held.ensure_held()


async def test_lost_lock_is_detected(redis: FakeAsyncRedis) -> None:
    locks = ConversationLocks(redis, ttl_s=0.3, wait_timeout_s=0.05)
    async with locks.for_conversation("lost") as held:
        await redis.set("lock:conversation:lost", "someone-else")
        await asyncio.sleep(0.25)  # next watchdog tick fails to extend
        with pytest.raises(LockLostError):
            held.ensure_held()


async def test_critical_sections_never_overlap(redis: FakeAsyncRedis) -> None:
    locks = ConversationLocks(redis, ttl_s=5, wait_timeout_s=10)
    active = 0
    max_active = 0
    order: list[int] = []

    async def worker(i: int) -> None:
        nonlocal active, max_active
        async with locks.for_conversation("shared"):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.005)
            order.append(i)
            active -= 1

    await asyncio.gather(*(worker(i) for i in range(25)))
    assert max_active == 1
    assert sorted(order) == list(range(25))


async def test_different_conversations_do_not_block_each_other(redis: FakeAsyncRedis) -> None:
    locks = ConversationLocks(redis, ttl_s=5, wait_timeout_s=0.05)
    async with locks.for_conversation("one"), locks.for_conversation("two"):
        pass
