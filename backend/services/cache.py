"""Session-scoped result cache.

This is layer 2 of two deliberately distinct caches:

    layer 1  charts.get_parsed_table   (database, isotope, dataset, value, field)
             raw parsed tuples; avoids re-running the 60s JANIS subprocess;
             process-global, public upstream data, no user identity.

    layer 2  this module                session_id:query_hash
             numpy SeriesArrays, pre-sorted and render-ready; avoids
             re-parsing and re-sorting on every slider movement.

Layer 1 is intentionally left alone - JANIS results are effectively
immutable, so its lifetime is nothing like an analysis session's.

The cache instance is process-local. See backend/deployment.py for why the
application is pinned to a single worker, and what would have to change to
lift that.
"""

import os
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Protocol

from ..deployment import env_int


class DataCache(Protocol):
    """The swap point for Redis, a filesystem cache, or any shared store."""

    def get(self, key: str) -> Any | None: ...

    def put(
        self,
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
        size: int | None = None,
    ) -> None: ...

    def delete(self, key: str) -> None: ...


class _Entry:
    __slots__ = ("value", "expires_at", "nbytes")

    def __init__(self, value: Any, expires_at: float, nbytes: int):
        self.value = value
        self.expires_at = expires_at
        self.nbytes = nbytes


class TTLMemoryCache:
    """Bounded, expiring, thread-safe LRU cache.

    Thread safety is mandatory rather than defensive: every Dash callback
    body is dispatched through `dash_ui.callbacks.offload`, which runs it in
    anyio's worker threadpool, so two browser tabs genuinely execute
    `get`/`put` concurrently on different OS threads, and
    `OrderedDict.move_to_end` plus eviction is a read-modify-write that is
    not atomic. (Without that offload Dash would run callbacks inline on the
    event loop, serializing everything - the locking here would still be
    correct, just unexercised.)

    Both bounds are enforced. An entry-count bound alone is meaningless when
    one entry can be 5 x 30,000 x 3 x 8 bytes = 3.6 MB and another is 40 KB.
    """

    def __init__(
        self,
        max_entries: int | None = None,
        max_bytes: int | None = None,
        ttl_seconds: float | None = None,
    ):
        self.max_entries = max_entries or env_int("QUERY_CACHE_MAX_ENTRIES", 32, minimum=1)
        self.max_bytes = max_bytes or (
            env_int("QUERY_CACHE_MAX_MB", 256, minimum=1) * 1024 * 1024
        )
        self.ttl_seconds = (
            ttl_seconds
            if ttl_seconds is not None
            else env_int("QUERY_CACHE_TTL_SECONDS", 1800, minimum=1)
        )
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    # `time.monotonic`, never `time.time`: a wall-clock step must not
    # extend entries indefinitely or expire them all at once.
    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= self._now():
                self._discard(key)
                return None
            self._entries.move_to_end(key)
            return entry.value

    def put(
        self,
        key: str,
        value: Any,
        *,
        ttl: float | None = None,
        size: int | None = None,
    ) -> None:
        lifetime = self.ttl_seconds if ttl is None else ttl
        entry = _Entry(
            value=value,
            expires_at=self._now() + lifetime,
            nbytes=max(0, int(size or 0)),
        )
        with self._lock:
            self._discard(key)
            self._entries[key] = entry
            self._bytes += entry.nbytes
            self._evict()

    def delete(self, key: str) -> None:
        with self._lock:
            self._discard(key)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return self._bytes

    def _discard(self, key: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._bytes -= entry.nbytes

    def _evict(self) -> None:
        """Drop expired entries, then LRU-evict until both bounds hold."""
        now = self._now()
        for key in [
            key
            for key, entry in self._entries.items()
            if entry.expires_at <= now
        ]:
            self._discard(key)

        while self._entries and (
            len(self._entries) > self.max_entries or self._bytes > self.max_bytes
        ):
            oldest = next(iter(self._entries))
            self._discard(oldest)


@lru_cache(maxsize=1)
def get_cache() -> DataCache:
    """Return the process cache.

    The only module-level global here, and it holds no user-specific state
    itself - per-user state lives inside the instance, partitioned by the
    session id component of every key. Swapping in Redis or a filesystem
    cache means changing this one function.
    """
    backend = os.getenv("QUERY_CACHE_BACKEND", "memory").lower()
    if backend != "memory":
        raise ValueError(
            f"Unsupported QUERY_CACHE_BACKEND={backend!r}. Only 'memory' is "
            f"implemented; see the scaling notes in backend/deployment.py."
        )
    return TTLMemoryCache()
