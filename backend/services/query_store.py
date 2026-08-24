"""Bridge between the session cache and the JANIS-backed series builder.

The browser holds only a small handle (session id + the query itself + per
series metadata); the 30,000-point arrays never leave the server. Because the
handle embeds the full query, a cache miss is self-healing: `load_query`
re-materializes from layer 1, which is normally a sub-millisecond
`charts.get_parsed_table` hit and only reaches JANIS if that expired too.
That single decision turns TTL expiry, LRU eviction, and multi-worker routing
from user-visible failures into invisible latency.
"""

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from itertools import product

from .. import charts
from ..deployment import max_concurrent_queries, query_budget_seconds
from . import analytics
from .cache import get_cache
from .errors import JanisError

logger = logging.getLogger(__name__)

SESSION_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

# Admission gate: how many users may have a query running at once. Sized
# from the same setting as `jar_runner._JANIS_SEMAPHORE`, which is correct
# because a query runs its series sequentially and so holds at most one
# subprocess slot at a time - see deployment.max_concurrent_queries().
#
# Callbacks reach this from anyio worker threads (backend/dash_ui/callbacks.py
# offloads every body with `run_in_threadpool`), so arrivals genuinely overlap
# and blocking here parks a worker thread rather than the event loop.
_QUEUE_GATE: threading.Semaphore | None = None
_GATE_LOCK = threading.Lock()


def _queue_gate() -> threading.Semaphore:
    global _QUEUE_GATE
    with _GATE_LOCK:
        if _QUEUE_GATE is None:
            _QUEUE_GATE = threading.Semaphore(max_concurrent_queries())
        return _QUEUE_GATE


def _reset_queue_gate() -> None:
    """Drop the memoized gate so a test can re-read the environment."""
    global _QUEUE_GATE
    with _GATE_LOCK:
        _QUEUE_GATE = None


def new_session_id() -> str:
    return uuid.uuid4().hex


def valid_session_id(session_id: str | None) -> bool:
    """A session id is a cache partition key, never an authorization token.

    Strict validation is what makes the `session:query` key scheme
    injection-proof: a ':' cannot appear in a valid id, so a crafted value
    cannot collide with another session's partition.
    """
    return bool(session_id) and bool(SESSION_ID_PATTERN.match(session_id))


def ensure_session_id(session_id: str | None) -> str:
    return session_id if valid_session_id(session_id) else new_session_id()


def query_hash(query: charts.CrossSectionQuery) -> str:
    canonical = json.dumps(
        query.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.blake2b(
        canonical.encode("utf-8"), digest_size=16
    ).hexdigest()


def cache_key(session_id: str, query: charts.CrossSectionQuery) -> str:
    return f"{session_id}:{query_hash(query)}"


@dataclass(frozen=True, slots=True)
class QueryHandle:
    """What the browser holds. Small enough for a dcc.Store, by design."""

    session_id: str
    query_id: str
    series: list[analytics.SeriesArrays]

    @property
    def total_points(self) -> int:
        return sum(item.size for item in self.series)


def _nbytes(series: list[analytics.SeriesArrays]) -> int:
    total = 0
    for item in series:
        total += item.energy.nbytes + item.sigma.nbytes
        if item.stddev is not None:
            total += item.stddev.nbytes
    return total


def load_query(
    session_id: str, query: charts.CrossSectionQuery
) -> QueryHandle:
    """Return the series for `query`, from cache when possible.

    Raises `DomainError` subclasses (`SelectionUnavailableError`,
    `DataUnavailableError`, `JanisError`) with messages safe to display.
    """
    key = cache_key(session_id, query)
    cache = get_cache()

    cached = cache.get(key)
    if cached is not None:
        return QueryHandle(session_id, query_hash(query), cached)

    # One deadline covers queueing AND execution. Splitting them would let a
    # request wait a full budget and then run for another, doubling worst-case
    # latency past the proxy read timeout the budget exists to stay under.
    deadline = time.monotonic() + query_budget_seconds()

    gate = _queue_gate()
    if not gate.acquire(timeout=max(0.0, deadline - time.monotonic())):
        logger.warning("no JANIS slot within budget; rejecting query %s", key)
        raise JanisError(
            f"Waited {query_budget_seconds()}s for a free query slot and none "
            f"opened. The server is already running "
            f"{max_concurrent_queries()} queries; try again shortly, or ask "
            f"an operator to raise JANIS_MAX_CONCURRENT_QUERIES."
        )

    try:
        raw = _build_series_within_budget(query, deadline)
    finally:
        gate.release()

    series = [
        analytics.series_from_points(
            item["key"],
            item["database"],
            item["isotope"],
            item["dataset"],
            item["points"],
        )
        for item in raw
    ]
    cache.put(key, series, size=_nbytes(series))
    return QueryHandle(session_id, query_hash(query), series)


def _build_series_within_budget(
    query: charts.CrossSectionQuery, deadline: float
) -> list[dict]:
    """Run the query, abandoning it if it runs past `deadline`.

    JANIS is bounded at 60s per invocation and a 5-series xs_stddev query is
    up to 10 invocations, so a single Run can approach 600s. The deadline is
    checked between series so a hopeless query fails fast with an actionable
    message instead of a proxy returning an opaque gateway error while the
    subprocess keeps running.

    `deadline` is a `time.monotonic()` value set by `load_query` before it
    queued for a slot, so time spent waiting comes out of the same allowance
    as time spent running.
    """
    budget = query_budget_seconds()

    charts._validate_available_selections(query)
    include_stddev = query.reaction_type == "xs_stddev"
    series: list[dict] = []

    for database, isotope, dataset in product(
        query.databases, query.isotopes, query.datasets
    ):
        if series and time.monotonic() > deadline:
            raise JanisError(
                f"This comparison exceeded the {budget}s query budget after "
                f"{len(series)} of {query.series_count} series. Run fewer "
                f"series or narrow the selection."
            )
        try:
            points = charts.build_records(
                database, isotope, dataset, query.field, include_stddev
            )
        except RuntimeError as error:
            # jar_runner raises RuntimeError for timeouts and non-zero exits.
            message = str(error)
            if "timed out" in message:
                raise JanisError(
                    f"JANIS did not respond within "
                    f"{charts.jar_runner.JANIS_TIMEOUT_SECONDS}s for "
                    f"{database}/{isotope}/{dataset}. Try again or choose a "
                    f"different dataset."
                ) from error
            if include_stddev and "Std. deviation" in message:
                # Common and expected: many evaluated libraries publish no
                # covariance data for a given reaction. Say so plainly
                # instead of surfacing the raw Java exception.
                raise JanisError(
                    f"{database} has no standard-deviation data for "
                    f"{isotope} {dataset}. Switch Reaction values back to "
                    f"\"Cross section\", or choose a library that "
                    f"publishes uncertainties."
                ) from error
            if "Can't find data" in message or f"No {isotope} in" in message:
                # Backstop for a reaction node that lists a dataset but holds
                # no table for it. charts._validate_quantity normally catches
                # this first with a fuller message; this keeps any case it
                # misses readable rather than a raw Java exception.
                raise JanisError(
                    f"{database} has no data for {isotope} {dataset}. "
                    f"Choose a different dataset, or drop {database} from "
                    f"the comparison."
                ) from error
            raise JanisError(f"JANIS failed: {message}") from error

        series.append(
            {
                "key": f"{database}|{isotope}|{dataset}",
                "database": database,
                "isotope": isotope,
                "dataset": dataset,
                "points": points,
            }
        )

    return series
