"""Concurrency: the event loop stays free and the query limit binds.

No other module in this suite can catch this class of bug, because they all
drive one callback at a time. The failure being guarded against is that Dash's
FastAPI backend runs a *sync* callback body inline on the event loop
(`response_data = ctx.run(partial_func)` in
`dash/backends/_fastapi.py::serve_callback`), so one slow callback blocks every
other request in the process - other callbacks, `/api/*`, and the `/api/health`
the container healthcheck depends on. `dash_ui.callbacks.offload` is what
prevents it.

Every assertion here is on **absolute timestamps, not per-request durations**.
A duration-only assertion passes even when the loop is completely blocked: the
request cannot start until the loop frees up, and then completes instantly, so
it measures a short duration and looks healthy. Recording when each request
*began* relative to a common origin is what distinguishes the two.
"""

import asyncio
import threading
import time
import unittest
from unittest.mock import patch

import httpx

from backend import deployment
from backend.main import app
from backend.services import query_store
from backend.services.cache import get_cache

# Long enough to dominate scheduling jitter, short enough to keep the suite
# quick. Assertions compare against fractions of it rather than absolutes.
DELAY = 0.4

FAKE_TABLE = tuple((10.0**exponent, 2.0 + exponent) for exponent in range(-4, 4))

# Distinct isotopes give distinct cache keys, so concurrent queries genuinely
# do the work instead of the second one hitting the first one's cache entry.
ISOTOPES = ["Co59", "Fe56", "U235", "H1", "Pu239"]


class JanisTracker:
    """Stands in for the JANIS subprocess and records real overlap.

    Counting how many calls are inside the fake at once is the direct
    measurement of "how many users can query JANIS simultaneously" - far
    stronger than timing the HTTP requests, which only shows when they were
    dispatched, not when work actually began.
    """

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(DELAY)
        with self._guard:
            self.active -= 1
        return FAKE_TABLE


class ConcurrencyTestCase(unittest.IsolatedAsyncioTestCase):
    """Drives the real ASGI app over the real Dash wire protocol."""

    async def asyncSetUp(self):
        get_cache().clear()
        query_store._reset_queue_gate()
        self.origin = time.monotonic()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://testserver",
            timeout=120,
        )
        deps = (await self.client.get("/_dash-dependencies")).json()
        self.run_cb = next(
            dep
            for dep in deps
            for item in dep["inputs"]
            if f"{item['id']}.{item['property']}" == "run-btn.n_clicks"
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        query_store._reset_queue_gate()

    def at(self) -> float:
        return time.monotonic() - self.origin

    @staticmethod
    def _outputs(output: str):
        return [
            {"id": part.rpartition(".")[0], "property": part.rpartition(".")[2]}
            for part in output[2:-2].split("...")
        ]

    def _payload(self, session_id: str, isotope: str) -> dict:
        values = {"run-btn.n_clicks": 1}
        state = {
            "session-store.data": {"sid": session_id},
            "comparison-mode.value": "single",
            "databases.value": "DB-A",
            "isotopes.value": isotope,
            "datasets.value": "MT102",
            "reaction-type.value": "xs",
        }
        return {
            "output": self.run_cb["output"],
            "outputs": self._outputs(self.run_cb["output"]),
            "inputs": [
                {
                    "id": spec["id"],
                    "property": spec["property"],
                    "value": values.get(f"{spec['id']}.{spec['property']}"),
                }
                for spec in self.run_cb["inputs"]
            ],
            "state": [
                {
                    "id": spec["id"],
                    "property": spec["property"],
                    "value": state.get(f"{spec['id']}.{spec['property']}"),
                }
                for spec in self.run_cb.get("state", [])
            ],
            "changedPropIds": ["run-btn.n_clicks"],
        }

    async def run_query(self, index: int) -> dict:
        """Fire one Run and report when it started and finished."""
        started = self.at()
        response = await self.client.post(
            "/_dash-update-component",
            json=self._payload(str(index) * 32, ISOTOPES[index]),
        )
        self.assertEqual(response.status_code, 200, response.text)
        return {
            "started": started,
            "finished": self.at(),
            "body": response.json()["response"],
        }

    def janis_stubbed(self, tracker, limit):
        """Stub the JANIS calls, pin the limit, and re-size the gate.

        `query_store` does `from ..deployment import max_concurrent_queries`,
        so it holds its own name binding - patching only
        `backend.deployment.max_concurrent_queries` leaves the admission gate
        reading the real value. Both bindings have to be patched.

        The patches go live immediately and unwind at cleanup, so the gate is
        reset here - AFTER they are active - and is therefore sized from the
        patched limit rather than the real environment. Callers used to have
        to remember that ordering themselves, between a manual `start()` loop
        and a `try/finally` `stop()` loop.
        """
        for target, kwargs in (
            ("backend.deployment.max_concurrent_queries", {"return_value": limit}),
            (
                "backend.services.query_store.max_concurrent_queries",
                {"return_value": limit},
            ),
            ("backend.jar_runner.list_databases", {"return_value": ["DB-A"]}),
            ("backend.jar_runner.list_isotopes", {"return_value": ISOTOPES}),
            ("backend.jar_runner.list_all_datasets", {"return_value": ["MT102"]}),
            (
                "backend.jar_runner.list_quantities",
                {"return_value": ["xs", "xs_stddev"]},
            ),
            ("backend.charts.get_parsed_table", {"side_effect": tracker}),
        ):
            self.enterContext(patch(target, **kwargs))
        query_store._reset_queue_gate()


class EventLoopTests(ConcurrencyTestCase):
    async def test_health_answers_while_a_query_is_running(self):
        """The regression guard for the blocked event loop.

        Before `offload`, a health check fired 0.5s into a 3s query could not
        even begin until the query finished - which is what makes a container
        healthcheck fail and an orchestrator restart the process mid-query.
        """

        async def health():
            await asyncio.sleep(DELAY / 4)
            started = self.at()
            response = await self.client.get("/api/health")
            return started, response

        self.janis_stubbed(JanisTracker(), limit=2)
        query, (health_started, response) = await asyncio.gather(
            self.run_query(0), health()
        )

        self.assertEqual(response.status_code, 200)
        # Health began while the query was still in flight, not after it.
        self.assertLess(health_started, query["finished"])
        self.assertGreater(query["finished"], DELAY * 0.8)


class QueryLimitTests(ConcurrencyTestCase):
    async def _fire(self, count: int, limit: int):
        tracker = JanisTracker()
        self.janis_stubbed(tracker, limit)
        results = await asyncio.gather(
            *[self.run_query(i) for i in range(count)]
        )
        return results, tracker

    async def test_queries_run_in_parallel_up_to_the_limit(self):
        results, tracker = await self._fire(count=3, limit=3)
        self.assertEqual(tracker.max_active, 3)
        # Three parallel slots means one round, not three.
        self.assertLess(max(r["finished"] for r in results), DELAY * 2)

    async def test_a_limit_of_one_serializes_queries(self):
        """Proves the setting binds rather than the parallelism being free."""
        results, tracker = await self._fire(count=2, limit=1)
        self.assertEqual(tracker.max_active, 1)
        self.assertGreater(max(r["finished"] for r in results), DELAY * 1.8)

    async def test_a_query_over_the_limit_waits_and_then_succeeds(self):
        """Over-limit arrivals queue; they are not turned away."""
        results, tracker = await self._fire(count=3, limit=2)
        self.assertEqual(tracker.max_active, 2)
        for result in results:
            status = result["body"]["status-line"]["children"]
            self.assertIn("Loaded 1 series", status)
        # Three queries through two slots takes two rounds, not one.
        self.assertGreater(max(r["finished"] for r in results), DELAY * 1.8)


class QueueBudgetTests(unittest.TestCase):
    def test_waiting_for_a_slot_is_bounded_by_the_query_budget(self):
        """Queueing and running share one deadline.

        Two budgets - one to wait, one to run - would double worst-case
        latency and blow past the proxy read timeout the budget exists to
        stay under.
        """
        query_store._reset_queue_gate()
        self.addCleanup(query_store._reset_queue_gate)

        with patch(
            "backend.deployment.max_concurrent_queries", return_value=1
        ), patch(
            "backend.services.query_store.max_concurrent_queries",
            return_value=1,
        ), patch(
            "backend.services.query_store.query_budget_seconds", return_value=1
        ):
            gate = query_store._queue_gate()
            gate.acquire()  # occupy the only slot
            self.addCleanup(gate.release)

            import backend.charts as charts

            query = charts.CrossSectionQuery(
                databases=["A"], isotopes=["I"], datasets=["D"]
            )
            started = time.monotonic()
            with self.assertRaises(query_store.JanisError) as context:
                query_store.load_query("a" * 32, query)
            waited = time.monotonic() - started

        self.assertIn("free query slot", str(context.exception))
        self.assertIn("JANIS_MAX_CONCURRENT_QUERIES", str(context.exception))
        # Gave up at the budget rather than blocking forever.
        self.assertGreaterEqual(waited, 0.9)
        self.assertLess(waited, 3.0)


class SettingTests(unittest.TestCase):
    def test_legacy_names_still_configure_the_limit(self):
        """An existing deployment must not silently change behaviour."""
        for legacy in ("JANIS_QUEUE_DEPTH", "JANIS_MAX_CONCURRENCY"):
            with self.subTest(legacy=legacy), patch.dict(
                "os.environ", {legacy: "5"}, clear=False
            ):
                import os

                os.environ.pop("JANIS_MAX_CONCURRENT_QUERIES", None)
                self.assertEqual(deployment.max_concurrent_queries(), 5)

    def test_new_name_wins_over_legacy_names(self):
        with patch.dict(
            "os.environ",
            {"JANIS_MAX_CONCURRENT_QUERIES": "3", "JANIS_QUEUE_DEPTH": "9"},
            clear=False,
        ):
            self.assertEqual(deployment.max_concurrent_queries(), 3)

    def test_the_limit_never_drops_below_one(self):
        for value in ("0", "-4", "nonsense"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"JANIS_MAX_CONCURRENT_QUERIES": value}, clear=False
            ):
                self.assertGreaterEqual(deployment.max_concurrent_queries(), 1)


if __name__ == "__main__":
    unittest.main()
