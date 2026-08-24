"""Enforcement tests for the two accepted deployment limits.

These exist so that a later scale-out or ingress change fails loudly rather
than silently running N concurrent JANIS subprocesses or truncating queries
at a proxy. See backend/deployment.py.
"""

import os
import unittest
from unittest.mock import patch

from backend import deployment


class SingleWorkerGuardTests(unittest.TestCase):
    """The JANIS semaphore is process-local, so >1 worker is unsupported."""

    def test_single_worker_is_accepted(self):
        with patch.dict(os.environ, {}, clear=True):
            deployment.enforce_single_worker()

    def test_explicit_single_worker_is_accepted(self):
        with patch.dict(os.environ, {"WEB_CONCURRENCY": "1"}, clear=True):
            deployment.enforce_single_worker()

    def test_multiple_workers_fail_fast(self):
        for variable in (
            "WEB_CONCURRENCY",
            "UVICORN_WORKERS",
            "GUNICORN_WORKERS",
        ):
            with self.subTest(variable=variable):
                with patch.dict(os.environ, {variable: "4"}, clear=True):
                    with self.assertRaises(
                        deployment.UnsupportedDeploymentError
                    ) as caught:
                        deployment.enforce_single_worker()
                # The message has to say why, not just "unsupported".
                self.assertIn("JANIS", str(caught.exception))
                self.assertIn("--workers 1", str(caught.exception))

    def test_override_permits_multiple_workers_but_warns(self):
        with patch.dict(
            os.environ,
            {"WEB_CONCURRENCY": "4", "ALLOW_UNSAFE_MULTIWORKER": "1"},
            clear=True,
        ):
            with self.assertLogs(deployment.logger, level="WARNING") as logs:
                deployment.enforce_single_worker()
        self.assertIn("per-process", "\n".join(logs.output))

    def test_unparseable_worker_count_does_not_crash(self):
        with patch.dict(os.environ, {"WEB_CONCURRENCY": "many"}, clear=True):
            deployment.enforce_single_worker()


class QueryBudgetTests(unittest.TestCase):
    """The budget must sit below the smallest proxy read timeout."""

    def test_default_budget(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(deployment.query_budget_seconds(), 180)

    def test_budget_is_configurable(self):
        # e.g. behind Cloudflare, whose non-Enterprise 100s origin cap
        # cannot be raised.
        with patch.dict(
            os.environ, {"JANIS_QUERY_BUDGET_SECONDS": "90"}, clear=True
        ):
            self.assertEqual(deployment.query_budget_seconds(), 90)

    def test_invalid_budget_falls_back_to_default(self):
        with patch.dict(
            os.environ, {"JANIS_QUERY_BUDGET_SECONDS": "none"}, clear=True
        ):
            self.assertEqual(deployment.query_budget_seconds(), 180)

    def test_budget_is_never_zero_or_negative(self):
        with patch.dict(
            os.environ, {"JANIS_QUERY_BUDGET_SECONDS": "0"}, clear=True
        ):
            self.assertGreaterEqual(deployment.query_budget_seconds(), 1)


if __name__ == "__main__":
    unittest.main()
