"""Tier 1: Dash and FastAPI coexist in one ASGI process.

The layout assertions below check that every element id the callback graph
depends on actually exists, without needing a browser. They are what replaced
the SPA's HTML-contract test at cutover.

No selenium and no pytest - `dash[testing]` pins selenium<=4.2.0 and would
also start Dash on its own server thread, which is a different code path from
the FastAPI-hosted one we ship.
"""

import json
import os
import re
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.dash_ui import app as dash_app_module
from backend.dash_ui.app import create_dash
from backend.main import app, dash_app

_CONFIG_PATTERN = re.compile(
    r'<script id="_dash-config" type="application/json">(.*?)</script>',
    re.DOTALL,
)


def dash_config(html: str) -> dict:
    """Extract the `_dash-config` JSON the renderer bootstraps from."""
    match = _CONFIG_PATTERN.search(html)
    if match is None:  # pragma: no cover - would fail the calling assertion
        raise AssertionError("served page has no _dash-config block")
    return json.loads(match.group(1))

# Component ids the layout must expose. Phase 4 extends this list as the
# selection and analysis panels are built; a callback referencing an id that
# is not here will be caught by CallbackGraphTests below.
REQUIRED_LAYOUT_IDS = {
    # stores and page shell
    "page-root", "app-location", "session-store", "query-store", "axis-store",
    "download-filtered", "download-processed",
    # selection panel
    "comparison-mode", "databases", "databases-hint", "field",
    "isotopes", "isotopes-hint", "datasets", "datasets-hint",
    "reaction-type", "run-btn", "export-btn", "export-processed-btn",
    "status-line", "selection-summary", "selection-section",
    # analysis panel
    "analysis-section", "reset-controls",
    "energy-min", "energy-max", "value-min", "value-max",
    "chart-tabs", "x-scale", "y-scale",
    "bin-controls", "group-structure-mode",
    "automatic-group-controls", "bin-count",
    "group-energy-min", "group-energy-max",
    "standard-group-controls", "group-structure-preset",
    "custom-group-controls", "bin-edges",
    "bar-mode", "kde-controls", "bandwidth",
    "series-legend",
    # ratio / difference panel
    "comparison-controls", "comparison-metric", "comparison-reference",
    "comparison-targets", "comparison-swap", "comparison-formula",
    # chart panel
    "chart-section", "chart-notice", "chart-error", "chart-graph",
}


def collect_ids(node) -> set[str]:
    """Recursively collect every `props.id` in a serialized Dash layout."""
    found: set[str] = set()
    if isinstance(node, list):
        for item in node:
            found |= collect_ids(item)
        return found
    if not isinstance(node, dict):
        return found

    props = node.get("props", {})
    identifier = props.get("id")
    if isinstance(identifier, str):
        found.add(identifier)
    children = props.get("children")
    if children is not None:
        found |= collect_ids(children)
    return found


class CoexistenceTests(unittest.TestCase):
    """The API, its docs, and the Dash UI all serve from one application."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_dash_uses_the_native_fastapi_backend(self):
        """The highest-value assertion in the suite.

        Dash still ships Flask as a base dependency. If a future release
        changed its server detection and fell back to the WSGI backend, the
        only symptom would be event-loop blocking under load - not an error.
        """
        self.assertIsNotNone(dash_app, "Dash did not mount")
        self.assertIs(dash_app.server, app)
        self.assertEqual(type(dash_app.backend).__name__, "FastAPIDashServer")

    def test_health_route_still_works(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_health_reports_the_single_process_limit(self):
        payload = self.client.get("/api/health").json()
        self.assertIs(payload["single_process"], True)
        self.assertIsInstance(payload["query_budget_seconds"], int)

    def test_openapi_schema_is_intact(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]
        # Mounting Dash must not shadow or drop the API surface.
        self.assertIn("/api/cross-sections/query", paths)
        self.assertIn("/api/databases", paths)
        self.assertIn("/api/health", paths)

    def test_docs_are_reachable(self):
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_unknown_api_paths_404_instead_of_serving_the_ui(self):
        """Dash at "/" is a catch-all; /api/* must not fall through to it.

        Without the guard in main.py these answer with the app shell and a
        200, which an API client cannot tell apart from success. Covers the
        routes retired at cutover as well as plain typos.

        Must run inside the lifespan context: Dash registers its catch-all at
        startup, so a bare TestClient would 404 these anyway and the
        assertion would hold even with the guard deleted.
        """
        with TestClient(app) as client:
            for path in (
                "/api/fields",
                "/api/cross-section",
                "/api/cross-section/export",
                "/api/does-not-exist",
            ):
                with self.subTest(path=path):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 404)
                    self.assertIn(
                        "application/json", response.headers["content-type"]
                    )

    def test_unknown_ui_paths_serve_the_app_shell(self):
        """The other half of the rule: client-side routing needs the catch-all.

        This is also the assertion that pins *when* Dash's catch-all appears -
        at lifespan startup, not at import.
        """
        with TestClient(app) as client:
            response = client.get("/some/deep/link")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_dash_index_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("_dash-renderer", response.text)

    def test_dash_serves_its_stylesheet(self):
        response = self.client.get("/assets/app.css")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/css", response.headers["content-type"])

    def test_served_page_advertises_the_requests_prefix(self):
        """A wrong prefix renders once, then 404s on every interaction.

        Read the embedded config rather than string-matching the HTML: Dash
        escapes forward slashes as \\u002f in that JSON blob.
        """
        config = dash_config(self.client.get("/").text)
        self.assertEqual(config["requests_pathname_prefix"], "/")


class LayoutContractTests(unittest.TestCase):
    """Every component id the callback graph depends on must be rendered."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_layout_endpoint_serves_json(self):
        response = self.client.get("/_dash-layout")
        self.assertEqual(response.status_code, 200)

    def test_layout_exposes_every_required_id(self):
        layout = self.client.get("/_dash-layout").json()
        present = collect_ids(layout)
        missing = REQUIRED_LAYOUT_IDS - present
        self.assertEqual(missing, set(), f"layout is missing ids: {missing}")


class CallbackGraphTests(unittest.TestCase):
    """Static validator: no callback may reference an id that isn't rendered.

    This catches typo'd component ids at test time instead of as a silent
    no-op in the browser.
    """

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_every_callback_target_exists_in_the_layout(self):
        layout_ids = collect_ids(self.client.get("/_dash-layout").json())
        dependencies = self.client.get("/_dash-dependencies").json()

        referenced: set[str] = set()
        for callback in dependencies:
            entries = list(callback.get("inputs", []))
            entries += list(callback.get("state", []))
            outputs = callback.get("output", "")
            for group in (entries, callback.get("outputs", [])):
                for item in group:
                    identifier = item.get("id")
                    if isinstance(identifier, str):
                        referenced.add(identifier)
            # Single-output callbacks encode the target as "id.property".
            if isinstance(outputs, str) and outputs and ".." not in outputs:
                referenced.add(outputs.rsplit(".", 1)[0])

        unknown = {name for name in referenced if name and name not in layout_ids}
        self.assertEqual(
            unknown, set(), f"callbacks reference ids not in the layout: {unknown}"
        )

    def test_run_button_locks_itself_while_a_query_is_in_flight(self):
        """The Run button must not be clickable twice for one query.

        Every press costs an admission slot and a `java -jar Janis.jar` JVM, so
        a double-click runs the same query twice and burns both default slots.
        The guard is a single `running=` keyword on the callback: drop it and
        the old behaviour returns silently, with no other test noticing. Hence
        this assertion against the served spec rather than the source.
        """
        dependencies = self.client.get("/_dash-dependencies").json()
        matches = [
            callback
            for callback in dependencies
            if any(
                f"{item['id']}.{item['property']}" == "run-btn.n_clicks"
                for item in callback.get("inputs", [])
            )
        ]
        self.assertEqual(len(matches), 1, "expected exactly one Run callback")
        running = matches[0].get("running")
        self.assertIsNotNone(
            running, "the Run callback lost its `running=` spec"
        )

        self.assertIs(running["running"]["run-btn.disabled"], True)
        # False, never True: a wrong value here strands the button disabled
        # and makes the app unusable, which is far worse than the edge case
        # it would be guarding against. See the note in callbacks.py.
        self.assertIs(running["runningOff"]["run-btn.disabled"], False)

        # The progress text is the half that stops users pressing Run again.
        self.assertTrue(running["running"]["status-line.children"])
        self.assertEqual(
            running["runningOff"]["run-btn.children"], "Run comparison"
        )


class PrefixTests(unittest.TestCase):
    """The index and its assets must work at the root and under a sub-path.

    Root is what ships. The sub-path case stays covered because a proxy that
    mounts this app somewhere other than / is the one configuration where a
    wrong prefix renders the page once and then 404s every interaction.
    """

    def _mount(self, prefix: str) -> TestClient:
        throwaway = FastAPI()

        @throwaway.get("/api/health")
        def health():  # pragma: no cover - trivial probe
            return {"status": "ok"}

        create_dash(throwaway, prefix=prefix)
        return TestClient(throwaway)

    def test_custom_prefix(self):
        """Sub-path hosting still works, for a proxy that mounts us under one."""
        client = self._mount("/dash/")
        self.assertEqual(client.get("/dash/").status_code, 200)
        self.assertEqual(client.get("/dash/assets/app.css").status_code, 200)
        self.assertEqual(client.get("/api/health").status_code, 200)

    def test_root_prefix(self):
        client = self._mount("/")
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/assets/app.css").status_code, 200)
        # The whole point of registering Dash last: the API still wins.
        self.assertEqual(client.get("/api/health").status_code, 200)


if __name__ == "__main__":
    unittest.main()


class PrefixEnvironmentTests(unittest.TestCase):
    """Regression: a set-but-empty prefix variable must behave as unset.

    `docker compose` renders `"${DASH_REQUESTS_PREFIX:-}"` as an empty
    string, not an absent variable. Reading it with a plain
    `os.getenv(name, default)` yielded "" -> "/", so a UI served under a
    sub-path advertised "/" for its bundles instead. The requests then missed
    Dash's routes entirely and the page hung on "Loading..." with nothing
    resembling an error.
    """

    def test_empty_requests_prefix_falls_back_to_the_url_prefix(self):
        with patch.dict(
            os.environ,
            {"DASH_URL_PREFIX": "/dash/", "DASH_REQUESTS_PREFIX": ""},
            clear=False,
        ):
            self.assertEqual(dash_app_module.url_prefix(), "/dash/")
            self.assertEqual(dash_app_module.requests_prefix(), "/dash/")

    def test_empty_url_prefix_falls_back_to_the_default(self):
        with patch.dict(os.environ, {"DASH_URL_PREFIX": ""}, clear=False):
            self.assertEqual(dash_app_module.url_prefix(), "/")

    def test_whitespace_is_treated_as_unset(self):
        with patch.dict(
            os.environ,
            {"DASH_URL_PREFIX": "", "DASH_REQUESTS_PREFIX": "   "},
            clear=False,
        ):
            self.assertEqual(dash_app_module.requests_prefix(), "/")

    def test_an_explicit_requests_prefix_is_still_honoured(self):
        with patch.dict(
            os.environ,
            {"DASH_URL_PREFIX": "/dash/", "DASH_REQUESTS_PREFIX": "/explorer/"},
            clear=False,
        ):
            self.assertEqual(dash_app_module.requests_prefix(), "/explorer/")

    def test_served_bundles_use_the_same_prefix_as_the_page(self):
        """The end-to-end invariant the unit tests above protect."""
        html = TestClient(app).get("/").text
        config = dash_config(html)
        prefix = config["requests_pathname_prefix"]
        scripts = re.findall(r'<script src="([^"]+)"', html)
        suites = [s for s in scripts if "_dash-component-suites" in s]
        self.assertTrue(suites, "no component-suite scripts in the page")
        for src in suites:
            self.assertTrue(
                src.startswith(prefix),
                f"bundle {src} does not start with prefix {prefix}",
            )
