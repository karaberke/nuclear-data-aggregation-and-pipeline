"""Construct the Dash application on an existing FastAPI server."""

import os
from pathlib import Path

from dash import Dash
from fastapi import FastAPI

from .callbacks import register
from .layout import build_layout

DEFAULT_URL_PREFIX = "/"

# Absolute, so asset resolution does not depend on which module calls
# create_dash. Dash derives assets_folder from the *caller's* root path, so a
# relative value resolves against backend/ when main.py mounts the app and
# would silently 404 every stylesheet.
ASSETS_FOLDER = str(Path(__file__).resolve().parent / "assets")


def _normalize_prefix(prefix: str) -> str:
    """Dash requires a pathname prefix with both a leading and trailing slash."""
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    return prefix


def _env_prefix(name: str, default: str) -> str:
    """Read a prefix variable, treating set-but-empty as unset.

    `os.getenv(name, default)` returns "" when the variable is defined but
    blank - which is what `docker compose` injects for
    `"${DASH_REQUESTS_PREFIX:-}"`. Normalizing "" produces "/", so a UI served
    under a sub-path would tell the browser to fetch its bundles from the
    root instead, and the page would hang on "Loading..." forever.
    """
    value = os.getenv(name) or ""
    return _normalize_prefix(value.strip() or default)


def url_prefix() -> str:
    return _env_prefix("DASH_URL_PREFIX", DEFAULT_URL_PREFIX)


def requests_prefix() -> str:
    """Public-facing prefix, which differs from `url_prefix` only when the UI
    is reached under a path this process does not itself serve - a
    path-rewriting proxy, or sub-path hosting alongside another app (see the
    deployment notes in README.md)."""
    return _env_prefix("DASH_REQUESTS_PREFIX", url_prefix())


def create_dash(server: FastAPI, prefix: str | None = None) -> Dash:
    """Mount a Dash UI onto `server` and return it.

    Passing a `FastAPI` instance makes Dash select its native ASGI backend
    (`dash.backends._fastapi.FastAPIDashServer`), so the UI and the REST API
    share one event loop and one process. `dash_app.server is server` holds
    afterwards; `tests/test_dash_smoke.py` asserts both facts, because a
    silent fall back to the Flask/WSGI backend would surface only as
    event-loop blocking under load.
    """
    routes_prefix = _normalize_prefix(prefix) if prefix else url_prefix()
    reqs_prefix = requests_prefix() if prefix is None else routes_prefix

    dash_app = Dash(
        server=server,
        routes_pathname_prefix=routes_prefix,
        requests_pathname_prefix=reqs_prefix,
        assets_folder=ASSETS_FOLDER,
        suppress_callback_exceptions=True,
        title="Nuclear Data Explorer",
        update_title=None,
    )
    dash_app.layout = build_layout()
    register(dash_app)
    return dash_app
