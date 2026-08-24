import logging
import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware

from . import deployment, jar_runner
from .charts import router as charts_router

# uvicorn configures its own loggers but leaves the root logger bare, so
# application log records would otherwise be dropped silently - including the
# deployment-limits line below, which operators need to see in container logs.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s:     %(name)s - %(message)s",
    )

# Fail fast before anything else is constructed - see backend/deployment.py.
deployment.enforce_single_worker()

app = FastAPI(title="Nuclear Data Process API")
app.include_router(charts_router)

# The application's own compression. This is what allowed the nginx reverse
# proxy to be removed - gzip was its only load-bearing feature here. Dash
# callback payloads are large and highly compressible, so do not drop this.
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/api/databases")
def get_databases():
    try:
        return jar_runner.list_databases()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/isotopes")
def get_isotopes(
    database: Annotated[list[str], Query(min_length=1, max_length=5)],
    field: str = Query("SIG"),
):
    try:
        return jar_runner.list_common_isotopes(database, field)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/datasets")
def get_datasets(
    database: Annotated[list[str], Query(min_length=1, max_length=5)],
    isotope: Annotated[list[str], Query(min_length=1, max_length=5)],
    field: str = Query("SIG"),
):
    if len(database) > 1 and len(isotope) > 1:
        raise HTTPException(
            status_code=422,
            detail="only databases or isotopes may contain multiple values",
        )
    try:
        return jar_runner.list_common_datasets(database, isotope, field)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reaction-types")
def get_reaction_types():
    return jar_runner.REACTION_TYPES


@app.get("/api/table")
def get_table(
    database: str,
    isotope: str,
    dataset: str,
    value: str,
    field: str = "SIG",
):
    try:
        return jar_runner.get_table(database, isotope, dataset, value, field)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
def health():
    """Liveness probe.

    `single_process` is additive - existing clients that only read `status`
    are unaffected - and exposes the single-worker deployment limit so an
    operator can see it from outside the container.
    """
    return {
        "status": "ok",
        "single_process": True,
        "pid": os.getpid(),
        "query_budget_seconds": deployment.query_budget_seconds(),
        "max_concurrent_queries": deployment.max_concurrent_queries(),
    }


@app.api_route(
    "/api/{unmatched:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    include_in_schema=False,
)
def api_not_found(unmatched: str):
    """Return a real 404 for unknown /api/* paths.

    Mounting Dash at "/" makes it a catch-all: without this, a typo'd or
    retired endpoint falls through to the UI and answers with the app shell
    and a 200, which an API client cannot distinguish from success. Declared
    after every real /api route and before create_dash, so it only ever
    catches what nothing else claimed.
    """
    raise HTTPException(status_code=404, detail=f"No such endpoint: /api/{unmatched}")


# --------------------------------------------------------------------------
# Dash UI.
#
# Registered LAST, deliberately. Starlette matches routes in registration
# order, and the Dash prefix is "/" - effectively a root catch-all. Mounting
# it after every FastAPI route is what keeps /api/*, /docs, /openapi.json and
# /redoc winning. tests/test_dash_smoke.py asserts this, so moving this block
# up will fail the suite rather than production.
#
# No try/except: a duplicate callback output or a layout typo raises at
# import time and should. Dash is the product now, so a process that boots
# without a UI is not a degraded success - it is a failure that must be
# visible immediately. ENABLE_DASH=0 is the deliberate API-only lever.
# --------------------------------------------------------------------------

dash_app = None

if deployment.env_flag("ENABLE_DASH", default=True):
    # Deliberately deferred, not an oversight: importing this at module scope
    # would pull Dash and Plotly into API-only mode, which ENABLE_DASH=0 exists
    # to avoid. Every other function-level import in this codebase was hoisted.
    from .dash_ui.app import create_dash

    dash_app = create_dash(app)

deployment.log_deployment_limits()
