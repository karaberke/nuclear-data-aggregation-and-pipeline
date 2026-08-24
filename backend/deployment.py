"""Deployment-limit enforcement.

This application is single-container, single-worker only. Two pieces of state
are process-local and have no shared-store implementation:

* `jar_runner._JANIS_SEMAPHORE` and the `query_store` admission gate, both
  sized from `max_concurrent_queries()`. With N worker processes the limit
  becomes N x its configured value, which is the exact resource exhaustion
  the gates exist to prevent. This is the binding constraint.
* `services.cache` and `charts.get_parsed_table` - per-process caches. These
  only degrade (extra JANIS calls, extra memory); they do not corrupt.

Scaling horizontally requires replacing *both* the cache backend and the
JANIS gate. Doing one without the other is worse than doing neither.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

MULTIWORKER_OVERRIDE = "ALLOW_UNSAFE_MULTIWORKER"

CONCURRENT_QUERIES = "JANIS_MAX_CONCURRENT_QUERIES"
DEFAULT_CONCURRENT_QUERIES = 2

# Superseded names, honoured so an existing deployment does not silently
# change behaviour on upgrade. Both used to mean a piece of what the single
# setting now means: JANIS_QUEUE_DEPTH gated admissions, JANIS_MAX_CONCURRENCY
# gated subprocesses. They were always the same number in practice - see
# max_concurrent_queries().
_LEGACY_CONCURRENCY_NAMES = ("JANIS_QUEUE_DEPTH", "JANIS_MAX_CONCURRENCY")


class UnsupportedDeploymentError(RuntimeError):
    """Raised at import time when the process is started multi-worker."""


def env_int(name: str, default: int, minimum: int | None = None) -> int:
    """Read an integer setting, falling back to `default` on anything unusable.

    The single reader for every integer environment setting in the codebase -
    `charts` and `services.cache` import it rather than keeping their own
    copies, which had drifted into three near-identical implementations.

    `minimum` clamps *both* the parsed value and the fallback. Clamping only
    the parsed value - what the previous copies did - meant a garbage string
    could yield a smaller number than any value the operator could actually
    type, which is the opposite of what a floor is for.
    """
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return value if minimum is None else max(minimum, value)


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean setting, the companion to `env_int`.

    Every boolean setting in this app is documented as "set to 1", and three
    call sites each spelled that out by hand - two as `== "1"` and one as the
    inverted `!= "1"`, which reads backwards at a glance. This is the single
    reader, so the convention cannot drift again.

    Anything other than the recognized true/false spellings falls back to
    `default`, matching `env_int`'s "never crash on a typo" behaviour: an
    operator fat-fingering a setting should get the documented default, not a
    process that refuses to boot.

    Set-but-empty therefore means "unset", which is deliberate and matches
    both `dash_ui.app._env_prefix` and compose's own `${VAR:-default}`
    expansion - `docker compose` substitutes an empty string for an undefined
    variable, and reading that as an explicit "off" would silently disable a
    feature nobody chose to disable. (The previous inline `== "1"` checks read
    empty as false.)
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _configured_worker_count() -> int:
    """Highest worker count implied by the environment.

    `WEB_CONCURRENCY` is honoured by both uvicorn and gunicorn;
    `UVICORN_WORKERS` and `GUNICORN_WORKERS` are checked as well because
    orchestration templates set them interchangeably.
    """
    return max(
        env_int("WEB_CONCURRENCY", 1),
        env_int("UVICORN_WORKERS", 1),
        env_int("GUNICORN_WORKERS", 1),
    )


def _running_under_gunicorn() -> bool:
    return "gunicorn" in sys.modules or "gunicorn" in os.getenv(
        "SERVER_SOFTWARE", ""
    ).lower()


def enforce_single_worker() -> None:
    """Fail fast when started with more than one worker process.

    `ALLOW_UNSAFE_MULTIWORKER=1` overrides this. The override exists so the
    guard can never be the thing blocking a legitimate future migration to a
    shared cache and a distributed JANIS gate - but it has to be typed
    deliberately, which is the point.
    """
    if env_flag(MULTIWORKER_OVERRIDE):
        logger.warning(
            "%s=1: single-worker guard disabled. The JANIS concurrency limit "
            "and the query cache are per-process and will not be enforced "
            "across workers.",
            MULTIWORKER_OVERRIDE,
        )
        return

    workers = _configured_worker_count()
    if workers > 1:
        raise UnsupportedDeploymentError(
            f"This application supports exactly one worker process, but the "
            f"environment requests {workers}. The JANIS concurrency semaphore "
            f"and the query cache are process-local, so N workers run N x "
            f"{CONCURRENT_QUERIES} concurrent JANIS subprocesses. Run with "
            f"--workers 1, or set "
            f"{MULTIWORKER_OVERRIDE}=1 if you have replaced both the cache "
            f"backend and the JANIS gate with shared implementations."
        )

    if _running_under_gunicorn():
        logger.warning(
            "Running under gunicorn. Ensure it is configured with exactly one "
            "worker; this application's JANIS gate is process-local."
        )


def query_budget_seconds() -> int:
    """Wall-clock budget for one whole multi-series query.

    Must be set below the *smallest* read timeout in the deployed proxy chain
    (nginx `proxy_read_timeout`, AWS ALB `idle_timeout`, GCP `timeoutSec`,
    Azure `requestTimeout`, and so on). Cloudflare caps origin responses at
    100 s on non-Enterprise plans and that cap cannot be raised.

    When the budget is hit the app returns an actionable message rather than
    letting a proxy return an opaque gateway error while the JANIS subprocess
    keeps running with nobody left to receive the result.
    """
    return env_int("JANIS_QUERY_BUDGET_SECONDS", 180, minimum=1)


def max_concurrent_queries() -> int:
    """How many users may have a JANIS query running at the same time.

    One number, because the two gates it sizes are 1:1 by construction:
    `query_store._build_series_within_budget` walks its database/isotope/
    dataset product *sequentially*, so one query holds at most one
    `java -jar Janis.jar` subprocess at any instant. N concurrent queries is
    therefore exactly N concurrent JVMs, and configuring the admission gate
    separately from the subprocess semaphore could only express combinations
    that cannot occur.

    Each slot costs a JVM, so this is the memory dial. Raise it when the
    container has headroom; leave it at 1 on a small host.
    """
    for legacy in _LEGACY_CONCURRENCY_NAMES:
        if os.getenv(CONCURRENT_QUERIES) is None and os.getenv(legacy) is not None:
            logger.warning(
                "%s is superseded by %s; using the value %s=%s. Update your "
                "environment - the old name will stop being read.",
                legacy,
                CONCURRENT_QUERIES,
                legacy,
                os.getenv(legacy),
            )
            return env_int(legacy, DEFAULT_CONCURRENT_QUERIES, minimum=1)
    return env_int(CONCURRENT_QUERIES, DEFAULT_CONCURRENT_QUERIES, minimum=1)


def log_deployment_limits() -> None:
    """Emit one startup line so both limits are visible in container logs."""
    logger.info(
        "Deployment limits: single worker (pid=%s); %s=%s concurrent JANIS "
        "queries; query budget=%ss - the budget MUST be below the smallest "
        "read timeout in the proxy chain in front of this process.",
        os.getpid(),
        CONCURRENT_QUERIES,
        max_concurrent_queries(),
        query_budget_seconds(),
    )
