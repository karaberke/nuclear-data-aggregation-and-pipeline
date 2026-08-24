# Must match `requires-python` in pyproject.toml and the interpreter the test
# suite runs on. The parity fixtures are bit-exact, so a split between the
# tested stack and the shipped one voids them - see pyproject.toml.
FROM python:3.14-slim

WORKDIR /app

# --- Runtime dependencies: headless Java for JANIS + TLS certificates ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN java -version

# --- Corporate root + intermediate CA bundle so HTTPS requests inside the
# container aren't rejected as untrusted by TLS-inspecting proxies ---
COPY corporate-ca-bundle.pem /usr/local/share/ca-certificates/corporate-ca-bundle.crt
RUN update-ca-certificates

# --- Install uv ---
# Version-pinned, not :latest. Everything else here is pinned hard
# (uv.lock --frozen, dash<5, one Python), so a floating build tool was the
# single unreproducible input left in the image.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

# --- Python deps: copy lockfiles first for caching ---
COPY pyproject.toml uv.lock ./
# --system-certs tells uv to trust the OS certificate store (which now
# includes the Zscaler CA above) instead of its own bundled cert list.
# --no-default-groups keeps the `dev` dependency group (httpx, used only by
# the test suite) out of the production image; uv installs it by default.
# UV_COMPILE_BYTECODE trades a little build time for faster process start -
# worth it here because the container healthcheck begins probing 10s in.
# UV_LINK_MODE=copy silences the hardlink warning when the cache and the venv
# are on different layers.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --system-certs --no-default-groups

# --- App code ---
COPY backend/ ./backend

# Drop root. Nothing here writes to the image at runtime - the caches are all
# in-process - so an unprivileged user is sufficient and removes the default
# of running a network-facing service as uid 0.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# curl is not present in python:3.14-slim, so probe with the stdlib.
# This is the single definition - docker-compose.yml deliberately does not
# restate it, because a compose healthcheck silently overrides the image's
# and the two then drift apart unnoticed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" \
    || exit 1

# --workers 1 is REQUIRED, not a default worth tuning. The JANIS gates and the
# query cache are process-local, so N workers allow N x
# JANIS_MAX_CONCURRENT_QUERIES concurrent JANIS subprocesses.
# backend/deployment.py fails fast if the environment asks for more.
# See "Deployment limits" in README.md.
CMD ["uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
