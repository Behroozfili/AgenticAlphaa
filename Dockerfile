# syntax=docker/dockerfile:1
#
# Dockerfile — AgenticAlpha
# =========================
# Multi-stage build for the FastAPI multi-agent service.
#
#   Entry point : uvicorn api.main:app   (main.py is only a stub)
#   Python      : 3.12   (pyproject.toml → requires-python = ">=3.12")
#   Deps        : uv + committed uv.lock  (reproducible install)
#
# Notes
# -----
# * uvicorn is declared only in requirements.txt (not pyproject.toml/uv.lock),
#   yet it is what actually boots the app, so it is installed explicitly below.
# * The ML stack (torch / transformers / sentence-transformers) is inherently
#   large; the multi-stage layout keeps build-only tooling out of the final
#   image, but the runtime image is still sizeable because of these wheels.

# ─────────────────────────────────────────────────────────────
# Stage 1 — Builder: resolve & install dependencies into a venv
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Build tooling for any package without a prebuilt wheel (kept out of runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# uv is this repo's dependency manager (uv.lock is committed).
RUN pip install --no-cache-dir uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Copy only the dependency manifests first so this layer is cached and only
# rebuilt when the dependencies actually change.
COPY pyproject.toml uv.lock ./

# Install the locked dependencies into /opt/venv.
#   --frozen             : fail if uv.lock is out of sync (no silent drift)
#   --no-install-project : the root package is a stub; we run from source
RUN uv sync --frozen --no-install-project

# uvicorn is the ASGI server the app is started with but is missing from
# pyproject.toml/uv.lock (it lives only in requirements.txt). Add it so the
# runtime image can actually launch the application.
RUN uv pip install --python /opt/venv/bin/python "uvicorn>=0.34.0"

# ─────────────────────────────────────────────────────────────
# Stage 2 — Runtime: minimal image running as a non-root user
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# libgomp1 is required by torch's OpenMP runtime on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN groupadd --system app \
    && useradd --system --gid app --create-home app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Bring in the pre-built virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Copy the application source (respecting .dockerignore).
COPY . .

# Drop privileges.
RUN chown -R app:app /app
USER app

EXPOSE 3000

# The app spawns MCP tool servers as `python tools/**/**_server.py` subprocesses
# and serves frontend/alpha-agent-app.html, so the full source tree is present.
CMD ["uvicorn", "api.main:app","--reload", "--port", "3000"]
