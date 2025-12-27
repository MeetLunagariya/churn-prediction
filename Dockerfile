# syntax=docker/dockerfile:1.7

# ---- builder ----
FROM python:3.11-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/python \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

# Pin uv to a known version for reproducible image builds.
COPY --from=ghcr.io/astral-sh/uv:0.5.13 /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first (this layer caches when pyproject/lock don't change).
COPY pyproject.toml uv.lock README.md .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Bring in the source and install the project (still no dev extras).
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---- runtime ----
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Non-root user for the running process.
RUN groupadd --system app && useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/scripts /app/scripts
COPY --from=builder /app/configs /app/configs

# Model artifact is expected to be mounted or built at runtime — see
# docker-compose.yml. Adding `models/` here keeps the path resolvable.
RUN mkdir -p /app/models /app/data /app/mlruns && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3).raise_for_status()" || exit 1

CMD ["uvicorn", "churn.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
