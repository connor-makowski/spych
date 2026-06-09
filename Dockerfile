# syntax = docker/dockerfile:1

## Uncomment the version of python you want to test against
# FROM python:3.11-bookworm
FROM python:3.12-bookworm
# FROM python:3.13-bookworm
# FROM python:3.14-bookworm

# Grab the prebuilt uv binary instead of pip-installing it
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Compile .pyc at install time (faster startup), copy instead of hardlink
# (the cache mount is a different filesystem, so hardlinks would fail)
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app/

# Install dependencies ONLY first — this layer caches until your deps change
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --extra kokoro --extra chatterbox

# Now copy the actual project and install it (cheap, deps already done)
COPY spych/ /app/spych/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra kokoro --extra chatterbox --extra dev

# Put the venv on PATH so you don't need `uv run` everywhere
ENV PATH="/app/.venv/bin:$PATH"

CMD ["/bin/bash"]