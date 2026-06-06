FROM python:3.12-slim-bookworm

# Install uv by copying it from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker caching
COPY . .

# Install dependencies (without the project itself)
RUN uv add triton && \
    uv sync --frozen --no-cache --no-install-project

# Final sync to install the project
RUN uv sync --frozen --no-cache

ENV TRITON_INTERPRET=1

CMD uv run cs336_systems/triton/weighted_sum.py
