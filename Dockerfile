FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy uv project files
COPY pyproject.toml uv.lock ./

# Install dependencies without installing the local project
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY src ./src
COPY app ./app
COPY model ./model

EXPOSE 8000

# Start FastAPI
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]