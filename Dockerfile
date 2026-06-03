# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_HTTP_TIMEOUT=120 \
    UV_LINK_MODE=copy \
    UV_TORCH_BACKEND=cpu \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app/src"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml requirements-docker.txt README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv \
    && uv pip install --torch-backend cpu -r requirements-docker.txt

COPY configs ./configs
COPY scripts ./scripts
COPY src ./src
COPY deploy.py pipeline.py streamlit_app.py ./

RUN uv pip install --no-deps -e . \
    && mkdir -p data/raw data/processed models reports mlruns mlflow-data

EXPOSE 8000 8501 5000

CMD ["uvicorn", "ucf_crime_recognition.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
