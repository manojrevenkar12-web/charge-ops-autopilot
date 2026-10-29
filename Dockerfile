# One image for the CSMS, the autopilot API and the Azure ML jobs.
FROM python:3.13-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
RUN pip install --no-cache-dir uv==0.9.5 && useradd --create-home --uid 10001 app
WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
COPY kb ./kb
COPY sql ./sql
RUN uv sync --frozen --no-dev
USER app
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 9000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["cpa", "serve-api"]
