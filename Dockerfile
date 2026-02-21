# Stage 1: Builder
FROM python:3.12-slim AS builder

RUN pip install uv --no-cache-dir

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install .

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ src/

USER appuser

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8090

CMD ["uvicorn", "rokid_bridge.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8090"]
