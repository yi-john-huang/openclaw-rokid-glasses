# Technology Stack

## Architecture
**Type**: Hexagonal (Ports & Adapters) — inbound port: Rokid HTTP/SSE; outbound port: OpenAI-compatible HTTP
**Language**: Python 3.12
**Module System**: Python packages (src layout)
**Framework**: FastAPI (async, ASGI)
**Build Tool**: Hatchling (via pyproject.toml)

## Core Technologies
- **Runtime**: Python 3.12
- **Language**: Python (typed, strict mypy)
- **Framework**: FastAPI + Uvicorn (ASGI server)
- **HTTP Client**: httpx (async, streaming)
- **Config**: pydantic-settings (env vars + .env file)
- **Testing**: pytest + pytest-asyncio + pytest-httpx

## Development Environment
- **Runtime Version**: Python >= 3.12
- **Package Manager**: uv (fast pip-compatible resolver)
- **Testing Framework**: pytest with asyncio_mode="auto"

## Dependencies

### Production Dependencies
| Package | Purpose |
|---------|---------|
| `fastapi>=0.115` | ASGI web framework, routing, validation |
| `uvicorn[standard]>=0.32` | ASGI server (with websocket + http/2 extras) |
| `httpx>=0.28` | Async HTTP client for upstream relay + streaming |
| `pydantic-settings>=2.6` | Settings from environment variables / .env |
| `pydantic>=2.10` | Request/response models and validation |

### Development Dependencies
| Package | Purpose |
|---------|---------|
| `pytest>=8.3` | Test runner |
| `pytest-asyncio>=0.24` | Async test support |
| `pytest-httpx>=0.35` | Mock httpx in tests |
| `ruff>=0.8` | Linting + formatting |
| `mypy>=1.13` | Static type checking (strict mode) |

## Development Commands
```bash
# Create virtualenv and install all deps (including dev)
uv venv && uv pip install -e ".[dev]"

# Run unit tests
uv run pytest tests/unit/ -v

# Run integration tests
uv run pytest tests/integration/ -v

# Run all tests with coverage
uv run pytest --cov=rokid_bridge --cov-report=term-missing

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Start dev server
ROKID_ACCESS_KEY=test UPSTREAM_TOKEN=test \
  uv run uvicorn rokid_bridge.app:create_app --factory --reload --port 8090
```

## Quality Assurance
- **Linting**: ruff (E, F, I, UP, B, SIM rules)
- **Type Checking**: mypy strict mode on src/
- **Testing**: pytest; minimum 80% coverage; 100% on auth and security paths
- **Security**: OWASP-aligned; constant-time HMAC; no secrets in logs

## Deployment Configuration
- **Containerization**: Docker (python:3.12-slim, multi-stage build, non-root user, read-only FS)
- **Orchestration**: docker compose (standalone) + docker-compose.override.yml (co-deploy with secure stack)
- **Routing**: Cloudflare Tunnel or Caddy/Nginx path-based routing (`/rokid/*` → `:8090`)
- **CI/CD**: Not yet configured (future: GitHub Actions)

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ROKID_ACCESS_KEY` | ✅ | — | Rokid HMAC key from Lingzhu Platform |
| `UPSTREAM_TOKEN` | ✅ | — | Bearer token for Secure Stack |
| `UPSTREAM_URL` | | `http://localhost:8080` | Secure Stack base URL |
| `ROKID_AGENT_ID` | | `""` | OpenClaw agent to route to |
| `ROKID_RATE_LIMIT` | | `30` | Max req/min per device |
| `ROKID_REPLAY_WINDOW` | | `300` | Replay protection window (seconds) |
| `ROKID_MAX_HISTORY_TURNS` | | `20` | Conversation history depth |
| `ROKID_IMAGE_DETAIL` | | `low` | OpenAI vision detail level |
| `PORT` | | `8090` | Bridge listening port |
