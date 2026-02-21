# openclaw-rokid-glasses

A lightweight SSE bridge that connects **Rokid AR Glasses** (via the Lingzhu/Rokid AI App platform) to [`openclaw-secure-stack`](https://github.com/yi-john-huang/openclaw-secure-stack). It acts as a protocol adapter: translating Rokid's proprietary request format into OpenAI-compatible chat completions and streaming SSE responses back to the glasses.

> **openclaw-secure-stack is never modified.** All governance, sanitisation, and audit remain in the existing stack. This bridge only handles protocol translation.

## Use Cases

| Use Case | Flow |
|----------|------|
| Voice conversation | Speech-to-text → Bridge → Secure Stack → AI agent → SSE reply on AR display |
| Camera photo analysis | Glasses photo → Bridge (image conversion) → vision model → SSE reply |
| Voice command control | Spoken command → Bridge → agent action → SSE confirmation |

## Architecture

```
Rokid Glasses → Rokid AI App → POST https://your-domain.com/rokid/chat
                                         │
                              ┌──────────▼──────────────┐
                              │  Rokid Bridge  (:8090)   │
                              │  Bearer auth + replay    │
                              │  Rate limiting           │
                              │  Image conversion        │
                              │  Conversation history    │
                              │  AR system prompt        │
                              └──────────┬──────────────┘
                                         │ POST /v1/chat/completions
                                         │ Bearer token + stream:true
                              ┌──────────▼──────────────┐
                              │  Secure Stack  (:8080)   │
                              │  Sanitise + govern       │
                              │  Audit + scan            │
                              └──────────┬──────────────┘
                                         │
                              ┌──────────▼──────────────┐
                              │  OpenClaw  (:3000)       │
                              └─────────────────────────┘
```

**Module responsibilities:**

| Module | Responsibility |
|--------|---------------|
| `app.py` | HTTP routing, pipeline wiring, rate limiting |
| `config.py` | Environment variable validation (pydantic-settings) |
| `models.py` | Pydantic request/response shapes |
| `rokid_auth.py` | Bearer token verification, replay protection |
| `image_handler.py` | Camera payload → OpenAI vision content parts |
| `history.py` | Per-device in-memory conversation history (TTL) |
| `relay.py` | SSE streaming relay to Secure Stack |

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker + Docker Compose (for deployment)
- A running `openclaw-secure-stack` instance

---

## Deployment

### Option A — Standalone (bridge + existing Secure Stack on same host)

Use this when `openclaw-secure-stack` is already running separately and you want the bridge to reach it over `localhost` or `host.docker.internal`.

**1. Copy and fill environment file**

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Required
ROKID_ACCESS_KEY=your-rokid-ak-from-lingzhu-platform
UPSTREAM_TOKEN=your-openclaw-bearer-token

# Optional — change if Secure Stack is not on localhost:8080
UPSTREAM_URL=http://host.docker.internal:8080

# Optional tuning
ROKID_AGENT_ID=          # leave empty for default agent
ROKID_RATE_LIMIT=30      # max requests/min per device
ROKID_REPLAY_WINDOW=300  # replay protection window in seconds
ROKID_MAX_HISTORY_TURNS=20
ROKID_IMAGE_DETAIL=low   # low|high (OpenAI vision detail)
PORT=8090
```

**2. Build and start**

```bash
docker compose up -d --build
```

**3. Verify**

```bash
curl http://localhost:8090/health
# {"status":"ok","service":"rokid-bridge"}
```

---

### Option B — Co-deploy alongside openclaw-secure-stack

Use this when both services run in the same Docker Compose project and communicate over an internal network (no host port exposure needed for the bridge).

**1. Ensure the Secure Stack's internal network exists**

The override file expects a network named `openclaw-secure-stack_internal`. Verify:

```bash
docker network ls | grep openclaw
```

If the network has a different name, edit `docker-compose.override.yml` accordingly.

**2. Copy and fill environment file** (same as Option A, `UPSTREAM_URL` will be overridden)

```bash
cp .env.example .env
# Fill ROKID_ACCESS_KEY and UPSTREAM_TOKEN
```

**3. Start with the override file**

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d --build
```

The override sets `UPSTREAM_URL=http://openclaw-secure-stack:8080` and removes the host port binding so the bridge is only reachable inside the Docker network.

---

### Cloudflare Tunnel routing

Add a path-based rule to route `/rokid/*` to the bridge:

| Path prefix | Backend |
|-------------|---------|
| `/*` | `localhost:8080` (existing Secure Stack) |
| `/rokid/*` | `localhost:8090` (Rokid Bridge) |

On the **Lingzhu Platform** (灵珠平台), set the custom agent endpoint to:

```
https://your-domain.com/rokid/chat
```

The bridge listens at `/chat` internally; the tunnel strips the `/rokid` prefix before forwarding.

> If your tunnel does **not** strip the prefix automatically, configure your reverse proxy (Caddy/Nginx) to rewrite `/rokid/chat` → `/chat` before forwarding to port 8090.

---

### Alternative: Caddy reverse proxy

```caddy
your-domain.com {
    handle /rokid/* {
        uri strip_prefix /rokid
        reverse_proxy localhost:8090
    }
    handle {
        reverse_proxy localhost:8080
    }
}
```

---

## Configuration Reference

All configuration is via environment variables (or `.env` file).

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `ROKID_ACCESS_KEY` | ✅ | — | Bearer token from Lingzhu Platform — sent by Rokid AI App as `Authorization: Bearer <key>` |
| `UPSTREAM_TOKEN` | ✅ | — | Bearer token for `openclaw-secure-stack` |
| `UPSTREAM_URL` | | `http://localhost:8080` | Base URL of the Secure Stack |
| `ROKID_AGENT_ID` | | `""` | OpenClaw agent ID to route to; empty = default agent |
| `ROKID_RATE_LIMIT` | | `30` | Max requests per minute per device |
| `ROKID_REPLAY_WINDOW` | | `300` | Seconds within which a timestamp is considered fresh |
| `ROKID_MAX_HISTORY_TURNS` | | `20` | Number of conversation turns to keep per device |
| `ROKID_IMAGE_DETAIL` | | `low` | OpenAI vision detail level (`low` or `high`) |
| `PORT` | | `8090` | Port the bridge listens on |

---

## API Reference

### `GET /health`

Liveness probe. No authentication required.

```bash
curl http://localhost:8090/health
```

```json
{"status": "ok", "service": "rokid-bridge"}
```

---

### `POST /chat`

Main endpoint. Accepts a Rokid AI App request, authenticates it, and streams an SSE response.

**Headers:**

```
Authorization: Bearer <ROKID_ACCESS_KEY>
Content-Type: application/json
```

**Request body:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "device_id": "rokid-sn-XXXXXXXX",
  "type": "text",
  "text": "What is the weather like today?",
  "timestamp": 1708300000
}
```

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `request_id` | string | ✅ | Unique request ID (UUID) |
| `device_id` | string | ✅ | Rokid device serial number |
| `type` | `text` \| `image` \| `text_with_image` | ✅ | Request type |
| `text` | string | for `text`, `text_with_image` | Speech-to-text result |
| `image.data` | string | for `image`, `text_with_image` | Base64-encoded image bytes |
| `image.mime_type` | string | for `image`, `text_with_image` | `image/jpeg` or `image/png` |
| `timestamp` | integer | ✅ | Unix seconds (replay protection) |

**Response:** `text/event-stream` — OpenAI-format SSE chunks passthrough from Secure Stack.

```
data: {"choices":[{"delta":{"content":"The weather"}}]}

data: {"choices":[{"delta":{"content":" is sunny."}}]}

data: [DONE]
```

**Error responses:**

| Status | Condition |
|--------|-----------|
| 401 | Missing, wrong, or expired token; expired or future timestamp |
| 422 | Missing required fields or invalid image |
| 429 | Rate limit exceeded (includes `Retry-After` header) |

---

### `POST /clear-history`

Clears the conversation history for a specific device. Requires authentication and a fresh timestamp.

**Headers:** same as `/chat`

**Request body:**

```json
{
  "device_id": "rokid-sn-XXXXXXXX",
  "timestamp": 1708300000
}
```

**Response:**

```json
{"cleared": true, "device_id": "rokid-sn-XXXXXXXX"}
```

---

## Operations

### View logs

```bash
# Follow live logs
docker compose logs -f rokid-bridge

# Last 100 lines
docker compose logs --tail=100 rokid-bridge
```

### Health check

```bash
curl http://localhost:8090/health
```

The container also runs an internal Docker health check every 30 seconds. Check its status:

```bash
docker inspect --format='{{.State.Health.Status}}' rokid-bridge
```

### Restart

```bash
docker compose restart rokid-bridge
```

### Stop and remove

```bash
docker compose down
```

### Update to a new version

```bash
git pull
docker compose up -d --build
```

### Clear a device's conversation history

```bash
DEVICE_ID="rokid-sn-XXXXXXXX"
TIMESTAMP=$(date +%s)

curl -X POST http://localhost:8090/clear-history \
  -H "Authorization: Bearer $ROKID_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"device_id\": \"$DEVICE_ID\", \"timestamp\": $TIMESTAMP}"
```

### Smoke test

Run the included smoke test script against a live bridge:

```bash
ROKID_ACCESS_KEY=your-key bash scripts/smoke-test.sh
```

This verifies:
1. `/health` returns 200
2. A wrong token is rejected with 401
3. A valid SSE request streams a response

To test against a remote host:

```bash
BASE_URL=https://your-domain.com/rokid ROKID_ACCESS_KEY=your-key bash scripts/smoke-test.sh
```

### Monitor rate limiting

A `429` response includes a `Retry-After` header (in seconds) indicating when the device's window resets:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 42
{"detail": "Rate limit exceeded"}
```

---

## Development

### Setup

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv and install all deps
uv venv && uv pip install -e ".[dev]"
```

### Run tests

```bash
# Unit tests only (fast, no I/O)
uv run pytest tests/unit/ -v

# Integration tests (mocked httpx, no running server needed)
uv run pytest tests/integration/ -v

# Full suite with coverage
uv run pytest tests/ --cov=rokid_bridge --cov-report=term-missing
```

### Lint and type check

```bash
uv run ruff check src/ tests/    # lint
uv run ruff format src/ tests/   # format
uv run mypy src/                 # strict type check
```

### Local dev server (no Docker)

```bash
ROKID_ACCESS_KEY=test-key UPSTREAM_TOKEN=test-token UPSTREAM_URL=http://localhost:8080 \
  uv run uvicorn rokid_bridge.app:create_app --factory --reload --port 8090
```

### Project structure

```
openclaw-rokid-glasses/
├── src/rokid_bridge/      # Application source
│   ├── app.py             # FastAPI factory + routes + rate limiter
│   ├── config.py          # Settings (pydantic-settings)
│   ├── models.py          # Request/response Pydantic models
│   ├── rokid_auth.py      # Auth + replay protection
│   ├── image_handler.py   # Camera image conversion
│   ├── history.py         # Per-device conversation history
│   └── relay.py           # SSE streaming relay
├── tests/
│   ├── unit/              # Isolated module tests (mocked I/O)
│   └── integration/       # Full endpoint tests (mocked httpx)
├── .spec/                 # Spec-Driven Development artefacts
│   ├── steering/          # Product, tech, structure docs
│   └── specs/rokid-glasses-bridge/  # Requirements, design, tasks
├── Dockerfile
├── docker-compose.yml              # Standalone deployment
└── docker-compose.override.yml     # Co-deploy with Secure Stack
```

### Security notes

- **Bearer token** is compared using `hmac.compare_digest` (constant-time) to prevent timing attacks
- **Replay protection**: requests older than `ROKID_REPLAY_WINDOW` seconds are rejected
- **Secrets** use `pydantic.SecretStr` — never appear in logs or `repr()`
- **Docker hardening**: non-root user (UID 1000), read-only filesystem, `cap_drop: ALL`, `no-new-privileges`
- **Conversation history** stores text only — base64 image data is never persisted

---

## License

[MIT](LICENSE)
