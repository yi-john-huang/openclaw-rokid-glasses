# Technical Design: rokid-glasses-bridge

**Feature**: `rokid-glasses-bridge`
**Status**: Draft — Pending Approval
**Created**: 2026-02-20
**Author**: Architect Agent (SDD Phase 3)
**Requirements Reference**: `.spec/specs/rokid-glasses-bridge/requirements.md`

---

## Table of Contents

1. [Architecture Diagram](#1-architecture-diagram)
2. [Module Design](#2-module-design)
   - 2.1 [config.py](#21-configpy)
   - 2.2 [models.py](#22-modelspy)
   - 2.3 [rokid_auth.py](#23-rokid_authpy)
   - 2.4 [image_handler.py](#24-image_handlerpy)
   - 2.5 [history.py](#25-historypy)
   - 2.6 [relay.py](#26-relaypy)
   - 2.7 [app.py](#27-apppy)
3. [Data Models](#3-data-models)
4. [ADR-001: Authentication Design](#4-adr-001-authentication-design)
5. [SSE Streaming Design](#5-sse-streaming-design)
6. [Conversation History Design](#6-conversation-history-design)
7. [Rate Limiter Design](#7-rate-limiter-design)
8. [AR System Prompt](#8-ar-system-prompt)
9. [Request Processing Pipeline](#9-request-processing-pipeline)
10. [Docker and Deployment Design](#10-docker-and-deployment-design)
11. [Interface Contracts](#11-interface-contracts)
12. [Security Design Summary](#12-security-design-summary)
13. [Testing Strategy](#13-testing-strategy)
14. [Open Questions and Risks](#14-open-questions-and-risks)
15. [Approval Checklist](#15-approval-checklist)

---

## 1. Architecture Diagram

### 1.1 Component Diagram

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Rokid Glasses Device                                                │
 │  (Lingzhu / Rokid AI App Platform)                                   │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 │  POST https://domain.com/rokid/chat
                                 │  Authorization: Bearer <ROKID_ACCESS_KEY>
                                 │  Content-Type: application/json
                                 │  Body: RokidChatRequest (JSON)
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Cloudflare Tunnel / Caddy / Nginx                                   │
 │  Path routing: /rokid/* → localhost:8090                             │
 │  (strips /rokid prefix before forwarding)                           │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 │  POST /chat
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Rokid Bridge  (:8090)  — python:3.12-slim  — non-root, RO FS       │
 │                                                                      │
 │  ┌─────────────────────────────────────────────────────────────────┐ │
 │  │  FastAPI ASGI App  (app.py — inbound adapter)                   │ │
 │  │                                                                 │ │
 │  │  POST /chat                   POST /clear-history  GET /health  │ │
 │  │       │                              │                  │       │ │
 │  │       ▼                              ▼                  ▼       │ │
 │  │  ┌──────────┐  FR-3        ┌──────────────┐    ┌─────────────┐ │ │
 │  │  │   Rate   │  Sliding     │   Auth +     │    │   Health    │ │ │
 │  │  │ Limiter  │  window      │   Replay     │    │   Check     │ │ │
 │  │  │(app.py)  │  per device  │(rokid_auth)  │    │  (app.py)   │ │ │
 │  │  └────┬─────┘              └──────┬───────┘    └─────────────┘ │ │
 │  │       │ [429 if exceeded]         │                             │ │
 │  │       ▼                           │ [401 if invalid]            │ │
 │  │  ┌──────────┐                     ▼                             │ │
 │  │  │   Auth   │◄────────────────────┘                             │ │
 │  │  │  Verify  │  Bearer token                                     │ │
 │  │  │(rokid_auth)  constant-time                                   │ │
 │  │  └────┬─────┘                                                   │ │
 │  │       │ [401 if invalid]                                        │ │
 │  │       ▼                                                         │ │
 │  │  ┌──────────┐     ┌───────────────┐     ┌──────────────────┐   │ │
 │  │  │  Image   │     │    History    │     │   Relay          │   │ │
 │  │  │ Handler  │────►│    Store      │────►│  (relay.py)      │   │ │
 │  │  │(image_   │     │  (history.py) │     │  outbound adapter│   │ │
 │  │  │ handler) │     │  per-device   │     │  httpx streaming │   │ │
 │  │  └──────────┘     │  in-memory    │     └────────┬─────────┘   │ │
 │  │  Base64 →         │  dict + TTL   │              │             │ │
 │  │  vision format    └───────────────┘              │             │ │
 │  └──────────────────────────────────────────────────┼─────────────┘ │
 └────────────────────────────────────────────────────┬┼───────────────┘
                                                      ││  POST /v1/chat/completions
                                                      ││  Authorization: Bearer <UPSTREAM_TOKEN>
                                                      ││  stream=true
                                                      ▼│
 ┌──────────────────────────────────────────────────────────────────────┐
 │  openclaw-secure-stack  (:8080)                                      │
 │  OpenAI-compatible chat completions — UNMODIFIED                     │
 │  Returns: text/event-stream  (SSE chunks)                            │
 └──────────────────────────────────────────────────────────────────────┘
                                                       │
                                              SSE relay (passthrough)
                                                       │
                                                       ▼
                                             Rokid Glasses (client)
                                             text/event-stream response
```

### 1.2 Docker Network Topology

#### Standalone Deployment

```
  Host machine
  ┌─────────────────────────────────────────────────────────────┐
  │  docker network: rokid-bridge-net (bridge)                  │
  │                                                             │
  │  ┌─────────────────────────┐                               │
  │  │  rokid-bridge           │                               │
  │  │  container              │  port mapping: 8090:8090      │
  │  │  (non-root, RO FS)      │◄──────────────────────────────┤ external
  │  └────────────┬────────────┘                               │
  │               │  http://host.docker.internal:8080           │
  │               │  (or configured UPSTREAM_URL)               │
  └───────────────┼─────────────────────────────────────────────┘
                  │
                  ▼ (out-of-docker, or separate stack)
             openclaw-secure-stack:8080
```

#### Co-Deploy with openclaw-secure-stack

```
  docker network: openclaw-net (shared, external)
  ┌────────────────────────────────────────────────────────────────┐
  │                                                                │
  │  ┌─────────────────────────┐    ┌──────────────────────────┐  │
  │  │  rokid-bridge           │    │  openclaw-secure-stack   │  │
  │  │  :8090                  │───►│  :8080                   │  │
  │  │  (non-root, RO FS)      │    │  (unmodified)            │  │
  │  └─────────────────────────┘    └──────────────────────────┘  │
  │  UPSTREAM_URL=http://openclaw-secure-stack:8080                │
  │                                                                │
  └────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Design

The package structure follows Hexagonal Architecture: `app.py` is the inbound HTTP adapter, `relay.py` is the outbound HTTP adapter, and the domain modules (`rokid_auth`, `image_handler`, `history`) contain pure business logic with no framework coupling.

### 2.1 `config.py`

**Responsibility (SRP):** Read, validate, and expose all environment configuration. No business logic — this module is the single source of truth for every runtime setting.

**SOLID Application:** Dependency Inversion — all other modules receive a `Settings` instance rather than reading `os.environ` directly, enabling easy override in tests.

```python
# src/rokid_bridge/config.py

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Required secrets — SecretStr prevents accidental logging (NFR-2)
    rokid_access_key: SecretStr = Field(
        ...,
        description="Shared Bearer token issued by Rokid/Lingzhu Platform",
    )
    upstream_token: SecretStr = Field(
        ...,
        description="Bearer token for openclaw-secure-stack",
    )

    # Upstream routing
    upstream_url: str = Field(
        default="http://localhost:8080",
        description="Base URL of the openclaw-secure-stack",
    )

    # Optional agent routing
    rokid_agent_id: str = Field(
        default="",
        description="OpenClaw agent ID to route to; empty string omits the field",
    )

    # Rate limiting (FR-3)
    rokid_rate_limit: int = Field(
        default=30,
        ge=1,
        description="Maximum requests per device per 60-second sliding window",
    )

    # Replay protection (FR-2)
    rokid_replay_window: int = Field(
        default=300,
        ge=1,
        description="Seconds within which a request timestamp is considered valid",
    )

    # Conversation history (FR-7)
    rokid_max_history_turns: int = Field(
        default=20,
        ge=1,
        description="Maximum user+assistant turn pairs to keep per device",
    )

    # Image handling (FR-5)
    rokid_image_detail: str = Field(
        default="low",
        description="OpenAI vision detail level: 'low' or 'high'",
    )

    # Server port
    port: int = Field(
        default=8090,
        description="Uvicorn listening port",
    )


def get_settings() -> Settings:
    """Return a cached Settings instance.

    Called once at app startup and injected everywhere via FastAPI dependency.
    The lru_cache on the caller side is intentionally NOT placed here to
    allow test overrides via dependency_overrides.
    """
    return Settings()
```

**Key Design Decisions:**
- `SecretStr` for `rokid_access_key` and `upstream_token` ensures `.get_secret_value()` must be called explicitly; accidental `str(settings.rokid_access_key)` renders as `'**********'` — satisfying NFR-2.
- No `@lru_cache` on `get_settings()` itself; the caller in `app.py` uses FastAPI's `Depends(get_settings)` with an optional override for tests.
- Validation via `ge=1` on numeric fields catches misconfigured zeros at startup rather than at runtime.

**Dependencies:** `pydantic`, `pydantic-settings`

**Error Contracts:**
- `ValidationError` (pydantic) raised at import time if required env vars are absent — process exits with clear message before serving any traffic.

---

### 2.2 `models.py`

**Responsibility (SRP):** Define all Pydantic request and response models. No business logic — only structural validation and field coercion.

**SOLID Application:** Interface Segregation — separate models for each endpoint's concern; no "god model" that serves all endpoints.

```python
# src/rokid_bridge/models.py

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RokidRequestType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_WITH_IMAGE = "text_with_image"


class RokidImagePayload(BaseModel):
    """Embedded image object within a Rokid request."""

    model_config = ConfigDict(extra="forbid")

    data: str = Field(..., description="Base64-encoded image bytes")
    mime_type: str = Field(..., description="MIME type, e.g. 'image/jpeg'")


class RokidChatRequest(BaseModel):
    """Inbound request from Rokid/Lingzhu platform."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="UUID v4 from Rokid platform")
    device_id: str = Field(..., description="Opaque device identifier")
    type: RokidRequestType
    text: str = Field(default="", description="Speech-to-text result (may be empty for image-only)")
    image: RokidImagePayload | None = Field(default=None)
    timestamp: int = Field(..., description="Unix seconds; used for replay protection")

    @field_validator("text")
    @classmethod
    def text_not_blank_when_required(cls, v: str, info: object) -> str:
        # Blank text is accepted at model level; the route handler enforces
        # the type-specific requirement (FR-4 AC-6, FR-6 AC-3).
        return v


class RokidClearRequest(BaseModel):
    """Body for the /clear-history endpoint."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(..., description="Device whose history to clear")
    timestamp: int = Field(..., description="Unix seconds; used for replay protection")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["ok"] = "ok"
    service: Literal["rokid-bridge"] = "rokid-bridge"


class ClearHistoryResponse(BaseModel):
    """Response body for POST /clear-history."""

    cleared: bool
    device_id: str


# ---------------------------------------------------------------------------
# OpenAI upstream request shapes (what we POST to Secure Stack)
# These are not Pydantic models — they are plain TypedDicts or dicts because
# we build them dynamically. Defining them as TypedDicts satisfies mypy --strict.
# ---------------------------------------------------------------------------

from typing import TypedDict, Union


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlDetail(TypedDict):
    url: str
    detail: str


class ImageUrlWrapper(TypedDict):
    image_url: ImageUrlDetail


class ImageContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrlDetail


ContentPart = Union[TextContentPart, ImageContentPart]


class OpenAIMessage(TypedDict, total=False):
    role: str
    content: Union[str, list[ContentPart]]


class UpstreamRequest(TypedDict, total=False):
    model: str
    messages: list[OpenAIMessage]
    stream: bool
    agent_id: str
```

**Key Design Decisions:**
- `extra="forbid"` on all inbound models satisfies FR-12 AC-7, rejecting unexpected fields.
- `RokidRequestType` as a `str` enum gives Pydantic the validation for free; unrecognized `type` values produce HTTP 422 automatically.
- The blank `text` field is allowed at model validation level because `type == "image"` requests legitimately have no text; business logic in the route enforces the constraint per type.
- OpenAI upstream request shapes use `TypedDict` rather than Pydantic models — they are never parsed from untrusted input, only constructed by us, so runtime validation overhead is unnecessary (YAGNI, KISS).

**Dependencies:** `pydantic`

**Error Contracts:**
- `pydantic.ValidationError` — raised on malformed inbound JSON; FastAPI converts to HTTP 422 automatically.

---

### 2.3 `rokid_auth.py`

**Responsibility (SRP):** Authenticate incoming requests by verifying the Bearer token and enforcing the replay window. No routing, no rate limiting, no logging — only the authentication decision.

**SOLID Application:** Open/Closed — the module exports two pure functions (`verify_bearer_token`, `check_replay_window`); callers can compose them without modifying this module.

```python
# src/rokid_bridge/rokid_auth.py

import hmac
import time

from rokid_bridge.config import Settings


class AuthError(Exception):
    """Raised when authentication or replay check fails.

    Attributes:
        status_code: HTTP status to return (401 or 400).
        detail: Human-readable reason (generic enough not to leak internals).
    """

    def __init__(self, detail: str, status_code: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def verify_bearer_token(
    authorization_header: str | None,
    settings: Settings,
) -> None:
    """Verify that the Authorization: Bearer <token> header matches the configured AK.

    Uses hmac.compare_digest for constant-time comparison (NFR-1).

    Args:
        authorization_header: Raw value of the Authorization header, or None.
        settings: Application configuration (provides rokid_access_key).

    Raises:
        AuthError: If header is missing, malformed, or token does not match.
    """
    if not authorization_header:
        raise AuthError("Unauthorized")

    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError("Unauthorized")

    expected = settings.rokid_access_key.get_secret_value()
    # hmac.compare_digest requires both arguments to be the same type.
    # encode to bytes to satisfy type checker and guarantee constant-time comparison.
    if not hmac.compare_digest(
        expected.encode("utf-8"),
        token.encode("utf-8"),
    ):
        raise AuthError("Unauthorized")


def check_replay_window(
    body_timestamp: int,
    settings: Settings,
    *,
    now: float | None = None,
) -> None:
    """Verify that the request timestamp is within the acceptable replay window.

    Args:
        body_timestamp: Unix seconds from the JSON request body.
        settings: Provides ROKID_REPLAY_WINDOW.
        now: Current UTC time as float (Unix seconds). Defaults to time.time().
             Accepts an override to enable deterministic unit testing.

    Raises:
        AuthError: If the timestamp is expired or too far in the future.
    """
    current_time = now if now is not None else time.time()
    age = current_time - body_timestamp

    if age > settings.rokid_replay_window:
        raise AuthError("Request expired")
    if age < -60:  # 60-second future tolerance (clock skew)
        raise AuthError("Request timestamp invalid")
```

**Key Design Decisions:**
- The `now` parameter on `check_replay_window` makes the function purely testable without monkey-patching `time.time` globally — a clean dependency seam.
- `verify_bearer_token` does not raise different errors for "missing header" vs "wrong token" — both return the same generic `"Unauthorized"` message to prevent information leakage (FR-1 AC-8).
- Authentication happens before timestamp checking in the call order (see Section 9) — but within this module, they are separate functions, keeping each function's responsibility minimal.
- `hmac.compare_digest` is used even though the token is not an HMAC-computed value; the function is defined as "compares two strings in constant time," which is exactly what we need regardless of what the strings represent.

**Dependencies:** `hmac`, `time` (stdlib only — zero external dependencies for the auth layer)

**Error Contracts:**
- `AuthError(detail, status_code=401)` — all auth failures; `status_code=400` for malformed timestamp (caught in route before calling this module, but documented here for completeness).

---

### 2.4 `image_handler.py`

**Responsibility (SRP):** Validate and convert a Rokid image payload into the OpenAI vision content format. No network I/O, no history, no auth — only data transformation.

**SOLID Application:** Single Responsibility and Open/Closed — the handler defines the transformation; adding support for a new image format (e.g., `image/webp`) requires only extending `ALLOWED_MIME_TYPES`, not modifying conversion logic.

```python
# src/rokid_bridge/image_handler.py

import base64

from rokid_bridge.models import (
    ContentPart,
    ImageContentPart,
    ImageUrlDetail,
    RokidImagePayload,
    TextContentPart,
)

ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png"})
MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB


class ImageValidationError(ValueError):
    """Raised when the image payload is invalid.

    Attributes:
        status_code: HTTP status code to return (413 or 422).
        detail: Error message for the client.
    """

    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def validate_and_build_image_part(
    image: RokidImagePayload,
    detail: str,
) -> ImageContentPart:
    """Validate a Rokid image payload and convert it to an OpenAI image_url content part.

    Args:
        image: The image payload from the Rokid request.
        detail: OpenAI vision detail level ('low' or 'high').

    Returns:
        An OpenAI-compatible ImageContentPart dict.

    Raises:
        ImageValidationError: If MIME type is unsupported, base64 is invalid,
                              or decoded size exceeds MAX_IMAGE_BYTES.
    """
    if image.mime_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError("Unsupported image format")

    try:
        decoded = base64.b64decode(image.data, validate=True)
    except Exception:
        raise ImageValidationError("Invalid base64 image data")

    if len(decoded) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image too large", status_code=413)

    url = f"data:{image.mime_type};base64,{image.data}"
    return ImageContentPart(
        type="image_url",
        image_url=ImageUrlDetail(url=url, detail=detail),
    )


def build_image_content(
    image: RokidImagePayload,
    detail: str,
) -> list[ContentPart]:
    """Build a content array for an image-only request (FR-5)."""
    return [validate_and_build_image_part(image, detail)]


def build_text_with_image_content(
    text: str,
    image: RokidImagePayload,
    detail: str,
) -> list[ContentPart]:
    """Build a content array for a combined text+image request (FR-6)."""
    text_part = TextContentPart(type="text", text=text)
    image_part = validate_and_build_image_part(image, detail)
    return [text_part, image_part]
```

**Key Design Decisions:**
- `frozenset` for `ALLOWED_MIME_TYPES` provides O(1) lookup and is immutable — extending formats is a one-line change (OCP).
- `base64.b64decode(validate=True)` performs strict validation; without `validate=True`, Python silently ignores non-base64 characters.
- Byte-level size check on the decoded bytes (not the base64 string length) ensures accurate 20 MB enforcement.
- The data URL is assembled from the *original* base64 string (not re-encoded from decoded bytes) to avoid unnecessary memory churn.
- Image base64 data is NOT stored in history (see Section 6); this module never touches the history store.

**Dependencies:** `base64` (stdlib), `rokid_bridge.models`

**Error Contracts:**
- `ImageValidationError(detail, status_code=422)` — unsupported MIME or invalid base64.
- `ImageValidationError(detail, status_code=413)` — image exceeds size limit.

---

### 2.5 `history.py`

**Responsibility (SRP):** Manage per-device conversation history in memory with TTL eviction and bounded depth. No auth, no HTTP, no image conversion — only CRUD on message lists.

**SOLID Application:** Single Responsibility — the `HistoryStore` class is the only place that knows the session key scheme, eviction logic, and TTL policy.

```python
# src/rokid_bridge/history.py

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

from rokid_bridge.models import OpenAIMessage

SESSION_KEY_PREFIX: str = "rokid"
HISTORY_TTL_SECONDS: int = 3600  # 1 hour (FR-7, AC condition)


@dataclass
class _Session:
    """Internal session record for one device."""

    messages: Deque[OpenAIMessage] = field(default_factory=deque)
    last_access: float = field(default_factory=time.time)


class HistoryStore:
    """In-memory per-device conversation history store.

    Session key scheme: "rokid:{device_id}"
    Each session holds an alternating sequence of user+assistant messages.
    TTL eviction is lazy: checked on every access and during append.
    """

    def __init__(self, max_turns: int, ttl_seconds: int = HISTORY_TTL_SECONDS) -> None:
        """
        Args:
            max_turns: Maximum user+assistant turn pairs per device.
            ttl_seconds: Seconds of inactivity before session is evicted.
        """
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, _Session] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_messages(self, device_id: str) -> list[OpenAIMessage]:
        """Retrieve current history for a device; returns empty list if none.

        Side effect: evicts expired sessions lazily.

        Args:
            device_id: Opaque Rokid device identifier.

        Returns:
            Ordered list of OpenAI message dicts (role + content).
        """
        key = self._make_key(device_id)
        self._evict_if_expired(key)
        session = self._sessions.get(key)
        if session is None:
            return []
        session.last_access = time.time()
        return list(session.messages)

    def append_turn(
        self,
        device_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """Append a completed user+assistant turn to history.

        If depth exceeds max_turns, the oldest turn pair is evicted first.
        Image base64 data is NEVER stored; only text summaries are stored.

        Args:
            device_id: Opaque Rokid device identifier.
            user_content: Plain text of the user's query (stripped of image data).
            assistant_content: Full assistant response text assembled from SSE chunks.
        """
        key = self._make_key(device_id)
        self._evict_if_expired(key)
        if key not in self._sessions:
            self._sessions[key] = _Session()
        session = self._sessions[key]
        session.last_access = time.time()

        # Evict oldest turn pair if at capacity
        while len(session.messages) >= self._max_turns * 2:
            session.messages.popleft()  # evict user message
            if session.messages:
                session.messages.popleft()  # evict paired assistant message

        session.messages.append(OpenAIMessage(role="user", content=user_content))
        session.messages.append(OpenAIMessage(role="assistant", content=assistant_content))

    def clear(self, device_id: str) -> None:
        """Delete all history for a device (idempotent).

        Args:
            device_id: Opaque Rokid device identifier.
        """
        key = self._make_key(device_id)
        self._sessions.pop(key, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(device_id: str) -> str:
        return f"{SESSION_KEY_PREFIX}:{device_id}"

    def _evict_if_expired(self, key: str) -> None:
        session = self._sessions.get(key)
        if session is not None:
            age = time.time() - session.last_access
            if age > self._ttl_seconds:
                del self._sessions[key]
```

**Key Design Decisions:**
- `collections.deque` is used for `messages` because `popleft()` is O(1) — evicting the oldest message is always constant time regardless of history depth.
- TTL eviction is lazy — no background thread is required (YAGNI). The eviction check happens on every `get_messages` and `append_turn`, which ensures memory is reclaimed without blocking request processing (FR-7 AC-5).
- The `_Session` dataclass is a private implementation detail; callers receive only `list[OpenAIMessage]`.
- Image base64 content is deliberately NOT stored: `user_content` in `append_turn` receives only the text portion (see Section 6 for details). This prevents unbounded memory growth from repeated camera requests.
- The `while` loop (not `if`) in `append_turn` handles the edge case where `max_turns` is reduced at runtime — it evicts until the invariant is satisfied.
- `clear()` uses `dict.pop(key, None)` — idempotent by design, satisfying FR-10 AC-2.

**Dependencies:** `time`, `collections` (stdlib), `rokid_bridge.models`

**Error Contracts:** No exceptions raised; all operations are defensive and idempotent.

---

### 2.6 `relay.py`

**Responsibility (SRP):** Build the upstream OpenAI-compatible request payload and relay the response as an SSE stream. No auth, no rate limiting, no image conversion — only upstream communication and SSE passthrough.

**SOLID Application:** Open/Closed — the relay constructs the messages array from provided components; swapping in a different upstream API format requires only modifying `build_upstream_request`, not the streaming loop.

```python
# src/rokid_bridge/relay.py

import json
from collections.abc import AsyncGenerator

import httpx

from rokid_bridge.config import Settings
from rokid_bridge.models import ContentPart, OpenAIMessage, UpstreamRequest

AR_SYSTEM_PROMPT: str = (
    "You are an AI assistant for AR smart glasses. "
    "Be concise. Use short sentences. Avoid markdown. "
    "The user sees your reply on a small transparent display."
)

UPSTREAM_PATH: str = "/v1/chat/completions"
UPSTREAM_TIMEOUT_SECONDS: float = 30.0


class UpstreamError(Exception):
    """Raised when the upstream secure stack returns an error response.

    Attributes:
        status_code: HTTP status from upstream.
        body: Raw response body text (truncated to 500 chars in logs).
    """

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Upstream error {status_code}")
        self.status_code = status_code
        self.body = body


def build_upstream_request(
    *,
    history: list[OpenAIMessage],
    user_content: str | list[ContentPart],
    settings: Settings,
) -> UpstreamRequest:
    """Assemble the OpenAI chat completions request body.

    Message order: [system_prompt] + history + [new_user_message].
    The system prompt is NEVER stored in history (FR-8 AC-4).

    Args:
        history: Existing conversation messages for this device.
        user_content: Either a plain text string (text request) or a
                      content array (image/text_with_image request).
        settings: Application configuration.

    Returns:
        A dict conforming to the OpenAI chat completions API.
    """
    system_message = OpenAIMessage(role="system", content=AR_SYSTEM_PROMPT)
    user_message = OpenAIMessage(role="user", content=user_content)

    messages: list[OpenAIMessage] = [system_message, *history, user_message]

    request: UpstreamRequest = {
        "messages": messages,
        "stream": True,
    }
    if settings.rokid_agent_id:
        request["agent_id"] = settings.rokid_agent_id

    return request


async def stream_upstream(
    client: httpx.AsyncClient,
    upstream_request: UpstreamRequest,
    settings: Settings,
) -> AsyncGenerator[str, None]:
    """Stream SSE chunks from the upstream secure stack.

    Yields raw SSE lines (e.g., 'data: {...}') as they arrive.
    The caller is responsible for appending to history on completion.

    Args:
        client: Shared httpx.AsyncClient (injected, not created here).
        upstream_request: Pre-built request body dict.
        settings: Provides upstream_url and upstream_token.

    Yields:
        Raw SSE line strings received from upstream.

    Raises:
        UpstreamError: If the upstream returns a non-2xx response.
        httpx.TimeoutException: Converted to 504 by the caller.
        httpx.ConnectError: Converted to 502 by the caller.
    """
    url = f"{settings.upstream_url}{UPSTREAM_PATH}"
    headers = {
        "Authorization": f"Bearer {settings.upstream_token.get_secret_value()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with client.stream(
        "POST",
        url,
        json=upstream_request,
        headers=headers,
        timeout=UPSTREAM_TIMEOUT_SECONDS,
    ) as response:
        if response.status_code >= 300:
            body = await response.aread()
            raise UpstreamError(response.status_code, body.decode("utf-8", errors="replace"))

        async for line in response.aiter_lines():
            yield line


def extract_assistant_text(sse_chunks: list[str]) -> str:
    """Assemble the assistant's full response from collected SSE chunk lines.

    Called after streaming completes to build the history entry.
    Parses 'data: {...}' lines and extracts delta content fields.

    Args:
        sse_chunks: All SSE lines received during streaming.

    Returns:
        Concatenated assistant response text.
    """
    parts: list[str] = []
    for line in sse_chunks:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                parts.append(content)
        except (json.JSONDecodeError, IndexError, KeyError):
            continue
    return "".join(parts)
```

**Key Design Decisions:**
- `httpx.AsyncClient` is injected rather than instantiated here; the client is created once in `create_app()` and reused across requests — enabling connection pooling and clean lifecycle management.
- `async for line in response.aiter_lines()` yields as soon as a newline is received from upstream — zero per-chunk buffering (FR-9 AC-3).
- SSE chunks are yielded as raw strings; `app.py` wraps them in `StreamingResponse` without re-parsing — the relay is a transparent passthrough (FR-9 AC-2).
- `extract_assistant_text` operates on the collected chunk list *after* streaming completes — history is updated only on success (FR-7 AC-8). The list is accumulated in the generator wrapper in `app.py`.
- `AR_SYSTEM_PROMPT` is a module-level constant in `relay.py` (not an env var) — see Section 8 for rationale.
- `UpstreamError` carries `status_code` so `app.py` can forward the exact HTTP status to the Rokid client (NFR-5).

**Dependencies:** `json` (stdlib), `httpx`, `rokid_bridge.config`, `rokid_bridge.models`

**Error Contracts:**
- `UpstreamError(status_code, body)` — upstream returned non-2xx.
- `httpx.TimeoutException` — caller maps to HTTP 504.
- `httpx.ConnectError` — caller maps to HTTP 502.

---

### 2.7 `app.py`

**Responsibility (SRP):** Wire together all components, define HTTP routes, manage per-device rate limiting, and handle error-to-HTTP-response mapping. No auth logic, no image parsing, no history storage — only composition and routing.

**SOLID Application:** Dependency Inversion — all stateful dependencies (`HistoryStore`, `httpx.AsyncClient`, `Settings`) are created in `create_app()` and injected into routes via closure or `Depends`. Routes do not instantiate anything directly.

```python
# src/rokid_bridge/app.py  (signatures and structure — not exhaustive implementation)

import logging
import time
from collections import deque
from collections.abc import AsyncGenerator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from rokid_bridge.config import Settings, get_settings
from rokid_bridge.history import HistoryStore
from rokid_bridge.image_handler import (
    ImageValidationError,
    build_image_content,
    build_text_with_image_content,
)
from rokid_bridge.models import (
    ClearHistoryResponse,
    HealthResponse,
    RokidChatRequest,
    RokidClearRequest,
    RokidRequestType,
)
from rokid_bridge.relay import (
    UpstreamError,
    build_upstream_request,
    extract_assistant_text,
    stream_upstream,
)
from rokid_bridge.rokid_auth import AuthError, check_replay_window, verify_bearer_token

logger = logging.getLogger(__name__)

RATE_WINDOW_SECONDS: int = 60

# ---------------------------------------------------------------------------
# Rate Limiter (FR-3) — sliding window, O(1) amortised per operation
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-device sliding window rate limiter.

    Internal structure: dict mapping device_id → deque of request timestamps.
    Each deque holds timestamps (float) within the current window.
    """

    def __init__(self, max_requests: int, window_seconds: int = RATE_WINDOW_SECONDS) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def check_and_record(self, device_id: str) -> tuple[bool, float]:
        """Check rate limit for a device and record this request if allowed.

        Returns:
            (allowed, retry_after_seconds) — retry_after is 0.0 if allowed.
        """
        now = time.time()
        if device_id not in self._buckets:
            self._buckets[device_id] = deque()
        bucket = self._buckets[device_id]

        # Evict timestamps outside the current window (O(n) worst case but
        # bounded by max_requests, so effectively O(1) amortised)
        cutoff = now - self._window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self._max:
            # Time until the oldest request exits the window
            retry_after = self._window - (now - bucket[0])
            return False, max(retry_after, 0.0)

        bucket.append(now)
        return True, 0.0


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI application factory.

    Creates all stateful components once and closes over them in route handlers.
    This pattern enables dependency injection in tests via settings override.

    Args:
        settings: Optional Settings override (for tests). If None, loads from env.

    Returns:
        Configured FastAPI application instance.
    """
    _settings = settings or get_settings()
    _history = HistoryStore(max_turns=_settings.rokid_max_history_turns)
    _rate_limiter = RateLimiter(max_requests=_settings.rokid_rate_limit)
    _http_client = httpx.AsyncClient()

    app = FastAPI(title="Rokid Bridge", version="0.1.0")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await _http_client.aclose()

    # -----------------------------------------------------------------------
    # Authentication dependency — shared by /chat and /clear-history
    # -----------------------------------------------------------------------

    def _require_auth(request: Request) -> None:
        """FastAPI dependency: verify Bearer token. Raises HTTP 401 on failure."""
        try:
            verify_bearer_token(request.headers.get("Authorization"), _settings)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.post("/chat", dependencies=[Depends(_require_auth)])
    async def chat(body: RokidChatRequest, request: Request) -> StreamingResponse:
        """Main chat endpoint. See Section 9 for pipeline order."""
        ...

    @app.post("/clear-history", dependencies=[Depends(_require_auth)])
    async def clear_history(body: RokidClearRequest) -> ClearHistoryResponse:
        """Clear per-device history."""
        ...

    @app.get("/health")
    async def health() -> HealthResponse:
        """Liveness probe — no auth, no I/O."""
        return HealthResponse()

    return app
```

**Key Design Decisions:**
- `create_app()` factory pattern (not a module-level `app` object) enables test isolation — each test can call `create_app(settings=override)` without shared state between tests.
- `_require_auth` is defined as a closure inside `create_app()` so it captures `_settings` — no global state, no import-time side effects.
- The `RateLimiter` is defined in `app.py` rather than its own module because it is intrinsically tied to the HTTP routing concern — it wraps request handling, not a domain concept. (YAGNI: a separate module would add abstraction with no benefit at this scale.)
- `httpx.AsyncClient` is shared across requests and closed on app shutdown — avoids per-request TCP handshake overhead.
- Error handling follows the pattern: domain exceptions (`AuthError`, `ImageValidationError`, `UpstreamError`, `httpx.*Error`) are caught in route handlers and mapped to appropriate HTTP responses.

**Dependencies:** All other `rokid_bridge` modules, `fastapi`, `httpx`, `logging`

**Error Contracts:** Route handlers catch all domain exceptions and map to `HTTPException` or `JSONResponse`. Uncaught exceptions bubble to FastAPI's default 500 handler.

---

## 3. Data Models

### 3.1 RokidChatRequest

| Field | Type | Required | Validation | Maps To |
|-------|------|----------|------------|---------|
| `request_id` | `str` | Yes | Non-empty string | Logged for observability |
| `device_id` | `str` | Yes | Non-empty string | History key, rate limit key |
| `type` | `RokidRequestType` | Yes | Enum: text/image/text_with_image | Route dispatch |
| `text` | `str` | No (default `""`) | Blank enforcement per-type in route | User message content |
| `image` | `RokidImagePayload \| None` | No | Required when type=image or text_with_image | Image handler |
| `timestamp` | `int` | Yes | Unix seconds integer | Replay window check |

### 3.2 RokidImagePayload

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `data` | `str` | Yes | Valid base64, decoded size <= 20 MB |
| `mime_type` | `str` | Yes | Must be `image/jpeg` or `image/png` |

### 3.3 RokidClearRequest

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `device_id` | `str` | Yes | Non-empty string |
| `timestamp` | `int` | Yes | Unix seconds; replay window enforced |

### 3.4 HealthResponse

```json
{"status": "ok", "service": "rokid-bridge"}
```

### 3.5 ClearHistoryResponse

```json
{"cleared": true, "device_id": "<device_id>"}
```

### 3.6 Upstream Request Shape (sent to Secure Stack)

```json
{
  "messages": [
    {"role": "system", "content": "<AR_SYSTEM_PROMPT>"},
    {"role": "user",   "content": "<previous user turn>"},
    {"role": "assistant", "content": "<previous assistant turn>"},
    {"role": "user",   "content": "<current user turn>"}
  ],
  "stream": true,
  "agent_id": "<ROKID_AGENT_ID>"
}
```

The `agent_id` field is omitted when `ROKID_AGENT_ID` is an empty string.

For image and text_with_image requests, the last user message `content` is a content array:

```json
[
  {"type": "text", "text": "What is this?"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "detail": "low"}}
]
```

---

## 4. ADR-001: Authentication Design

### Context

The Rokid/Lingzhu platform issues an Access Key (AK) to developers. This AK is configured on the Lingzhu platform as the credential for the endpoint. When Rokid glasses make a request to the bridge, the platform includes the AK in the HTTP `Authorization` header as a Bearer token.

The original requirements assumed HMAC-SHA256 body signing (the AK is used to compute a signature over the request body). Research confirmed the actual Lingzhu platform behavior: the AK is sent directly as a Bearer token — it is a shared secret, not an HMAC signing key.

### Decision

Authenticate incoming requests by:
1. Extracting the token from `Authorization: Bearer <token>`.
2. Comparing it to `settings.rokid_access_key.get_secret_value()` using `hmac.compare_digest()`.

The `hmac.compare_digest()` function is used even for a direct token comparison (not an HMAC digest comparison) because it provides constant-time comparison behavior that prevents timing-based oracle attacks (NFR-1).

### Consequences

**Positive:**
- Simpler implementation — no per-request HMAC computation over the body.
- Fewer moving parts — no timestamp header dependency for auth (timestamp is in the body for replay protection).
- `hmac.compare_digest` satisfies the constant-time requirement with zero additional code.

**Negative:**
- The AK is a long-lived shared secret. If it leaks, all devices using that key are compromised until it is rotated.
- Mitigation: `SecretStr` in Settings (NFR-2), non-root Docker container (NFR-3), and Cloudflare Tunnel (TLS in transit).

### Alternatives Considered

**HMAC-SHA256 Body Signing:**
- Would require Rokid to compute `HMAC-SHA256(AK, "{timestamp}.{body_bytes}")` and send the result in a header.
- Confirmed that Lingzhu platform does NOT do this — it sends the AK directly.
- Not viable given platform behavior.

**API Key in Query Parameter:**
- Rejected: query parameters appear in access logs and CDN caches, violating NFR-2.

### Replay Protection (Complementary to Auth)

The `timestamp` field in the JSON body is checked *after* auth verification:
- Must be within `ROKID_REPLAY_WINDOW` seconds of current server time (default 300s).
- Must not be more than 60 seconds in the future (clock skew tolerance).
- This prevents an attacker who captures a valid Bearer-token request from replaying it indefinitely.

---

## 5. SSE Streaming Design

### 5.1 End-to-End Flow

```
Rokid Client                   Bridge (:8090)              Secure Stack (:8080)
     │                              │                              │
     │ POST /chat                   │                              │
     │─────────────────────────────►│                              │
     │                              │ POST /v1/chat/completions    │
     │                              │─────────────────────────────►│
     │                              │                              │
     │                              │◄── SSE chunk 1 ─────────────│
     │◄── SSE chunk 1 ──────────────│  (yield from aiter_lines)   │
     │                              │◄── SSE chunk 2 ─────────────│
     │◄── SSE chunk 2 ──────────────│                              │
     │                              │◄── data: [DONE] ────────────│
     │◄── data: [DONE] ─────────────│                              │
     │                              │  (close upstream connection) │
     │                              │  extract_assistant_text()    │
     │                              │  history.append_turn()       │
     │ (connection closed)          │                              │
```

### 5.2 Streaming Generator with History Collection

The route handler uses a generator that:
1. Accumulates chunk lines in a local list (in-memory, lightweight strings).
2. Yields each line to `StreamingResponse` immediately (zero buffering delay).
3. After the upstream closes, calls `extract_assistant_text()` on the accumulated list.
4. Updates history only on successful completion.

```python
async def _sse_generator(
    client: httpx.AsyncClient,
    upstream_request: UpstreamRequest,
    settings: Settings,
    history: HistoryStore,
    device_id: str,
    user_text_for_history: str,
) -> AsyncGenerator[str, None]:
    """Streaming generator: relay SSE and collect for history."""
    collected: list[str] = []
    success = False
    try:
        async for line in stream_upstream(client, upstream_request, settings):
            collected.append(line)
            yield f"{line}\n"  # SSE line + newline passthrough
        success = True
    except UpstreamError as exc:
        yield f"data: {{'error': 'upstream error {exc.status_code}'}}\n"
    except httpx.TimeoutException:
        yield "data: {'error': 'upstream timeout'}\n"
    finally:
        if success and collected:
            assistant_text = extract_assistant_text(collected)
            history.append_turn(device_id, user_text_for_history, assistant_text)
```

**Why the `finally` block handles history:** Python generators guarantee that `finally` runs when the generator is fully consumed or garbage collected — this gives us a reliable hook to update history after streaming, without requiring the route handler to coordinate separately.

### 5.3 SSE Chunk Format

The bridge is a transparent passthrough. Each chunk from upstream:
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"},"index":0}]}

```
Is forwarded verbatim. The bridge does not re-parse or transform SSE chunk payloads (only `extract_assistant_text` parses them post-stream for history storage).

### 5.4 StreamingResponse Configuration

```python
return StreamingResponse(
    _sse_generator(...),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # Disable Nginx/Caddy buffering
    },
)
```

`X-Accel-Buffering: no` is critical — without it, some reverse proxies buffer the SSE stream and the Rokid client receives chunks in large batches rather than individually.

### 5.5 Mid-Stream Error Handling

If the upstream connection drops mid-stream:
- The `async for` loop in `stream_upstream` raises `httpx.RemoteProtocolError` or `httpx.ReadError`.
- The `except` clause in `_sse_generator` emits a final error SSE event.
- `success` remains `False`, so history is NOT updated.
- The generator returns, closing the downstream connection to the Rokid client.

---

## 6. Conversation History Design

### 6.1 Session Key Scheme

All history is keyed by: `"rokid:{device_id}"`

The prefix namespace prevents any future collision if the same in-memory dict is reused for other session types (OCP — open to extension without modifying the `HistoryStore` interface).

### 6.2 In-Memory Data Structure

```
_sessions: dict[str, _Session]

_Session:
  messages: deque[OpenAIMessage]   # ordered, bounded
  last_access: float               # Unix timestamp, updated on every access
```

A `collections.deque` is used over a `list` because:
- `popleft()` (evict oldest) is O(1) on deque, O(n) on list.
- `append()` (add newest) is O(1) on both, but deque is the idiomatic choice for FIFO eviction.

### 6.3 Why Image Base64 is NOT Stored

Storing base64 image data in history would:
- Add up to 20 MB per image per turn to in-memory storage.
- Cause uncontrolled memory growth with frequent camera requests.
- Violate KISS — the upstream LLM does not use prior image turns for context in typical deployments.

**Decision:** When `type == "image"`, the `user_content` parameter to `history.append_turn()` is set to a descriptive placeholder such as `"[image request]"` rather than the base64 data. When `type == "text_with_image"`, the `user_content` is the text portion only. The image data is never written to `_sessions`.

This is documented in `append_turn`'s docstring and enforced by the route handler — `image_handler.py` does not call `history.py` directly.

### 6.4 Truncation Policy

Eviction removes the oldest **turn pair** (user + assistant) as a unit, preserving role alternation invariant. Removing only one of the pair would create an orphaned message that confuses the upstream model.

```
Before eviction (max_turns=2, 5 messages — invariant violation prevented by while loop):
  [user_0, asst_0, user_1, asst_1, user_2]  ← would be broken

After eviction (remove user_0 + asst_0):
  [user_1, asst_1, user_2]  ← new assistant appended after this
```

The `while` loop (not `if`) ensures that if `max_turns` is lowered at runtime (e.g., via config reload), the deque is brought within bounds in a single call.

### 6.5 TTL Eviction

TTL is checked lazily on every `get_messages()` and `append_turn()` call. The `last_access` timestamp is updated on successful retrieval, not on failed eviction checks. This means:
- A device that has been silent for > 1 hour has its session evicted on the next request — the next request starts with empty history.
- No background timer or asyncio task is required (YAGNI).
- Memory is bounded: at most `N` devices with active sessions, where `N` is the number of distinct `device_id` values seen within any 1-hour window.

---

## 7. Rate Limiter Design

### 7.1 Algorithm: Sliding Window with Deque

Each device gets a `deque[float]` of request timestamps within the current window.

On each `check_and_record(device_id)` call:
1. Compute `cutoff = now - window_seconds`.
2. Pop all timestamps from the left of the deque that are `<= cutoff` (expired).
3. If `len(deque) >= max_requests`: return `(False, retry_after)`.
4. Else: append `now` and return `(True, 0.0)`.

### 7.2 Complexity

- **Space**: O(max_requests) per device — the deque never exceeds `max_requests` entries.
- **Time (amortised)**: Each timestamp is pushed exactly once and popped exactly once → O(1) amortised per call.
- **Worst case**: O(max_requests) — all entries are expired and must be popped.

### 7.3 Retry-After Header

When a device is rate-limited, the response includes:
```
HTTP/1.1 429 Too Many Requests
Retry-After: 42
```

`retry_after = window_seconds - (now - bucket[0])` where `bucket[0]` is the oldest timestamp in the window.

### 7.4 Placement in Pipeline

Rate limiting is checked **before** auth in the design overview diagram, but **after** auth in the actual code pipeline (see Section 9). This is intentional and is explained in Section 9.

### 7.5 Why Not a Library

A dedicated rate-limiting library (e.g., `slowapi`) would add a dependency and force a specific integration pattern. The deque-based implementation is 20 lines of code and is fully testable in isolation — KISS and YAGNI apply.

---

## 8. AR System Prompt

### 8.1 Exact Prompt Text

```
You are an AI assistant for AR smart glasses. Be concise. Use short sentences. Avoid markdown. The user sees your reply on a small transparent display.
```

This constant is defined as `AR_SYSTEM_PROMPT` in `relay.py`.

### 8.2 Rationale for Constant (Not Env Var)

**Why a constant:**
- The prompt is a **product requirement**, not a deployment parameter. Changing it requires understanding its impact on UX (FR-8 AC-3), which warrants a code change and review — not a silent env var override.
- Making it configurable would violate YAGNI — there is no current requirement to vary the prompt per environment.
- A constant is testable by asserting `AR_SYSTEM_PROMPT` in the messages payload (FR-8 testability note).

**Why in `relay.py` (not `models.py` or `config.py`):**
- It is consumed only in `build_upstream_request()` — co-locating the constant with its consumer (SRP, locality of reference).
- Placing it in `config.py` would falsely imply it is user-configurable.
- Placing it in `models.py` would couple the data model layer to presentation concerns.

**What the prompt achieves:**
- "Be concise / short sentences" — reduces multi-sentence prose that wraps awkwardly on a small display.
- "Avoid markdown" — eliminates `**bold**`, `- bullets`, `` `code` `` which render as raw characters on non-markdown displays.
- "Small transparent display" — contextualizes the constraint so the model self-regulates verbosity.

---

## 9. Request Processing Pipeline

### 9.1 POST /chat Pipeline

```
Step   Operation              Module            Failure → HTTP
─────  ─────────────────────  ────────────────  ───────────────────────────
  1    Parse JSON body        FastAPI/Pydantic   422 Unprocessable Entity
       (RokidChatRequest)
  2    Rate limit check       RateLimiter        429 Too Many Requests
       (device_id)            (app.py)           + Retry-After header
  3    Auth verify            rokid_auth.py      401 Unauthorized
       (Bearer token)
  4    Replay check           rokid_auth.py      401 Request expired / invalid
       (body.timestamp)
  5    Type-specific          app.py + image_    422 (blank text, missing image,
       payload validation     handler.py         bad MIME, bad base64)
                                                 413 (image too large)
  6    Retrieve history       history.py         — (returns empty list if none)
  7    Build upstream         relay.py           — (pure construction)
       request
  8    Stream upstream        relay.py           502/504 on connect/timeout error
       (httpx streaming)                         SSE error event on mid-stream drop
  9    Relay SSE chunks       app.py             SSE error event on failure
       to Rokid client        StreamingResponse
 10    Collect chunks,        relay.py           — (history not updated on failure)
       update history         history.py
```

### 9.2 Justification for Pipeline Order

**Rate Limiting before Auth (Step 2 before Step 3):**

This ordering is deliberate for DoS protection:
- An attacker sending thousands of unauthenticated requests would cause HMAC computation (or token comparison) to run on every request, consuming CPU.
- With rate limiting first, the rate limiter rejects excess requests with a cheap deque lookup before any auth computation happens.
- **However**, the rate limiter keys on `device_id`, which comes from the (already-parsed) JSON body. An unauthenticated attacker can still flood with many `device_id` values to bypass per-device limits. This is an accepted trade-off at v0.1 scale (YAGNI: IP-based rate limiting at the CDN/Cloudflare Tunnel layer is the correct mitigation for that attack vector).

**Auth before Replay (Step 3 before Step 4):**
- Checking the timestamp before auth would allow timing oracle attacks where an attacker probes timestamp validity without a valid token.
- Auth first, timestamp second prevents this (as specified in FR-2 AC-6).

**History retrieved after auth (Step 6 after Step 3-4):**
- History lookup is in-memory and O(1), but it is still wasteful to do for unauthenticated requests.
- The ordering also prevents an unauthenticated request from refreshing a device's TTL timer.

### 9.3 POST /clear-history Pipeline

```
Step   Operation              Module            Failure → HTTP
─────  ─────────────────────  ────────────────  ───────────────────────────
  1    Parse JSON body        FastAPI/Pydantic   422 Unprocessable Entity
       (RokidClearRequest)
  2    Auth verify            rokid_auth.py      401 Unauthorized
       (Bearer token)
  3    Replay check           rokid_auth.py      401 Request expired / invalid
       (body.timestamp)
  4    Clear history          history.py         — (idempotent)
  5    Return 200             app.py             —
```

Auth is reused via the `_require_auth` FastAPI dependency — the same function serves both `/chat` and `/clear-history` (DRY principle in action).

---

## 10. Docker and Deployment Design

### 10.1 Multi-Stage Dockerfile

```dockerfile
# Stage 1: Builder — installs dependencies into a venv
FROM python:3.12-slim AS builder

# Install uv
RUN pip install uv --no-cache-dir

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

# Create venv and install production deps only (no [dev] extras — NFR-3 AC-4)
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install .

# Stage 2: Runtime — minimal image, no build tools
FROM python:3.12-slim AS runtime

# Security hardening: non-root user (NFR-3 AC-1)
RUN groupadd -r appgroup && \
    useradd -r -g appgroup -u 1000 appuser

# Copy only the venv from builder stage (no uv, no pip, no build tools)
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Copy source (no tests, no spec files, no .env)
COPY src/ src/

# Switch to non-root before CMD (NFR-3 AC-1)
USER appuser

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port (documentation only; actual binding via docker-compose)
EXPOSE 8090

CMD ["uvicorn", "rokid_bridge.app:create_app", "--factory",
     "--host", "0.0.0.0", "--port", "8090"]
```

**Multi-stage rationale:**
- Stage 1 includes build tools and uv; Stage 2 does not — minimizes attack surface.
- Only the compiled venv is copied, excluding pip, wheel, and uv from the runtime image.
- `pyproject.toml` is copied before `src/` to leverage Docker layer caching — rebuilds only reinstall deps when `pyproject.toml` changes.

### 10.2 docker-compose.yml (Standalone)

```yaml
# docker-compose.yml
version: "3.9"
services:
  rokid-bridge:
    build: .
    image: rokid-bridge:latest
    ports:
      - "8090:8090"
    read_only: true                     # NFR-3 AC-2
    tmpfs:
      - /tmp:size=64m,mode=1777        # NFR-3 AC-3
    environment:
      - ROKID_ACCESS_KEY=${ROKID_ACCESS_KEY}
      - UPSTREAM_TOKEN=${UPSTREAM_TOKEN}
      - UPSTREAM_URL=${UPSTREAM_URL:-http://host.docker.internal:8080}
      - ROKID_AGENT_ID=${ROKID_AGENT_ID:-}
      - ROKID_RATE_LIMIT=${ROKID_RATE_LIMIT:-30}
      - ROKID_REPLAY_WINDOW=${ROKID_REPLAY_WINDOW:-300}
      - ROKID_MAX_HISTORY_TURNS=${ROKID_MAX_HISTORY_TURNS:-20}
      - ROKID_IMAGE_DETAIL=${ROKID_IMAGE_DETAIL:-low}
      - PORT=${PORT:-8090}
    restart: unless-stopped
    cap_drop:
      - ALL                             # Drop all Linux capabilities
    security_opt:
      - no-new-privileges:true
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8090/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
```

### 10.3 docker-compose.override.yml (Co-Deploy)

```yaml
# docker-compose.override.yml
# Used when rokid-bridge is deployed alongside openclaw-secure-stack
version: "3.9"
services:
  rokid-bridge:
    networks:
      - openclaw-net
    environment:
      - UPSTREAM_URL=http://openclaw-secure-stack:8080
    ports: []  # Remove external port mapping; internal traffic only

networks:
  openclaw-net:
    external: true  # Pre-existing network created by openclaw-secure-stack
```

**Co-deploy rationale:** When on the same Docker network, service discovery uses container names — `http://openclaw-secure-stack:8080` — and no port is exposed externally. Cloudflare Tunnel provides the external HTTPS entry point.

### 10.4 Security Hardening Summary

| Measure | Config Location | Satisfies |
|---------|----------------|-----------|
| Non-root user (UID 1000) | Dockerfile `USER appuser` | NFR-3 AC-1 |
| Read-only root filesystem | `docker-compose.yml read_only: true` | NFR-3 AC-2 |
| `/tmp` as tmpfs | `docker-compose.yml tmpfs` | NFR-3 AC-3 |
| No dev dependencies in image | Dockerfile multi-stage | NFR-3 AC-4 |
| All Linux capabilities dropped | `docker-compose.yml cap_drop` | Defense-in-depth |
| `no-new-privileges` | `docker-compose.yml security_opt` | Defense-in-depth |

---

## 11. Interface Contracts

### 11.1 POST /chat

**Request:**
```
POST /chat
Authorization: Bearer <ROKID_ACCESS_KEY>
Content-Type: application/json

{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "device_id": "rokid-serial-abc123",
  "type": "text",
  "text": "What is the weather like today?",
  "timestamp": 1708300000
}
```

**Success Response:**
```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"It"},"index":0}]}

data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":" is"},"index":0}]}

data: [DONE]

```

**Error Responses:**

| Condition | Status | Body |
|-----------|--------|------|
| Missing/invalid Bearer token | 401 | `{"detail": "Unauthorized"}` |
| Expired timestamp | 401 | `{"detail": "Request expired"}` |
| Future timestamp | 401 | `{"detail": "Request timestamp invalid"}` |
| Rate limit exceeded | 429 | `{"detail": "Rate limit exceeded"}` + `Retry-After: N` header |
| Invalid JSON / missing fields | 422 | FastAPI validation error format |
| Blank text on text request | 422 | `{"detail": "Text is required for type 'text'"}` |
| Missing image on image request | 422 | `{"detail": "Image is required for type 'image'"}` |
| Unsupported MIME type | 422 | `{"detail": "Unsupported image format"}` |
| Invalid base64 | 422 | `{"detail": "Invalid base64 image data"}` |
| Image > 20 MB | 413 | `{"detail": "Image too large"}` |
| Upstream 4xx/5xx | upstream status | Upstream error body (forwarded) |
| Upstream timeout | 504 | `{"detail": "Upstream timeout"}` |
| Upstream connection refused | 502 | `{"detail": "Upstream unavailable"}` |

### 11.2 POST /clear-history

**Request:**
```
POST /clear-history
Authorization: Bearer <ROKID_ACCESS_KEY>
Content-Type: application/json

{
  "device_id": "rokid-serial-abc123",
  "timestamp": 1708300000
}
```

**Success Response:**
```
HTTP/1.1 200 OK
Content-Type: application/json

{"cleared": true, "device_id": "rokid-serial-abc123"}
```

**Error Responses:** Same auth/replay errors as `/chat`; no 413/422 for payload beyond missing required fields.

### 11.3 GET /health

**Request:**
```
GET /health
```

**Success Response:**
```
HTTP/1.1 200 OK
Content-Type: application/json

{"status": "ok", "service": "rokid-bridge"}
```

No auth required. No upstream I/O. Always responds within 100ms (FR-11 AC-3).

---

## 12. Security Design Summary

### 12.1 Threat Model

| Threat | Mitigation |
|--------|-----------|
| Token theft (replay with valid token) | Timestamp replay window (FR-2); TTL: 300s default |
| Timing attack on token comparison | `hmac.compare_digest` (NFR-1) |
| Brute-force token guessing | Rate limiting per device (FR-3); Cloudflare Tunnel DDoS protection |
| Secret leakage in logs | `SecretStr` type; log redaction (NFR-2) |
| Container privilege escalation | Non-root user + cap_drop + no-new-privileges (NFR-3) |
| Filesystem tampering from container | Read-only root FS (NFR-3) |
| Oversized image DoS | 20 MB hard limit with 413 response (FR-5 AC-4) |
| Malformed JSON body DoS | FastAPI/Pydantic parsing with `extra="forbid"` (FR-12) |
| Upstream SSRF | `UPSTREAM_URL` is a configured env var, not user-controlled |

### 12.2 Secret Handling Flow

```
Environment variable
  ROKID_ACCESS_KEY=<actual-value>
       │
       ▼
  pydantic-settings
  Settings.rokid_access_key: SecretStr
       │
       ├── str(settings.rokid_access_key)       → "**********"  (safe)
       ├── repr(settings.rokid_access_key)      → "SecretStr('**********')"  (safe)
       └── settings.rokid_access_key.get_secret_value()  → "<actual-value>"  (used only in rokid_auth.py)
```

The `.get_secret_value()` call appears in exactly ONE location: `rokid_auth.verify_bearer_token()`. Loggers, error responses, and all other code receive the masked representation.

### 12.3 Security-Critical Code Paths

These paths require 100% test coverage (NFR-8):
- `rokid_auth.verify_bearer_token()` — all branches
- `rokid_auth.check_replay_window()` — valid, expired, future timestamp cases
- Auth error handling in `app.py` route handlers — mapping `AuthError` to HTTP 401

---

## 13. Testing Strategy

### 13.1 Test Pyramid

```
                    ┌───────────────────────────┐
                    │   Integration Tests (15%)  │
                    │   tests/integration/        │
                    │   TestClient + mock httpx   │
                    ├───────────────────────────┤
                    │     Unit Tests (85%)        │
                    │     tests/unit/             │
                    │     Pure functions, mocks   │
                    └───────────────────────────┘
```

### 13.2 Unit Test Boundaries

| Test File | Module Under Test | What to Mock | Key Scenarios |
|-----------|-------------------|--------------|---------------|
| `test_rokid_auth.py` | `rokid_auth.py` | `time.time` (via `now=` param) | Valid token, missing header, wrong token, expired ts, future ts, clock edge cases |
| `test_image_handler.py` | `image_handler.py` | Nothing (pure functions) | Valid JPEG/PNG, unsupported MIME, invalid b64, too large, exact size boundary |
| `test_history.py` | `history.py` | `time.time` (monkey-patch) | Append, get, clear, truncation at max_turns, TTL eviction, cross-device isolation |
| `test_relay.py` | `relay.py` | `httpx.AsyncClient` (pytest-httpx) | System prompt first, history included, agent_id present/absent, [DONE] parsing, extract_assistant_text edge cases |

### 13.3 Integration Test Approach

**File:** `tests/integration/test_endpoints.py`
**Framework:** `fastapi.testclient.TestClient` (sync) + `pytest-httpx` for mocking outbound httpx calls

```python
# Test fixture pattern
@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        rokid_access_key=SecretStr("test-ak-12345"),
        upstream_token=SecretStr("test-upstream-token"),
        upstream_url="http://mock-upstream",
    )

@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    app = create_app(settings=test_settings)
    return TestClient(app)
```

**Key integration scenarios:**

| Scenario | Type | Mocks Needed |
|----------|------|-------------|
| Full chat flow (text) | Happy path | httpx returning SSE chunks |
| Full chat flow (image) | Happy path | httpx returning SSE chunks |
| Full chat flow (text_with_image) | Happy path | httpx returning SSE chunks |
| Auth rejected (wrong token) | Failure | None |
| Replay rejected (old timestamp) | Failure | None |
| Rate limit exceeded | Failure | None |
| History persists across requests | State | httpx mock |
| Clear history then chat = empty history | State | httpx mock |
| Upstream 500 forwarded | Upstream error | httpx returning 500 |
| Upstream timeout → 504 | Upstream error | httpx timeout |
| Upstream refused → 502 | Upstream error | httpx connect error |
| Health check (no auth) | Liveness | None |

### 13.4 What to Mock vs. What to Test Real

**Mock:**
- `httpx.AsyncClient` outbound calls — use `pytest-httpx` fixtures to simulate upstream SSE responses without running the secure stack.
- `time.time` — inject `now=` parameter in `check_replay_window` and monkey-patch in `HistoryStore` tests.

**Test Real (do not mock):**
- Pydantic model parsing — tests should exercise actual model validation.
- `hmac.compare_digest` — test with real known-good and known-bad token pairs.
- `base64.b64decode` — test with real valid and invalid base64 strings.
- FastAPI routing and dependency injection — integration tests use real FastAPI app via `TestClient`.

### 13.5 Coverage Targets

| Module | Target | Rationale |
|--------|--------|-----------|
| `rokid_auth.py` | 100% lines + branches | Security-critical path (NFR-8) |
| `app.py` (auth paths) | 100% | All auth error handling branches |
| `image_handler.py` | 95%+ | All validation branches |
| `history.py` | 90%+ | TTL + truncation edge cases |
| `relay.py` | 85%+ | SSE parsing, upstream error mapping |
| `config.py` | 80%+ | Validation at startup |
| **Overall** | 80%+ | Minimum project standard |

### 13.6 TDD Cycle per Task

Each implementation task follows Red → Green → Refactor:
1. Write the failing test first (defines the contract).
2. Write minimum code to pass the test.
3. Refactor for clarity without breaking tests.
4. Run `mypy --strict` and `ruff check` before committing.

---

## 14. Open Questions and Risks

### 14.1 Confirmed vs. Uncertain

| # | Item | Status | Mitigation |
|---|------|--------|-----------|
| OQ-1 | Rokid sends `Authorization: Bearer <AK>` | Confirmed per research | Design built on this |
| OQ-2 | Timestamp is in the JSON body (`"timestamp": int`) | Confirmed | `RokidChatRequest.timestamp` field |
| OQ-3 | `request_id` is always present (UUID v4) | Assumed — Rokid platform docs confirm this | Model has it required; if missing → 422 |
| OQ-4 | `device_id` format | Opaque string assumed | No format validation — any non-empty string accepted |
| OQ-5 | Rokid retries on SSE error event | Unknown | Bridge emits well-formed SSE error event; Rokid behavior TBD |
| OQ-6 | Maximum concurrent devices in production | Unknown | In-memory `dict` unbounded; mitigation: Cloudflare Tunnel device limits |
| OQ-7 | Lingzhu Platform AK rotation mechanism | Unknown | AK is env var — rotation requires container restart or secret management (v0.2) |

### 14.2 Runtime Mitigations for Uncertain Items

**OQ-4 (device_id format):** The bridge accepts any non-empty string as `device_id`. If Rokid sends a device ID that collides with another device's ID (theoretical), the attacker would need a valid AK to exploit it. The risk is negligible given AK protection.

**OQ-5 (retry behavior):** The SSE error event format is:
```
data: {"error": "upstream error <status_code>"}

```
If Rokid does not handle this gracefully, the user simply needs to re-invoke the AI. This is a UX concern for v0.2, not a correctness concern.

**OQ-6 (concurrent devices):** The `_sessions` dict grows with distinct device IDs. With TTL eviction, the maximum in-memory footprint is bounded by the number of unique devices active within the TTL window. For typical AR glasses deployments (tens to hundreds of concurrent devices), this is not a concern at v0.1. Horizontal scaling (Redis-backed history) is explicitly deferred to v0.2 (YAGNI).

### 14.3 Deferred Features (YAGNI — explicitly out of scope for v0.1)

| Feature | Reason Deferred | When to Revisit |
|---------|----------------|-----------------|
| Persistent history (Redis/DB) | No requirement for cross-restart continuity | v0.2, if user feedback demands it |
| Horizontal scaling | Single-instance model; no requirement | v0.2, when load exceeds single-instance capacity |
| IP-based rate limiting | Per-device is sufficient; CF Tunnel handles IP-level DDoS | v0.2 |
| WebSocket support | Rokid platform uses HTTP POST + SSE | Only if Rokid changes protocol |
| AK rotation without restart | Requires secrets management integration | v0.2 |
| Metrics endpoint (Prometheus) | Structured logs satisfy observability at v0.1 | v0.2 |
| HMAC-SHA256 body signing | Lingzhu platform does not produce HMAC signatures | Only if Rokid changes auth scheme |

---

## 15. Approval Checklist

### Architecture Quality

- [ ] Architecture diagram included with full component detail and Docker topology
- [ ] All 7 modules have SRP statements, signatures with type hints, and error contracts
- [ ] Hexagonal architecture (ports & adapters) is clearly implemented
- [ ] No circular imports between modules (`app.py` imports all; no other module imports `app.py`)

### SOLID Principles Coverage

- [ ] **SRP**: Each module has one reason to change (documented per module)
- [ ] **OCP**: `ALLOWED_MIME_TYPES` frozenset enables format extension without modifying handler logic
- [ ] **LSP**: No inheritance hierarchy; TypedDicts are substitutable (not applicable here — flat design)
- [ ] **ISP**: Each endpoint uses a focused Pydantic model, not a shared "god model"
- [ ] **DIP**: `create_app()` injects `Settings`, `HistoryStore`, `httpx.AsyncClient` — no direct env access in routes

### DRY, KISS, YAGNI

- [ ] Auth dependency (`_require_auth`) is shared between `/chat` and `/clear-history` (DRY)
- [ ] No over-engineering: rate limiter is 20 lines, not a separate library (KISS)
- [ ] Deferred features explicitly listed in Section 14.3 (YAGNI)

### Security

- [ ] Constant-time token comparison documented and justified (ADR-001)
- [ ] Replay window design covers both expired and future timestamps
- [ ] `SecretStr` usage prevents secret leakage via logging
- [ ] Docker hardening (non-root, RO FS, cap_drop) documented with compose config

### Data Models

- [ ] All Pydantic models defined with `extra="forbid"` (FR-12 AC-7)
- [ ] Upstream request shapes defined as TypedDicts (satisfies `mypy --strict`)
- [ ] `RokidRequestType` enum ensures valid type values at parse time

### SSE Streaming

- [ ] Zero-buffering relay design documented (generator + `aiter_lines`)
- [ ] History updated only after successful stream completion (finally block)
- [ ] Mid-stream error handling produces SSE error event then closes

### Testing

- [ ] All 7 modules have identified unit test boundaries
- [ ] Integration test fixtures documented (TestClient + pytest-httpx)
- [ ] Coverage targets specified per module (100% for auth, 80%+ overall)
- [ ] TDD cycle (Red-Green-Refactor) explicitly required

### Interface Contracts

- [ ] All HTTP error codes documented with body format for `/chat`
- [ ] `/clear-history` contract defined (auth, replay, idempotent clear)
- [ ] `/health` contract defined (no auth, no I/O, <100ms)

### Open Questions

- [ ] All uncertain items listed with runtime mitigations
- [ ] Deferred features explicitly scoped out with rationale

---

**Approved by**: _____________________________ **Date**: _____________
