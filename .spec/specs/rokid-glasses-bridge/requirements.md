# Requirements: rokid-glasses-bridge

**Feature**: `rokid-glasses-bridge`
**Status**: Draft — Pending Approval
**Created**: 2026-02-20
**Author**: Planner Agent (SDD Phase 2)

---

## Overview

The Rokid Bridge is a standalone FastAPI service (port 8090) that acts as a security relay and protocol adapter between Rokid AR Glasses (running the Lingzhu/Rokid AI App platform) and the `openclaw-secure-stack` backend (port 8080). It authenticates, validates, transforms, and streams all Rokid device interactions without modifying the upstream secure stack.

### Request Flow

```
Rokid Glasses → POST /rokid/chat (HMAC-signed JSON)
                    ↓
            Rokid Bridge (:8090)
            [auth → rate-limit → transform → relay]
                    ↓ POST /v1/chat/completions (stream=true)
            Secure Stack (:8080) [UNMODIFIED]
                    ↓ SSE stream
            Rokid Bridge relays SSE
                    ↓ SSE stream
            Rokid Glasses client
```

---

## Constraints

1. **Language & Runtime**: Python 3.12; no older Python versions supported.
2. **Framework**: FastAPI (async/ASGI); no Flask, Django, or synchronous frameworks.
3. **HTTP Client**: `httpx` (async) for all outbound requests; no `requests` library.
4. **Configuration**: `pydantic-settings` exclusively; no direct `os.environ` access outside the settings module.
5. **Upstream Immutability**: The `openclaw-secure-stack` (port 8080) MUST NOT be modified in any way.
6. **Type Safety**: All source code MUST pass `mypy --strict`; no untyped `Any` except at Pydantic model boundaries.
7. **Linting**: All code MUST pass `ruff check` with rules E, F, I, UP, B, SIM enabled.
8. **Containerization**: Docker image MUST be built from `python:3.12-slim` with a non-root user and read-only filesystem.
9. **In-Memory Only**: Conversation history is stored in-memory only; no database or file system persistence required for v0.1.
10. **No WebSocket**: Rokid devices use HTTP POST + SSE; WebSocket is not required.
11. **Package Manager**: `uv` for dependency management; `hatchling` as the build backend.

---

## Assumptions

1. Rokid devices always include `x-rokid-signature` and `x-rokid-timestamp` HTTP headers on every request.
2. The `ROKID_ACCESS_KEY` secret is provisioned via environment variable; it is never embedded in source code or Docker images.
3. The `openclaw-secure-stack` upstream accepts `Authorization: Bearer <token>` and returns OpenAI-compatible SSE chunks.
4. Device clocks may drift by up to 60 seconds; the replay window accounts for this.
5. A single bridge instance handles all device traffic; horizontal scaling is a future concern (v0.2+).
6. Images sent by Rokid devices are JPEG or PNG format; other formats are rejected.
7. The bridge and secure stack are co-located on the same Docker network; TLS between them is not required.
8. The Lingzhu platform sends `request_id` (UUID v4), `device_id` (opaque string), `type` (enum), `text` (string), `image` (optional object), and `timestamp` (Unix seconds integer).

---

## Functional Requirements

---

### FR-1: HMAC Authentication

**EARS Pattern**: Event-Driven

WHEN a POST request arrives at `/rokid/chat`,
THEN the system SHALL verify the `x-rokid-signature` header against an HMAC-SHA256 digest of `{timestamp}.{request_body_bytes}` using the `ROKID_ACCESS_KEY` secret.

**Acceptance Criteria**:
1. A request with a valid signature MUST be accepted (2xx response delivered downstream).
2. A request with a missing `x-rokid-signature` header MUST be rejected with HTTP 401.
3. A request with a missing `x-rokid-timestamp` header MUST be rejected with HTTP 401.
4. A request with an incorrect signature MUST be rejected with HTTP 401.
5. The comparison MUST use `hmac.compare_digest` (constant-time) to prevent timing attacks.
6. The HMAC algorithm MUST be SHA-256.
7. The signature input MUST be the exact bytes: `{timestamp_str}.{raw_request_body}` (timestamp as ASCII decimal string, dot separator, then raw body bytes).
8. The response body for auth failures MUST be `{"detail": "Unauthorized"}` with no further details.

**Testability**: Unit-testable in `tests/unit/test_rokid_auth.py` with known key/payload/signature triples. Coverage: 100%.

---

### FR-2: Replay Attack Protection

**EARS Pattern**: Unwanted Behavior

IF the `x-rokid-timestamp` value represents a Unix timestamp older than `ROKID_REPLAY_WINDOW` seconds from the server's current UTC time,
THEN the system SHALL reject the request with HTTP 401 and message `{"detail": "Request expired"}`.

IF the `x-rokid-timestamp` value represents a Unix timestamp more than 60 seconds in the future (clock skew allowance),
THEN the system SHALL reject the request with HTTP 401 and message `{"detail": "Request timestamp invalid"}`.

**Acceptance Criteria**:
1. A timestamp within ±60 seconds of server time (and within the replay window) MUST be accepted.
2. A timestamp older than `ROKID_REPLAY_WINDOW` seconds (default: 300) MUST be rejected.
3. A timestamp more than 60 seconds in the future MUST be rejected.
4. A non-integer or malformed timestamp header value MUST be rejected with HTTP 400.
5. The replay window MUST be configurable via `ROKID_REPLAY_WINDOW` environment variable.
6. Timestamp validation MUST occur after HMAC verification (to prevent oracle attacks).

**Testability**: Unit-testable via time-mocked tests in `tests/unit/test_rokid_auth.py`. Coverage: 100%.

---

### FR-3: Per-Device Rate Limiting

**EARS Pattern**: State-Driven

WHILE a device identified by `device_id` has exceeded `ROKID_RATE_LIMIT` requests within the current 60-second sliding window,
THE system SHALL reject subsequent requests from that device with HTTP 429 and body `{"detail": "Rate limit exceeded"}`.

**Acceptance Criteria**:
1. Each device's request count is tracked independently, keyed by `device_id`.
2. The sliding window duration is 60 seconds (fixed).
3. The maximum requests per window is configurable via `ROKID_RATE_LIMIT` (default: 30).
4. A device below the limit MUST NOT be affected by another device exceeding it.
5. After the sliding window expires, the device's count resets automatically.
6. Rate limiting MUST be applied after HMAC/replay validation (authenticated requests only).
7. The response MUST include a `Retry-After` header indicating seconds until the window resets.

**Testability**: Integration-testable via repeated requests in `tests/integration/test_endpoints.py`. Unit-testable with time-mocked sliding window logic.

---

### FR-4: Text Request Handling

**EARS Pattern**: Event-Driven

WHEN the validated Rokid request has `type == "text"`,
THEN the system SHALL construct an OpenAI-compatible chat completion request containing:
- the AR system prompt (see FR-8) as the first message,
- the per-device conversation history (see FR-7) as preceding messages,
- a new `{"role": "user", "content": request.text}` message appended last,
and relay it to the upstream secure stack with `stream=true`.

**Acceptance Criteria**:
1. The upstream request MUST use `POST /v1/chat/completions` with `Content-Type: application/json`.
2. The `Authorization` header sent upstream MUST be `Bearer {UPSTREAM_TOKEN}`.
3. If `ROKID_AGENT_ID` is set, the request MUST include `"agent_id": "{ROKID_AGENT_ID}"` in the body.
4. The request body MUST include `"stream": true`.
5. The user message content MUST be exactly `request.text` with no modification.
6. An empty or whitespace-only `text` field on a `type == "text"` request MUST be rejected with HTTP 422.

**Testability**: Integration-testable with mocked httpx in `tests/integration/test_endpoints.py`.

---

### FR-5: Image Request Handling

**EARS Pattern**: Event-Driven

WHEN the validated Rokid request has `type == "image"`,
THEN the system SHALL convert the `image.data` (base64 string) and `image.mime_type` into an OpenAI vision-format content array and relay it upstream with `stream=true`.

**Acceptance Criteria**:
1. The OpenAI content array MUST be of the format:
   ```json
   [{"type": "image_url", "image_url": {"url": "data:{mime_type};base64,{image_data}", "detail": "{ROKID_IMAGE_DETAIL}"}}]
   ```
2. `ROKID_IMAGE_DETAIL` controls the `detail` field; default is `"low"`.
3. If `image.mime_type` is not one of `["image/jpeg", "image/png"]`, the request MUST be rejected with HTTP 422 and message `{"detail": "Unsupported image format"}`.
4. If the decoded image bytes exceed 20 MB, the request MUST be rejected with HTTP 413 and message `{"detail": "Image too large"}`.
5. The base64 decoding MUST validate that `image.data` is valid base64; invalid base64 MUST be rejected with HTTP 422.
6. A missing or null `image` field on a `type == "image"` request MUST be rejected with HTTP 422.
7. Conversation history MUST be included in the upstream request (same as FR-4).

**Testability**: Unit-testable in `tests/unit/test_image_handler.py` for conversion logic. Integration-testable for the full flow.

---

### FR-6: Text-with-Image Request Handling

**EARS Pattern**: Event-Driven

WHEN the validated Rokid request has `type == "text_with_image"`,
THEN the system SHALL construct an OpenAI vision message combining both the text and image into a single content array and relay it upstream with `stream=true`.

**Acceptance Criteria**:
1. The content array MUST contain both a text part and an image_url part:
   ```json
   [
     {"type": "text", "text": "{request.text}"},
     {"type": "image_url", "image_url": {"url": "data:{mime_type};base64,{data}", "detail": "{ROKID_IMAGE_DETAIL}"}}
   ]
   ```
2. All image validation rules from FR-5 (MIME type, size, base64 validity) apply identically.
3. An empty or whitespace-only `text` field on a `type == "text_with_image"` request MUST be rejected with HTTP 422.
4. A missing or null `image` field on a `type == "text_with_image"` request MUST be rejected with HTTP 422.
5. Conversation history MUST be included (same as FR-4).

**Testability**: Integration-testable with combined payload. Unit-testable for content construction logic.

---

### FR-7: Per-Device Conversation History

**EARS Pattern**: Ubiquitous / State-Driven

The system SHALL maintain per-device conversation history as an ordered list of `{"role": "user"|"assistant", "content": "..."}` messages, keyed by `device_id`.

WHEN a request is successfully relayed upstream and a complete response is received,
THEN the system SHALL append the user's message and the assistant's full response to that device's history.

WHILE a device's history length exceeds `ROKID_MAX_HISTORY_TURNS` turns (one turn = one user + one assistant message pair),
THE system SHALL evict the oldest turn before appending the new one.

IF a device has had no activity for more than 1 hour (TTL),
THEN the system SHALL evict that device's history from memory.

**Acceptance Criteria**:
1. History is retrieved before each upstream request and prepended after the system prompt.
2. History MUST be stored as individual messages (not merged), preserving role alternation.
3. The maximum depth is configurable via `ROKID_MAX_HISTORY_TURNS` (default: 20 turns = 40 messages).
4. Eviction MUST remove the oldest user+assistant pair as a unit (not individual messages).
5. TTL eviction MUST occur lazily (on next access or on a background sweep); it MUST NOT block request processing.
6. History for a device MUST be cleared when the `/rokid/clear-history` endpoint is called (FR-10).
7. History MUST be isolated per `device_id`; no cross-device leakage is permitted.
8. The system MUST NOT append to history if the upstream request fails (error response).

**Testability**: Unit-testable in `tests/unit/test_history.py` for all CRUD, TTL, and truncation paths.

---

### FR-8: AR Response Formatting (System Prompt Injection)

**EARS Pattern**: Ubiquitous

The system SHALL prepend a fixed AR-optimized system prompt as the first message in every upstream chat completions request.

**Acceptance Criteria**:
1. The system prompt MUST be the first message in the `messages` array sent upstream (before history and user message).
2. The system prompt role MUST be `"system"`.
3. The system prompt content MUST instruct the model to produce concise responses suitable for a small transparent AR display, without Markdown, bullet points, or lengthy prose.
4. The system prompt MUST NOT be stored in conversation history.
5. The system prompt MUST be configurable as a constant within the codebase (not an environment variable).

**Testability**: Unit-testable by inspecting the `messages` payload sent to the upstream mock.

---

### FR-9: SSE Streaming Relay

**EARS Pattern**: Event-Driven

WHEN the upstream secure stack returns an SSE stream in response to a chat completions request,
THEN the system SHALL relay each SSE chunk to the Rokid client immediately as it is received, without buffering the entire response.

**Acceptance Criteria**:
1. The bridge response to Rokid MUST have `Content-Type: text/event-stream`.
2. Each `data: {...}` chunk received from upstream MUST be forwarded verbatim to the Rokid client.
3. The relay MUST use async streaming (not buffered reads) to minimize latency.
4. The bridge MUST close the downstream connection when the upstream stream ends (i.e., when `data: [DONE]` is received or the upstream connection closes).
5. The assistant's full response text MUST be assembled from SSE chunks and stored in history after streaming completes (see FR-7).
6. If the upstream connection drops mid-stream, the bridge MUST send an SSE error event to the client and close the connection.
7. The relay overhead (bridge processing time excluding upstream latency) MUST be less than 20 ms per request.

**Testability**: Integration-testable with a mocked streaming httpx response in `tests/integration/test_endpoints.py`.

---

### FR-10: History Clear Endpoint

**EARS Pattern**: Event-Driven

WHEN a POST request is received at `/rokid/clear-history` with a valid `device_id` in the request body,
THEN the system SHALL delete all stored conversation history for that device and return HTTP 200 with body `{"cleared": true, "device_id": "{device_id}"}`.

**Acceptance Criteria**:
1. The endpoint MUST require HMAC authentication (same FR-1 rules apply).
2. Clearing a device with no existing history MUST still return HTTP 200 (idempotent).
3. The endpoint MUST accept `{"device_id": "..."}` as a JSON body.
4. After clearing, subsequent requests from that device MUST start with an empty history.
5. Clearing device A's history MUST NOT affect device B's history.

**Testability**: Integration-testable via sequential request pairs (clear then chat).

---

### FR-11: Health Check Endpoint

**EARS Pattern**: Ubiquitous

The system SHALL expose a `/health` endpoint that returns the service's liveness status.

**Acceptance Criteria**:
1. `GET /health` MUST return HTTP 200 with body `{"status": "ok", "service": "rokid-bridge"}`.
2. The endpoint MUST NOT require authentication.
3. The endpoint MUST respond within 100 ms under any load condition.
4. The endpoint MUST NOT perform any I/O (database, upstream HTTP) — it is a pure liveness check.

**Testability**: Integration-testable; always-green unless the process is dead.

---

### FR-12: Invalid Payload Rejection

**EARS Pattern**: Unwanted Behavior

IF the request body does not conform to the expected Rokid JSON schema (missing required fields, wrong types, unrecognized `type` value),
THEN the system SHALL return HTTP 422 with a structured validation error body.

**Acceptance Criteria**:
1. A missing `request_id` field MUST produce HTTP 422.
2. A missing `device_id` field MUST produce HTTP 422.
3. A `type` value not in `["text", "image", "text_with_image"]` MUST produce HTTP 422.
4. A non-string `text` field (when required) MUST produce HTTP 422.
5. A non-integer `timestamp` field MUST produce HTTP 400 (caught before Pydantic validation, in auth layer).
6. The error body MUST follow FastAPI's default validation error format: `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}`.
7. Pydantic models MUST use `model_config = ConfigDict(extra="forbid")` to reject unknown fields.

**Testability**: Unit-testable via Pydantic model parsing tests; integration-testable via malformed payloads.

---

## Non-Functional Requirements

---

### NFR-1: Security — Constant-Time Authentication

**EARS Pattern**: Ubiquitous

The system SHALL use `hmac.compare_digest` for all HMAC signature comparisons to prevent timing-based side-channel attacks.

**Acceptance Criteria**:
1. No string equality operator (`==`) is used anywhere in the authentication path for comparing secrets or signatures.
2. Static analysis (ruff/mypy) MUST NOT detect any `==` comparison involving the HMAC digest.
3. 100% test coverage on `rokid_auth.py` MUST be maintained.

---

### NFR-2: Security — Secret Isolation

**EARS Pattern**: Ubiquitous

The system SHALL ensure that secrets (`ROKID_ACCESS_KEY`, `UPSTREAM_TOKEN`) are never written to logs, error responses, or SSE output.

**Acceptance Criteria**:
1. No log statement at any level (DEBUG through CRITICAL) emits the value of any secret.
2. `pydantic-settings` model MUST mark secret fields with `SecretStr` type.
3. Error responses to clients MUST NOT include environment variable names or values.

---

### NFR-3: Security — Docker Hardening

**EARS Pattern**: Ubiquitous

The system SHALL run inside a Docker container as a non-root user with a read-only root filesystem.

**Acceptance Criteria**:
1. The Dockerfile MUST create and switch to a non-root user (e.g., `appuser` with UID 1000) before the `CMD` instruction.
2. The `docker-compose.yml` MUST include `read_only: true` for the bridge service.
3. Any writable temporary directories required at runtime (e.g., `/tmp`) MUST be explicitly declared as `tmpfs` volumes.
4. The Docker image MUST NOT include development dependencies (`[dev]` extras) in the final stage.

---

### NFR-4: Performance — Relay Latency

**EARS Pattern**: Ubiquitous

The system SHALL add less than 20 ms of overhead per request, excluding upstream processing time.

**Acceptance Criteria**:
1. HMAC computation, replay check, rate-limit lookup, history retrieval, and message construction combined MUST complete in under 20 ms.
2. SSE chunk relay MUST not introduce per-chunk buffering delays.
3. History operations (read, append, evict) MUST be O(n) or better where n is `ROKID_MAX_HISTORY_TURNS`.

---

### NFR-5: Reliability — Upstream Error Propagation

**EARS Pattern**: Unwanted Behavior

IF the upstream secure stack returns a non-2xx HTTP status code,
THEN the system SHALL return the upstream status code and error body to the Rokid client without modification.

**Acceptance Criteria**:
1. A 429 from upstream MUST be forwarded as 429 to the Rokid client.
2. A 500 from upstream MUST be forwarded as 500 to the Rokid client.
3. A connection timeout to the upstream MUST result in HTTP 504 (Gateway Timeout) to the Rokid client.
4. A connection refused to the upstream MUST result in HTTP 502 (Bad Gateway) to the Rokid client.
5. History MUST NOT be updated on any upstream error.

---

### NFR-6: Reliability — Graceful Shutdown

**EARS Pattern**: Event-Driven

WHEN the bridge process receives SIGTERM or SIGINT,
THEN the system SHALL complete all in-flight SSE streams before terminating (graceful shutdown).

**Acceptance Criteria**:
1. Uvicorn MUST be started with graceful shutdown support (default behavior).
2. In-flight SSE connections MUST be drained before the process exits.
3. New connections MUST be rejected after the shutdown signal is received.

---

### NFR-7: Observability — Structured Logging

**EARS Pattern**: Ubiquitous

The system SHALL emit structured log lines for every inbound request and outbound upstream call.

**Acceptance Criteria**:
1. Each request log MUST include: `device_id`, `request_id`, `type`, timestamp, and response HTTP status.
2. Authentication failures MUST be logged at WARN level with reason (but without the signature value or secret key).
3. Upstream errors MUST be logged at ERROR level with status code and response body (truncated to 500 chars).
4. Log format MUST be structured JSON or a consistent key=value format.
5. Log output MUST go to stdout (not file) for Docker compatibility.

---

### NFR-8: Code Quality

**EARS Pattern**: Ubiquitous

The system SHALL maintain a minimum of 80% test coverage across all modules, with 100% coverage on `rokid_auth.py` and authentication code paths in `app.py`.

**Acceptance Criteria**:
1. `pytest --cov=rokid_bridge --cov-report=term-missing` MUST report >= 80% overall coverage.
2. `rokid_auth.py` MUST report 100% line and branch coverage.
3. `mypy --strict src/` MUST exit with code 0 (no type errors).
4. `ruff check src/ tests/` MUST exit with code 0 (no lint violations).

---

## Summary Table

| ID | Area | Pattern | Priority |
|----|------|---------|----------|
| FR-1 | HMAC Authentication | Event-Driven | Critical |
| FR-2 | Replay Protection | Unwanted | Critical |
| FR-3 | Rate Limiting | State-Driven | High |
| FR-4 | Text Request Handling | Event-Driven | Critical |
| FR-5 | Image Request Handling | Event-Driven | High |
| FR-6 | Text+Image Request Handling | Event-Driven | High |
| FR-7 | Conversation History | Ubiquitous/State-Driven | High |
| FR-8 | AR System Prompt | Ubiquitous | High |
| FR-9 | SSE Streaming Relay | Event-Driven | Critical |
| FR-10 | History Clear Endpoint | Event-Driven | Medium |
| FR-11 | Health Check | Ubiquitous | Medium |
| FR-12 | Invalid Payload Rejection | Unwanted | High |
| NFR-1 | Constant-Time Auth | Ubiquitous | Critical |
| NFR-2 | Secret Isolation | Ubiquitous | Critical |
| NFR-3 | Docker Hardening | Ubiquitous | High |
| NFR-4 | Relay Latency < 20ms | Ubiquitous | High |
| NFR-5 | Upstream Error Propagation | Unwanted | High |
| NFR-6 | Graceful Shutdown | Event-Driven | Medium |
| NFR-7 | Structured Logging | Ubiquitous | Medium |
| NFR-8 | Code Quality Gates | Ubiquitous | High |

---

## Approval Checklist

- [ ] All requirements use valid EARS format (Ubiquitous / Event-Driven / State-Driven / Unwanted)
- [ ] Each requirement is independently testable
- [ ] Acceptance criteria are specific and measurable
- [ ] All 13 architecture areas from the feature brief are covered
- [ ] Constraints and assumptions are documented
- [ ] Non-functional requirements address security, performance, and reliability
- [ ] No requirement depends on a future phase or unresolved assumption
- [ ] Approved by: _____________________________ on: _____________
