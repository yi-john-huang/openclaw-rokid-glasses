# TDD Task Breakdown: rokid-glasses-bridge

**Feature**: `rokid-glasses-bridge`
**Status**: Approved
**Date**: 2026-02-20
**Author**: TDD Guide Agent (SDD Phase 4)
**Design Reference**: `.spec/specs/rokid-glasses-bridge/design.md`
**Requirements Reference**: `.spec/specs/rokid-glasses-bridge/requirements.md`

---

## Implementation Order Overview

Tasks are ordered so that each task's dependencies are satisfied before it begins. Because all domain modules are pure (no framework coupling), tests can be written against interfaces immediately using mocks for any missing dependency.

```
T-01 (scaffold)
  └─► T-02 (config)
        └─► T-03 (models)
              ├─► T-04 (auth verify) ──► T-05 (auth replay)
              ├─► T-06 (image handler)
              ├─► T-07 (history)
              └─► T-08 (relay build) ──► T-09 (extract text) ──► T-10 (stream)
                    └─► T-11 (rate limiter)
                          └─► T-12 (app scaffold + health)
                                ├─► T-13 (chat text integration)
                                ├─► T-14 (chat image integration)
                                ├─► T-15 (auth integration)
                                ├─► T-16 (rate limit integration)
                                ├─► T-17 (clear history)
                                └─► T-18 (upstream errors)
                                      └─► T-19 (Docker)
                                            └─► T-20 (smoke test)
```

---

## Tasks

---

### T-01: Project Scaffolding

**Module**: `pyproject.toml`, `src/rokid_bridge/__init__.py`, directory structure
**Test File**: N/A (verified by install command)
**Complexity**: Low
**Depends On**: none
**Satisfies**: Constraint 1, 2, 3, 4, 11 (language, framework, HTTP client, config, package manager)

#### Red Phase — Write These Failing Tests First

No automated test for scaffolding itself. Verification is via install and import:

```python
# Verification (run manually, not a pytest test):
# uv venv && uv pip install -e ".[dev]"
# python -c "import rokid_bridge; print('ok')"
```

The first real test in T-02 will fail with `ModuleNotFoundError` if this task is incomplete — that IS the red phase for the scaffold.

#### Green Phase — Minimum Implementation

Create the following structure:

```
openclaw-rokid-glasses/
├── pyproject.toml
├── src/
│   └── rokid_bridge/
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   └── __init__.py
│   └── integration/
│       └── __init__.py
└── .env.example
```

`pyproject.toml` must include:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "rokid-bridge"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.32",
    "httpx>=0.28",
    "pydantic-settings>=2.6",
    "pydantic>=2.10",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-httpx>=0.35",
    "ruff>=0.8",
    "mypy>=1.13",
    "pytest-cov>=6.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/rokid_bridge"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
strict = true
python_version = "3.12"

[tool.ruff]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

`src/rokid_bridge/__init__.py` is empty (zero bytes).

#### Refactor Checklist
- [ ] SOLID principle verified: SRP — each file has one purpose; `__init__.py` is intentionally empty
- [ ] No duplication: single `pyproject.toml` is the single source of dependency truth
- [ ] Directory layout matches `src/` layout convention from tech.md
- [ ] `.env.example` documents all env vars from `Settings` (added in T-02)

---

### T-02: Settings and Configuration

**Module**: `src/rokid_bridge/config.py`
**Test File**: `tests/unit/test_config.py`
**Complexity**: Low
**Depends On**: T-01
**Satisfies**: Constraint 4, NFR-2, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_config.py
import pytest
from pydantic import SecretStr
from pydantic_core import ValidationError


# Test 1: Required fields raise ValidationError when missing
def test_settings_missing_rokid_access_key_raises(monkeypatch):
    monkeypatch.delenv("ROKID_ACCESS_KEY", raising=False)
    monkeypatch.delenv("UPSTREAM_TOKEN", raising=False)
    from rokid_bridge.config import Settings
    with pytest.raises(ValidationError):
        Settings()


# Test 2: Required fields are accepted when provided
def test_settings_valid_required_fields(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "my-secret-key")
    monkeypatch.setenv("UPSTREAM_TOKEN", "my-upstream-token")
    from rokid_bridge.config import Settings
    s = Settings()
    assert s.rokid_access_key.get_secret_value() == "my-secret-key"
    assert s.upstream_token.get_secret_value() == "my-upstream-token"


# Test 3: SecretStr masks value in str() representation
def test_settings_secret_str_masked(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "supersecret")
    monkeypatch.setenv("UPSTREAM_TOKEN", "anothersecret")
    from rokid_bridge.config import Settings
    s = Settings()
    assert "supersecret" not in str(s.rokid_access_key)
    assert "supersecret" not in repr(s.rokid_access_key)


# Test 4: Default values are applied correctly
def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    from rokid_bridge.config import Settings
    s = Settings()
    assert s.upstream_url == "http://localhost:8080"
    assert s.rokid_agent_id == ""
    assert s.rokid_rate_limit == 30
    assert s.rokid_replay_window == 300
    assert s.rokid_max_history_turns == 20
    assert s.rokid_image_detail == "low"
    assert s.port == 8090


# Test 5: Zero rate limit is rejected (ge=1 constraint)
def test_settings_zero_rate_limit_rejected(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    monkeypatch.setenv("ROKID_RATE_LIMIT", "0")
    from rokid_bridge.config import Settings
    with pytest.raises(ValidationError):
        Settings()


# Test 6: get_settings returns a Settings instance
def test_get_settings_returns_settings(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    from rokid_bridge.config import Settings, get_settings
    result = get_settings()
    assert isinstance(result, Settings)


# Test 7: Custom upstream URL is accepted
def test_settings_custom_upstream_url(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    monkeypatch.setenv("UPSTREAM_URL", "http://custom-stack:9090")
    from rokid_bridge.config import Settings
    s = Settings()
    assert s.upstream_url == "http://custom-stack:9090"
```

#### Green Phase — Minimum Implementation

Implement `src/rokid_bridge/config.py` exactly as specified in design Section 2.1:
- `Settings(BaseSettings)` class with all eight fields
- `SecretStr` on `rokid_access_key` and `upstream_token`
- `ge=1` validators on `rokid_rate_limit`, `rokid_replay_window`, `rokid_max_history_turns`
- `get_settings()` function (no `@lru_cache` — test overrides depend on fresh instantiation)
- `SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)`

#### Refactor Checklist
- [ ] SOLID principle verified: DIP — `get_settings()` returns an instance; callers do not import `os.environ`
- [ ] No duplication: all env var defaults are defined once in `Field(default=...)`
- [ ] Type annotations complete: `mypy --strict` passes on `config.py`
- [ ] `ruff check` passes

---

### T-03: Pydantic Data Models

**Module**: `src/rokid_bridge/models.py`
**Test File**: `tests/unit/test_models.py`
**Complexity**: Low
**Depends On**: T-01
**Satisfies**: FR-4, FR-5, FR-6, FR-12, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_models.py
import pytest
from pydantic import ValidationError


# Test 1: Valid text request is accepted
def test_rokid_chat_request_valid_text():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    req = RokidChatRequest(
        request_id="550e8400-e29b-41d4-a716-446655440000",
        device_id="device-abc",
        type=RokidRequestType.TEXT,
        text="Hello",
        timestamp=1708300000,
    )
    assert req.device_id == "device-abc"
    assert req.type == RokidRequestType.TEXT


# Test 2: Unknown fields are rejected (extra="forbid")
def test_rokid_chat_request_extra_field_rejected():
    from rokid_bridge.models import RokidRequestType
    with pytest.raises(ValidationError):
        from rokid_bridge.models import RokidChatRequest
        RokidChatRequest(
            request_id="id",
            device_id="dev",
            type=RokidRequestType.TEXT,
            text="hi",
            timestamp=1708300000,
            unknown_field="bad",
        )


# Test 3: Invalid enum value for type is rejected
def test_rokid_chat_request_invalid_type_rejected():
    from rokid_bridge.models import RokidChatRequest
    with pytest.raises(ValidationError):
        RokidChatRequest(
            request_id="id",
            device_id="dev",
            type="voice",  # invalid
            text="hi",
            timestamp=1708300000,
        )


# Test 4: Missing required field request_id raises ValidationError
def test_rokid_chat_request_missing_request_id():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    with pytest.raises(ValidationError):
        RokidChatRequest(
            device_id="dev",
            type=RokidRequestType.TEXT,
            text="hi",
            timestamp=1708300000,
        )


# Test 5: RokidImagePayload extra field rejected
def test_rokid_image_payload_extra_field_rejected():
    from rokid_bridge.models import RokidImagePayload
    with pytest.raises(ValidationError):
        RokidImagePayload(data="abc", mime_type="image/jpeg", extra="bad")


# Test 6: Valid RokidClearRequest is accepted
def test_rokid_clear_request_valid():
    from rokid_bridge.models import RokidClearRequest
    req = RokidClearRequest(device_id="dev", timestamp=1708300000)
    assert req.device_id == "dev"


# Test 7: HealthResponse has correct literal values
def test_health_response_literals():
    from rokid_bridge.models import HealthResponse
    h = HealthResponse()
    assert h.status == "ok"
    assert h.service == "rokid-bridge"


# Test 8: ClearHistoryResponse fields are set correctly
def test_clear_history_response_fields():
    from rokid_bridge.models import ClearHistoryResponse
    r = ClearHistoryResponse(cleared=True, device_id="dev-123")
    assert r.cleared is True
    assert r.device_id == "dev-123"


# Test 9: text defaults to empty string when omitted
def test_rokid_chat_request_text_defaults_empty():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    req = RokidChatRequest(
        request_id="id",
        device_id="dev",
        type=RokidRequestType.IMAGE,
        timestamp=1708300000,
    )
    assert req.text == ""


# Test 10: image defaults to None when omitted
def test_rokid_chat_request_image_defaults_none():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    req = RokidChatRequest(
        request_id="id",
        device_id="dev",
        type=RokidRequestType.TEXT,
        text="hi",
        timestamp=1708300000,
    )
    assert req.image is None
```

#### Green Phase — Minimum Implementation

Implement `src/rokid_bridge/models.py` exactly as specified in design Section 2.2:
- `RokidRequestType(str, Enum)` with TEXT, IMAGE, TEXT_WITH_IMAGE values
- `RokidImagePayload(BaseModel)` with `extra="forbid"`, `data` and `mime_type` fields
- `RokidChatRequest(BaseModel)` with `extra="forbid"` and all six fields
- `RokidClearRequest(BaseModel)` with `extra="forbid"`, `device_id`, `timestamp`
- `HealthResponse(BaseModel)` with `Literal["ok"]` and `Literal["rokid-bridge"]`
- `ClearHistoryResponse(BaseModel)` with `cleared: bool` and `device_id: str`
- All TypedDicts: `TextContentPart`, `ImageUrlDetail`, `ImageUrlWrapper`, `ImageContentPart`, `ContentPart`, `OpenAIMessage`, `UpstreamRequest`

#### Refactor Checklist
- [ ] SOLID principle verified: ISP — each model is focused on one endpoint's concern; no shared "god model"
- [ ] No duplication: `extra="forbid"` applies consistently via `ConfigDict`
- [ ] Type annotations complete: `mypy --strict` passes on `models.py`
- [ ] `ruff check` passes; enum members use UPPER_SNAKE_CASE

---

### T-04: Bearer Token Authentication

**Module**: `src/rokid_bridge/rokid_auth.py` — `verify_bearer_token` function
**Test File**: `tests/unit/test_rokid_auth.py`
**Complexity**: Low
**Depends On**: T-02
**Satisfies**: FR-1, NFR-1, NFR-8 (100% coverage target)

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_rokid_auth.py
import pytest
from pydantic import SecretStr


def _make_settings(key: str = "correct-token"):
    """Helper: create a Settings-like object with a known access key."""
    from rokid_bridge.config import Settings
    return Settings(
        rokid_access_key=SecretStr(key),
        upstream_token=SecretStr("upstream"),
    )


# Test 1: Valid Bearer token is accepted (no exception raised)
def test_verify_bearer_token_valid():
    from rokid_bridge.rokid_auth import verify_bearer_token
    settings = _make_settings("correct-token")
    # Should not raise
    verify_bearer_token("Bearer correct-token", settings)


# Test 2: Missing Authorization header raises AuthError
def test_verify_bearer_token_missing_header():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    settings = _make_settings()
    with pytest.raises(AuthError) as exc_info:
        verify_bearer_token(None, settings)
    assert exc_info.value.status_code == 401


# Test 3: Wrong scheme (not Bearer) raises AuthError
def test_verify_bearer_token_wrong_scheme():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    settings = _make_settings("correct-token")
    with pytest.raises(AuthError) as exc_info:
        verify_bearer_token("Basic correct-token", settings)
    assert exc_info.value.status_code == 401


# Test 4: Wrong token value raises AuthError
def test_verify_bearer_token_wrong_token():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    settings = _make_settings("correct-token")
    with pytest.raises(AuthError) as exc_info:
        verify_bearer_token("Bearer wrong-token", settings)
    assert exc_info.value.status_code == 401


# Test 5: Empty token after "Bearer " raises AuthError
def test_verify_bearer_token_empty_token():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    settings = _make_settings("correct-token")
    with pytest.raises(AuthError):
        verify_bearer_token("Bearer ", settings)


# Test 6: Error detail is generic "Unauthorized" (no info leak)
def test_verify_bearer_token_error_detail_generic():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    settings = _make_settings("correct-token")
    with pytest.raises(AuthError) as exc_info:
        verify_bearer_token("Bearer wrong", settings)
    assert exc_info.value.detail == "Unauthorized"


# Test 7: AuthError is raised (not raw ValueError or other exception)
def test_verify_bearer_token_raises_auth_error_type():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    settings = _make_settings()
    with pytest.raises(AuthError):
        verify_bearer_token(None, settings)


# Test 8: Token comparison uses hmac.compare_digest (constant-time)
# We verify indirectly: a token that is a prefix of the correct token still fails
def test_verify_bearer_token_prefix_attack_fails():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    settings = _make_settings("correct-token-long")
    with pytest.raises(AuthError):
        verify_bearer_token("Bearer correct-token", settings)
```

#### Green Phase — Minimum Implementation

Implement `src/rokid_bridge/rokid_auth.py` — the `AuthError` class and `verify_bearer_token` function:

```python
class AuthError(Exception):
    def __init__(self, detail: str, status_code: int = 401) -> None: ...

def verify_bearer_token(
    authorization_header: str | None,
    settings: Settings,
) -> None:
    # 1. Return AuthError if header is None or empty
    # 2. Split on first space: scheme, _, token
    # 3. Validate scheme.lower() == "bearer" and token non-empty
    # 4. hmac.compare_digest(expected.encode(), token.encode())
    # 5. Raise AuthError("Unauthorized") on any failure
```

#### Refactor Checklist
- [ ] SOLID principle verified: SRP — function only authenticates; no rate limiting, logging, or routing
- [ ] No duplication: single `AuthError` class used for all failure modes
- [ ] 100% line and branch coverage confirmed via `pytest --cov`
- [ ] No `==` operator used for secret comparison (ruff/manual check)
- [ ] `mypy --strict` passes

---

### T-05: Replay Window Protection

**Module**: `src/rokid_bridge/rokid_auth.py` — `check_replay_window` function
**Test File**: `tests/unit/test_rokid_auth.py` (continued)
**Complexity**: Low
**Depends On**: T-04
**Satisfies**: FR-2, NFR-1, NFR-8 (100% coverage target)

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_rokid_auth.py (continued)
from pydantic import SecretStr


def _make_settings_replay(window: int = 300):
    from rokid_bridge.config import Settings
    return Settings(
        rokid_access_key=SecretStr("k"),
        upstream_token=SecretStr("t"),
        rokid_replay_window=window,
    )


# Test 9: Timestamp within window is accepted (no exception)
def test_check_replay_window_valid():
    from rokid_bridge.rokid_auth import check_replay_window
    settings = _make_settings_replay(window=300)
    now = 1708300000.0
    # timestamp is 100 seconds old — well within 300s window
    check_replay_window(int(now) - 100, settings, now=now)


# Test 10: Timestamp older than window raises AuthError "Request expired"
def test_check_replay_window_expired():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    settings = _make_settings_replay(window=300)
    now = 1708300000.0
    with pytest.raises(AuthError) as exc_info:
        check_replay_window(int(now) - 301, settings, now=now)
    assert exc_info.value.detail == "Request expired"


# Test 11: Timestamp more than 60s in the future raises AuthError
def test_check_replay_window_future_timestamp():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    settings = _make_settings_replay()
    now = 1708300000.0
    with pytest.raises(AuthError) as exc_info:
        check_replay_window(int(now) + 61, settings, now=now)
    assert exc_info.value.detail == "Request timestamp invalid"


# Test 12: Timestamp exactly at window boundary is rejected
def test_check_replay_window_exactly_at_boundary_expired():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    settings = _make_settings_replay(window=300)
    now = 1708300000.0
    # age == 301 > 300, so expired
    with pytest.raises(AuthError):
        check_replay_window(int(now) - 301, settings, now=now)


# Test 13: Timestamp exactly within window (age == window) is rejected (> not >=)
def test_check_replay_window_at_window_limit():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    settings = _make_settings_replay(window=300)
    now = 1708300000.0
    # age == 300 is NOT > 300, so accepted
    check_replay_window(int(now) - 300, settings, now=now)  # should not raise


# Test 14: Timestamp exactly 60s in future is accepted (boundary)
def test_check_replay_window_exactly_60s_future_accepted():
    from rokid_bridge.rokid_auth import check_replay_window
    settings = _make_settings_replay()
    now = 1708300000.0
    # age == -60, not < -60, so accepted
    check_replay_window(int(now) + 60, settings, now=now)  # should not raise


# Test 15: now= injection works (deterministic without monkeypatching time.time)
def test_check_replay_window_now_injection():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    settings = _make_settings_replay(window=10)
    # Use a fixed now value — no dependency on real clock
    check_replay_window(1000000, settings, now=1000005.0)  # 5s old, within 10s window
    with pytest.raises(AuthError):
        check_replay_window(1000000, settings, now=1000015.0)  # 15s old, outside 10s window
```

#### Green Phase — Minimum Implementation

Add `check_replay_window` to `src/rokid_bridge/rokid_auth.py`:

```python
def check_replay_window(
    body_timestamp: int,
    settings: Settings,
    *,
    now: float | None = None,
) -> None:
    current_time = now if now is not None else time.time()
    age = current_time - body_timestamp
    if age > settings.rokid_replay_window:
        raise AuthError("Request expired")
    if age < -60:
        raise AuthError("Request timestamp invalid")
```

#### Refactor Checklist
- [ ] SOLID principle verified: OCP — `check_replay_window` and `verify_bearer_token` are separate, composable functions; callers chain them without modifying this module
- [ ] No duplication: `now=` injection pattern used for testability without global monkey-patching
- [ ] 100% line and branch coverage on all four branches (valid, expired, future, boundary)
- [ ] `mypy --strict` passes

---

### T-06: Image Handler

**Module**: `src/rokid_bridge/image_handler.py`
**Test File**: `tests/unit/test_image_handler.py`
**Complexity**: Medium
**Depends On**: T-03
**Satisfies**: FR-5, FR-6, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_image_handler.py
import base64
import pytest


def _make_jpeg_payload(size_bytes: int = 100) -> str:
    """Return base64-encoded fake JPEG data of given decoded byte size."""
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * (size_bytes - 3)).decode()


def _make_png_payload(size_bytes: int = 100) -> str:
    """Return base64-encoded fake PNG data."""
    return base64.b64encode(b"\x89PNG" + b"\x00" * (size_bytes - 4)).decode()


# Test 1: Valid JPEG image is accepted and returns ImageContentPart
def test_validate_and_build_jpeg_valid():
    from rokid_bridge.image_handler import validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    payload = RokidImagePayload(data=_make_jpeg_payload(), mime_type="image/jpeg")
    result = validate_and_build_image_part(payload, "low")
    assert result["type"] == "image_url"
    assert result["image_url"]["detail"] == "low"
    assert result["image_url"]["url"].startswith("data:image/jpeg;base64,")


# Test 2: Valid PNG image is accepted
def test_validate_and_build_png_valid():
    from rokid_bridge.image_handler import validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    payload = RokidImagePayload(data=_make_png_payload(), mime_type="image/png")
    result = validate_and_build_image_part(payload, "high")
    assert result["image_url"]["detail"] == "high"
    assert result["image_url"]["url"].startswith("data:image/png;base64,")


# Test 3: Unsupported MIME type raises ImageValidationError with 422
def test_validate_unsupported_mime_raises():
    from rokid_bridge.image_handler import ImageValidationError, validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    payload = RokidImagePayload(data=_make_jpeg_payload(), mime_type="image/gif")
    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_build_image_part(payload, "low")
    assert exc_info.value.status_code == 422
    assert "Unsupported" in exc_info.value.detail


# Test 4: Invalid base64 data raises ImageValidationError with 422
def test_validate_invalid_base64_raises():
    from rokid_bridge.image_handler import ImageValidationError, validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    payload = RokidImagePayload(data="not-valid-base64!!!", mime_type="image/jpeg")
    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_build_image_part(payload, "low")
    assert exc_info.value.status_code == 422


# Test 5: Exactly 20 MB decoded image is accepted (boundary)
def test_validate_exactly_20mb_accepted():
    from rokid_bridge.image_handler import MAX_IMAGE_BYTES, validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    data = base64.b64encode(b"\x00" * MAX_IMAGE_BYTES).decode()
    payload = RokidImagePayload(data=data, mime_type="image/jpeg")
    # Should not raise
    validate_and_build_image_part(payload, "low")


# Test 6: Image exceeding 20 MB raises ImageValidationError with 413
def test_validate_oversized_image_raises_413():
    from rokid_bridge.image_handler import MAX_IMAGE_BYTES, ImageValidationError, validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    data = base64.b64encode(b"\x00" * (MAX_IMAGE_BYTES + 1)).decode()
    payload = RokidImagePayload(data=data, mime_type="image/jpeg")
    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_build_image_part(payload, "low")
    assert exc_info.value.status_code == 413


# Test 7: build_image_content returns a list with one ImageContentPart
def test_build_image_content_shape():
    from rokid_bridge.image_handler import build_image_content
    from rokid_bridge.models import RokidImagePayload
    payload = RokidImagePayload(data=_make_jpeg_payload(), mime_type="image/jpeg")
    result = build_image_content(payload, "low")
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["type"] == "image_url"


# Test 8: build_text_with_image_content returns list [TextPart, ImagePart]
def test_build_text_with_image_content_shape():
    from rokid_bridge.image_handler import build_text_with_image_content
    from rokid_bridge.models import RokidImagePayload
    payload = RokidImagePayload(data=_make_jpeg_payload(), mime_type="image/jpeg")
    result = build_text_with_image_content("What is this?", payload, "low")
    assert len(result) == 2
    assert result[0]["type"] == "text"
    assert result[0]["text"] == "What is this?"
    assert result[1]["type"] == "image_url"


# Test 9: webp MIME type is rejected (not in ALLOWED_MIME_TYPES)
def test_validate_webp_rejected():
    from rokid_bridge.image_handler import ImageValidationError, validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    payload = RokidImagePayload(data=_make_jpeg_payload(), mime_type="image/webp")
    with pytest.raises(ImageValidationError):
        validate_and_build_image_part(payload, "low")
```

#### Green Phase — Minimum Implementation

Implement `src/rokid_bridge/image_handler.py` exactly as specified in design Section 2.4:
- `ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png"})`
- `MAX_IMAGE_BYTES: int = 20 * 1024 * 1024`
- `ImageValidationError(ValueError)` with `detail` and `status_code` attributes
- `validate_and_build_image_part(image, detail) -> ImageContentPart`
- `build_image_content(image, detail) -> list[ContentPart]`
- `build_text_with_image_content(text, image, detail) -> list[ContentPart]`

Use `base64.b64decode(image.data, validate=True)` for strict validation.

#### Refactor Checklist
- [ ] SOLID principle verified: OCP — `ALLOWED_MIME_TYPES` is a `frozenset`; adding `image/webp` is a one-line extension
- [ ] No duplication: `validate_and_build_image_part` is called by both `build_image_content` and `build_text_with_image_content`
- [ ] Size check uses decoded byte length (not base64 string length)
- [ ] Data URL uses original `image.data` string (no re-encode)
- [ ] `mypy --strict` passes; `ruff check` passes

---

### T-07: Conversation History Store

**Module**: `src/rokid_bridge/history.py`
**Test File**: `tests/unit/test_history.py`
**Complexity**: Medium
**Depends On**: T-03
**Satisfies**: FR-7, FR-10, NFR-4, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_history.py
import time
import pytest


# Test 1: get_messages returns empty list for unknown device
def test_get_messages_empty_for_unknown_device():
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=5)
    result = store.get_messages("device-unknown")
    assert result == []


# Test 2: append_turn stores user+assistant messages
def test_append_turn_stores_messages():
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=5)
    store.append_turn("device-1", "Hello", "Hi there")
    messages = store.get_messages("device-1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hi there"


# Test 3: get_messages after multiple appends preserves order
def test_get_messages_preserves_order():
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=10)
    store.append_turn("dev", "Q1", "A1")
    store.append_turn("dev", "Q2", "A2")
    messages = store.get_messages("dev")
    assert messages[0]["content"] == "Q1"
    assert messages[1]["content"] == "A1"
    assert messages[2]["content"] == "Q2"
    assert messages[3]["content"] == "A2"


# Test 4: Truncation evicts oldest turn pair when max_turns exceeded
def test_append_turn_truncates_at_max_turns():
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=2)
    store.append_turn("dev", "Q1", "A1")
    store.append_turn("dev", "Q2", "A2")
    store.append_turn("dev", "Q3", "A3")  # should evict Q1/A1
    messages = store.get_messages("dev")
    assert len(messages) == 4  # 2 turns * 2 messages each
    assert messages[0]["content"] == "Q2"
    assert messages[-1]["content"] == "A3"


# Test 5: TTL eviction clears session after expiry (monkeypatch time.time)
def test_ttl_eviction_on_access(monkeypatch):
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=5, ttl_seconds=10)
    store.append_turn("dev", "Q1", "A1")

    # Advance time past TTL
    original_time = time.time
    monkeypatch.setattr(time, "time", lambda: original_time() + 20)

    # Next access should evict the session
    result = store.get_messages("dev")
    assert result == []


# Test 6: clear removes history for device (idempotent)
def test_clear_removes_history():
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=5)
    store.append_turn("dev", "Q1", "A1")
    store.clear("dev")
    assert store.get_messages("dev") == []


# Test 7: clear on non-existent device does not raise
def test_clear_nonexistent_device_no_error():
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=5)
    store.clear("never-existed")  # should not raise


# Test 8: History is isolated per device_id
def test_history_isolated_per_device():
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=5)
    store.append_turn("device-A", "QA", "AA")
    store.append_turn("device-B", "QB", "AB")
    assert store.get_messages("device-A")[0]["content"] == "QA"
    assert store.get_messages("device-B")[0]["content"] == "QB"
    assert len(store.get_messages("device-A")) == 2
    assert len(store.get_messages("device-B")) == 2


# Test 9: Session key uses "rokid:" prefix
def test_session_key_prefix():
    from rokid_bridge.history import HistoryStore, SESSION_KEY_PREFIX
    assert SESSION_KEY_PREFIX == "rokid"
    store = HistoryStore(max_turns=5)
    store.append_turn("my-device", "Q", "A")
    # Internal key should be "rokid:my-device"
    assert "rokid:my-device" in store._sessions


# Test 10: deque popleft is used (O(1) eviction verified by type)
def test_history_uses_deque():
    from collections import deque
    from rokid_bridge.history import HistoryStore
    store = HistoryStore(max_turns=5)
    store.append_turn("dev", "Q", "A")
    session = store._sessions["rokid:dev"]
    assert isinstance(session.messages, deque)
```

#### Green Phase — Minimum Implementation

Implement `src/rokid_bridge/history.py` exactly as specified in design Section 2.5:
- `SESSION_KEY_PREFIX: str = "rokid"`
- `HISTORY_TTL_SECONDS: int = 3600`
- `_Session` dataclass with `messages: Deque[OpenAIMessage]` and `last_access: float`
- `HistoryStore` class with `__init__(max_turns, ttl_seconds)`, `get_messages`, `append_turn`, `clear`, `_make_key`, `_evict_if_expired`
- Use `while` loop (not `if`) in `append_turn` for eviction
- `clear` uses `dict.pop(key, None)` for idempotency

#### Refactor Checklist
- [ ] SOLID principle verified: SRP — `HistoryStore` is the only class managing session lifecycle
- [ ] No duplication: `_make_key` and `_evict_if_expired` are extracted helpers
- [ ] `deque` used for O(1) `popleft` eviction (verified in test 10)
- [ ] `while` loop (not `if`) handles runtime `max_turns` reduction
- [ ] `mypy --strict` passes; `ruff check` passes

---

### T-08: Relay — build_upstream_request

**Module**: `src/rokid_bridge/relay.py` — `build_upstream_request` and `AR_SYSTEM_PROMPT`
**Test File**: `tests/unit/test_relay.py`
**Complexity**: Low
**Depends On**: T-02, T-03
**Satisfies**: FR-4, FR-5, FR-6, FR-8, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_relay.py
import pytest
from pydantic import SecretStr


def _make_settings(agent_id: str = ""):
    from rokid_bridge.config import Settings
    return Settings(
        rokid_access_key=SecretStr("k"),
        upstream_token=SecretStr("t"),
        rokid_agent_id=agent_id,
    )


# Test 1: System prompt is always the first message
def test_build_upstream_system_prompt_first():
    from rokid_bridge.relay import AR_SYSTEM_PROMPT, build_upstream_request
    settings = _make_settings()
    result = build_upstream_request(history=[], user_content="Hello", settings=settings)
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == AR_SYSTEM_PROMPT


# Test 2: History messages appear between system prompt and user message
def test_build_upstream_history_in_middle():
    from rokid_bridge.models import OpenAIMessage
    from rokid_bridge.relay import build_upstream_request
    settings = _make_settings()
    history = [
        OpenAIMessage(role="user", content="prev Q"),
        OpenAIMessage(role="assistant", content="prev A"),
    ]
    result = build_upstream_request(history=history, user_content="New Q", settings=settings)
    messages = result["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == "prev Q"
    assert messages[2]["content"] == "prev A"
    assert messages[3]["role"] == "user"
    assert messages[3]["content"] == "New Q"


# Test 3: User message is always the last message
def test_build_upstream_user_message_last():
    from rokid_bridge.relay import build_upstream_request
    settings = _make_settings()
    result = build_upstream_request(history=[], user_content="Final Q", settings=settings)
    assert result["messages"][-1]["role"] == "user"
    assert result["messages"][-1]["content"] == "Final Q"


# Test 4: agent_id is included when ROKID_AGENT_ID is set
def test_build_upstream_agent_id_included_when_set():
    from rokid_bridge.relay import build_upstream_request
    settings = _make_settings(agent_id="my-agent-123")
    result = build_upstream_request(history=[], user_content="Q", settings=settings)
    assert result.get("agent_id") == "my-agent-123"


# Test 5: agent_id is omitted when ROKID_AGENT_ID is empty string
def test_build_upstream_agent_id_omitted_when_empty():
    from rokid_bridge.relay import build_upstream_request
    settings = _make_settings(agent_id="")
    result = build_upstream_request(history=[], user_content="Q", settings=settings)
    assert "agent_id" not in result


# Test 6: stream is always True in the upstream request
def test_build_upstream_stream_is_true():
    from rokid_bridge.relay import build_upstream_request
    settings = _make_settings()
    result = build_upstream_request(history=[], user_content="Q", settings=settings)
    assert result["stream"] is True


# Test 7: list content (image) is accepted as user_content
def test_build_upstream_list_content_accepted():
    from rokid_bridge.models import TextContentPart
    from rokid_bridge.relay import build_upstream_request
    settings = _make_settings()
    content: list = [TextContentPart(type="text", text="What is this?")]
    result = build_upstream_request(history=[], user_content=content, settings=settings)
    last_msg = result["messages"][-1]
    assert isinstance(last_msg["content"], list)


# Test 8: AR_SYSTEM_PROMPT contains AR-specific instructions
def test_ar_system_prompt_content():
    from rokid_bridge.relay import AR_SYSTEM_PROMPT
    prompt_lower = AR_SYSTEM_PROMPT.lower()
    assert "concise" in prompt_lower or "short" in prompt_lower
    assert "markdown" in prompt_lower
```

#### Green Phase — Minimum Implementation

Implement the first portion of `src/rokid_bridge/relay.py`:
- `AR_SYSTEM_PROMPT: str` constant (module-level, not an env var)
- `UPSTREAM_PATH: str = "/v1/chat/completions"`
- `UPSTREAM_TIMEOUT_SECONDS: float = 30.0`
- `UpstreamError(Exception)` with `status_code` and `body` attributes
- `build_upstream_request(*, history, user_content, settings) -> UpstreamRequest`

The function assembles `[system_message] + history + [user_message]` and conditionally adds `agent_id`.

#### Refactor Checklist
- [ ] SOLID principle verified: OCP — `build_upstream_request` is a pure function; adding a new message type requires no modification
- [ ] No duplication: `AR_SYSTEM_PROMPT` is defined once at module level
- [ ] `AR_SYSTEM_PROMPT` co-located with its consumer `build_upstream_request` (locality of reference, SRP)
- [ ] `mypy --strict` passes; `ruff check` passes

---

### T-09: Relay — extract_assistant_text

**Module**: `src/rokid_bridge/relay.py` — `extract_assistant_text` function
**Test File**: `tests/unit/test_relay.py` (continued)
**Complexity**: Low
**Depends On**: T-08
**Satisfies**: FR-7 (AC-5 — assemble assistant text from SSE chunks), FR-9, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_relay.py (continued)

# Test 9: Single data chunk is extracted
def test_extract_assistant_text_single_chunk():
    from rokid_bridge.relay import extract_assistant_text
    chunks = ['data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}']
    assert extract_assistant_text(chunks) == "Hello"


# Test 10: Multiple chunks are concatenated in order
def test_extract_assistant_text_multiple_chunks():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        'data: {"choices":[{"delta":{"content":"He"},"index":0}]}',
        'data: {"choices":[{"delta":{"content":"llo"},"index":0}]}',
    ]
    assert extract_assistant_text(chunks) == "Hello"


# Test 11: [DONE] sentinel terminates extraction
def test_extract_assistant_text_done_terminates():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"should-be-ignored"},"index":0}]}',
    ]
    assert extract_assistant_text(chunks) == "Hi"


# Test 12: Non-data lines (empty lines, comment lines) are ignored
def test_extract_assistant_text_non_data_lines_ignored():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        "",
        ": keep-alive",
        'data: {"choices":[{"delta":{"content":"World"},"index":0}]}',
    ]
    assert extract_assistant_text(chunks) == "World"


# Test 13: Malformed JSON in data line is skipped gracefully
def test_extract_assistant_text_malformed_json_skipped():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        "data: {not-valid-json",
        'data: {"choices":[{"delta":{"content":"OK"},"index":0}]}',
    ]
    assert extract_assistant_text(chunks) == "OK"


# Test 14: Empty chunk list returns empty string
def test_extract_assistant_text_empty_chunks():
    from rokid_bridge.relay import extract_assistant_text
    assert extract_assistant_text([]) == ""


# Test 15: Chunk with empty content string contributes nothing
def test_extract_assistant_text_empty_content_skipped():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        'data: {"choices":[{"delta":{"content":""},"index":0}]}',
        'data: {"choices":[{"delta":{"content":"A"},"index":0}]}',
    ]
    assert extract_assistant_text(chunks) == "A"
```

#### Green Phase — Minimum Implementation

Add `extract_assistant_text(sse_chunks: list[str]) -> str` to `relay.py`:

```python
def extract_assistant_text(sse_chunks: list[str]) -> str:
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

#### Refactor Checklist
- [ ] SOLID principle verified: SRP — pure function, no side effects, no I/O
- [ ] No duplication: single parsing loop with clear responsibilities
- [ ] All edge cases covered: empty list, [DONE], malformed JSON, non-data lines
- [ ] `mypy --strict` passes

---

### T-10: Relay — stream_upstream

**Module**: `src/rokid_bridge/relay.py` — `stream_upstream` async generator
**Test File**: `tests/unit/test_relay.py` (continued)
**Complexity**: High
**Depends On**: T-08, T-09
**Satisfies**: FR-9, NFR-4, NFR-5, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_relay.py (continued)
import pytest
import httpx
from pytest_httpx import HTTPXMock
from pydantic import SecretStr


def _make_relay_settings():
    from rokid_bridge.config import Settings
    return Settings(
        rokid_access_key=SecretStr("k"),
        upstream_token=SecretStr("upstream-token"),
        upstream_url="http://mock-upstream",
    )


# Test 16: Successful SSE stream yields all lines
@pytest.mark.asyncio
async def test_stream_upstream_yields_lines(httpx_mock: HTTPXMock):
    from rokid_bridge.relay import stream_upstream
    from rokid_bridge.models import UpstreamRequest
    settings = _make_relay_settings()
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n'
        "data: [DONE]\n"
    )
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=sse_body.encode(),
        status_code=200,
    )
    async with httpx.AsyncClient() as client:
        upstream_req: UpstreamRequest = {"messages": [], "stream": True}
        lines = [line async for line in stream_upstream(client, upstream_req, settings)]
    assert any("Hi" in line for line in lines)
    assert any("[DONE]" in line for line in lines)


# Test 17: Non-2xx upstream response raises UpstreamError
@pytest.mark.asyncio
async def test_stream_upstream_non_2xx_raises(httpx_mock: HTTPXMock):
    from rokid_bridge.relay import UpstreamError, stream_upstream
    from rokid_bridge.models import UpstreamRequest
    settings = _make_relay_settings()
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        status_code=500,
        content=b"Internal Server Error",
    )
    async with httpx.AsyncClient() as client:
        upstream_req: UpstreamRequest = {"messages": [], "stream": True}
        with pytest.raises(UpstreamError) as exc_info:
            async for _ in stream_upstream(client, upstream_req, settings):
                pass
    assert exc_info.value.status_code == 500


# Test 18: Upstream 429 raises UpstreamError with status_code 429
@pytest.mark.asyncio
async def test_stream_upstream_429_raises(httpx_mock: HTTPXMock):
    from rokid_bridge.relay import UpstreamError, stream_upstream
    from rokid_bridge.models import UpstreamRequest
    settings = _make_relay_settings()
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        status_code=429,
        content=b"Rate limited",
    )
    async with httpx.AsyncClient() as client:
        upstream_req: UpstreamRequest = {"messages": [], "stream": True}
        with pytest.raises(UpstreamError) as exc_info:
            async for _ in stream_upstream(client, upstream_req, settings):
                pass
    assert exc_info.value.status_code == 429


# Test 19: Timeout raises httpx.TimeoutException
@pytest.mark.asyncio
async def test_stream_upstream_timeout_raises(httpx_mock: HTTPXMock):
    from rokid_bridge.relay import stream_upstream
    from rokid_bridge.models import UpstreamRequest
    settings = _make_relay_settings()
    httpx_mock.add_exception(
        httpx.ReadTimeout("upstream timed out"),
        url="http://mock-upstream/v1/chat/completions",
    )
    async with httpx.AsyncClient() as client:
        upstream_req: UpstreamRequest = {"messages": [], "stream": True}
        with pytest.raises(httpx.TimeoutException):
            async for _ in stream_upstream(client, upstream_req, settings):
                pass


# Test 20: Authorization header sent to upstream uses upstream_token
@pytest.mark.asyncio
async def test_stream_upstream_sends_auth_header(httpx_mock: HTTPXMock):
    from rokid_bridge.relay import stream_upstream
    from rokid_bridge.models import UpstreamRequest
    settings = _make_relay_settings()
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=b"data: [DONE]\n",
        status_code=200,
    )
    async with httpx.AsyncClient() as client:
        upstream_req: UpstreamRequest = {"messages": [], "stream": True}
        async for _ in stream_upstream(client, upstream_req, settings):
            pass
    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer upstream-token"
```

#### Green Phase — Minimum Implementation

Add `stream_upstream` async generator to `relay.py`:

```python
async def stream_upstream(
    client: httpx.AsyncClient,
    upstream_request: UpstreamRequest,
    settings: Settings,
) -> AsyncGenerator[str, None]:
    url = f"{settings.upstream_url}{UPSTREAM_PATH}"
    headers = {
        "Authorization": f"Bearer {settings.upstream_token.get_secret_value()}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    async with client.stream("POST", url, json=upstream_request,
                              headers=headers, timeout=UPSTREAM_TIMEOUT_SECONDS) as response:
        if response.status_code >= 300:
            body = await response.aread()
            raise UpstreamError(response.status_code, body.decode("utf-8", errors="replace"))
        async for line in response.aiter_lines():
            yield line
```

#### Refactor Checklist
- [ ] SOLID principle verified: SRP — `stream_upstream` only handles network I/O; no history logic here
- [ ] No duplication: `UPSTREAM_PATH` and `UPSTREAM_TIMEOUT_SECONDS` are module-level constants
- [ ] `httpx.AsyncClient` is injected (not created inside the function) — enables connection pooling
- [ ] `mypy --strict` passes — `AsyncGenerator[str, None]` annotation required

---

### T-11: Rate Limiter

**Module**: `src/rokid_bridge/app.py` — `RateLimiter` class
**Test File**: `tests/unit/test_rate_limiter.py`
**Complexity**: Medium
**Depends On**: T-01
**Satisfies**: FR-3, NFR-4, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/unit/test_rate_limiter.py
import time
import pytest


# Test 1: First request is always allowed
def test_rate_limiter_first_request_allowed():
    from rokid_bridge.app import RateLimiter
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    allowed, retry_after = limiter.check_and_record("device-1")
    assert allowed is True
    assert retry_after == 0.0


# Test 2: Requests below limit are allowed
def test_rate_limiter_below_limit_allowed():
    from rokid_bridge.app import RateLimiter
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        allowed, _ = limiter.check_and_record("device-1")
        assert allowed is True


# Test 3: Request at limit is rejected
def test_rate_limiter_at_limit_rejected():
    from rokid_bridge.app import RateLimiter
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        limiter.check_and_record("device-1")
    allowed, retry_after = limiter.check_and_record("device-1")
    assert allowed is False
    assert retry_after > 0


# Test 4: Window slides — old entries expire
def test_rate_limiter_window_slides(monkeypatch):
    from rokid_bridge.app import RateLimiter
    limiter = RateLimiter(max_requests=2, window_seconds=10)
    # Record 2 requests at time=0
    current_time = [0.0]
    monkeypatch.setattr(time, "time", lambda: current_time[0])
    limiter.check_and_record("dev")
    limiter.check_and_record("dev")
    # At time=0, limit is hit
    allowed, _ = limiter.check_and_record("dev")
    assert allowed is False
    # Advance time past the window
    current_time[0] = 11.0
    allowed, _ = limiter.check_and_record("dev")
    assert allowed is True


# Test 5: Retry-After is positive when rejected
def test_rate_limiter_retry_after_positive():
    from rokid_bridge.app import RateLimiter
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.check_and_record("dev")
    allowed, retry_after = limiter.check_and_record("dev")
    assert allowed is False
    assert 0 < retry_after <= 60


# Test 6: Different devices are tracked independently
def test_rate_limiter_devices_independent():
    from rokid_bridge.app import RateLimiter
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    limiter.check_and_record("device-A")
    limiter.check_and_record("device-A")  # device-A is now over limit

    # device-B is unaffected
    allowed, _ = limiter.check_and_record("device-B")
    assert allowed is True


# Test 7: retry_after is 0.0 for allowed requests
def test_rate_limiter_retry_after_zero_when_allowed():
    from rokid_bridge.app import RateLimiter
    limiter = RateLimiter(max_requests=5, window_seconds=60)
    _, retry_after = limiter.check_and_record("dev")
    assert retry_after == 0.0
```

#### Green Phase — Minimum Implementation

Add `RateLimiter` class to `src/rokid_bridge/app.py`:

```python
class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = RATE_WINDOW_SECONDS) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = {}

    def check_and_record(self, device_id: str) -> tuple[bool, float]:
        now = time.time()
        if device_id not in self._buckets:
            self._buckets[device_id] = deque()
        bucket = self._buckets[device_id]
        cutoff = now - self._window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()
        if len(bucket) >= self._max:
            retry_after = self._window - (now - bucket[0])
            return False, max(retry_after, 0.0)
        bucket.append(now)
        return True, 0.0
```

Also add `RATE_WINDOW_SECONDS: int = 60` module constant.

#### Refactor Checklist
- [ ] SOLID principle verified: SRP — `RateLimiter` only manages rate windows; no auth or routing
- [ ] No duplication: single `_buckets` dict; single eviction loop
- [ ] O(1) amortised: each timestamp is appended once and popped once
- [ ] `mypy --strict` passes; deque type annotation is `dict[str, deque[float]]`

---

### T-12: FastAPI App — Scaffolding and Health Endpoint

**Module**: `src/rokid_bridge/app.py` — `create_app()` factory and `GET /health`
**Test File**: `tests/integration/test_endpoints.py`
**Complexity**: Medium
**Depends On**: T-02, T-03, T-07, T-11
**Satisfies**: FR-11, NFR-6, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/integration/test_endpoints.py
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr


@pytest.fixture
def test_settings():
    from rokid_bridge.config import Settings
    return Settings(
        rokid_access_key=SecretStr("test-ak-12345"),
        upstream_token=SecretStr("test-upstream-token"),
        upstream_url="http://mock-upstream",
    )


@pytest.fixture
def client(test_settings):
    from rokid_bridge.app import create_app
    app = create_app(settings=test_settings)
    return TestClient(app)


# Test 1: GET /health returns 200 with correct body
def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "rokid-bridge"


# Test 2: GET /health does not require Authorization header
def test_health_no_auth_required(client):
    response = client.get("/health")
    assert response.status_code == 200


# Test 3: create_app returns a FastAPI instance
def test_create_app_returns_fastapi(test_settings):
    from fastapi import FastAPI
    from rokid_bridge.app import create_app
    app = create_app(settings=test_settings)
    assert isinstance(app, FastAPI)


# Test 4: create_app with no settings loads from environment (smoke)
def test_create_app_accepts_none_settings(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    from rokid_bridge.app import create_app
    app = create_app(settings=None)
    assert app is not None
```

#### Green Phase — Minimum Implementation

Implement `src/rokid_bridge/app.py`:
- All imports from design Section 2.7
- `RATE_WINDOW_SECONDS: int = 60` constant
- `RateLimiter` class (from T-11)
- `create_app(settings: Settings | None = None) -> FastAPI` factory:
  - Instantiates `_settings`, `_history`, `_rate_limiter`, `_http_client`
  - Registers `@app.on_event("shutdown")` to close httpx client
  - Defines `_require_auth(request: Request) -> None` as inner dependency
  - Registers `GET /health` route returning `HealthResponse()`
  - Registers `POST /chat` and `POST /clear-history` as stubs (returning 501 for now)

#### Refactor Checklist
- [ ] SOLID principle verified: DIP — `create_app()` injects all dependencies; no module-level app object
- [ ] No duplication: `_require_auth` is defined once inside `create_app()` and reused via `Depends`
- [ ] Factory pattern verified: calling `create_app()` twice creates independent instances
- [ ] `mypy --strict` passes; `ruff check` passes

---

### T-13: FastAPI App — /chat Text Flow (Integration)

**Module**: `src/rokid_bridge/app.py` — `/chat` route (text type)
**Test File**: `tests/integration/test_endpoints.py` (continued)
**Complexity**: High
**Depends On**: T-04, T-05, T-08, T-09, T-10, T-12
**Satisfies**: FR-4, FR-7, FR-8, FR-9, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/integration/test_endpoints.py (continued)
import time
from pytest_httpx import HTTPXMock


def _valid_chat_body(text: str = "Hello", type_: str = "text") -> dict:
    return {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "device_id": "test-device",
        "type": type_,
        "text": text,
        "timestamp": int(time.time()),
    }


def _auth_headers() -> dict:
    return {"Authorization": "Bearer test-ak-12345"}


# Test 5: Valid text request returns 200 with text/event-stream
def test_chat_text_returns_event_stream(client, httpx_mock: HTTPXMock):
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hi"},"index":0}]}\n'
        "data: [DONE]\n"
    )
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=sse_body.encode(),
        status_code=200,
    )
    response = client.post(
        "/chat",
        json=_valid_chat_body(),
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


# Test 6: SSE chunks are forwarded in the response body
def test_chat_text_sse_chunks_forwarded(client, httpx_mock: HTTPXMock):
    sse_body = (
        'data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}\n'
        "data: [DONE]\n"
    )
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=sse_body.encode(),
        status_code=200,
    )
    response = client.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
    assert "Hello" in response.text


# Test 7: History is updated after successful text stream
def test_chat_text_history_updated_after_stream(test_settings, httpx_mock: HTTPXMock):
    from rokid_bridge.app import create_app
    app = create_app(settings=test_settings)
    c = TestClient(app)
    sse_body = (
        'data: {"choices":[{"delta":{"content":"World"},"index":0}]}\n'
        "data: [DONE]\n"
    )
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=sse_body.encode(),
        status_code=200,
    )
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=sse_body.encode(),
        status_code=200,
    )
    # First request
    c.post("/chat", json=_valid_chat_body("Q1"), headers=_auth_headers())
    # Second request — upstream should receive history
    c.post("/chat", json=_valid_chat_body("Q2"), headers=_auth_headers())
    second_request = httpx_mock.get_requests()[1]
    import json
    body = json.loads(second_request.content)
    # System prompt + Q1 + A1 + Q2 = 4 messages
    assert len(body["messages"]) == 4
```

#### Green Phase — Minimum Implementation

Implement the full `/chat` route in `create_app()`:
1. Rate limit check via `_rate_limiter.check_and_record(body.device_id)`
2. Replay check via `check_replay_window(body.timestamp, _settings)`
3. Type-specific payload validation for `text` type (reject blank text)
4. History retrieval via `_history.get_messages(body.device_id)`
5. Build upstream request via `build_upstream_request`
6. Return `StreamingResponse(_sse_generator(...), media_type="text/event-stream")`
7. `_sse_generator` accumulates chunks and calls `history.append_turn` on success

Include the `_sse_generator` inner async generator function from design Section 5.2.

#### Refactor Checklist
- [ ] SOLID principle verified: pipeline order matches design Section 9.1 exactly
- [ ] No duplication: `_sse_generator` handles all streaming; route handler delegates
- [ ] `StreamingResponse` includes `Cache-Control: no-cache` and `X-Accel-Buffering: no`
- [ ] History updated only on `success = True` (not on error)
- [ ] `mypy --strict` passes

---

### T-14: FastAPI App — /chat Image and text_with_image Flows (Integration)

**Module**: `src/rokid_bridge/app.py` — `/chat` route (image and text_with_image types)
**Test File**: `tests/integration/test_endpoints.py` (continued)
**Complexity**: Medium
**Depends On**: T-06, T-13
**Satisfies**: FR-5, FR-6, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/integration/test_endpoints.py (continued)
import base64


def _make_jpeg_b64(size_bytes: int = 100) -> str:
    return base64.b64encode(b"\xff\xd8\xff" + b"\x00" * (size_bytes - 3)).decode()


def _valid_image_body(type_: str = "image", text: str = "") -> dict:
    return {
        "request_id": "550e8400-e29b-41d4-a716-446655440001",
        "device_id": "test-device",
        "type": type_,
        "text": text,
        "image": {"data": _make_jpeg_b64(), "mime_type": "image/jpeg"},
        "timestamp": int(time.time()),
    }


def _sse_response() -> bytes:
    return (
        b'data: {"choices":[{"delta":{"content":"OK"},"index":0}]}\n'
        b"data: [DONE]\n"
    )


# Test 8: image type request returns 200
def test_chat_image_returns_200(client, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=_sse_response(),
        status_code=200,
    )
    response = client.post(
        "/chat",
        json=_valid_image_body("image"),
        headers=_auth_headers(),
    )
    assert response.status_code == 200


# Test 9: text_with_image type request returns 200
def test_chat_text_with_image_returns_200(client, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=_sse_response(),
        status_code=200,
    )
    response = client.post(
        "/chat",
        json=_valid_image_body("text_with_image", text="What is this?"),
        headers=_auth_headers(),
    )
    assert response.status_code == 200


# Test 10: Blank text on text type returns 422
def test_chat_blank_text_on_text_type_returns_422(client):
    response = client.post(
        "/chat",
        json=_valid_chat_body(text="   ", type_="text"),
        headers=_auth_headers(),
    )
    assert response.status_code == 422


# Test 11: Missing image on image type returns 422
def test_chat_missing_image_on_image_type_returns_422(client):
    body = _valid_chat_body(type_="image")
    body.pop("image", None)
    response = client.post("/chat", json=body, headers=_auth_headers())
    assert response.status_code == 422


# Test 12: Invalid base64 returns 422
def test_chat_invalid_base64_returns_422(client):
    body = _valid_image_body("image")
    body["image"]["data"] = "not-valid-base64!!!"
    response = client.post("/chat", json=body, headers=_auth_headers())
    assert response.status_code == 422


# Test 13: Oversized image returns 413
def test_chat_oversized_image_returns_413(client):
    from rokid_bridge.image_handler import MAX_IMAGE_BYTES
    body = _valid_image_body("image")
    body["image"]["data"] = base64.b64encode(b"\x00" * (MAX_IMAGE_BYTES + 1)).decode()
    response = client.post("/chat", json=body, headers=_auth_headers())
    assert response.status_code == 413
```

#### Green Phase — Minimum Implementation

Add type-dispatch logic inside the `/chat` route for image and text_with_image cases:
- `RokidRequestType.TEXT`: reject if `body.text.strip() == ""`
- `RokidRequestType.IMAGE`: reject if `body.image is None`; call `build_image_content`; use `"[image request]"` as `user_text_for_history`
- `RokidRequestType.TEXT_WITH_IMAGE`: reject if blank text or None image; call `build_text_with_image_content`; use `body.text` as `user_text_for_history`
- Catch `ImageValidationError` and raise `HTTPException(status_code=exc.status_code, detail=exc.detail)`

#### Refactor Checklist
- [ ] SOLID principle verified: SRP — route delegates to `image_handler`; does not reimplement validation
- [ ] No duplication: `ImageValidationError` caught once and mapped to `HTTPException`
- [ ] `user_text_for_history` never contains base64 image data
- [ ] `mypy --strict` passes

---

### T-15: FastAPI App — Auth and Replay Integration Tests

**Module**: `src/rokid_bridge/app.py` — `_require_auth` dependency and replay wiring
**Test File**: `tests/integration/test_endpoints.py` (continued)
**Complexity**: Low
**Depends On**: T-04, T-05, T-13
**Satisfies**: FR-1, FR-2, NFR-1, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/integration/test_endpoints.py (continued)

# Test 14: Missing Authorization header returns 401
def test_chat_missing_auth_returns_401(client):
    response = client.post("/chat", json=_valid_chat_body())
    assert response.status_code == 401


# Test 15: Wrong Bearer token returns 401
def test_chat_wrong_token_returns_401(client):
    response = client.post(
        "/chat",
        json=_valid_chat_body(),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"


# Test 16: Expired timestamp returns 401
def test_chat_expired_timestamp_returns_401(client):
    body = _valid_chat_body()
    body["timestamp"] = int(time.time()) - 400  # older than default 300s window
    response = client.post("/chat", json=body, headers=_auth_headers())
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


# Test 17: Future timestamp (> 60s) returns 401
def test_chat_future_timestamp_returns_401(client):
    body = _valid_chat_body()
    body["timestamp"] = int(time.time()) + 120  # 120s in the future
    response = client.post("/chat", json=body, headers=_auth_headers())
    assert response.status_code == 401


# Test 18: Auth error detail does not leak the actual token
def test_chat_auth_error_does_not_leak_token(client):
    response = client.post(
        "/chat",
        json=_valid_chat_body(),
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert "test-ak-12345" not in response.text
    assert "wrong-token" not in response.text
```

#### Green Phase — Minimum Implementation

Auth code is already implemented in T-04 and T-05. This task wires them into routes:
- `_require_auth` inner dependency in `create_app()` calls `verify_bearer_token`
- In the `/chat` route handler, call `check_replay_window(body.timestamp, _settings)` after rate limit check and auth dependency
- Catch `AuthError` from replay check and raise `HTTPException(401, detail=exc.detail)`

#### Refactor Checklist
- [ ] SOLID principle verified: DIP — auth logic lives in `rokid_auth.py`; route handler only wires it
- [ ] No duplication: `_require_auth` dependency is defined once and reused by both routes
- [ ] Error detail strings match interface contract in design Section 11.1
- [ ] `mypy --strict` passes

---

### T-16: FastAPI App — Rate Limiting Integration Tests

**Module**: `src/rokid_bridge/app.py` — `RateLimiter` wired into `/chat` route
**Test File**: `tests/integration/test_endpoints.py` (continued)
**Complexity**: Medium
**Depends On**: T-11, T-15
**Satisfies**: FR-3, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/integration/test_endpoints.py (continued)

def _make_low_limit_client(test_settings):
    from rokid_bridge.config import Settings
    from rokid_bridge.app import create_app
    from pydantic import SecretStr
    low_limit_settings = Settings(
        rokid_access_key=SecretStr("test-ak-12345"),
        upstream_token=SecretStr("test-upstream-token"),
        upstream_url="http://mock-upstream",
        rokid_rate_limit=2,
    )
    return TestClient(create_app(settings=low_limit_settings))


# Test 19: First N requests within limit are allowed
def test_rate_limit_first_requests_allowed(httpx_mock: HTTPXMock, test_settings):
    sse_body = b'data: {"choices":[{"delta":{"content":"OK"},"index":0}]}\ndata: [DONE]\n'
    httpx_mock.add_response(url="http://mock-upstream/v1/chat/completions", content=sse_body, status_code=200)
    httpx_mock.add_response(url="http://mock-upstream/v1/chat/completions", content=sse_body, status_code=200)
    c = _make_low_limit_client(test_settings)
    for _ in range(2):
        response = c.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
        assert response.status_code == 200


# Test 20: (N+1)th request returns 429 with Retry-After header
def test_rate_limit_exceeded_returns_429(httpx_mock: HTTPXMock, test_settings):
    sse_body = b'data: {"choices":[{"delta":{"content":"OK"},"index":0}]}\ndata: [DONE]\n'
    httpx_mock.add_response(url="http://mock-upstream/v1/chat/completions", content=sse_body, status_code=200)
    httpx_mock.add_response(url="http://mock-upstream/v1/chat/completions", content=sse_body, status_code=200)
    c = _make_low_limit_client(test_settings)
    for _ in range(2):
        c.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
    response = c.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert response.json()["detail"] == "Rate limit exceeded"


# Test 21: Different device_id is not rate limited by first device
def test_rate_limit_different_device_not_affected(httpx_mock: HTTPXMock, test_settings):
    sse_body = b'data: {"choices":[{"delta":{"content":"OK"},"index":0}]}\ndata: [DONE]\n'
    httpx_mock.add_response(url="http://mock-upstream/v1/chat/completions", content=sse_body, status_code=200)
    c = _make_low_limit_client(test_settings)
    body_a = _valid_chat_body()
    body_a["device_id"] = "device-A"
    body_b = _valid_chat_body()
    body_b["device_id"] = "device-B"
    for _ in range(3):
        c.post("/chat", json=body_a, headers=_auth_headers())
    response = c.post("/chat", json=body_b, headers=_auth_headers())
    assert response.status_code == 200
```

#### Green Phase — Minimum Implementation

Wire rate limiter into the `/chat` route (already instantiated in T-12):

```python
# In /chat route handler, before replay check:
allowed, retry_after = _rate_limiter.check_and_record(body.device_id)
if not allowed:
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded",
        headers={"Retry-After": str(int(retry_after))},
    )
```

#### Refactor Checklist
- [ ] SOLID principle verified: rate limit check is Step 2 in pipeline (after JSON parse, before auth replay)
- [ ] No duplication: `_rate_limiter` is instantiated once in `create_app()`
- [ ] `Retry-After` header is an integer string (RFC 7231 compliance)
- [ ] `mypy --strict` passes

---

### T-17: FastAPI App — /clear-history Endpoint

**Module**: `src/rokid_bridge/app.py` — `POST /clear-history` route
**Test File**: `tests/integration/test_endpoints.py` (continued)
**Complexity**: Low
**Depends On**: T-07, T-15
**Satisfies**: FR-10, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/integration/test_endpoints.py (continued)

def _clear_body(device_id: str = "test-device") -> dict:
    return {"device_id": device_id, "timestamp": int(time.time())}


# Test 22: Valid clear request returns 200 with cleared=true
def test_clear_history_returns_200(client):
    response = client.post(
        "/clear-history",
        json=_clear_body(),
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["cleared"] is True
    assert data["device_id"] == "test-device"


# Test 23: Auth is required for /clear-history
def test_clear_history_requires_auth(client):
    response = client.post("/clear-history", json=_clear_body())
    assert response.status_code == 401


# Test 24: Replay check is enforced for /clear-history
def test_clear_history_expired_timestamp_returns_401(client):
    body = _clear_body()
    body["timestamp"] = int(time.time()) - 400
    response = client.post("/clear-history", json=body, headers=_auth_headers())
    assert response.status_code == 401


# Test 25: Clearing non-existent device returns 200 (idempotent)
def test_clear_history_nonexistent_device_returns_200(client):
    response = client.post(
        "/clear-history",
        json=_clear_body(device_id="never-existed"),
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    assert response.json()["cleared"] is True


# Test 26: After clearing, subsequent chat starts with empty history
def test_clear_then_chat_starts_fresh(test_settings, httpx_mock: HTTPXMock):
    from rokid_bridge.app import create_app
    app = create_app(settings=test_settings)
    c = TestClient(app)
    sse_body = b'data: {"choices":[{"delta":{"content":"A1"},"index":0}]}\ndata: [DONE]\n'
    httpx_mock.add_response(url="http://mock-upstream/v1/chat/completions", content=sse_body, status_code=200)
    httpx_mock.add_response(url="http://mock-upstream/v1/chat/completions", content=sse_body, status_code=200)

    # First chat — builds history
    c.post("/chat", json=_valid_chat_body("Q1"), headers=_auth_headers())
    # Clear history
    c.post("/clear-history", json=_clear_body(), headers=_auth_headers())
    # Second chat — should have no history
    c.post("/chat", json=_valid_chat_body("Q2"), headers=_auth_headers())
    second_request = httpx_mock.get_requests()[1]  # index 1 = second chat
    import json
    body = json.loads(second_request.content)
    # Only system prompt + Q2 = 2 messages (no history from Q1)
    assert len(body["messages"]) == 2
```

#### Green Phase — Minimum Implementation

Implement the `/clear-history` route in `create_app()`:

```python
@app.post("/clear-history", dependencies=[Depends(_require_auth)])
async def clear_history(body: RokidClearRequest) -> ClearHistoryResponse:
    try:
        check_replay_window(body.timestamp, _settings)
    except AuthError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    _history.clear(body.device_id)
    return ClearHistoryResponse(cleared=True, device_id=body.device_id)
```

#### Refactor Checklist
- [ ] SOLID principle verified: DRY — same `_require_auth` dependency reused from `/chat`
- [ ] No duplication: `check_replay_window` called identically in both routes
- [ ] `clear()` is idempotent (verified in test 25)
- [ ] `mypy --strict` passes

---

### T-18: FastAPI App — Upstream Error Handling (Integration)

**Module**: `src/rokid_bridge/app.py` — error handling in `_sse_generator` and route handler
**Test File**: `tests/integration/test_endpoints.py` (continued)
**Complexity**: Medium
**Depends On**: T-10, T-13
**Satisfies**: NFR-5, NFR-8

#### Red Phase — Write These Failing Tests First

```python
# tests/integration/test_endpoints.py (continued)

# Test 27: Upstream 500 returns 500 to client
def test_upstream_500_forwarded(client, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        status_code=500,
        content=b"Internal Server Error",
    )
    response = client.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
    assert response.status_code == 500


# Test 28: Upstream 429 returns 429 to client
def test_upstream_429_forwarded(client, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        status_code=429,
        content=b"Too Many Requests",
    )
    response = client.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
    assert response.status_code == 429


# Test 29: Upstream timeout returns 504
def test_upstream_timeout_returns_504(client, httpx_mock: HTTPXMock):
    httpx_mock.add_exception(
        httpx.ReadTimeout("timeout"),
        url="http://mock-upstream/v1/chat/completions",
    )
    response = client.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
    assert response.status_code == 504


# Test 30: Upstream connection refused returns 502
def test_upstream_connection_refused_returns_502(client, httpx_mock: HTTPXMock):
    httpx_mock.add_exception(
        httpx.ConnectError("connection refused"),
        url="http://mock-upstream/v1/chat/completions",
    )
    response = client.post("/chat", json=_valid_chat_body(), headers=_auth_headers())
    assert response.status_code == 502


# Test 31: History is NOT updated after upstream error
def test_upstream_error_history_not_updated(test_settings, httpx_mock: HTTPXMock):
    from rokid_bridge.app import create_app
    app = create_app(settings=test_settings)
    c = TestClient(app)
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        status_code=500,
        content=b"Error",
    )
    httpx_mock.add_response(
        url="http://mock-upstream/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"A"},"index":0}]}\ndata: [DONE]\n',
        status_code=200,
    )
    c.post("/chat", json=_valid_chat_body("Q1"), headers=_auth_headers())
    c.post("/chat", json=_valid_chat_body("Q2"), headers=_auth_headers())
    second_request = httpx_mock.get_requests()[1]
    import json
    body = json.loads(second_request.content)
    # Only system + Q2 (no history from failed Q1 request)
    assert len(body["messages"]) == 2
```

#### Green Phase — Minimum Implementation

Add error handling around the `stream_upstream` call in the `/chat` route:
- Catch `UpstreamError` before starting streaming: `raise HTTPException(status_code=exc.status_code, detail=...)`
- Catch `httpx.TimeoutException`: `raise HTTPException(status_code=504, detail="Upstream timeout")`
- Catch `httpx.ConnectError`: `raise HTTPException(status_code=502, detail="Upstream unavailable")`
- Inside `_sse_generator`, keep `success = False` flag; set to `True` only after full stream completes

#### Refactor Checklist
- [ ] SOLID principle verified: NFR-5 — all four upstream error types mapped correctly
- [ ] No duplication: error-to-HTTP mapping is consistent between route and generator
- [ ] `success` flag ensures history is only updated on complete success
- [ ] `mypy --strict` passes

---

### T-19: Dockerfile and docker-compose

**Module**: `Dockerfile`, `docker-compose.yml`, `docker-compose.override.yml`, `.env.example`
**Test File**: N/A (manual verification steps)
**Complexity**: Medium
**Depends On**: T-18
**Satisfies**: NFR-3, Constraint 8

#### Red Phase — Write These Failing Tests First

No automated pytest tests for Docker infrastructure. Verification is via manual build and run commands:

```bash
# Verification step 1: Build succeeds
docker build -t rokid-bridge .

# Verification step 2: Container runs as non-root (UID 1000)
docker run --rm rokid-bridge id
# Expected: uid=1000(appuser) gid=1000(appgroup)

# Verification step 3: Health check endpoint responds
docker run --rm -e ROKID_ACCESS_KEY=test -e UPSTREAM_TOKEN=test \
  -p 8090:8090 rokid-bridge &
sleep 2
curl -f http://localhost:8090/health
# Expected: {"status":"ok","service":"rokid-bridge"}

# Verification step 4: docker-compose with read_only works
docker compose up -d
docker compose exec rokid-bridge touch /test-write 2>&1
# Expected: "touch: cannot touch '/test-write': Read-only file system"
docker compose down
```

#### Green Phase — Minimum Implementation

Create `Dockerfile` exactly as specified in design Section 10.1:
- Stage 1 (`builder`): `python:3.12-slim`, install `uv`, copy `pyproject.toml` + `src/`, run `uv venv + uv pip install .`
- Stage 2 (`runtime`): `python:3.12-slim`, create `appuser:appgroup` UID 1000, copy `/opt/venv` from builder, copy `src/`, set `USER appuser`, set `PATH`, `CMD uvicorn ... --factory`

Create `docker-compose.yml` exactly as in design Section 10.2:
- `read_only: true`, `tmpfs: [/tmp:size=64m,mode=1777]`, all env vars, `cap_drop: [ALL]`, `no-new-privileges:true`, healthcheck

Create `docker-compose.override.yml` exactly as in design Section 10.3.

Create `.env.example` with all variables from `Settings` listed with placeholder values and comments.

#### Refactor Checklist
- [ ] NFR-3 AC-1: non-root user created and switched to before CMD
- [ ] NFR-3 AC-2: `read_only: true` in compose
- [ ] NFR-3 AC-3: `/tmp` tmpfs declared in compose
- [ ] NFR-3 AC-4: multi-stage build excludes dev dependencies
- [ ] `pyproject.toml` is copied before `src/` for Docker layer cache efficiency
- [ ] No `.env` file is copied into the image (secrets from environment only)

---

### T-20: End-to-End Smoke Test Script

**Module**: `scripts/smoke-test.sh`
**Test File**: N/A (manual verification script)
**Complexity**: Low
**Depends On**: T-19
**Satisfies**: Manual integration verification (all FRs)

#### Red Phase — Write These Failing Tests First

The smoke test script itself is the "test." Before the service is running, all curl commands will fail — that is the RED phase. After T-19 is complete, the script should pass — that is GREEN.

```bash
#!/usr/bin/env bash
# scripts/smoke-test.sh
# Manual end-to-end smoke test for rokid-bridge
# Usage: ROKID_ACCESS_KEY=<key> UPSTREAM_TOKEN=<token> ./scripts/smoke-test.sh

set -euo pipefail

BRIDGE_URL="${BRIDGE_URL:-http://localhost:8090}"
ACCESS_KEY="${ROKID_ACCESS_KEY:?ROKID_ACCESS_KEY must be set}"

echo "=== Smoke Test: rokid-bridge ==="
echo "Target: $BRIDGE_URL"

# Step 1: Health check
echo ""
echo "1. Health check..."
response=$(curl -sf "$BRIDGE_URL/health")
echo "   Response: $response"
echo "$response" | grep -q '"status":"ok"' || { echo "FAIL: health check"; exit 1; }
echo "   PASS"

# Step 2: Valid text request (SSE chunks expected)
echo ""
echo "2. Valid text request with correct Bearer token..."
TIMESTAMP=$(date +%s)
BODY="{\"request_id\":\"test-1\",\"device_id\":\"smoke-device\",\"type\":\"text\",\"text\":\"Hello\",\"timestamp\":$TIMESTAMP}"
response=$(curl -sf -N \
  -H "Authorization: Bearer $ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d "$BODY" \
  "$BRIDGE_URL/chat" | head -5)
echo "   First 5 SSE lines:"
echo "$response"
echo "$response" | grep -q "data:" || { echo "FAIL: no SSE data received"; exit 1; }
echo "   PASS"

# Step 3: Wrong token returns 401
echo ""
echo "3. Wrong token returns 401..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer wrong-token" \
  -H "Content-Type: application/json" \
  -d "$BODY" \
  "$BRIDGE_URL/chat")
[ "$HTTP_CODE" = "401" ] || { echo "FAIL: expected 401, got $HTTP_CODE"; exit 1; }
echo "   PASS"

# Step 4: Missing auth returns 401
echo ""
echo "4. Missing Authorization returns 401..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Content-Type: application/json" \
  -d "$BODY" \
  "$BRIDGE_URL/chat")
[ "$HTTP_CODE" = "401" ] || { echo "FAIL: expected 401, got $HTTP_CODE"; exit 1; }
echo "   PASS"

echo ""
echo "=== All smoke tests PASSED ==="
```

#### Green Phase — Minimum Implementation

Create `scripts/smoke-test.sh` with the content above. Make it executable:
```bash
chmod +x scripts/smoke-test.sh
```

Create `scripts/` directory if it does not exist.

#### Refactor Checklist
- [ ] Script uses `set -euo pipefail` for safe bash execution
- [ ] `ROKID_ACCESS_KEY` is read from environment (never hardcoded in script)
- [ ] Script provides clear PASS/FAIL output for each step
- [ ] Script is idempotent (can be run multiple times safely)

---

## Implementation Order Summary

```
T-01 (scaffold)
  └─► T-02 (config)
        └─► T-03 (models)
              ├─► T-04 (auth verify)
              │     └─► T-05 (auth replay)
              ├─► T-06 (image handler)
              ├─► T-07 (history)
              └─► T-08 (relay build)
                    ├─► T-09 (extract text)
                    │     └─► T-10 (stream upstream)
                    └─► T-11 (rate limiter)
                          └─► T-12 (app scaffold + health)
                                ├─► T-13 (chat text integration)
                                │     └─► T-14 (chat image/text_with_image)
                                ├─► T-15 (auth wiring integration)
                                ├─► T-16 (rate limit wiring integration)
                                ├─► T-17 (clear history)
                                └─► T-18 (upstream errors)
                                      └─► T-19 (Docker)
                                            └─► T-20 (smoke test)
```

**Parallelizable after T-03:** T-04/T-05, T-06, T-07, and T-08 through T-10 can all be worked on in parallel after T-03 is complete.

---

## Test Pyramid Summary

| Test File | Type | Estimated Tests | Module(s) Covered |
|-----------|------|----------------|-------------------|
| `tests/unit/test_config.py` | Unit | 7 | `config.py` |
| `tests/unit/test_models.py` | Unit | 10 | `models.py` |
| `tests/unit/test_rokid_auth.py` | Unit | 15 | `rokid_auth.py` |
| `tests/unit/test_image_handler.py` | Unit | 9 | `image_handler.py` |
| `tests/unit/test_history.py` | Unit | 10 | `history.py` |
| `tests/unit/test_relay.py` | Unit | 20 | `relay.py` |
| `tests/unit/test_rate_limiter.py` | Unit | 7 | `app.py` (RateLimiter) |
| `tests/integration/test_endpoints.py` | Integration | 31 | `app.py` + all modules |
| **Total** | | **~109 tests** | |

**Unit tests (71%)**: 78 tests covering individual functions and classes in isolation.
**Integration tests (29%)**: 31 tests covering the full request pipeline via `TestClient`.

---

## Verification Commands

Run these commands after all tasks are complete to verify the full implementation:

```bash
# Install all dependencies
uv venv && uv pip install -e ".[dev]"

# Run unit tests with verbose output
uv run pytest tests/unit/ -v --cov=rokid_bridge --cov-report=term-missing

# Run integration tests
uv run pytest tests/integration/ -v

# Run all tests with coverage report
uv run pytest --cov=rokid_bridge --cov-report=term-missing
# Expected: >= 80% overall; 100% on rokid_auth.py

# Type check (must exit 0)
uv run mypy src/

# Lint check (must exit 0)
uv run ruff check src/ tests/

# Build Docker image
docker build -t rokid-bridge .

# Run Docker smoke test
ROKID_ACCESS_KEY=test UPSTREAM_TOKEN=test ./scripts/smoke-test.sh
```

---

## Approval Checklist

### TDD Methodology
- [ ] Every task has a RED phase with concrete, runnable `def test_...` stubs
- [ ] Every task has a GREEN phase describing minimum implementation
- [ ] Every task has a REFACTOR checklist with SOLID principle verification
- [ ] No task skips the RED phase

### Task Quality
- [ ] All 20 tasks are present (T-01 through T-20)
- [ ] Tasks are ordered by dependency (no task depends on a later task)
- [ ] Test imports use exact module paths from design.md
- [ ] Complexity ratings are honest (High for T-10, T-13; Medium for others; Low for pure functions)

### Coverage Alignment
- [ ] `rokid_auth.py` tests target 100% coverage (T-04, T-05: 15 tests)
- [ ] `image_handler.py` tests target 95%+ coverage (T-06: 9 tests)
- [ ] `history.py` tests target 90%+ coverage (T-07: 10 tests)
- [ ] `relay.py` tests target 85%+ coverage (T-08, T-09, T-10: 20 tests)
- [ ] Overall target >= 80% coverage (T-13 through T-18: 31 integration tests)

### Interface Contract Alignment
- [ ] Test assertions reference exact HTTP status codes from design Section 11
- [ ] Test assertions reference exact error detail strings from design Section 11.1
- [ ] SSE response format tested in T-13 matches design Section 5.3
- [ ] History update behavior tested in T-13, T-17, T-18 matches design Section 6

### Steering Document Compliance
- [ ] SOLID principles verified per task (SRP, OCP, ISP, DIP as applicable)
- [ ] DRY: `_require_auth` reuse in T-15/T-17; `validate_and_build_image_part` in T-06
- [ ] KISS: no over-engineered solutions proposed
- [ ] YAGNI: no tests for features not in requirements

---

**Approved by**: _____________________________ **Date**: _____________
