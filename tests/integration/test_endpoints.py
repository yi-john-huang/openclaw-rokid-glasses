"""Integration tests for the Rokid Bridge FastAPI endpoints.

Tests cover:
- T-12: Health endpoint
- T-13: /chat text flow
- T-14: /chat image and text+image flows
- T-15: Auth and replay protection
- T-16: Rate limiting
- T-17: /clear-history endpoint
- T-18: Upstream error handling
"""
import base64
import json
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_settings(**kwargs):  # type: ignore[no-untyped-def]
    from rokid_bridge.config import Settings

    defaults = dict(
        rokid_access_key=SecretStr("test-ak"),
        upstream_token=SecretStr("test-ut"),
        upstream_url="http://mock-upstream:8080",
        rokid_rate_limit=100,
    )
    defaults.update(kwargs)
    return Settings(**defaults)


def _valid_headers(token: str = "test-ak") -> dict:  # type: ignore[type-arg]
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _text_body(text: str = "Hello", device_id: str = "dev-001") -> dict:  # type: ignore[type-arg]
    return {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "device_id": device_id,
        "type": "text",
        "text": text,
        "timestamp": int(time.time()),
    }


def _image_body(device_id: str = "dev-001", text: str | None = None) -> dict:  # type: ignore[type-arg]
    fake_image = base64.b64encode(b"fake-image-data").decode()
    body: dict = {  # type: ignore[type-arg]
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "device_id": device_id,
        "type": "image",
        "image": {"data": fake_image, "mime_type": "image/jpeg"},
        "timestamp": int(time.time()),
    }
    if text is not None:
        body["type"] = "text_with_image"
        body["text"] = text
    return body


def _clear_body(device_id: str = "dev-001") -> dict:  # type: ignore[type-arg]
    return {"device_id": device_id, "timestamp": int(time.time())}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():  # type: ignore[no-untyped-def]
    from rokid_bridge.app import create_app

    app = create_app(settings=_make_settings())
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def rate_limited_client():  # type: ignore[no-untyped-def]
    from rokid_bridge.app import create_app

    app = create_app(settings=_make_settings(rokid_rate_limit=2))
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# T-12: Health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_200(client):  # type: ignore[no-untyped-def]
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_returns_ok_status(client):  # type: ignore[no-untyped-def]
    resp = client.get("/health")
    assert resp.json()["status"] == "ok"


def test_health_returns_service_name(client):  # type: ignore[no-untyped-def]
    resp = client.get("/health")
    assert resp.json()["service"] == "rokid-bridge"


def test_health_no_auth_required(client):  # type: ignore[no-untyped-def]
    # No Authorization header — must still return 200
    resp = client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# T-13: /chat text flow
# ---------------------------------------------------------------------------


def test_chat_text_returns_event_stream(client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    body = _text_body()
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_chat_text_sse_chunks_forwarded(client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    body = _text_body()
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert "Hello" in resp.text


def test_chat_text_history_updated_after_stream(client, httpx_mock):  # type: ignore[no-untyped-def]
    """After a successful chat, the next request includes history."""
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"Reply1"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"Reply2"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    body = _text_body("First message", "dev-history-test")
    client.post("/chat", json=body, headers=_valid_headers())
    body2 = _text_body("Second message", "dev-history-test")
    client.post("/chat", json=body2, headers=_valid_headers())
    # Verify second request included history
    requests = httpx_mock.get_requests()
    second_body = json.loads(requests[1].content)
    messages = second_body["messages"]
    roles = [m["role"] for m in messages]
    assert "assistant" in roles  # history was included


# ---------------------------------------------------------------------------
# T-14: /chat image and text+image flows
# ---------------------------------------------------------------------------


def test_chat_image_returns_event_stream(client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"I see"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    resp = client.post("/chat", json=_image_body(), headers=_valid_headers())
    assert resp.status_code == 200


def test_chat_text_with_image_returns_event_stream(client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"I see"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    resp = client.post("/chat", json=_image_body(text="What is this?"), headers=_valid_headers())
    assert resp.status_code == 200


def test_chat_blank_text_returns_422(client):  # type: ignore[no-untyped-def]
    body = _text_body(text="   ")
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 422


def test_chat_image_type_missing_image_returns_422(client):  # type: ignore[no-untyped-def]
    body = {
        "request_id": "id",
        "device_id": "dev",
        "type": "image",
        "timestamp": int(time.time()),
    }
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 422


def test_chat_invalid_mime_returns_422(client):  # type: ignore[no-untyped-def]
    body = {
        "request_id": "id",
        "device_id": "dev",
        "type": "image",
        "image": {"data": "aGVsbG8=", "mime_type": "image/gif"},
        "timestamp": int(time.time()),
    }
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# T-15: Auth and replay integration tests
# ---------------------------------------------------------------------------


def test_chat_missing_auth_returns_401(client):  # type: ignore[no-untyped-def]
    body = _text_body()
    resp = client.post("/chat", json=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 401


def test_chat_wrong_token_returns_401(client):  # type: ignore[no-untyped-def]
    body = _text_body()
    resp = client.post("/chat", json=body, headers=_valid_headers(token="wrong-token"))
    assert resp.status_code == 401


def test_chat_expired_timestamp_returns_401(client):  # type: ignore[no-untyped-def]
    body = {
        "request_id": "id",
        "device_id": "dev",
        "type": "text",
        "text": "hi",
        "timestamp": int(time.time()) - 400,  # expired
    }
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 401


def test_chat_future_timestamp_returns_401(client):  # type: ignore[no-untyped-def]
    body = {
        "request_id": "id",
        "device_id": "dev",
        "type": "text",
        "text": "hi",
        "timestamp": int(time.time()) + 200,  # too far future
    }
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 401


def test_chat_auth_error_body_is_generic(client):  # type: ignore[no-untyped-def]
    body = _text_body()
    resp = client.post("/chat", json=body, headers=_valid_headers(token="wrong"))
    assert resp.json()["detail"] == "Unauthorized"


# ---------------------------------------------------------------------------
# T-16: Rate limiting integration tests
# ---------------------------------------------------------------------------


def test_rate_limit_exceeded_returns_429(rate_limited_client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    statuses = []
    for _ in range(4):
        body = _text_body(device_id="rate-dev")
        resp = rate_limited_client.post("/chat", json=body, headers=_valid_headers())
        statuses.append(resp.status_code)
    assert 429 in statuses


def test_rate_limit_has_retry_after_header(rate_limited_client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b"data: [DONE]\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b"data: [DONE]\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    for _ in range(3):
        body = _text_body(device_id="rate-dev-2")
        resp = rate_limited_client.post("/chat", json=body, headers=_valid_headers())
        if resp.status_code == 429:
            assert "retry-after" in resp.headers
            break


def test_rate_limit_different_device_not_affected(rate_limited_client, httpx_mock):  # type: ignore[no-untyped-def]
    # rate_limit=2: dev-A gets 2 successes, 3rd is rate-limited (no upstream call)
    # dev-B gets 1 success — 3 total upstream calls needed
    for _ in range(3):
        httpx_mock.add_response(
            method="POST",
            url="http://mock-upstream:8080/v1/chat/completions",
            content=b"data: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"},
        )
    # Exhaust rate limit for dev-A
    for _ in range(3):
        rate_limited_client.post(
            "/chat", json=_text_body(device_id="dev-A"), headers=_valid_headers()
        )
    # dev-B should still work
    resp = rate_limited_client.post(
        "/chat", json=_text_body(device_id="dev-B"), headers=_valid_headers()
    )
    assert resp.status_code != 429


# ---------------------------------------------------------------------------
# T-17: /clear-history endpoint
# ---------------------------------------------------------------------------


def test_clear_history_returns_200(client):  # type: ignore[no-untyped-def]
    body = _clear_body()
    resp = client.post("/clear-history", json=body, headers=_valid_headers())
    assert resp.status_code == 200


def test_clear_history_response_body(client):  # type: ignore[no-untyped-def]
    body = _clear_body("dev-clear-test")
    resp = client.post("/clear-history", json=body, headers=_valid_headers())
    data = resp.json()
    assert data["cleared"] is True
    assert data["device_id"] == "dev-clear-test"


def test_clear_history_no_auth_returns_401(client):  # type: ignore[no-untyped-def]
    body = _clear_body()
    resp = client.post("/clear-history", json=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 401


def test_clear_history_nonexistent_device_returns_200(client):  # type: ignore[no-untyped-def]
    body = _clear_body("nonexistent-device")
    resp = client.post("/clear-history", json=body, headers=_valid_headers())
    assert resp.status_code == 200


def test_clear_history_then_chat_starts_fresh(client, httpx_mock):  # type: ignore[no-untyped-def]
    """After clearing history, next chat has no history in upstream request."""
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"R1"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"R2"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    dev = "dev-clear-fresh"
    # First chat builds history
    client.post("/chat", json=_text_body("Hi", dev), headers=_valid_headers())
    # Clear history
    client.post("/clear-history", json=_clear_body(dev), headers=_valid_headers())
    # Second chat
    client.post("/chat", json=_text_body("Again", dev), headers=_valid_headers())
    requests = httpx_mock.get_requests()
    # The second chat request (index 1, post-clear) should have only system + user (no history)
    second_body = json.loads(requests[1].content)
    messages = second_body["messages"]
    assert all(m["role"] != "assistant" for m in messages)


# ---------------------------------------------------------------------------
# T-18: Upstream error handling
# ---------------------------------------------------------------------------


def test_upstream_500_forwarded_as_sse_error(client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        status_code=500,
        content=b"Internal Server Error",
    )
    resp = client.post("/chat", json=_text_body(), headers=_valid_headers())
    assert resp.status_code == 200  # SSE response starts OK
    assert "error" in resp.text.lower()


def test_upstream_error_does_not_update_history(client, httpx_mock):  # type: ignore[no-untyped-def]
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        status_code=500,
        content=b"error",
    )
    httpx_mock.add_response(
        method="POST",
        url="http://mock-upstream:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    dev = "dev-error-test"
    client.post("/chat", json=_text_body("First", dev), headers=_valid_headers())
    client.post("/chat", json=_text_body("Second", dev), headers=_valid_headers())
    requests = httpx_mock.get_requests()
    second_body = json.loads(requests[1].content)
    messages = second_body["messages"]
    # No assistant message in history (first request failed)
    assert all(m["role"] != "assistant" for m in messages)


# ---------------------------------------------------------------------------
# Coverage gap: lifespan shutdown (lines 95-96)
# ---------------------------------------------------------------------------


def test_app_lifespan_shutdown_closes_http_client():  # type: ignore[no-untyped-def]
    """Entering and exiting TestClient as context manager triggers lifespan shutdown."""
    from rokid_bridge.app import create_app

    app = create_app(settings=_make_settings())
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200
    # Exiting the context manager triggers the lifespan shutdown (lines 95-96).
    # No assertion needed beyond confirming no exception was raised on aclose().


# ---------------------------------------------------------------------------
# Coverage gap: text_with_image validation branches (lines 160, 164, 171-172)
# ---------------------------------------------------------------------------


def test_chat_text_with_image_missing_image_returns_422(client):  # type: ignore[no-untyped-def]
    """type=text_with_image with no image field must return 422 (line 160)."""
    body = {
        "request_id": "id",
        "device_id": "dev-001",
        "type": "text_with_image",
        "text": "describe this",
        "timestamp": int(time.time()),
    }
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 422
    assert "image is required" in resp.json()["detail"].lower()


def test_chat_text_with_image_blank_text_returns_422(client):  # type: ignore[no-untyped-def]
    """type=text_with_image with blank text must return 422 (line 164)."""
    fake_image = base64.b64encode(b"fake-image-data").decode()
    body = {
        "request_id": "id",
        "device_id": "dev-001",
        "type": "text_with_image",
        "text": "   ",
        "image": {"data": fake_image, "mime_type": "image/jpeg"},
        "timestamp": int(time.time()),
    }
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 422
    assert "text is required" in resp.json()["detail"].lower()


def test_chat_text_with_image_invalid_image_returns_422(client):  # type: ignore[no-untyped-def]
    """type=text_with_image with invalid base64 image must return 422 (lines 171-172)."""
    body = {
        "request_id": "id",
        "device_id": "dev-001",
        "type": "text_with_image",
        "text": "describe this",
        "image": {"data": "!!!not-valid-base64!!!", "mime_type": "image/jpeg"},
        "timestamp": int(time.time()),
    }
    resp = client.post("/chat", json=body, headers=_valid_headers())
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Coverage gap: clear-history replay protection (lines 208-209)
# ---------------------------------------------------------------------------


def test_clear_history_expired_timestamp_returns_401(client):  # type: ignore[no-untyped-def]
    """Expired timestamp on /clear-history must return 401 (lines 208-209)."""
    body = {"device_id": "dev-001", "timestamp": int(time.time()) - 400}
    resp = client.post("/clear-history", json=body, headers=_valid_headers())
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Coverage gap: SSE generator network error paths (lines 240-245)
# ---------------------------------------------------------------------------


def test_upstream_timeout_returns_sse_error_event(client, httpx_mock):  # type: ignore[no-untyped-def]
    """httpx.TimeoutException must yield an SSE error event (lines 240-242)."""
    import httpx as _httpx

    httpx_mock.add_exception(_httpx.TimeoutException("upstream timed out"))
    resp = client.post("/chat", json=_text_body(), headers=_valid_headers())
    assert resp.status_code == 200
    assert "upstream timeout" in resp.text


def test_upstream_connect_error_returns_sse_error_event(client, httpx_mock):  # type: ignore[no-untyped-def]
    """httpx.ConnectError must yield an SSE error event (lines 243-245)."""
    import httpx as _httpx

    httpx_mock.add_exception(_httpx.ConnectError("connection refused"))
    resp = client.post("/chat", json=_text_body(), headers=_valid_headers())
    assert resp.status_code == 200
    assert "upstream unavailable" in resp.text
