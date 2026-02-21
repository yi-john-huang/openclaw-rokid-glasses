import logging
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
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
    ContentPart,
    HealthResponse,
    OpenAIMessage,
    RokidChatRequest,
    RokidClearRequest,
    RokidRequestType,
    UpstreamRequest,
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


class RateLimiter:
    """Per-device sliding window rate limiter.

    Internal structure: dict mapping device_id -> deque of request timestamps.
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

        cutoff = now - self._window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self._max:
            retry_after = self._window - (now - bucket[0])
            return False, max(retry_after, 0.0)

        bucket.append(now)
        return True, 0.0


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

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await _http_client.aclose()

    app = FastAPI(
        title="Rokid Bridge",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )

    def _require_auth(request: Request) -> None:
        """Verify Bearer token. Raises HTTP 401 on failure."""
        try:
            verify_bearer_token(request.headers.get("Authorization"), _settings)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness probe — no auth, no I/O."""
        return HealthResponse()

    @app.post("/chat")
    async def chat(body: RokidChatRequest, request: Request) -> StreamingResponse:
        """Main chat endpoint. Pipeline: rate-limit -> auth -> replay -> validate -> stream."""
        # Pipeline Step 2: Rate limit
        allowed, retry_after = _rate_limiter.check_and_record(body.device_id)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(int(retry_after) + 1)},
            )

        # Pipeline Step 3: Auth
        _require_auth(request)

        # Pipeline Step 4: Replay check
        try:
            check_replay_window(body.timestamp, _settings)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

        # Pipeline Step 5: Type-specific payload validation + build messages
        user_content: str | list[ContentPart]
        user_text_for_history: str

        if body.type == RokidRequestType.TEXT:
            if not body.text.strip():
                raise HTTPException(status_code=422, detail="Text is required for type 'text'")
            user_content = body.text
            user_text_for_history = body.text

        elif body.type == RokidRequestType.IMAGE:
            if body.image is None:
                raise HTTPException(status_code=422, detail="Image is required for type 'image'")
            try:
                user_content = build_image_content(body.image, _settings.rokid_image_detail)
            except ImageValidationError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
            user_text_for_history = "[image request]"

        elif body.type == RokidRequestType.TEXT_WITH_IMAGE:
            if body.image is None:
                raise HTTPException(
                    status_code=422, detail="Image is required for type 'text_with_image'"
                )
            if not body.text.strip():
                raise HTTPException(
                    status_code=422, detail="Text is required for type 'text_with_image'"
                )
            try:
                user_content = build_text_with_image_content(
                    body.text, body.image, _settings.rokid_image_detail
                )
            except ImageValidationError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
            user_text_for_history = body.text

        else:  # pragma: no cover  — StrEnum validation makes this unreachable
            raise HTTPException(status_code=422, detail=f"Unknown request type: {body.type}")

        # Pipeline Step 6: Retrieve history
        history_messages: list[OpenAIMessage] = _history.get_messages(body.device_id)

        # Pipeline Step 7: Build upstream request
        upstream_req = build_upstream_request(
            history=history_messages,
            user_content=user_content,
            settings=_settings,
        )

        # Pipeline Steps 8-10: Stream and update history
        return StreamingResponse(
            _sse_generator(
                _http_client,
                upstream_req,
                _settings,
                _history,
                body.device_id,
                user_text_for_history,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/clear-history", response_model=ClearHistoryResponse)
    async def clear_history(body: RokidClearRequest, request: Request) -> ClearHistoryResponse:
        """Clear per-device history."""
        _require_auth(request)
        try:
            check_replay_window(body.timestamp, _settings)
        except AuthError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        _history.clear(body.device_id)
        return ClearHistoryResponse(cleared=True, device_id=body.device_id)

    return app


async def _sse_generator(
    client: httpx.AsyncClient,
    upstream_request: UpstreamRequest,
    settings: Settings,
    history: HistoryStore,
    device_id: str,
    user_text_for_history: str,
) -> AsyncGenerator[str, None]:
    """Streaming generator: relay SSE chunks and collect for history update."""
    collected: list[str] = []
    success = False
    try:
        async for line in stream_upstream(client, upstream_request, settings):
            collected.append(line)
            yield f"{line}\n"
        success = True
    except UpstreamError as exc:
        logger.error(
            "Upstream error %d for device %s: %s",
            exc.status_code,
            device_id,
            exc.body[:500],
        )
        yield f'data: {{"error": "upstream error", "status": {exc.status_code}}}\n\n'
    except httpx.TimeoutException:
        logger.error("Upstream timeout for device %s", device_id)
        yield 'data: {"error": "upstream timeout"}\n\n'
    except httpx.ConnectError:
        logger.error("Upstream connection refused for device %s", device_id)
        yield 'data: {"error": "upstream unavailable"}\n\n'
    finally:
        if success and collected:
            assistant_text = extract_assistant_text(collected)
            if assistant_text:
                history.append_turn(device_id, user_text_for_history, assistant_text)
