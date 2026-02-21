import pytest
from pydantic import SecretStr


def _make_settings(**kwargs):
    from rokid_bridge.config import Settings
    defaults = dict(rokid_access_key=SecretStr("ak"), upstream_token=SecretStr("ut"),
                    upstream_url="http://mock:8080")
    defaults.update(kwargs)
    return Settings(**defaults)


# ---------------------------------------------------------------------------
# T-08: build_upstream_request
# ---------------------------------------------------------------------------


def test_build_upstream_request_system_prompt_first():
    from rokid_bridge.relay import AR_SYSTEM_PROMPT, build_upstream_request
    result = build_upstream_request(history=[], user_content="Hello", settings=_make_settings())
    assert result["messages"][0]["role"] == "system"
    assert result["messages"][0]["content"] == AR_SYSTEM_PROMPT


def test_build_upstream_request_user_message_last():
    from rokid_bridge.relay import build_upstream_request
    result = build_upstream_request(history=[], user_content="Hello", settings=_make_settings())
    assert result["messages"][-1]["role"] == "user"
    assert result["messages"][-1]["content"] == "Hello"


def test_build_upstream_request_history_in_middle():
    from rokid_bridge.relay import build_upstream_request
    history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "resp"}]
    result = build_upstream_request(
        history=history, user_content="New Q", settings=_make_settings()
    )
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_build_upstream_request_agent_id_included_when_set():
    from rokid_bridge.relay import build_upstream_request
    result = build_upstream_request(history=[], user_content="Hi",
                                     settings=_make_settings(rokid_agent_id="my-agent"))
    assert result.get("agent_id") == "my-agent"


def test_build_upstream_request_agent_id_omitted_when_empty():
    from rokid_bridge.relay import build_upstream_request
    result = build_upstream_request(history=[], user_content="Hi",
                                     settings=_make_settings(rokid_agent_id=""))
    assert "agent_id" not in result


def test_build_upstream_request_stream_is_true():
    from rokid_bridge.relay import build_upstream_request
    result = build_upstream_request(history=[], user_content="Hi", settings=_make_settings())
    assert result["stream"] is True


def test_build_upstream_request_list_content_passthrough():
    from rokid_bridge.relay import build_upstream_request
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ]
    result = build_upstream_request(history=[], user_content=content, settings=_make_settings())
    assert result["messages"][-1]["content"] == content


def test_build_upstream_request_empty_history():
    from rokid_bridge.relay import build_upstream_request
    result = build_upstream_request(history=[], user_content="Hello", settings=_make_settings())
    assert len(result["messages"]) == 2  # system + user


# ---------------------------------------------------------------------------
# T-09: extract_assistant_text
# ---------------------------------------------------------------------------


def test_extract_assistant_text_single_chunk():
    from rokid_bridge.relay import extract_assistant_text
    chunks = ['data: {"choices":[{"delta":{"content":"Hello"}}]}', "data: [DONE]"]
    assert extract_assistant_text(chunks) == "Hello"


def test_extract_assistant_text_multiple_chunks():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "data: [DONE]",
    ]
    assert extract_assistant_text(chunks) == "Hello world"


def test_extract_assistant_text_done_stops_processing():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        'data: {"choices":[{"delta":{"content":"before"}}]}',
        "data: [DONE]",
        'data: {"choices":[{"delta":{"content":"after"}}]}',
    ]
    assert extract_assistant_text(chunks) == "before"


def test_extract_assistant_text_non_data_lines_ignored():
    from rokid_bridge.relay import extract_assistant_text
    chunks = ["", "comment line", 'data: {"choices":[{"delta":{"content":"hi"}}]}', "data: [DONE]"]
    assert extract_assistant_text(chunks) == "hi"


def test_extract_assistant_text_malformed_json_skipped():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        "data: {invalid json}",
        'data: {"choices":[{"delta":{"content":"ok"}}]}',
        "data: [DONE]",
    ]
    assert extract_assistant_text(chunks) == "ok"


def test_extract_assistant_text_empty_delta_skipped():
    from rokid_bridge.relay import extract_assistant_text
    chunks = [
        'data: {"choices":[{"delta":{}}]}',
        'data: {"choices":[{"delta":{"content":"end"}}]}',
        "data: [DONE]",
    ]
    assert extract_assistant_text(chunks) == "end"


def test_extract_assistant_text_empty_chunks_returns_empty():
    from rokid_bridge.relay import extract_assistant_text
    assert extract_assistant_text([]) == ""


# ---------------------------------------------------------------------------
# T-10: stream_upstream async generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_upstream_yields_lines(httpx_mock):
    import httpx

    from rokid_bridge.relay import build_upstream_request, stream_upstream
    settings = _make_settings()
    httpx_mock.add_response(
        method="POST",
        url="http://mock:8080/v1/chat/completions",
        content=b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n',
        headers={"Content-Type": "text/event-stream"},
    )
    request = build_upstream_request(history=[], user_content="Hello", settings=settings)
    async with httpx.AsyncClient() as client:
        lines = [line async for line in stream_upstream(client, request, settings)]
    assert any("Hi" in line for line in lines)


@pytest.mark.asyncio
async def test_stream_upstream_raises_upstream_error_on_non_2xx(httpx_mock):
    import httpx

    from rokid_bridge.relay import UpstreamError, build_upstream_request, stream_upstream
    settings = _make_settings()
    httpx_mock.add_response(
        method="POST",
        url="http://mock:8080/v1/chat/completions",
        status_code=500,
        content=b"Internal Server Error",
    )
    request = build_upstream_request(history=[], user_content="Hello", settings=settings)
    async with httpx.AsyncClient() as client:
        with pytest.raises(UpstreamError) as exc_info:
            async for _ in stream_upstream(client, request, settings):
                pass
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_stream_upstream_authorization_header_sent(httpx_mock):
    import httpx

    from rokid_bridge.relay import build_upstream_request, stream_upstream
    settings = _make_settings()
    httpx_mock.add_response(
        method="POST",
        url="http://mock:8080/v1/chat/completions",
        content=b"data: [DONE]\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    request = build_upstream_request(history=[], user_content="Hi", settings=settings)
    async with httpx.AsyncClient() as client:
        async for _ in stream_upstream(client, request, settings):
            pass
    sent_request = httpx_mock.get_request()
    assert sent_request.headers["Authorization"] == "Bearer ut"
