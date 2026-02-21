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
    system_message = OpenAIMessage(role="system", content=AR_SYSTEM_PROMPT)
    user_message = OpenAIMessage(role="user", content=user_content)
    messages: list[OpenAIMessage] = [system_message, *history, user_message]
    request: UpstreamRequest = {"messages": messages, "stream": True}
    if settings.rokid_agent_id:
        request["agent_id"] = settings.rokid_agent_id
    return request


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
    async with client.stream(
        "POST", url, json=upstream_request, headers=headers, timeout=UPSTREAM_TIMEOUT_SECONDS,
    ) as response:
        if response.status_code >= 300:
            body = await response.aread()
            raise UpstreamError(response.status_code, body.decode("utf-8", errors="replace"))
        async for line in response.aiter_lines():
            yield line


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
