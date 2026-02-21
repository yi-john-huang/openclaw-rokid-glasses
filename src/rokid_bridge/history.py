import time
from collections import deque
from dataclasses import dataclass, field

from rokid_bridge.models import OpenAIMessage

SESSION_KEY_PREFIX: str = "rokid"
HISTORY_TTL_SECONDS: int = 3600


@dataclass
class _Session:
    messages: deque[OpenAIMessage] = field(default_factory=deque)
    last_access: float = field(default_factory=time.time)


class HistoryStore:
    def __init__(self, max_turns: int, ttl_seconds: float = HISTORY_TTL_SECONDS) -> None:
        self._max_turns = max_turns
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, _Session] = {}

    def get_messages(self, device_id: str) -> list[OpenAIMessage]:
        key = self._make_key(device_id)
        self._evict_if_expired(key)
        session = self._sessions.get(key)
        if session is None:
            return []
        session.last_access = time.time()
        return list(session.messages)

    def append_turn(self, device_id: str, user_content: str, assistant_content: str) -> None:
        key = self._make_key(device_id)
        self._evict_if_expired(key)
        if key not in self._sessions:
            self._sessions[key] = _Session()
        session = self._sessions[key]
        session.last_access = time.time()
        while len(session.messages) >= self._max_turns * 2:
            session.messages.popleft()
            if session.messages:
                session.messages.popleft()
        session.messages.append(OpenAIMessage(role="user", content=user_content))
        session.messages.append(OpenAIMessage(role="assistant", content=assistant_content))

    def clear(self, device_id: str) -> None:
        self._sessions.pop(self._make_key(device_id), None)

    @staticmethod
    def _make_key(device_id: str) -> str:
        return f"{SESSION_KEY_PREFIX}:{device_id}"

    def _evict_if_expired(self, key: str) -> None:
        session = self._sessions.get(key)
        if session is not None and (time.time() - session.last_access) > self._ttl_seconds:
            del self._sessions[key]
