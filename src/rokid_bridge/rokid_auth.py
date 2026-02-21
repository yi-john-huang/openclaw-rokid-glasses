import hmac
import time

from rokid_bridge.config import Settings


class AuthError(Exception):
    def __init__(self, detail: str, status_code: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def verify_bearer_token(authorization_header: str | None, settings: Settings) -> None:
    if not authorization_header:
        raise AuthError("Unauthorized")
    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError("Unauthorized")
    expected = settings.rokid_access_key.get_secret_value()
    if not hmac.compare_digest(expected.encode("utf-8"), token.encode("utf-8")):
        raise AuthError("Unauthorized")


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
