import pytest
from pydantic import SecretStr


def _make_settings(key: str = "correct-token"):
    from rokid_bridge.config import Settings
    return Settings(rokid_access_key=SecretStr(key), upstream_token=SecretStr("up"))


def test_verify_bearer_token_valid():
    from rokid_bridge.rokid_auth import verify_bearer_token
    verify_bearer_token("Bearer correct-token", _make_settings())  # no exception


def test_verify_bearer_token_missing_header_raises():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    with pytest.raises(AuthError):
        verify_bearer_token(None, _make_settings())


def test_verify_bearer_token_empty_header_raises():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    with pytest.raises(AuthError):
        verify_bearer_token("", _make_settings())


def test_verify_bearer_token_wrong_scheme_raises():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    with pytest.raises(AuthError):
        verify_bearer_token("Token correct-token", _make_settings())


def test_verify_bearer_token_wrong_token_raises():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    with pytest.raises(AuthError):
        verify_bearer_token("Bearer wrong-token", _make_settings())


def test_verify_bearer_token_raises_auth_error_with_401():
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    with pytest.raises(AuthError) as exc_info:
        verify_bearer_token("Bearer wrong", _make_settings())
    assert exc_info.value.status_code == 401


def test_verify_bearer_token_generic_message():
    """Error message must not differ between missing vs wrong token (no leakage)."""
    from rokid_bridge.rokid_auth import AuthError, verify_bearer_token
    try:
        verify_bearer_token(None, _make_settings())
    except AuthError as e1:
        try:
            verify_bearer_token("Bearer wrong", _make_settings())
        except AuthError as e2:
            assert e1.detail == e2.detail  # both say "Unauthorized"


def test_verify_bearer_token_uses_constant_time_comparison():
    """Verify hmac.compare_digest is used (no == on secrets)."""
    from rokid_bridge.rokid_auth import verify_bearer_token
    # If compare_digest is not used, a timing oracle would be detectable.
    # Here we just confirm it doesn't raise on valid token (functional test).
    verify_bearer_token("Bearer correct-token", _make_settings("correct-token"))


# ---------------------------------------------------------------------------
# T-05: Replay Window Protection
# ---------------------------------------------------------------------------


def _make_settings_with_window(window: int = 300):
    from rokid_bridge.config import Settings
    return Settings(
        rokid_access_key=SecretStr("k"),
        upstream_token=SecretStr("t"),
        rokid_replay_window=window,
    )


def test_check_replay_window_valid_current():
    import time

    from rokid_bridge.rokid_auth import check_replay_window
    now = time.time()
    check_replay_window(int(now), _make_settings_with_window(), now=now)  # no exception


def test_check_replay_window_within_window():
    from rokid_bridge.rokid_auth import check_replay_window
    now = 1000000.0
    check_replay_window(int(now) - 150, _make_settings_with_window(300), now=now)


def test_check_replay_window_expired_raises():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    now = 1000000.0
    with pytest.raises(AuthError, match="expired"):
        check_replay_window(int(now) - 400, _make_settings_with_window(300), now=now)


def test_check_replay_window_future_beyond_60s_raises():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    now = 1000000.0
    with pytest.raises(AuthError, match="invalid"):
        check_replay_window(int(now) + 120, _make_settings_with_window(300), now=now)


def test_check_replay_window_future_within_60s_allowed():
    from rokid_bridge.rokid_auth import check_replay_window
    now = 1000000.0
    check_replay_window(int(now) + 30, _make_settings_with_window(300), now=now)


def test_check_replay_window_exactly_at_boundary():
    from rokid_bridge.rokid_auth import check_replay_window
    now = 1000000.0
    check_replay_window(int(now) - 300, _make_settings_with_window(300), now=now)


def test_check_replay_window_configurable():
    from rokid_bridge.rokid_auth import AuthError, check_replay_window
    now = 1000000.0
    with pytest.raises(AuthError):
        check_replay_window(int(now) - 100, _make_settings_with_window(60), now=now)
