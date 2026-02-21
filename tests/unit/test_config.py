import pytest
from pydantic_core import ValidationError


def test_settings_missing_required_fields_raises(monkeypatch):
    monkeypatch.delenv("ROKID_ACCESS_KEY", raising=False)
    monkeypatch.delenv("UPSTREAM_TOKEN", raising=False)
    from rokid_bridge.config import Settings
    with pytest.raises(ValidationError):
        Settings()


def test_settings_valid_required_fields(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "my-secret-key")
    monkeypatch.setenv("UPSTREAM_TOKEN", "my-upstream-token")
    from rokid_bridge.config import Settings
    s = Settings()
    assert s.rokid_access_key.get_secret_value() == "my-secret-key"
    assert s.upstream_token.get_secret_value() == "my-upstream-token"


def test_settings_secret_str_masked(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "supersecret")
    monkeypatch.setenv("UPSTREAM_TOKEN", "anothersecret")
    from rokid_bridge.config import Settings
    s = Settings()
    assert "supersecret" not in str(s.rokid_access_key)
    assert "supersecret" not in repr(s.rokid_access_key)


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


def test_settings_zero_rate_limit_rejected(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    monkeypatch.setenv("ROKID_RATE_LIMIT", "0")
    from rokid_bridge.config import Settings
    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_returns_settings(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    from rokid_bridge.config import Settings, get_settings
    result = get_settings()
    assert isinstance(result, Settings)


def test_settings_custom_upstream_url(monkeypatch):
    monkeypatch.setenv("ROKID_ACCESS_KEY", "k")
    monkeypatch.setenv("UPSTREAM_TOKEN", "t")
    monkeypatch.setenv("UPSTREAM_URL", "http://custom-stack:9090")
    from rokid_bridge.config import Settings
    s = Settings()
    assert s.upstream_url == "http://custom-stack:9090"
