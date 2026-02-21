import time


def _make_store(max_turns: int = 5, ttl: float = 3600.0):
    from rokid_bridge.history import HistoryStore
    return HistoryStore(max_turns=max_turns, ttl_seconds=ttl)


def test_get_empty_returns_empty_list():
    store = _make_store()
    assert store.get_messages("dev-1") == []


def test_append_turn_and_get():
    store = _make_store()
    store.append_turn("dev-1", "Hello", "Hi there")
    msgs = store.get_messages("dev-1")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "Hi there"


def test_multiple_turns_in_order():
    store = _make_store()
    store.append_turn("dev-1", "Q1", "A1")
    store.append_turn("dev-1", "Q2", "A2")
    msgs = store.get_messages("dev-1")
    assert [m["content"] for m in msgs] == ["Q1", "A1", "Q2", "A2"]


def test_truncation_evicts_oldest_pair(monkeypatch):
    store = _make_store(max_turns=2)
    store.append_turn("dev-1", "Q1", "A1")
    store.append_turn("dev-1", "Q2", "A2")
    store.append_turn("dev-1", "Q3", "A3")  # should evict Q1/A1
    msgs = store.get_messages("dev-1")
    assert len(msgs) == 4
    assert msgs[0]["content"] == "Q2"


def test_clear_removes_history():
    store = _make_store()
    store.append_turn("dev-1", "Hello", "Hi")
    store.clear("dev-1")
    assert store.get_messages("dev-1") == []


def test_clear_nonexistent_is_safe():
    store = _make_store()
    store.clear("nonexistent")  # must not raise


def test_get_returns_copy_not_reference():
    store = _make_store()
    store.append_turn("dev-1", "Hello", "Hi")
    msgs = store.get_messages("dev-1")
    msgs.append({"role": "user", "content": "Injected"})
    assert len(store.get_messages("dev-1")) == 2  # still 2, not 3


def test_cross_device_isolation():
    store = _make_store()
    store.append_turn("dev-A", "From A", "Reply A")
    store.append_turn("dev-B", "From B", "Reply B")
    assert store.get_messages("dev-A")[0]["content"] == "From A"
    assert store.get_messages("dev-B")[0]["content"] == "From B"


def test_ttl_eviction(monkeypatch):
    store = _make_store(ttl=0.01)
    store.append_turn("dev-1", "Hello", "Hi")
    time.sleep(0.05)
    assert store.get_messages("dev-1") == []


def test_session_key_uses_rokid_prefix():
    from rokid_bridge.history import SESSION_KEY_PREFIX
    assert SESSION_KEY_PREFIX == "rokid"
