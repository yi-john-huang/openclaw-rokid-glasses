import time


def _make_limiter(max_requests: int = 3, window: int = 60):
    from rokid_bridge.app import RateLimiter
    return RateLimiter(max_requests=max_requests, window_seconds=window)


def test_first_request_allowed():
    limiter = _make_limiter()
    allowed, retry_after = limiter.check_and_record("dev-1")
    assert allowed is True
    assert retry_after == 0.0


def test_requests_below_limit_allowed():
    limiter = _make_limiter(max_requests=3)
    for _ in range(3):
        allowed, _ = limiter.check_and_record("dev-1")
        assert allowed is True


def test_request_at_limit_rejected():
    limiter = _make_limiter(max_requests=3)
    for _ in range(3):
        limiter.check_and_record("dev-1")
    allowed, retry_after = limiter.check_and_record("dev-1")
    assert allowed is False
    assert retry_after > 0


def test_different_devices_independent():
    limiter = _make_limiter(max_requests=1)
    limiter.check_and_record("dev-A")
    # dev-A is now at limit
    allowed_a, _ = limiter.check_and_record("dev-A")
    allowed_b, _ = limiter.check_and_record("dev-B")
    assert allowed_a is False
    assert allowed_b is True


def test_window_expiry_resets_count():
    limiter = _make_limiter(max_requests=1, window=1)
    limiter.check_and_record("dev-1")
    allowed_before, _ = limiter.check_and_record("dev-1")
    assert allowed_before is False
    time.sleep(1.1)
    allowed_after, _ = limiter.check_and_record("dev-1")
    assert allowed_after is True


def test_retry_after_is_reasonable():
    limiter = _make_limiter(max_requests=1, window=60)
    limiter.check_and_record("dev-1")
    _, retry_after = limiter.check_and_record("dev-1")
    assert 0 < retry_after <= 60


def test_new_device_not_affected_by_other():
    limiter = _make_limiter(max_requests=2)
    limiter.check_and_record("dev-1")
    limiter.check_and_record("dev-1")
    limiter.check_and_record("dev-1")  # dev-1 at limit
    allowed, _ = limiter.check_and_record("dev-2")
    assert allowed is True
