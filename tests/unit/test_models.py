import pytest
from pydantic import ValidationError


def test_rokid_chat_request_valid_text():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    req = RokidChatRequest(
        request_id="550e8400-e29b-41d4-a716-446655440000",
        device_id="device-abc",
        type=RokidRequestType.TEXT,
        text="Hello",
        timestamp=1708300000,
    )
    assert req.device_id == "device-abc"
    assert req.type == RokidRequestType.TEXT


def test_rokid_chat_request_extra_field_rejected():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    with pytest.raises(ValidationError):
        RokidChatRequest(
            request_id="id", device_id="dev", type=RokidRequestType.TEXT,
            text="hi", timestamp=1708300000, unknown_field="bad",
        )


def test_rokid_chat_request_invalid_type_rejected():
    from rokid_bridge.models import RokidChatRequest
    with pytest.raises(ValidationError):
        RokidChatRequest(
            request_id="id", device_id="dev", type="voice", text="hi", timestamp=1708300000
        )


def test_rokid_chat_request_missing_request_id():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    with pytest.raises(ValidationError):
        RokidChatRequest(
            device_id="dev", type=RokidRequestType.TEXT, text="hi", timestamp=1708300000
        )


def test_rokid_image_payload_extra_field_rejected():
    from rokid_bridge.models import RokidImagePayload
    with pytest.raises(ValidationError):
        RokidImagePayload(data="abc", mime_type="image/jpeg", extra="bad")


def test_rokid_clear_request_valid():
    from rokid_bridge.models import RokidClearRequest
    req = RokidClearRequest(device_id="dev", timestamp=1708300000)
    assert req.device_id == "dev"


def test_health_response_literals():
    from rokid_bridge.models import HealthResponse
    h = HealthResponse()
    assert h.status == "ok"
    assert h.service == "rokid-bridge"


def test_clear_history_response_fields():
    from rokid_bridge.models import ClearHistoryResponse
    r = ClearHistoryResponse(cleared=True, device_id="dev-123")
    assert r.cleared is True
    assert r.device_id == "dev-123"


def test_rokid_chat_request_text_defaults_empty():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    req = RokidChatRequest(
        request_id="id", device_id="dev", type=RokidRequestType.IMAGE, timestamp=1708300000
    )
    assert req.text == ""


def test_rokid_chat_request_image_defaults_none():
    from rokid_bridge.models import RokidChatRequest, RokidRequestType
    req = RokidChatRequest(
        request_id="id", device_id="dev", type=RokidRequestType.TEXT,
        text="hi", timestamp=1708300000
    )
    assert req.image is None
