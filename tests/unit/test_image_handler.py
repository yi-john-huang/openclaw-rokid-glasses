import base64

import pytest


def _make_image(data: str = "aGVsbG8=", mime_type: str = "image/jpeg"):
    from rokid_bridge.models import RokidImagePayload
    return RokidImagePayload(data=data, mime_type=mime_type)


def test_validate_and_build_jpeg_valid():
    from rokid_bridge.image_handler import validate_and_build_image_part
    part = validate_and_build_image_part(_make_image(), "low")
    assert part["type"] == "image_url"
    assert "image_url" in part
    assert part["image_url"]["detail"] == "low"
    assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_validate_and_build_png_valid():
    from rokid_bridge.image_handler import validate_and_build_image_part
    part = validate_and_build_image_part(_make_image(mime_type="image/png"), "high")
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_unsupported_mime_raises():
    from rokid_bridge.image_handler import ImageValidationError, validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    img = RokidImagePayload.model_construct(data="aGVsbG8=", mime_type="image/gif")
    with pytest.raises(ImageValidationError, match="Unsupported"):
        validate_and_build_image_part(img, "low")


def test_invalid_base64_raises():
    from rokid_bridge.image_handler import ImageValidationError, validate_and_build_image_part
    from rokid_bridge.models import RokidImagePayload
    img = RokidImagePayload.model_construct(data="not!valid!base64!!!", mime_type="image/jpeg")
    with pytest.raises(ImageValidationError, match="base64"):
        validate_and_build_image_part(img, "low")


def test_oversized_image_raises_413():
    from rokid_bridge.image_handler import (
        MAX_IMAGE_BYTES,
        ImageValidationError,
        validate_and_build_image_part,
    )
    from rokid_bridge.models import RokidImagePayload
    # Create base64 that decodes to > MAX_IMAGE_BYTES
    oversized = base64.b64encode(b"x" * (MAX_IMAGE_BYTES + 1)).decode()
    img = RokidImagePayload.model_construct(data=oversized, mime_type="image/jpeg")
    with pytest.raises(ImageValidationError) as exc_info:
        validate_and_build_image_part(img, "low")
    assert exc_info.value.status_code == 413


def test_exact_max_size_is_allowed():
    from rokid_bridge.image_handler import MAX_IMAGE_BYTES, validate_and_build_image_part
    exact = base64.b64encode(b"x" * MAX_IMAGE_BYTES).decode()
    RokidImagePayload = __import__(
        "rokid_bridge.models", fromlist=["RokidImagePayload"]
    ).RokidImagePayload
    img_ok = RokidImagePayload.model_construct(data=exact, mime_type="image/jpeg")
    validate_and_build_image_part(img_ok, "low")  # must not raise


def test_build_image_content_returns_list():
    from rokid_bridge.image_handler import build_image_content
    parts = build_image_content(_make_image(), "low")
    assert len(parts) == 1
    assert parts[0]["type"] == "image_url"


def test_build_text_with_image_content_has_text_and_image():
    from rokid_bridge.image_handler import build_text_with_image_content
    parts = build_text_with_image_content("Describe this", _make_image(), "low")
    assert len(parts) == 2
    types = {p["type"] for p in parts}
    assert "text" in types
    assert "image_url" in types


def test_build_text_with_image_content_text_value():
    from rokid_bridge.image_handler import build_text_with_image_content
    parts = build_text_with_image_content("What is this?", _make_image(), "low")
    text_parts = [p for p in parts if p["type"] == "text"]
    assert text_parts[0]["text"] == "What is this?"
