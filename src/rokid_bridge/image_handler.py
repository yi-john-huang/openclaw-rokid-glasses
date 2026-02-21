import base64

from rokid_bridge.models import (
    ContentPart,
    ImageContentPart,
    ImageUrlDetail,
    RokidImagePayload,
    TextContentPart,
)

ALLOWED_MIME_TYPES: frozenset[str] = frozenset({"image/jpeg", "image/png"})
MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB


class ImageValidationError(ValueError):
    def __init__(self, detail: str, status_code: int = 422) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def validate_and_build_image_part(image: RokidImagePayload, detail: str) -> ImageContentPart:
    if image.mime_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError("Unsupported image format")
    try:
        decoded = base64.b64decode(image.data, validate=True)
    except Exception:
        raise ImageValidationError("Invalid base64 image data") from None
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ImageValidationError("Image too large", status_code=413)
    url = f"data:{image.mime_type};base64,{image.data}"
    return ImageContentPart(type="image_url", image_url=ImageUrlDetail(url=url, detail=detail))


def build_image_content(image: RokidImagePayload, detail: str) -> list[ContentPart]:
    return [validate_and_build_image_part(image, detail)]


def build_text_with_image_content(
    text: str, image: RokidImagePayload, detail: str
) -> list[ContentPart]:
    text_part = TextContentPart(type="text", text=text)
    image_part = validate_and_build_image_part(image, detail)
    return [text_part, image_part]
