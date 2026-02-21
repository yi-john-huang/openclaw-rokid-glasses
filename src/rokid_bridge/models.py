from enum import StrEnum
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RokidRequestType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TEXT_WITH_IMAGE = "text_with_image"


class RokidImagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str = Field(..., description="Base64-encoded image bytes")
    mime_type: str = Field(..., description="MIME type, e.g. 'image/jpeg'")


class RokidChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(..., description="UUID v4 from Rokid platform")
    device_id: str = Field(..., description="Opaque device identifier")
    type: RokidRequestType
    text: str = Field(default="", description="Speech-to-text result (may be empty for image-only)")
    image: RokidImagePayload | None = Field(default=None)
    timestamp: int = Field(..., description="Unix seconds; used for replay protection")

    @field_validator("text")
    @classmethod
    def text_not_blank_when_required(cls, v: str, info: object) -> str:
        return v


class RokidClearRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(..., description="Device whose history to clear")
    timestamp: int = Field(..., description="Unix seconds; used for replay protection")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["rokid-bridge"] = "rokid-bridge"


class ClearHistoryResponse(BaseModel):
    cleared: bool
    device_id: str


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrlDetail(TypedDict):
    url: str
    detail: str


class ImageUrlWrapper(TypedDict):
    image_url: ImageUrlDetail


class ImageContentPart(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrlDetail


ContentPart = TextContentPart | ImageContentPart


class OpenAIMessage(TypedDict, total=False):
    role: str
    content: str | list[ContentPart]


class UpstreamRequest(TypedDict, total=False):
    model: str
    messages: list[OpenAIMessage]
    stream: bool
    agent_id: str
