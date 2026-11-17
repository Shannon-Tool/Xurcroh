from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from xurcroh.errors import ConfigurationError


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextContentPart(RequestModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class ImageURL(RequestModel):
    url: str = Field(min_length=1)


class ImageContentPart(RequestModel):
    type: Literal["image_url"]
    image_url: ImageURL


class AudioData(RequestModel):
    data: str = Field(min_length=1)
    format: Literal["wav"] = "wav"


class AudioContentPart(RequestModel):
    type: Literal["input_audio"]
    input_audio: AudioData


ContentPart = TextContentPart | ImageContentPart | AudioContentPart


class InputMessage(RequestModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

    def plain_text(self) -> str:
        if isinstance(self.content, str):
            if not self.content:
                raise ConfigurationError("message content cannot be empty")
            return self.content
        if not self.content:
            raise ConfigurationError("message content cannot be empty")
        if all(isinstance(part, TextContentPart) for part in self.content):
            return "".join(part.text for part in self.content if isinstance(part, TextContentPart))
        raise ConfigurationError("this model accepts only text message content")


class CompletionRequest(RequestModel):
    model: str
    messages: list[InputMessage] = Field(min_length=1)
    temperature: float = Field(default=1.0, ge=0, le=10)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100_000)
    max_tokens: int = Field(default=128, ge=1, le=1_000_000)
    stream: bool = False
    timeout: float | None = Field(default=None, gt=0)
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    seed: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None

    @classmethod
    def validated(cls, **values: object) -> CompletionRequest:
        try:
            return cls.model_validate(values)
        except ValidationError as error:
            raise ConfigurationError(str(error)) from error


class ImageGenerationRequest(RequestModel):
    model: str
    prompt: str = Field(min_length=1)
    size: str = Field(default="1024x1024", pattern=r"^[1-9][0-9]*x[1-9][0-9]*$")
    image: str | None = None
    mask: str | None = None
    strength: float = Field(default=0.75, gt=0, le=1)
    steps: int = Field(default=30, ge=1, le=1000)
    seed: int | None = Field(default=None, ge=0)
    timeout: float | None = Field(default=None, gt=0)
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")

    @property
    def dimensions(self) -> tuple[int, int]:
        width, height = self.size.split("x")
        return int(width), int(height)


class VideoGenerationRequest(RequestModel):
    model: str
    prompt: str = Field(min_length=1)
    image: str | None = None
    frames: int = Field(default=81, ge=1, le=10_000)
    fps: int = Field(default=24, ge=1, le=240)
    size: str = Field(default="832x480", pattern=r"^[1-9][0-9]*x[1-9][0-9]*$")
    steps: int = Field(default=40, ge=1, le=1000)
    seed: int | None = Field(default=None, ge=0)
    timeout: float | None = Field(default=None, gt=0)
    request_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
