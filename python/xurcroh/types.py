from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(ResponseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionUsage(ResponseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ExecutionMetadata(ResponseModel):
    request_id: str
    engine: str
    worker_ids: list[str]
    duration_ms: float = Field(ge=0)
    security_mode: str
    seed: int | None = Field(default=None, ge=0)


class ChatChoice(ResponseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "cancelled"]


class ChatCompletion(ResponseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: CompletionUsage
    execution: ExecutionMetadata


class ChatDelta(ResponseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatChunkChoice(ResponseModel):
    index: int
    delta: ChatDelta
    finish_reason: Literal["stop", "length", "cancelled"] | None = None


class ChatCompletionChunk(ResponseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
    usage: CompletionUsage | None = None
    execution: ExecutionMetadata | None = None


class GeneratedImage(ResponseModel):
    b64_json: str
    mime_type: Literal["image/png"] = "image/png"


class ImageGenerationResponse(ResponseModel):
    created: int
    data: list[GeneratedImage]
    execution: ExecutionMetadata


class GeneratedVideo(ResponseModel):
    b64_json: str
    mime_type: Literal["video/mp4"] = "video/mp4"


class VideoGenerationResponse(ResponseModel):
    created: int
    data: list[GeneratedVideo]
    execution: ExecutionMetadata
