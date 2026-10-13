from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from routetune_core.providers import ApiModelConfig, OpenAICompatibleClient


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    content: str
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    provider_request_id: str | None
    latency_ms: float


@dataclass(frozen=True)
class GenerationOptions:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None


class ModelAdapter(Protocol):
    @property
    def model_ids(self) -> frozenset[str]: ...

    def complete(
        self,
        model_id: str,
        messages: Sequence[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> ChatResponse: ...


class OpenAICompatibleAdapter:
    def __init__(
        self,
        models: Sequence[ApiModelConfig],
        client: OpenAICompatibleClient | None = None,
    ) -> None:
        self._models: Mapping[str, ApiModelConfig] = {item.model_id: item for item in models}
        if len(self._models) != len(models):
            raise ValueError("model_id values must be unique")
        self._client = client or OpenAICompatibleClient()

    @property
    def model_ids(self) -> frozenset[str]:
        return frozenset(self._models)

    def complete(
        self,
        model_id: str,
        messages: Sequence[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> ChatResponse:
        try:
            model = self._models[model_id]
        except KeyError as exc:
            raise ValueError(f"no adapter configuration for model: {model_id}") from exc
        completion = self._client.complete(
            model,
            [{"role": item.role, "content": item.content} for item in messages],
            max_tokens_override=options.max_tokens if options else None,
            temperature_override=options.temperature if options else None,
            top_p_override=options.top_p if options else None,
            top_k_override=options.top_k if options else None,
        )
        return ChatResponse(
            content=completion.content,
            reasoning_content=completion.reasoning_content,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            provider_request_id=completion.request_id,
            latency_ms=completion.latency_ms,
        )
