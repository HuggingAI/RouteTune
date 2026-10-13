from __future__ import annotations

import hashlib
import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routetune_core.contracts import ContractError


class ProviderError(RuntimeError):
    """A sanitized model-provider error safe to print or store in a trace."""


@dataclass(frozen=True)
class ProviderConfig:
    provider_id: str
    base_url: str
    api_key_env: str
    extra_headers: Mapping[str, str]

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env, "").strip()
        if not value:
            raise ProviderError(f"required environment variable is not set: {self.api_key_env}")
        return value


@dataclass(frozen=True)
class ApiModelConfig:
    model_id: str
    provider: ProviderConfig
    api_model: str
    cost_rank: int
    max_tokens: int = 8
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    thinking: str | None = None
    reasoning_split: bool | None = None
    max_tokens_field: str = "max_tokens"
    route_label: str = "tier_5"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.cost_rank, bool) or not isinstance(self.cost_rank, int):
            raise ContractError("cost_rank must be an integer")
        if self.cost_rank <= 0:
            raise ContractError("cost_rank must be positive")
        if self.thinking not in {None, "adaptive", "enabled", "disabled"}:
            raise ContractError("thinking must be adaptive, enabled, or disabled")
        if self.max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ContractError("max_tokens_field must be max_tokens or max_completion_tokens")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ContractError("temperature must be between 0 and 2")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ContractError("top_p must be between 0 and 1")
        if self.top_k is not None and self.top_k < 1:
            raise ContractError("top_k must be positive")
        if not self.route_label.strip():
            raise ContractError("route_label must not be blank")

    def fingerprint(self) -> str:
        payload = {
            "api_model": self.api_model,
            "base_url": self.provider.base_url,
            "cost_rank": self.cost_rank,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens_field": self.max_tokens_field,
            "model_id": self.model_id,
            "provider": self.provider.provider_id,
            "reasoning_split": self.reasoning_split,
            "route_label": self.route_label,
            "tags": sorted(self.tags),
            "thinking": self.thinking,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class Completion:
    content: str
    reasoning_content: str | None
    prompt_tokens: int
    completion_tokens: int
    request_id: str | None
    latency_ms: float


def load_model_configs(path: Path) -> list[ApiModelConfig]:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    raw_providers = data.get("providers")
    raw_models = data.get("models")
    if not isinstance(raw_providers, dict) or not isinstance(raw_models, dict):
        raise ContractError("model config requires [providers.*] and [models.*] tables")

    providers: dict[str, ProviderConfig] = {}
    for provider_id, raw in raw_providers.items():
        if not isinstance(raw, dict):
            raise ContractError(f"provider {provider_id} must be a table")
        base_url_env = str(raw.get("base_url_env", "")).strip()
        base_url = os.environ.get(base_url_env, "").strip() if base_url_env else ""
        base_url = base_url or str(raw.get("base_url", "")).strip()
        api_key_env = str(raw.get("api_key_env", "")).strip()
        if not base_url.startswith(("https://", "http://")):
            raise ContractError(f"provider {provider_id} needs an HTTP(S) base_url")
        headers = raw.get("headers", {})
        if not isinstance(headers, dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
        ):
            raise ContractError(f"provider {provider_id} headers must be string pairs")
        providers[provider_id] = ProviderConfig(
            provider_id=provider_id,
            base_url=base_url.rstrip("/"),
            api_key_env=api_key_env,
            extra_headers=headers,
        )

    models: list[ApiModelConfig] = []
    for model_id, raw in raw_models.items():
        if not isinstance(raw, dict):
            raise ContractError(f"model {model_id} must be a table")
        provider_id = str(raw.get("provider", ""))
        if provider_id not in providers:
            raise ContractError(f"model {model_id} references unknown provider {provider_id}")
        raw_tags = raw.get("tags", [])
        if not isinstance(raw_tags, list) or any(
            not isinstance(value, str) or not value.strip() for value in raw_tags
        ):
            raise ContractError(f"model {model_id} tags must be a list of non-empty strings")
        models.append(
            ApiModelConfig(
                model_id=model_id,
                provider=providers[provider_id],
                api_model=str(raw.get("api_model", "")).strip(),
                cost_rank=int(raw.get("cost_rank", 0)),
                max_tokens=int(raw.get("max_tokens", 8)),
                temperature=(
                    float(raw["temperature"]) if raw.get("temperature") is not None else None
                ),
                top_p=float(raw["top_p"]) if raw.get("top_p") is not None else None,
                top_k=int(raw["top_k"]) if raw.get("top_k") is not None else None,
                thinking=str(raw["thinking"]) if raw.get("thinking") is not None else None,
                reasoning_split=(
                    bool(raw["reasoning_split"]) if "reasoning_split" in raw else None
                ),
                max_tokens_field=str(raw.get("max_tokens_field", "max_tokens")),
                route_label=str(raw.get("route_label", "tier_5")).strip(),
                tags=tuple(value.strip() for value in raw_tags),
            )
        )
    if len(models) < 2:
        raise ContractError("routing evaluation requires at least two configured models")
    if any(not model.api_model or model.max_tokens <= 0 for model in models):
        raise ContractError("each model needs api_model and a positive max_tokens")
    if len({model.cost_rank for model in models}) != len(models):
        raise ContractError("candidate models must have unique cost_rank values")
    return models


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without printing, returning, or persisting secret values."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped.removeprefix("export ").strip()
            if "=" not in stripped:
                raise ContractError(f"invalid env line at {path}:{line_number}")
            key, value = stripped.split("=", 1)
            key = key.strip()
            if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
                raise ContractError(f"invalid env key at {path}:{line_number}")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
        timeout_seconds: float = 60,
        max_attempts: int = 1,
    ) -> None:
        self._opener = opener
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts

    def complete(
        self,
        model: ApiModelConfig,
        messages: list[dict[str, str]],
        *,
        max_tokens_override: int | None = None,
        temperature_override: float | None = None,
        top_p_override: float | None = None,
        top_k_override: int | None = None,
        use_model_defaults: bool = True,
    ) -> Completion:
        payload = {
            "model": model.api_model,
            "messages": messages,
            "stream": False,
        }
        max_tokens = max_tokens_override
        if max_tokens is None and use_model_defaults:
            max_tokens = model.max_tokens
        if max_tokens is not None:
            payload[model.max_tokens_field] = max_tokens
        if temperature_override is not None:
            payload["temperature"] = temperature_override
        elif use_model_defaults and model.temperature is not None:
            payload["temperature"] = model.temperature
        if top_p_override is not None:
            payload["top_p"] = top_p_override
        elif use_model_defaults and model.top_p is not None:
            payload["top_p"] = model.top_p
        if top_k_override is not None:
            payload["top_k"] = top_k_override
        elif use_model_defaults and model.top_k is not None:
            payload["top_k"] = model.top_k
        if model.thinking is not None:
            payload["thinking"] = {"type": model.thinking}
        if model.reasoning_split is not None:
            payload["reasoning_split"] = model.reasoning_split
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "RouteTune/0.1",
            **model.provider.extra_headers,
        }
        api_key = model.provider.api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            f"{model.provider.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        for attempt in range(1, self._max_attempts + 1):
            try:
                with self._opener(request, timeout=self._timeout_seconds) as response:
                    body = json.loads(response.read())
                return _parse_completion(body, model, (time.monotonic() - started) * 1000)
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self._max_attempts:
                    raise ProviderError(_sanitized_http_error(exc)) from exc
            except urllib.error.URLError as exc:
                if attempt == self._max_attempts:
                    raise ProviderError(f"provider connection failed: {exc.reason}") from exc
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ProviderError("provider returned an invalid completion response") from exc
            time.sleep(min(2 ** (attempt - 1), 4))
        raise AssertionError("retry loop exited unexpectedly")


def _sanitized_http_error(exc: urllib.error.HTTPError) -> str:
    details: list[str] = []
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after and retry_after.isdigit():
        details.append(f"retry_after={retry_after}s")
    try:
        body = json.loads(exc.read() or b"{}")
    except (json.JSONDecodeError, OSError):
        body = {}
    if isinstance(body, dict):
        base_response = body.get("base_resp")
        error = body.get("error")
        if isinstance(base_response, dict):
            code = base_response.get("status_code")
            if isinstance(code, int):
                details.append(f"provider_code={code}")
        elif isinstance(error, dict):
            error_type = error.get("type")
            if isinstance(error_type, str) and error_type.replace("_", "").isalnum():
                details.append(f"provider_type={error_type}")
    suffix = f" ({', '.join(details)})" if details else ""
    return f"provider returned HTTP {exc.code}{suffix}"


def _parse_completion(
    body: Mapping[str, Any], model: ApiModelConfig, latency_ms: float
) -> Completion:
    if isinstance(body.get("error"), Mapping):
        raise ProviderError("provider returned an error response")
    message = body["choices"][0]["message"]
    content = message["content"]
    reasoning_content = message.get("reasoning_content") or message.get("reasoning")
    usage = body.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    if content is None and isinstance(reasoning_content, str):
        raise ProviderError("provider returned reasoning but no final answer")
    if not isinstance(content, str) or (
        reasoning_content is not None and not isinstance(reasoning_content, str)
    ):
        raise TypeError("completion content is not text")
    if not content.strip():
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            raise ProviderError("provider returned reasoning but no final answer")
        raise ProviderError("provider returned an empty final answer")
    return Completion(
        content=content,
        reasoning_content=reasoning_content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        request_id=str(body["id"]) if body.get("id") is not None else None,
        latency_ms=latency_ms,
    )
