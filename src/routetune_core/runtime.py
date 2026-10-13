from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from time import monotonic

from routetune_core.adapters import ChatMessage, GenerationOptions, ModelAdapter
from routetune_core.label_router import LabelRouter, LabelRoutingDecision
from routetune_core.model_pool import ModelPoolSnapshot, ModelPoolStore
from routetune_core.router import Router
from routetune_core.security import RedactionResult, SecurityPipeline, SensitiveFinding
from routetune_core.trace_store import TraceStore


class RuntimeRequestError(ValueError):
    def __init__(self, message: str, *, trace_id: str | None = None) -> None:
        super().__init__(message)
        self.trace_id = trace_id


class SecurityBlocked(RuntimeRequestError):
    pass


class ModelCallFailed(RuntimeError):
    def __init__(self, message: str, *, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


class RuntimeExecutionFailed(RuntimeError):
    def __init__(self, message: str, *, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


@dataclass(frozen=True)
class RuntimeResponse:
    trace_id: str
    request_id: str
    content: str
    route_label: str | None
    selected_model_id: str
    actual_model_id: str
    route_reason: str
    provider_fallback: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    deployment_id: str | None
    release_id: str | None


class RouteTuneRuntime:
    def __init__(
        self,
        *,
        router: Router | LabelRouter,
        adapter: ModelAdapter,
        security: SecurityPipeline,
        traces: TraceStore,
        model_pool: ModelPoolStore | ModelPoolSnapshot | None = None,
        deployment_id: str | None = None,
        release_id: str | None = None,
    ) -> None:
        if isinstance(router, Router):
            expected = frozenset(router.bundle.profiles)
            missing = expected - adapter.model_ids
            if missing:
                raise ValueError(f"adapter is missing routed models: {', '.join(sorted(missing))}")
        elif model_pool is None:
            raise ValueError("label routing requires a model pool")
        self.router = router
        self.adapter = adapter
        self.security = security
        self.traces = traces
        self.model_pool = model_pool
        self.deployment_id = deployment_id
        self.release_id = release_id

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        request_id: str | None = None,
        model_id_override: str | None = None,
        override_reason: str = "MANUAL_MODEL_OVERRIDE",
        allow_provider_fallback: bool = True,
        generation_options: GenerationOptions | None = None,
        trace_source: str = "api",
    ) -> RuntimeResponse:
        _validate_single_turn(messages)

        request_id = request_id or f"req_{uuid.uuid4().hex}"
        raw_serialized = json.dumps(
            [{"role": item.role, "content": item.content} for item in messages],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            redacted_messages, input_findings = self._redact_messages(messages)
        except Exception as error:
            trace_id = self.traces.start(
                request_id=request_id,
                input_hash=_hash(raw_serialized),
                input_text="[SECURITY_PIPELINE_ERROR]",
                findings=(),
                deployment_id=self.deployment_id,
                release_id=self.release_id,
                request_source=trace_source,
            )
            self.traces.fail(trace_id, _safe_error_code(error))
            raise RuntimeExecutionFailed(
                "input security pipeline failed", trace_id=trace_id
            ) from error
        redacted_serialized = json.dumps(
            [{"role": item.role, "content": item.content} for item in redacted_messages],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        trace_id = self.traces.start(
            request_id=request_id,
            input_hash=_hash(raw_serialized),
            input_text=redacted_serialized,
            findings=input_findings,
            deployment_id=self.deployment_id,
            release_id=self.release_id,
            request_source=trace_source,
        )

        blocked = self.security.blocked_types(
            RedactionResult(text=redacted_serialized, findings=input_findings)
        )
        if blocked:
            self.traces.fail(trace_id, f"INPUT_BLOCKED:{','.join(blocked)}", status="blocked")
            raise SecurityBlocked(
                f"request blocked by security policy: {', '.join(blocked)}", trace_id=trace_id
            )

        route_text = next(item.content for item in redacted_messages if item.role == "user")
        try:
            decision = self.router.decide(request_id, route_text)
            route_label: str | None = None
            if isinstance(decision, LabelRoutingDecision):
                route_label = decision.selected_route_label
                if model_id_override is not None:
                    assert self.model_pool is not None
                    model_entry = self.model_pool.get_entry(model_id_override, require_enabled=True)
                    route_label = model_entry.route_label
                    decision = dataclass_replace(
                        decision,
                        selected_route_label=route_label,
                        used_fallback=False,
                        reason=override_reason,
                    )
                    selected_model_id = model_id_override
                else:
                    assert self.model_pool is not None
                    try:
                        selected_model_id = self.model_pool.resolve(route_label).model_id
                    except ValueError:
                        labels = self.router.bundle.route_labels
                        start = labels.index(route_label)
                        resolved = next(
                            (
                                (label, self.model_pool.resolve(label).model_id)
                                for label in labels[start + 1 :]
                                if self.model_pool.list(
                                    route_label=label, enabled_only=True, online_only=True
                                )
                            ),
                            None,
                        )
                        if resolved is None:
                            raise RuntimeRequestError(
                                f"no model is available at or above route label: {route_label}",
                                trace_id=trace_id,
                            ) from None
                        route_label, selected_model_id = resolved
                        decision = dataclass_replace(
                            decision,
                            selected_route_label=route_label,
                            used_fallback=True,
                            reason="NO_MODEL_IN_TIER_ESCALATED_UPWARD",
                        )
            else:
                selected_model_id = decision.selected_model_id
                if model_id_override is not None:
                    if model_id_override not in self.adapter.model_ids:
                        raise RuntimeRequestError(
                            f"model is not available: {model_id_override}", trace_id=trace_id
                        )
                    decision = dataclass_replace(
                        decision,
                        selected_model_id=model_id_override,
                        used_fallback=False,
                        reason=override_reason,
                    )
                    selected_model_id = model_id_override
            self.traces.set_route(trace_id, decision)
            self.traces.set_selected_model(trace_id, selected_model_id)
        except RuntimeRequestError as error:
            self.traces.fail(trace_id, "INVALID_MODEL_OVERRIDE")
            raise error
        except Exception as error:
            self.traces.fail(trace_id, _safe_error_code(error))
            raise RuntimeExecutionFailed("routing failed", trace_id=trace_id) from error

        started = monotonic()
        actual_model_id = selected_model_id
        provider_fallback = False
        fallback_reason: str | None = None
        attempts = [selected_model_id]
        if allow_provider_fallback:
            attempts.extend(self._fallback_model_ids(selected_model_id, route_label))
        completion = None
        last_error: Exception | None = None
        for attempt_index, candidate_model_id in enumerate(attempts):
            actual_model_id = candidate_model_id
            try:
                if generation_options is None:
                    completion = self.adapter.complete(actual_model_id, redacted_messages)
                else:
                    completion = self.adapter.complete(
                        actual_model_id, redacted_messages, generation_options
                    )
                provider_fallback = attempt_index > 0
                break
            except Exception as error:
                last_error = error
                fallback_reason = fallback_reason or _safe_error_code(error)
        if completion is None:
            assert last_error is not None
            self.traces.fail(trace_id, _safe_error_code(last_error))
            raise ModelCallFailed(
                "all eligible model calls failed", trace_id=trace_id
            ) from last_error

        try:
            output = self.security.redact(completion.content)
        except Exception as error:
            self.traces.fail(trace_id, _safe_error_code(error))
            raise RuntimeExecutionFailed(
                "output security pipeline failed", trace_id=trace_id
            ) from error
        output_blocked = self.security.blocked_types(output)
        if output_blocked:
            self.traces.fail(
                trace_id, f"OUTPUT_BLOCKED:{','.join(output_blocked)}", status="blocked"
            )
            raise SecurityBlocked(
                f"model output blocked by security policy: {', '.join(output_blocked)}",
                trace_id=trace_id,
            )

        total_latency_ms = (monotonic() - started) * 1000
        self.traces.complete(
            trace_id,
            actual_model_id=actual_model_id,
            output_hash=_hash(completion.content),
            output_text=output.text,
            output_findings=output.findings,
            provider_request_id=completion.provider_request_id,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=total_latency_ms,
            fallback_reason=fallback_reason,
        )
        return RuntimeResponse(
            trace_id=trace_id,
            request_id=request_id,
            content=output.text,
            route_label=route_label,
            selected_model_id=selected_model_id,
            actual_model_id=actual_model_id,
            route_reason=decision.reason,
            provider_fallback=provider_fallback,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=total_latency_ms,
            deployment_id=self.deployment_id,
            release_id=self.release_id,
        )

    def _fallback_model_ids(self, selected_model_id: str, route_label: str | None) -> list[str]:
        if self.model_pool is None or not isinstance(self.router, LabelRouter):
            fallback_model_id = self.router.bundle.fallback_model_id
            return [] if selected_model_id == fallback_model_id else [fallback_model_id]
        labels = self.router.bundle.route_labels
        start = labels.index(route_label) if route_label in labels else len(labels) - 1
        candidates: list[str] = []
        for label in labels[start:]:
            entries = self.model_pool.list(route_label=label, enabled_only=True, online_only=True)
            candidates.extend(
                item["model_id"]
                for item in entries
                if item["model_id"] != selected_model_id and item["model_id"] not in candidates
            )
        return candidates

    def _redact_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[list[ChatMessage], tuple[SensitiveFinding, ...]]:
        redacted: list[ChatMessage] = []
        findings: list[SensitiveFinding] = []
        for message in messages:
            result = self.security.redact(message.content)
            redacted.append(ChatMessage(role=message.role, content=result.text))
            findings.extend(result.findings)
        return redacted, tuple(findings)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_single_turn(messages: list[ChatMessage]) -> None:
    roles = [item.role for item in messages]
    if roles not in (["user"], ["system", "user"]):
        raise RuntimeRequestError("single-turn messages must be [user] or [system, user]")
    if any(not item.content.strip() for item in messages):
        raise RuntimeRequestError("message content must not be blank")


def _safe_error_code(error: Exception) -> str:
    message = str(error).strip()
    if error.__class__.__name__ == "ProviderError" and message:
        return f"PROVIDER_ERROR:{message[:300]}"
    return f"{error.__class__.__name__.upper()}"
