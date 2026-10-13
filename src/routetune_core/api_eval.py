from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from routetune_core.benchmark import PROMPT_VERSION, BenchmarkItem, extract_choice
from routetune_core.contracts import Annotation, CandidateResult, ContractError, EvaluationRecord
from routetune_core.dataset import write_jsonl
from routetune_core.providers import ApiModelConfig, OpenAICompatibleClient, ProviderError


@dataclass(frozen=True)
class ApiEvaluationTrace:
    trace_id: str
    created_at: str
    item_id: str
    benchmark: str
    benchmark_revision: str
    category: str
    source: str
    model_id: str
    api_model: str
    provider_id: str
    cost_rank: int
    model_fingerprint: str
    prompt_version: str
    input_messages: tuple[dict[str, str], ...]
    expected_answer: str
    status: str
    raw_output: str | None
    raw_reasoning: str | None
    predicted_answer: str | None
    passed: bool | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    provider_request_id: str | None
    error: str | None

    @property
    def cache_key(self) -> tuple[str, str, str, str]:
        return (self.item_id, self.model_id, self.model_fingerprint, self.prompt_version)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_messages"] = list(self.input_messages)
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApiEvaluationTrace:
        messages = data.get("input_messages")
        if not isinstance(messages, list):
            raise ContractError("trace input_messages must be a list")
        return cls(
            trace_id=str(data["trace_id"]),
            created_at=str(data["created_at"]),
            item_id=str(data["item_id"]),
            benchmark=str(data["benchmark"]),
            benchmark_revision=str(data["benchmark_revision"]),
            category=str(data["category"]),
            source=str(data["source"]),
            model_id=str(data["model_id"]),
            api_model=str(data["api_model"]),
            provider_id=str(data["provider_id"]),
            cost_rank=int(data.get("cost_rank", 0)),
            model_fingerprint=str(data["model_fingerprint"]),
            prompt_version=str(data["prompt_version"]),
            input_messages=tuple(messages),
            expected_answer=str(data["expected_answer"]),
            status=str(data["status"]),
            raw_output=data.get("raw_output"),
            raw_reasoning=data.get("raw_reasoning"),
            predicted_answer=data.get("predicted_answer"),
            passed=data.get("passed"),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            latency_ms=float(data["latency_ms"]),
            provider_request_id=data.get("provider_request_id"),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class ApiEvaluationSummary:
    item_count: int
    model_count: int
    new_request_count: int
    cached_result_count: int
    failed_request_count: int
    complete_item_count: int
    trace_path: Path


def run_api_evaluation(
    items: Sequence[BenchmarkItem],
    models: Sequence[ApiModelConfig],
    trace_path: Path,
    *,
    client: OpenAICompatibleClient | None = None,
    max_requests: int = 40,
    min_interval_seconds: float = 0.0,
) -> ApiEvaluationSummary:
    if not items or len(models) < 2:
        raise ContractError("API evaluation needs items and at least two models")
    if max_requests <= 0 or min_interval_seconds < 0:
        raise ContractError("request limit must be positive; interval must be non-negative")

    client = client or OpenAICompatibleClient()
    existing = load_traces(trace_path)
    cache = {trace.cache_key: trace for trace in existing if trace.status == "ok"}
    new_requests = 0
    cached_results = 0
    failures = 0

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        for item in items:
            for model in models:
                key = (item.item_id, model.model_id, model.fingerprint(), PROMPT_VERSION)
                if key in cache:
                    cached_results += 1
                    continue
                if new_requests >= max_requests:
                    return _summary(
                        items, models, trace_path, cache, new_requests, cached_results, failures
                    )
                if new_requests and min_interval_seconds:
                    time.sleep(min_interval_seconds)
                trace = _execute_one(item, model, client)
                handle.write(json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                new_requests += 1
                if trace.status == "ok":
                    cache[key] = trace
                else:
                    failures += 1
    return _summary(items, models, trace_path, cache, new_requests, cached_results, failures)


def _execute_one(
    item: BenchmarkItem, model: ApiModelConfig, client: OpenAICompatibleClient
) -> ApiEvaluationTrace:
    messages = item.messages()
    started = time.monotonic()
    try:
        completion = client.complete(model, messages)
        predicted = extract_choice(completion.content, len(item.options))
        return ApiEvaluationTrace(
            trace_id=f"trace-{uuid.uuid4().hex}",
            created_at=datetime.now(UTC).isoformat(),
            item_id=item.item_id,
            benchmark=item.benchmark,
            benchmark_revision=item.benchmark_revision,
            category=item.category,
            source=item.source,
            model_id=model.model_id,
            api_model=model.api_model,
            provider_id=model.provider.provider_id,
            cost_rank=model.cost_rank,
            model_fingerprint=model.fingerprint(),
            prompt_version=PROMPT_VERSION,
            input_messages=tuple(messages),
            expected_answer=item.answer,
            status="ok",
            raw_output=completion.content,
            raw_reasoning=completion.reasoning_content,
            predicted_answer=predicted,
            passed=predicted == item.answer,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            latency_ms=completion.latency_ms,
            provider_request_id=completion.request_id,
            error=None,
        )
    except ProviderError as exc:
        return ApiEvaluationTrace(
            trace_id=f"trace-{uuid.uuid4().hex}",
            created_at=datetime.now(UTC).isoformat(),
            item_id=item.item_id,
            benchmark=item.benchmark,
            benchmark_revision=item.benchmark_revision,
            category=item.category,
            source=item.source,
            model_id=model.model_id,
            api_model=model.api_model,
            provider_id=model.provider.provider_id,
            cost_rank=model.cost_rank,
            model_fingerprint=model.fingerprint(),
            prompt_version=PROMPT_VERSION,
            input_messages=tuple(messages),
            expected_answer=item.answer,
            status="error",
            raw_output=None,
            raw_reasoning=None,
            predicted_answer=None,
            passed=None,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=(time.monotonic() - started) * 1000,
            provider_request_id=None,
            error=str(exc),
        )


def load_traces(path: Path) -> list[ApiEvaluationTrace]:
    if not path.exists():
        return []
    traces: list[ApiEvaluationTrace] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                traces.append(ApiEvaluationTrace.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, ContractError) as exc:
                raise ContractError(f"invalid trace at {path}:{line_number}: {exc}") from exc
    return traces


def build_evaluation_records(
    items: Sequence[BenchmarkItem],
    models: Sequence[ApiModelConfig],
    traces: Iterable[ApiEvaluationTrace],
) -> list[EvaluationRecord]:
    successful = {trace.cache_key: trace for trace in traces if trace.status == "ok"}
    records: list[EvaluationRecord] = []
    for item in items:
        candidates: list[CandidateResult] = []
        for model in models:
            key = (item.item_id, model.model_id, model.fingerprint(), PROMPT_VERSION)
            trace = successful.get(key)
            if trace is None:
                break
            if trace.raw_output is None or trace.passed is None:
                raise ContractError(f"trace for {item.item_id}/{model.model_id} is incomplete")
            candidates.append(
                CandidateResult(
                    model_id=model.model_id,
                    annotation=Annotation(
                        passed=trace.passed,
                        failure_reason=None if trace.passed else "incorrect_multiple_choice_answer",
                        schema_version="automatic-multiple-choice/v1",
                    ),
                    cost_rank=trace.cost_rank,
                    latency_ms=trace.latency_ms,
                    output_hash=hashlib.sha256(trace.raw_output.encode()).hexdigest(),
                )
            )
        if len(candidates) == len(models):
            records.append(
                EvaluationRecord(
                    request_id=item.item_id,
                    request_text=item.request_text(),
                    candidates=tuple(candidates),
                    metadata={
                        "benchmark": item.benchmark,
                        "benchmark_revision": item.benchmark_revision,
                        "category": item.category,
                        "source": item.source,
                    },
                )
            )
    return records


def write_evaluation_records(records: Sequence[EvaluationRecord], path: Path) -> None:
    if len(records) < 10:
        raise ContractError("at least 10 complete benchmark items are required for router training")
    write_jsonl(records, path)


def _summary(
    items: Sequence[BenchmarkItem],
    models: Sequence[ApiModelConfig],
    trace_path: Path,
    cache: dict[tuple[str, str, str, str], ApiEvaluationTrace],
    new_requests: int,
    cached_results: int,
    failures: int,
) -> ApiEvaluationSummary:
    complete_items = sum(
        all(
            (item.item_id, model.model_id, model.fingerprint(), PROMPT_VERSION) in cache
            for model in models
        )
        for item in items
    )
    return ApiEvaluationSummary(
        item_count=len(items),
        model_count=len(models),
        new_request_count=new_requests,
        cached_result_count=cached_results,
        failed_request_count=failures,
        complete_item_count=complete_items,
        trace_path=trace_path,
    )
