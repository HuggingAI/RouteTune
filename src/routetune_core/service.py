from __future__ import annotations

import os
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from routetune_core.adapters import ChatMessage, GenerationOptions
from routetune_core.batches import BatchManager, BatchStore
from routetune_core.contracts import Annotation, ContractError
from routetune_core.label_router import (
    LabelRouter,
    resolve_training_strategy,
    training_strategies,
)
from routetune_core.model_pool import (
    ModelAvailabilityMonitor,
    ModelPoolEntry,
    ModelPoolStore,
    RegistryAdapter,
    check_model_availability,
)
from routetune_core.releases import DeploymentManager, ReleaseStore
from routetune_core.runtime import (
    ModelCallFailed,
    RouteTuneRuntime,
    RuntimeExecutionFailed,
    RuntimeRequestError,
    SecurityBlocked,
)
from routetune_core.security_store import SecurityDictionaryStore
from routetune_core.trace_store import TraceStore


class MessageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user"]
    content: str = Field(min_length=1, max_length=100_000)


class RoutingMetadataInput(BaseModel):
    """Caller-supplied dimensions used only for grouped operational reporting."""

    model_config = ConfigDict(extra="forbid")
    language: str | None = Field(default=None, min_length=1, max_length=100)
    scenario: str | None = Field(default=None, min_length=1, max_length=100)
    tenant: str | None = Field(default=None, min_length=1, max_length=100)
    segment: str | None = Field(default=None, min_length=1, max_length=100)


class CompletionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "auto"
    messages: list[MessageInput] = Field(min_length=1, max_length=2)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=1_000_000)
    request_id: str | None = Field(default=None, min_length=1, max_length=200)
    metadata: RoutingMetadataInput | None = None

    @model_validator(mode="after")
    def validate_single_turn(self) -> CompletionInput:
        roles = [item.role for item in self.messages]
        if roles not in (["user"], ["system", "user"]):
            raise ValueError("single-turn messages must be [user] or [system, user]")
        return self


class AnnotationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    failure_reason: str | None = Field(default=None, max_length=200)
    score: float | None = Field(default=None, ge=1, le=5)
    comment: str | None = Field(default=None, max_length=2_000)
    annotator_id: str | None = Field(default=None, max_length=200)


class ModelPoolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=100)
    provider_id: str = Field(min_length=1, max_length=100)
    base_url: str = Field(min_length=1, max_length=500)
    api_key_env: str = Field(default="", max_length=100)
    api_model: str = Field(min_length=1, max_length=300)
    cost_rank: int = Field(ge=1, le=10_000)
    route_label: str = Field(min_length=1, max_length=100)
    tags: list[str] = Field(default_factory=list, max_length=30)
    enabled: bool = True
    thinking: Literal["adaptive", "enabled", "disabled"] | None = None
    reasoning_split: bool | None = None
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    max_tokens: int = Field(default=1024, ge=1, le=1_000_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=1_000_000)
    # Output-only operational fields are accepted on PUT so a GET response can
    # be edited and sent back without the client having to strip them.
    availability_status: Literal["unchecked", "available", "unavailable"] | None = None
    availability_checked_at: str | None = None
    availability_error: str | None = None
    availability_latency_ms: float | None = None


class BatchImportInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    jsonl: str = Field(min_length=1, max_length=10_000_000)
    mode: Literal["route", "compare"] = "route"
    model_tag: str | None = Field(default=None, max_length=100)
    model_ids: list[str] = Field(default_factory=list, max_length=20)
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=1_000_000)


class BatchFeedbackInput(AnnotationInput):
    trace_id: str = Field(min_length=1, max_length=100)
    retry_on_fail: bool = False
    retry_model_id: str | None = Field(default=None, max_length=100)


class RouterTrainingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["auto", "tfidf_logreg", "fasttext", "bert"] = "auto"


class ProviderCredentialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr


class SecurityDictionaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=200)
    terms: dict[str, list[str]]


class SecurityDictionaryStateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool


class ReleaseCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notes: str | None = Field(default=None, max_length=2_000)


_PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "siliconflow": "SILICONFLOW_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


def _model_check_interval_seconds() -> float | None:
    raw = os.environ.get("ROUTETUNE_MODEL_CHECK_INTERVAL_SECONDS", "1800").strip()
    try:
        value = float(raw)
    except ValueError:
        value = 1_800
    if value <= 0:
        return None
    return max(value, 60)


def create_app(
    runtime: RouteTuneRuntime,
    traces: TraceStore,
    *,
    model_pool: ModelPoolStore | None = None,
    batch_manager: BatchManager | None = None,
    security_dictionaries: SecurityDictionaryStore | None = None,
    deployments: DeploymentManager | None = None,
    api_key_env: str = "ROUTETUNE_API_KEY",
) -> FastAPI:
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="routetune-batch")
    training_batches: set[str] = set()
    training_lock = Lock()
    availability_monitor: ModelAvailabilityMonitor | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if availability_monitor is not None:
            availability_monitor.start()
        try:
            yield
        finally:
            if availability_monitor is not None:
                availability_monitor.stop()
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="RouteTune",
        version="0.1.0",
        description="Route requests to the lowest-cost model likely to meet the quality bar.",
        lifespan=lifespan,
    )
    static_dir = Path(__file__).with_name("static")
    model_pool = model_pool or ModelPoolStore(traces.path)
    security_dictionaries = security_dictionaries or SecurityDictionaryStore(traces.path)
    runtime.security.set_managed_dictionary_terms(security_dictionaries.combined_enabled_terms())
    if deployments is None and isinstance(runtime.router, LabelRouter):
        try:
            deployments = DeploymentManager(
                draft_runtime=runtime,
                model_pool=model_pool,
                dictionaries=security_dictionaries,
                traces=traces,
                releases=ReleaseStore(traces.path),
            )
        except TypeError:
            # Tests and embedders may use a custom adapter. They keep the legacy
            # single-runtime behavior until they provide a versioned deployment manager.
            deployments = None
    if batch_manager is None and isinstance(runtime.router, LabelRouter):
        batch_manager = BatchManager(
            store=BatchStore(traces.path),
            runtime=runtime,
            router=runtime.router,
            model_pool=model_pool,
            security=runtime.security,
            traces=traces,
        )
    model_pool.backfill_availability_from_batch_runs()
    monitor_client = (
        runtime.adapter.client if isinstance(runtime.adapter, RegistryAdapter) else None
    )
    monitor_interval = _model_check_interval_seconds()
    if monitor_client is not None and monitor_interval is not None:
        availability_monitor = ModelAvailabilityMonitor(
            model_pool,
            monitor_client,
            interval_seconds=monitor_interval,
        )
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = os.environ.get(api_key_env, "").strip()
        supplied = authorization or ""
        if expected and not secrets.compare_digest(supplied, f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    protected = [Depends(authorize)]

    @app.get("/health")
    def health() -> dict[str, object]:
        serving_runtime = deployments.active_runtime if deployments else runtime
        is_label_router = isinstance(serving_runtime.router, LabelRouter)
        pool_models = (
            serving_runtime.model_pool.list(enabled_only=True, online_only=True)
            if is_label_router and serving_runtime.model_pool is not None
            else []
        )
        online_models = (
            sorted(item["model_id"] for item in pool_models)
            if is_label_router
            else sorted(serving_runtime.adapter.model_ids)
        )
        routing_targets: dict[str, str] = {}
        if is_label_router and serving_runtime.model_pool is not None:
            for label in serving_runtime.router.bundle.route_labels:
                try:
                    routing_targets[label] = serving_runtime.model_pool.resolve(label).model_id
                except ValueError:
                    continue
        demo_models = sorted(
            str(item["model_id"]) for item in pool_models if "demo" in item.get("tags", [])
        )
        configured_real_models = sorted(
            str(item["model_id"])
            for item in pool_models
            if "demo" not in item.get("tags", [])
            and (not item.get("api_key_env") or item.get("api_key_available"))
        )
        if not is_label_router:
            provider_mode = "legacy"
        elif configured_real_models and demo_models:
            provider_mode = "live_with_demo_fallback"
        elif configured_real_models:
            provider_mode = "live"
        elif demo_models:
            provider_mode = "demo"
        else:
            provider_mode = "unavailable"
        return {
            "status": "ok",
            "bundle_version": serving_runtime.router.bundle.version,
            "cold_start": serving_runtime.router.bundle.training_record_count == 0,
            "route_labels": (serving_runtime.router.bundle.route_labels if is_label_router else []),
            "fallback_label": (
                serving_runtime.router.bundle.fallback_label if is_label_router else None
            ),
            "fallback_model": (
                None if is_label_router else serving_runtime.router.bundle.fallback_model_id
            ),
            "models": sorted(serving_runtime.adapter.model_ids),
            "online_models": online_models,
            "routing_targets": routing_targets,
            "provider_mode": provider_mode,
            "demo_models": demo_models,
            "configured_real_models": configured_real_models,
            "deployment_id": deployments.deployment_id if deployments else None,
            "release_id": deployments.active_release_id if deployments else None,
        }

    @app.post("/v1/chat/completions", dependencies=protected)
    def chat_completion(payload: CompletionInput) -> dict[str, object]:
        if payload.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported in the MVP")
        try:
            result = (deployments or runtime).complete(
                [ChatMessage(role=item.role, content=item.content) for item in payload.messages],
                request_id=payload.request_id,
                trace_metadata=(
                    payload.metadata.model_dump(exclude_none=True) if payload.metadata else None
                ),
                generation_options=(
                    GenerationOptions(
                        max_tokens=payload.max_tokens,
                        temperature=payload.temperature,
                        top_p=payload.top_p,
                        top_k=payload.top_k,
                    )
                    if any(
                        value is not None
                        for value in (
                            payload.max_tokens,
                            payload.temperature,
                            payload.top_p,
                            payload.top_k,
                        )
                    )
                    else None
                ),
            )
        except SecurityBlocked as exc:
            raise HTTPException(
                status_code=422, detail={"message": str(exc), "trace_id": exc.trace_id}
            ) from exc
        except RuntimeRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ModelCallFailed as exc:
            raise HTTPException(
                status_code=502,
                detail={"message": "upstream model call failed", "trace_id": exc.trace_id},
            ) from exc
        except RuntimeExecutionFailed as exc:
            raise HTTPException(
                status_code=500,
                detail={"message": "RouteTune runtime failed", "trace_id": exc.trace_id},
            ) from exc

        return {
            "id": result.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.actual_model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
            "routetune": {
                "trace_id": result.trace_id,
                "route_label": result.route_label,
                "selected_model": result.selected_model_id,
                "actual_model": result.actual_model_id,
                "route_reason": result.route_reason,
                "provider_fallback": result.provider_fallback,
                "latency_ms": result.latency_ms,
                "deployment_id": result.deployment_id,
                "release_id": result.release_id,
            },
        }

    @app.get("/v1/traces", dependencies=protected)
    def list_traces(limit: Annotated[int, Query(ge=1, le=200)] = 50) -> dict[str, object]:
        return {"data": traces.list(limit)}

    @app.get("/v1/traces/{trace_id}", dependencies=protected)
    def get_trace(trace_id: str) -> dict[str, object]:
        trace = traces.get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return trace

    @app.post("/v1/traces/{trace_id}/annotations", dependencies=protected)
    def create_annotation(trace_id: str, payload: AnnotationInput) -> dict[str, object]:
        try:
            annotation = Annotation(
                passed=payload.passed,
                failure_reason=payload.failure_reason,
                score=payload.score,
                comment=payload.comment,
            )
            return traces.annotate(trace_id, annotation, annotator_id=payload.annotator_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="trace not found") from exc
        except (ContractError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/stats", dependencies=protected)
    def stats(
        window: Annotated[Literal["24h", "7d", "30d", "all"], Query()] = "all",
        source: Annotated[Literal["api", "batch", "all"], Query()] = "api",
    ) -> dict[str, object]:
        return traces.stats(window, source)

    @app.get("/v1/releases", dependencies=protected)
    def list_releases() -> dict[str, object]:
        manager = _require_deployments(deployments)
        return {
            "data": manager.releases.list(),
            "active_release_id": manager.active_release_id,
        }

    @app.get("/v1/releases/{release_id}", dependencies=protected)
    def get_release(release_id: str) -> dict[str, object]:
        manager = _require_deployments(deployments)
        release = manager.releases.get(release_id)
        if release is None:
            raise HTTPException(status_code=404, detail="release not found")
        return release

    @app.get("/v1/releases/{release_id}/artifact", dependencies=protected)
    def download_release_artifact(release_id: str) -> StreamingResponse:
        manager = _require_deployments(deployments)
        try:
            release = manager.releases.load(release_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="release not found") from exc
        bundle_path = Path(str(release["bundle_path"]))
        files = [bundle_path / "label-bundle.joblib", bundle_path / "label-manifest.json"]
        if any(not path.is_file() for path in files):
            raise HTTPException(status_code=404, detail="release artifact is incomplete")
        archive = BytesIO()
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle_zip:
            for path in files:
                bundle_zip.write(path, arcname=path.name)
        archive.seek(0)
        filename = f"{release['bundle_version']}.zip"
        return StreamingResponse(
            archive,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.get("/v1/deployments/production", dependencies=protected)
    def get_production_deployment() -> dict[str, object]:
        manager = _require_deployments(deployments)
        deployment = manager.releases.deployment(manager.deployment_id)
        assert deployment is not None
        return deployment

    @app.post("/v1/releases/candidates", dependencies=protected)
    def create_release_candidate(payload: ReleaseCandidateInput) -> dict[str, object]:
        manager = _require_deployments(deployments)
        return manager.create_candidate(manager.draft_runtime.router.bundle, notes=payload.notes)

    @app.post("/v1/releases/{release_id}/publish", dependencies=protected)
    def publish_release(release_id: str) -> dict[str, object]:
        manager = _require_deployments(deployments)
        try:
            deployment = manager.publish(release_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="release not found") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deployment": deployment, "release": manager.releases.get(release_id)}

    @app.get("/v1/model-pool", dependencies=protected)
    def list_model_pool() -> dict[str, object]:
        monitor = (
            availability_monitor.status()
            if availability_monitor is not None
            else {"enabled": False, "interval_seconds": None}
        )
        return {"data": model_pool.list(), "availability_monitor": monitor}

    @app.post("/v1/model-pool", dependencies=protected)
    def create_model(payload: ModelPoolInput) -> dict[str, object]:
        if model_pool.get(payload.model_id) is not None:
            raise HTTPException(status_code=409, detail="model_id already exists")
        _validate_route_label(runtime, payload.route_label)
        try:
            return model_pool.create(_model_pool_entry(payload), replace=False)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/v1/model-pool/{model_id}", dependencies=protected)
    def update_model(model_id: str, payload: ModelPoolInput) -> dict[str, object]:
        if payload.model_id != model_id:
            raise HTTPException(status_code=409, detail="model_id cannot be changed")
        _validate_route_label(runtime, payload.route_label)
        try:
            return model_pool.update(model_id, _model_pool_entry(payload))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/model-pool/{model_id}/check", dependencies=protected)
    def check_model(model_id: str) -> dict[str, object]:
        try:
            entry = model_pool.get_entry(model_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model not found") from exc
        client = runtime.adapter.client if isinstance(runtime.adapter, RegistryAdapter) else None
        if client is None:
            raise HTTPException(status_code=409, detail="model availability check is unavailable")
        return check_model_availability(model_pool, client, entry.model_id)

    @app.post("/v1/providers/{provider_id}/credential", dependencies=protected)
    def set_provider_credential(
        provider_id: str, payload: ProviderCredentialInput
    ) -> dict[str, object]:
        api_key_env = _PROVIDER_KEY_ENV.get(provider_id)
        if api_key_env is None:
            matching = [item for item in model_pool.list() if item["provider_id"] == provider_id]
            if not matching:
                raise HTTPException(status_code=404, detail="unknown provider")
            api_key_env = str(matching[0]["api_key_env"])
        if not api_key_env:
            raise HTTPException(status_code=409, detail="provider uses no authentication")
        value = payload.api_key.get_secret_value().strip()
        if not value:
            raise HTTPException(status_code=422, detail="api_key must not be blank")
        os.environ[api_key_env] = value
        return {
            "provider_id": provider_id,
            "api_key_env": api_key_env,
            "available": True,
            "persistence": "process_only",
        }

    @app.get("/v1/providers", dependencies=protected)
    def list_providers() -> dict[str, object]:
        providers: dict[str, dict[str, object]] = {
            "openrouter": {
                "provider_id": "openrouter",
                "display_name": "OpenRouter",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
            },
            "siliconflow": {
                "provider_id": "siliconflow",
                "display_name": "硅基流动",
                "base_url": "https://api.siliconflow.cn/v1",
                "api_key_env": "SILICONFLOW_API_KEY",
            },
            "deepseek": {
                "provider_id": "deepseek",
                "display_name": "DeepSeek",
                "base_url": "https://api.deepseek.com",
                "api_key_env": "DEEPSEEK_API_KEY",
            },
            "minimax": {
                "provider_id": "minimax",
                "display_name": "MiniMax",
                "base_url": os.environ.get(
                    "MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"
                ),
                "api_key_env": "MINIMAX_API_KEY",
            },
        }
        for model in model_pool.list():
            providers.setdefault(
                str(model["provider_id"]),
                {
                    "provider_id": model["provider_id"],
                    "display_name": model["provider_id"],
                    "base_url": model["base_url"],
                    "api_key_env": model["api_key_env"],
                },
            )
        for provider in providers.values():
            key_env = str(provider["api_key_env"])
            provider["api_key_available"] = bool(key_env and os.environ.get(key_env, ""))
            provider["auth_mode"] = "bearer" if key_env else "none"
        return {"data": list(providers.values())}

    @app.get("/v1/security/dictionaries", dependencies=protected)
    def list_security_dictionaries() -> dict[str, object]:
        return {"data": security_dictionaries.list()}

    @app.post("/v1/security/dictionaries/import", dependencies=protected)
    def import_security_dictionary(payload: SecurityDictionaryInput) -> dict[str, object]:
        try:
            result = security_dictionaries.create(payload.name, payload.terms)
            runtime.security.set_managed_dictionary_terms(
                security_dictionaries.combined_enabled_terms()
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/v1/security/dictionaries/{dictionary_id}", dependencies=protected)
    def update_security_dictionary(
        dictionary_id: str, payload: SecurityDictionaryStateInput
    ) -> dict[str, object]:
        try:
            result = security_dictionaries.set_enabled(dictionary_id, payload.enabled)
            runtime.security.set_managed_dictionary_terms(
                security_dictionaries.combined_enabled_terms()
            )
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dictionary not found") from exc

    @app.get("/v1/batches", dependencies=protected)
    def list_batches() -> dict[str, object]:
        manager = _require_batch_manager(batch_manager)
        return {"data": manager.store.list_batches()}

    @app.post("/v1/batches/import", dependencies=protected)
    def import_batch(payload: BatchImportInput) -> dict[str, object]:
        manager = _require_batch_manager(batch_manager)
        try:
            return manager.import_jsonl(
                name=payload.name,
                content=payload.jsonl,
                mode=payload.mode,
                model_tag=payload.model_tag,
                model_ids=payload.model_ids,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/batches/{batch_id}", dependencies=protected)
    def get_batch(batch_id: str) -> dict[str, object]:
        manager = _require_batch_manager(batch_manager)
        batch = manager.details(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="batch not found")
        return batch

    @app.post("/v1/batches/{batch_id}/run", dependencies=protected)
    def run_batch(batch_id: str) -> dict[str, object]:
        manager = _require_batch_manager(batch_manager)
        batch = manager.store.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="batch not found")
        if batch["status"] != "ready":
            raise HTTPException(status_code=409, detail="batch has already been started")
        manager.store.set_batch_status(batch_id, "queued")
        executor.submit(manager.run, batch_id)
        return {"batch_id": batch_id, "status": "queued"}

    @app.post("/v1/batches/{batch_id}/items/{item_id}/feedback", dependencies=protected)
    def batch_feedback(
        batch_id: str, item_id: str, payload: BatchFeedbackInput
    ) -> dict[str, object]:
        manager = _require_batch_manager(batch_manager)
        batch = manager.details(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="batch not found")
        item = next((value for value in batch["items"] if value["item_id"] == item_id), None)
        if item is None or not any(run["trace_id"] == payload.trace_id for run in item["runs"]):
            raise HTTPException(status_code=404, detail="batch item trace not found")
        try:
            annotation = traces.annotate(
                payload.trace_id,
                Annotation(
                    passed=payload.passed,
                    failure_reason=payload.failure_reason,
                    score=payload.score,
                    comment=payload.comment,
                ),
                annotator_id=payload.annotator_id,
            )
            if not payload.passed and payload.retry_on_fail:
                executor.submit(manager.retry_item, batch_id, item_id, payload.retry_model_id)
        except (KeyError, ValueError, ContractError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "annotation": annotation,
            "retry_queued": not payload.passed and payload.retry_on_fail,
        }

    @app.post("/v1/batches/{batch_id}/items/{item_id}/retry", dependencies=protected)
    def retry_batch_item(
        batch_id: str, item_id: str, model_id: str | None = None
    ) -> dict[str, object]:
        manager = _require_batch_manager(batch_manager)
        batch = manager.store.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="batch not found")
        item = next((value for value in batch["items"] if value["item_id"] == item_id), None)
        if item is None:
            raise HTTPException(status_code=404, detail="batch item not found")
        if item["status"] in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="batch item retry is already queued")
        manager.store.set_item_status(item_id, "queued")
        manager.store.set_batch_status(batch_id, "queued")
        executor.submit(manager.retry_item, batch_id, item_id, model_id)
        return {"batch_id": batch_id, "item_id": item_id, "status": "queued"}

    @app.post("/v1/batches/{batch_id}/train", dependencies=protected)
    def train_batch(
        batch_id: str, payload: RouterTrainingInput | None = None
    ) -> dict[str, object]:
        manager = _require_batch_manager(batch_manager)
        strategy = payload.strategy if payload is not None else "auto"
        batch = manager.store.get_batch(batch_id)
        if batch is None:
            raise HTTPException(status_code=404, detail="batch not found")
        try:
            trainer_id = resolve_training_strategy(strategy, int(batch["item_count"]))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if deployments is not None:
            existing = deployments.releases.latest_for_source_batch(batch_id, trainer_id)
            if existing is not None:
                return {
                    "batch_id": batch_id,
                    "bundle_version": existing["bundle_version"],
                    "release_id": existing["release_id"],
                    "training_record_count": existing["training_record_count"],
                    "published": existing["status"] == "published",
                    "requested_strategy": strategy,
                    "trainer_id": existing["trainer_id"],
                    "reused": True,
                }
        with training_lock:
            if batch_id in training_batches:
                raise HTTPException(status_code=409, detail="batch training is already running")
            training_batches.add(batch_id)
        try:
            result = manager.train_candidate_bundle(batch_id, strategy=strategy)
            if deployments is not None:
                release = deployments.create_candidate_from_path(
                    Path(str(result["output"])), source_batch_id=batch_id
                )
                result["release_id"] = release["release_id"]
            return result
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="batch not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            with training_lock:
                training_batches.discard(batch_id)

    @app.get("/v1/router/trainers", dependencies=protected)
    def list_router_trainers() -> dict[str, object]:
        return {
            "router_family": "probabilistic-classifier",
            "concept": "text classifier with label probabilities and confidence fallback",
            "data": training_strategies(),
        }

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/admin")

    @app.get("/admin", include_in_schema=False)
    def admin() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    return app


def _model_pool_entry(payload: ModelPoolInput) -> ModelPoolEntry:
    extra_headers = (
        {
            "HTTP-Referer": "https://github.com/HuggingAI/RouteTune",
            "X-Title": "RouteTune",
        }
        if payload.provider_id == "openrouter"
        else {}
    )
    return ModelPoolEntry(
        model_id=payload.model_id,
        provider_id=payload.provider_id,
        base_url=payload.base_url,
        api_key_env=payload.api_key_env,
        api_model=payload.api_model,
        cost_rank=payload.cost_rank,
        max_tokens=payload.max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        route_label=payload.route_label,
        tags=tuple(sorted(set(payload.tags))),
        enabled=payload.enabled,
        thinking=payload.thinking,
        reasoning_split=payload.reasoning_split,
        max_tokens_field=payload.max_tokens_field,
        extra_headers=extra_headers,
    )


def _require_batch_manager(manager: BatchManager | None) -> BatchManager:
    if manager is None:
        raise HTTPException(status_code=409, detail="batch workflow requires label routing")
    return manager


def _require_deployments(manager: DeploymentManager | None) -> DeploymentManager:
    if manager is None:
        raise HTTPException(
            status_code=409,
            detail="versioned deployments require label routing with the registry adapter",
        )
    return manager


def _validate_route_label(runtime: RouteTuneRuntime, route_label: str) -> None:
    if (
        isinstance(runtime.router, LabelRouter)
        and route_label not in runtime.router.bundle.route_labels
    ):
        raise HTTPException(
            status_code=422,
            detail=f"route_label must be one of: {', '.join(runtime.router.bundle.route_labels)}",
        )
