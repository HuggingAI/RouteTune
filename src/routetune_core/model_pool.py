from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from routetune_core.adapters import ChatMessage, ChatResponse, GenerationOptions, ModelAdapter
from routetune_core.providers import (
    ApiModelConfig,
    OpenAICompatibleClient,
    ProviderConfig,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _routing_priority(entry: ModelPoolEntry) -> tuple[int, int, str]:
    """Prefer configured real providers while retaining demo models as a cold-start fallback."""
    is_demo = "demo" in entry.tags
    has_credential = not entry.api_key_env or bool(
        os.environ.get(entry.api_key_env, "").strip()
    )
    if has_credential and not is_demo:
        group = 0
    elif is_demo:
        group = 1
    else:
        group = 2
    return group, entry.cost_rank, entry.model_id


@dataclass(frozen=True)
class ModelPoolEntry:
    model_id: str
    provider_id: str
    base_url: str
    api_key_env: str
    api_model: str
    cost_rank: int
    max_tokens: int
    route_label: str
    tags: tuple[str, ...]
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    enabled: bool = True
    thinking: str | None = None
    reasoning_split: bool | None = None
    max_tokens_field: str = "max_tokens"
    extra_headers: dict[str, str] | None = None
    availability_status: str = "unchecked"
    availability_checked_at: str | None = None
    availability_error: str | None = None
    availability_latency_ms: float | None = None

    def to_api_config(self) -> ApiModelConfig:
        return ApiModelConfig(
            model_id=self.model_id,
            provider=ProviderConfig(
                provider_id=self.provider_id,
                base_url=self.base_url,
                api_key_env=self.api_key_env,
                extra_headers=self.extra_headers or {},
            ),
            api_model=self.api_model,
            cost_rank=self.cost_rank,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            thinking=self.thinking,
            reasoning_split=self.reasoning_split,
            max_tokens_field=self.max_tokens_field,
            route_label=self.route_label,
            tags=self.tags,
        )


class ModelPoolStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_pool (
                    model_id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    api_key_env TEXT NOT NULL,
                    api_model TEXT NOT NULL,
                    cost_rank INTEGER NOT NULL,
                    max_tokens INTEGER NOT NULL,
                    temperature REAL,
                    top_p REAL,
                    top_k INTEGER,
                    route_label TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    thinking TEXT,
                    reasoning_split INTEGER,
                    max_tokens_field TEXT NOT NULL,
                    extra_headers_json TEXT NOT NULL,
                    availability_status TEXT NOT NULL DEFAULT 'unchecked',
                    availability_checked_at TEXT,
                    availability_error TEXT,
                    availability_latency_ms REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(model_pool)").fetchall()
            }
            additions = {
                "temperature": "REAL",
                "top_p": "REAL",
                "top_k": "INTEGER",
                "availability_status": "TEXT NOT NULL DEFAULT 'unchecked'",
                "availability_checked_at": "TEXT",
                "availability_error": "TEXT",
                "availability_latency_ms": "REAL",
            }
            for column, declaration in additions.items():
                if column not in columns:
                    connection.execute(f"ALTER TABLE model_pool ADD COLUMN {column} {declaration}")

    def seed(self, models: list[ApiModelConfig]) -> None:
        for model in models:
            entry = ModelPoolEntry(
                model_id=model.model_id,
                provider_id=model.provider.provider_id,
                base_url=model.provider.base_url,
                api_key_env=model.provider.api_key_env,
                api_model=model.api_model,
                cost_rank=model.cost_rank,
                max_tokens=model.max_tokens,
                temperature=model.temperature,
                top_p=model.top_p,
                top_k=model.top_k,
                route_label=model.route_label,
                tags=tuple(sorted({"configured", *model.tags})),
                thinking=model.thinking,
                reasoning_split=model.reasoning_split,
                max_tokens_field=model.max_tokens_field,
                extra_headers=dict(model.provider.extra_headers),
            )
            self.create(entry, replace=False)

    def create(self, entry: ModelPoolEntry, *, replace: bool = False) -> dict[str, Any]:
        _validate_entry(entry)
        now = _now()
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        with self._connect() as connection:
            connection.execute(
                f"""
                {verb} INTO model_pool(
                    model_id, provider_id, base_url, api_key_env, api_model, cost_rank,
                    max_tokens, temperature, top_p, top_k, route_label, tags_json,
                    enabled, thinking, reasoning_split,
                    max_tokens_field, extra_headers_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.model_id,
                    entry.provider_id,
                    entry.base_url.rstrip("/"),
                    entry.api_key_env,
                    entry.api_model,
                    entry.cost_rank,
                    entry.max_tokens,
                    entry.temperature,
                    entry.top_p,
                    entry.top_k,
                    entry.route_label,
                    json.dumps(sorted(set(entry.tags)), ensure_ascii=False),
                    int(entry.enabled),
                    entry.thinking,
                    None if entry.reasoning_split is None else int(entry.reasoning_split),
                    entry.max_tokens_field,
                    json.dumps(entry.extra_headers or {}, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        result = self.get(entry.model_id)
        if result is None:
            raise RuntimeError("model pool write failed")
        return result

    def record_availability(
        self,
        model_id: str,
        *,
        available: bool,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        status = "available" if available else "unavailable"
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE model_pool
                SET availability_status = ?, availability_checked_at = ?,
                    availability_error = ?, availability_latency_ms = ?, updated_at = updated_at
                WHERE model_id = ?
                """,
                (status, _now(), None if available else (error or "model check failed")[:500],
                 latency_ms, model_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(model_id)
        result = self.get(model_id)
        assert result is not None
        return result

    def backfill_availability_from_batch_runs(self) -> None:
        """Initialize unchecked models from their latest terminal comparison call."""
        with self._connect() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if not {"batch_runs", "traces"}.issubset(tables):
                return
            rows = connection.execute(
                """
                SELECT r.requested_model_id AS model_id, r.status, r.error_code,
                       r.completed_at, t.status AS trace_status,
                       t.error_code AS trace_error, t.latency_ms
                FROM batch_runs r
                LEFT JOIN traces t ON t.trace_id = r.trace_id
                WHERE r.status IN ('completed', 'failed')
                  AND r.run_id = (
                    SELECT latest.run_id FROM batch_runs latest
                    WHERE latest.requested_model_id = r.requested_model_id
                      AND latest.status IN ('completed', 'failed')
                    ORDER BY latest.created_at DESC, latest.attempt DESC
                    LIMIT 1
                  )
                """
            ).fetchall()
            for row in rows:
                available = row["status"] == "completed" and row["trace_status"] == "completed"
                connection.execute(
                    """
                    UPDATE model_pool
                    SET availability_status = ?, availability_checked_at = ?,
                        availability_error = ?, availability_latency_ms = ?
                    WHERE model_id = ? AND availability_status = 'unchecked'
                    """,
                    (
                        "available" if available else "unavailable",
                        row["completed_at"] or _now(),
                        None
                        if available
                        else str(row["trace_error"] or row["error_code"] or "model call failed")[
                            :500
                        ],
                        row["latency_ms"],
                        row["model_id"],
                    ),
                )

    def update(self, model_id: str, entry: ModelPoolEntry) -> dict[str, Any]:
        if model_id != entry.model_id:
            raise ValueError("model_id cannot be changed")
        if self.get(model_id) is None:
            raise KeyError(model_id)
        return self.create(entry, replace=True)

    def get_entry(self, model_id: str, *, require_enabled: bool = False) -> ModelPoolEntry:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_pool WHERE model_id = ?", (model_id,)
            ).fetchone()
        if row is None:
            raise KeyError(model_id)
        entry = _entry_from_row(row)
        if require_enabled and not entry.enabled:
            raise ValueError(f"model is disabled: {model_id}")
        return entry

    def get(self, model_id: str) -> dict[str, Any] | None:
        try:
            entry = self.get_entry(model_id)
        except KeyError:
            return None
        return _entry_dict(entry)

    def list(
        self,
        *,
        tag: str | None = None,
        route_label: str | None = None,
        enabled_only: bool = False,
        online_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_pool ORDER BY cost_rank, model_id"
            ).fetchall()
        entries = [_entry_from_row(row) for row in rows]
        if enabled_only:
            entries = [entry for entry in entries if entry.enabled]
        if online_only:
            entries = [entry for entry in entries if "evaluation-only" not in entry.tags]
        if tag:
            entries = [entry for entry in entries if tag in entry.tags]
        if route_label:
            entries = [entry for entry in entries if entry.route_label == route_label]
        return [_entry_dict(entry) for entry in entries]

    def model_ids(self, *, enabled_only: bool = True) -> frozenset[str]:
        return frozenset(item["model_id"] for item in self.list(enabled_only=enabled_only))

    def snapshot(self) -> list[dict[str, Any]]:
        """Return the credential-free configuration used by an immutable release."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_pool ORDER BY cost_rank, model_id"
            ).fetchall()
        return [_entry_snapshot(_entry_from_row(row)) for row in rows]

    def next_higher_cost(self, model_id: str) -> ModelPoolEntry:
        current = self.get_entry(model_id)
        candidates = [
            self.get_entry(item["model_id"])
            for item in self.list(enabled_only=True)
            if int(item["cost_rank"]) > current.cost_rank
        ]
        if not candidates:
            raise ValueError(f"no enabled higher cost-rank model after {model_id}")
        return min(candidates, key=lambda item: (item.cost_rank, item.model_id))

    def resolve(
        self,
        route_label: str,
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> ModelPoolEntry:
        candidates = self.routing_candidates(route_label, exclude=exclude)
        if not candidates:
            raise ValueError(f"no enabled model is assigned to route label: {route_label}")
        return candidates[0]

    def routing_candidates(
        self,
        route_label: str,
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> list[ModelPoolEntry]:
        candidates = [
            self.get_entry(item["model_id"])
            for item in self.list(route_label=route_label, enabled_only=True, online_only=True)
            if item["model_id"] not in exclude
        ]
        return sorted(candidates, key=_routing_priority)


class RegistryAdapter(ModelAdapter):
    def __init__(
        self,
        pool: ModelPoolStore,
        client: OpenAICompatibleClient | None = None,
        status_store: ModelPoolStore | None = None,
    ) -> None:
        self.pool = pool
        self.client = client or OpenAICompatibleClient()
        self.status_store = status_store or (pool if isinstance(pool, ModelPoolStore) else None)

    @property
    def model_ids(self) -> frozenset[str]:
        return self.pool.model_ids(enabled_only=True)

    def complete(
        self,
        model_id: str,
        messages: list[ChatMessage],
        options: GenerationOptions | None = None,
    ) -> ChatResponse:
        model = self.pool.get_entry(model_id, require_enabled=True).to_api_config()
        try:
            completion = self.client.complete(
                model,
                [{"role": item.role, "content": item.content} for item in messages],
                max_tokens_override=options.max_tokens if options else None,
                temperature_override=options.temperature if options else None,
                top_p_override=options.top_p if options else None,
                top_k_override=options.top_k if options else None,
                use_model_defaults=True,
            )
        except Exception as exc:
            self._record_availability(
                model_id,
                available=False,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            raise
        self._record_availability(model_id, available=True, latency_ms=completion.latency_ms)
        return ChatResponse(
            content=completion.content,
            reasoning_content=completion.reasoning_content,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            provider_request_id=completion.request_id,
            latency_ms=completion.latency_ms,
        )

    def _record_availability(
        self,
        model_id: str,
        *,
        available: bool,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        if self.status_store is None:
            return
        try:
            self.status_store.record_availability(
                model_id,
                available=available,
                latency_ms=latency_ms,
                error=error,
            )
        except Exception:
            # Availability is telemetry and must never change completion behavior.
            return


def check_model_availability(
    pool: ModelPoolStore,
    client: OpenAICompatibleClient,
    model_id: str,
) -> dict[str, object]:
    """Send one deliberately tiny completion and persist the operational result."""
    entry = pool.get_entry(model_id, require_enabled=True)
    adapter = RegistryAdapter(pool, client=client)
    try:
        response = adapter.complete(
            model_id,
            [ChatMessage(role="user", content="Reply with OK only.")],
            GenerationOptions(max_tokens=min(entry.max_tokens, 32), temperature=0),
        )
        return {
            "model": pool.get(model_id),
            "available": True,
            "latency_ms": response.latency_ms,
        }
    except Exception:
        return {"model": pool.get(model_id), "available": False}


class ModelAvailabilityMonitor:
    """Sequential periodic probes; never adds model-call concurrency of its own."""

    def __init__(
        self,
        pool: ModelPoolStore,
        client: OpenAICompatibleClient,
        *,
        interval_seconds: float = 1_800,
    ) -> None:
        if interval_seconds < 60:
            raise ValueError("model availability interval must be at least 60 seconds")
        self.pool = pool
        self.client = client
        self.interval_seconds = interval_seconds
        self._stop = Event()
        self._lock = Lock()
        self._thread: Thread | None = None
        self._checking = False
        self._last_run_at: str | None = None
        self._last_checked_count = 0
        self._last_available_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(
            target=self._run,
            name="routetune-model-health",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def run_once(self) -> dict[str, object]:
        with self._lock:
            if self._checking:
                return self.status()
            self._checking = True
        checked = 0
        available = 0
        try:
            for model in self.pool.list(enabled_only=True):
                if self._stop.is_set():
                    break
                if model["api_key_env"] and not model["api_key_available"]:
                    continue
                result = check_model_availability(self.pool, self.client, model["model_id"])
                checked += 1
                available += int(bool(result["available"]))
        finally:
            with self._lock:
                self._checking = False
                self._last_run_at = _now()
                self._last_checked_count = checked
                self._last_available_count = available
        return self.status()

    def status(self) -> dict[str, object]:
        return {
            "enabled": True,
            "interval_seconds": self.interval_seconds,
            "checking": self._checking,
            "last_run_at": self._last_run_at,
            "last_checked_count": self._last_checked_count,
            "last_available_count": self._last_available_count,
        }

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.run_once()


class ModelPoolSnapshot:
    """Read-only model pool captured as part of a published release."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        parsed = [_entry_from_snapshot(item) for item in entries]
        self._entries = {entry.model_id: entry for entry in parsed}
        if len(self._entries) != len(parsed):
            raise ValueError("release model snapshot contains duplicate model_id values")

    def get_entry(self, model_id: str, *, require_enabled: bool = False) -> ModelPoolEntry:
        try:
            entry = self._entries[model_id]
        except KeyError as exc:
            raise KeyError(model_id) from exc
        if require_enabled and not entry.enabled:
            raise ValueError(f"model is disabled: {model_id}")
        return entry

    def list(
        self,
        *,
        tag: str | None = None,
        route_label: str | None = None,
        enabled_only: bool = False,
        online_only: bool = False,
    ) -> list[dict[str, Any]]:
        entries = sorted(self._entries.values(), key=lambda item: (item.cost_rank, item.model_id))
        if enabled_only:
            entries = [entry for entry in entries if entry.enabled]
        if online_only:
            entries = [entry for entry in entries if "evaluation-only" not in entry.tags]
        if tag:
            entries = [entry for entry in entries if tag in entry.tags]
        if route_label:
            entries = [entry for entry in entries if entry.route_label == route_label]
        return [_entry_dict(entry) for entry in entries]

    def model_ids(self, *, enabled_only: bool = True) -> frozenset[str]:
        return frozenset(item["model_id"] for item in self.list(enabled_only=enabled_only))

    def resolve(
        self,
        route_label: str,
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> ModelPoolEntry:
        candidates = self.routing_candidates(route_label, exclude=exclude)
        if not candidates:
            raise ValueError(f"no enabled model is assigned to route label: {route_label}")
        return candidates[0]

    def routing_candidates(
        self,
        route_label: str,
        *,
        exclude: frozenset[str] = frozenset(),
    ) -> list[ModelPoolEntry]:
        candidates = [
            entry
            for entry in self._entries.values()
            if entry.enabled
            and "evaluation-only" not in entry.tags
            and entry.route_label == route_label
            and entry.model_id not in exclude
        ]
        return sorted(candidates, key=_routing_priority)


def _validate_entry(entry: ModelPoolEntry) -> None:
    if not entry.model_id.strip() or not entry.api_model.strip():
        raise ValueError("model_id and api_model are required")
    if not entry.base_url.startswith(("https://", "http://")):
        raise ValueError("model base_url must use HTTP or HTTPS")
    if entry.cost_rank <= 0 or entry.max_tokens <= 0:
        raise ValueError("cost_rank and max_tokens must be positive")
    if any(not tag.strip() for tag in entry.tags):
        raise ValueError("model tags must not be blank")
    if not entry.route_label.strip():
        raise ValueError("route_label must not be blank")
    entry.to_api_config()


def _entry_from_row(row: sqlite3.Row) -> ModelPoolEntry:
    reasoning = row["reasoning_split"]
    return ModelPoolEntry(
        model_id=row["model_id"],
        provider_id=row["provider_id"],
        base_url=row["base_url"],
        api_key_env=row["api_key_env"],
        api_model=row["api_model"],
        cost_rank=int(row["cost_rank"]),
        max_tokens=int(row["max_tokens"]),
        temperature=(float(row["temperature"]) if row["temperature"] is not None else None),
        top_p=float(row["top_p"]) if row["top_p"] is not None else None,
        top_k=int(row["top_k"]) if row["top_k"] is not None else None,
        route_label=row["route_label"],
        tags=tuple(json.loads(row["tags_json"])),
        enabled=bool(row["enabled"]),
        thinking=row["thinking"],
        reasoning_split=None if reasoning is None else bool(reasoning),
        max_tokens_field=row["max_tokens_field"],
        extra_headers=json.loads(row["extra_headers_json"]),
        availability_status=row["availability_status"],
        availability_checked_at=row["availability_checked_at"],
        availability_error=row["availability_error"],
        availability_latency_ms=(
            float(row["availability_latency_ms"])
            if row["availability_latency_ms"] is not None
            else None
        ),
    )


def _entry_dict(entry: ModelPoolEntry) -> dict[str, Any]:
    return {
        "model_id": entry.model_id,
        "provider_id": entry.provider_id,
        "base_url": entry.base_url,
        "api_key_env": entry.api_key_env,
        "api_key_available": bool(os.environ.get(entry.api_key_env, "").strip()),
        "api_model": entry.api_model,
        "cost_rank": entry.cost_rank,
        "max_tokens": entry.max_tokens,
        "temperature": entry.temperature,
        "top_p": entry.top_p,
        "top_k": entry.top_k,
        "route_label": entry.route_label,
        "tags": list(entry.tags),
        "enabled": entry.enabled,
        "thinking": entry.thinking,
        "reasoning_split": entry.reasoning_split,
        "max_tokens_field": entry.max_tokens_field,
        "availability_status": entry.availability_status,
        "availability_checked_at": entry.availability_checked_at,
        "availability_error": entry.availability_error,
        "availability_latency_ms": entry.availability_latency_ms,
    }


def _entry_snapshot(entry: ModelPoolEntry) -> dict[str, Any]:
    return {
        "model_id": entry.model_id,
        "provider_id": entry.provider_id,
        "base_url": entry.base_url,
        "api_key_env": entry.api_key_env,
        "api_model": entry.api_model,
        "cost_rank": entry.cost_rank,
        "max_tokens": entry.max_tokens,
        "temperature": entry.temperature,
        "top_p": entry.top_p,
        "top_k": entry.top_k,
        "route_label": entry.route_label,
        "tags": list(entry.tags),
        "enabled": entry.enabled,
        "thinking": entry.thinking,
        "reasoning_split": entry.reasoning_split,
        "max_tokens_field": entry.max_tokens_field,
        "extra_headers": entry.extra_headers or {},
    }


def _entry_from_snapshot(value: dict[str, Any]) -> ModelPoolEntry:
    entry = ModelPoolEntry(
        model_id=str(value["model_id"]),
        provider_id=str(value["provider_id"]),
        base_url=str(value["base_url"]),
        api_key_env=str(value.get("api_key_env", "")),
        api_model=str(value["api_model"]),
        cost_rank=int(value["cost_rank"]),
        max_tokens=int(value.get("max_tokens", 1024)),
        temperature=(
            float(value["temperature"]) if value.get("temperature") is not None else None
        ),
        top_p=float(value["top_p"]) if value.get("top_p") is not None else None,
        top_k=int(value["top_k"]) if value.get("top_k") is not None else None,
        route_label=str(value["route_label"]),
        tags=tuple(str(item) for item in value.get("tags", [])),
        enabled=bool(value.get("enabled", True)),
        thinking=value.get("thinking"),
        reasoning_split=value.get("reasoning_split"),
        max_tokens_field=str(value.get("max_tokens_field", "max_tokens")),
        extra_headers={
            str(key): str(item) for key, item in dict(value.get("extra_headers", {})).items()
        },
    )
    _validate_entry(entry)
    return entry
