from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from routetune_core.adapters import ChatMessage, GenerationOptions
from routetune_core.contracts import Annotation, CandidateResult, EvaluationRecord
from routetune_core.label_router import LabelRouter, train_label_bundle
from routetune_core.model_pool import ModelPoolStore
from routetune_core.runtime import RouteTuneRuntime
from routetune_core.security import RedactionResult, SecurityPipeline, SensitiveFinding
from routetune_core.trace_store import TraceStore

BatchMode = Literal["route", "compare"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class BatchStore:
    def __init__(self, path: Path) -> None:
        self.path = path
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
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_tag TEXT,
                    item_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_models (
                    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
                    model_id TEXT NOT NULL,
                    PRIMARY KEY(batch_id, model_id)
                );
                CREATE TABLE IF NOT EXISTS batch_items (
                    item_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
                    external_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    input_hash TEXT NOT NULL,
                    messages_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    route_label TEXT,
                    assigned_model_id TEXT,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(batch_id, external_id)
                );
                CREATE INDEX IF NOT EXISTS batch_items_batch_idx
                    ON batch_items(batch_id, position);
                CREATE TABLE IF NOT EXISTS batch_runs (
                    run_id TEXT PRIMARY KEY,
                    item_id TEXT NOT NULL REFERENCES batch_items(item_id),
                    attempt INTEGER NOT NULL,
                    requested_model_id TEXT NOT NULL,
                    actual_model_id TEXT,
                    trace_id TEXT,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS batch_runs_item_idx
                    ON batch_runs(item_id, attempt);
                """
            )
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(batch_items)").fetchall()
            }
            if "generation_options_json" not in columns:
                connection.execute(
                    "ALTER TABLE batch_items ADD COLUMN generation_options_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                )

    def create_batch(
        self,
        *,
        name: str,
        mode: BatchMode,
        model_tag: str | None,
        model_ids: list[str],
        items: list[dict[str, Any]],
    ) -> str:
        batch_id = f"batch_{uuid.uuid4().hex}"
        now = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO batches(batch_id, name, mode, status, model_tag, item_count,
                                    created_at, updated_at)
                VALUES (?, ?, ?, 'ready', ?, ?, ?, ?)
                """,
                (batch_id, name, mode, model_tag, len(items), now, now),
            )
            connection.executemany(
                "INSERT INTO batch_models(batch_id, model_id) VALUES (?, ?)",
                [(batch_id, model_id) for model_id in model_ids],
            )
            connection.executemany(
                """
                INSERT INTO batch_items(
                    item_id, batch_id, external_id, position, input_hash, messages_json,
                    metadata_json, generation_options_json, route_label, assigned_model_id,
                    status, error_code, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item["item_id"],
                        batch_id,
                        item["external_id"],
                        item["position"],
                        item["input_hash"],
                        item["messages_json"],
                        item["metadata_json"],
                        item["generation_options_json"],
                        item.get("route_label"),
                        item.get("assigned_model_id"),
                        item["status"],
                        item.get("error_code"),
                        now,
                        now,
                    )
                    for item in items
                ],
            )
        return batch_id

    def list_batches(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT b.*,
                    (SELECT COUNT(*) FROM batch_items i
                        WHERE i.batch_id = b.batch_id AND i.status = 'completed')
                        AS completed_count,
                    (SELECT COUNT(*) FROM batch_items i
                        WHERE i.batch_id = b.batch_id AND i.status = 'failed')
                        AS failed_count,
                    (SELECT COUNT(*) FROM batch_items i
                        WHERE i.batch_id = b.batch_id AND i.status = 'blocked')
                        AS blocked_count,
                    CASE WHEN b.mode = 'compare' THEN b.item_count *
                        (SELECT COUNT(*) FROM batch_models m WHERE m.batch_id = b.batch_id)
                        ELSE b.item_count END AS expected_run_count,
                    (SELECT CASE WHEN b.mode = 'compare'
                        THEN COUNT(DISTINCT i.item_id || ':' || r.requested_model_id)
                        ELSE COUNT(DISTINCT i.item_id) END
                        FROM batch_runs r JOIN batch_items i ON i.item_id = r.item_id
                        WHERE i.batch_id = b.batch_id
                        AND r.status IN ('completed', 'failed')) AS finished_run_count,
                    (SELECT COUNT(*) FROM batch_runs r JOIN batch_items i
                        ON i.item_id = r.item_id WHERE i.batch_id = b.batch_id
                        AND r.status = 'failed') AS failed_run_count,
                    (SELECT CASE WHEN b.mode = 'compare'
                        THEN COUNT(DISTINCT i.item_id || ':' || r.requested_model_id)
                        ELSE COUNT(DISTINCT i.item_id) END FROM annotations a
                        JOIN batch_runs r ON r.trace_id = a.trace_id
                        JOIN batch_items i ON i.item_id = r.item_id
                        WHERE i.batch_id = b.batch_id) AS annotated_run_count
                FROM batches b ORDER BY b.created_at DESC
                """
            ).fetchall()
        return [_batch_dict(row) for row in rows]

    def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            batch = connection.execute(
                "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if batch is None:
                return None
            models = connection.execute(
                "SELECT model_id FROM batch_models WHERE batch_id = ? ORDER BY model_id",
                (batch_id,),
            ).fetchall()
            items = connection.execute(
                "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY position", (batch_id,)
            ).fetchall()
            item_ids = [row["item_id"] for row in items]
            runs: dict[str, list[dict[str, Any]]] = {item_id: [] for item_id in item_ids}
            if item_ids:
                placeholders = ",".join("?" for _ in item_ids)
                run_rows = connection.execute(
                    f"SELECT * FROM batch_runs WHERE item_id IN ({placeholders}) "
                    "ORDER BY item_id, attempt",
                    item_ids,
                ).fetchall()
                for row in run_rows:
                    runs[row["item_id"]].append(dict(row))
        result = dict(batch)
        result["model_ids"] = [row["model_id"] for row in models]
        result["items"] = [_item_dict(row, runs[row["item_id"]]) for row in items]
        return result

    def pending_items(self, batch_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM batch_items
                WHERE batch_id = ? AND status IN ('pending', 'failed')
                ORDER BY position
                """,
                (batch_id,),
            ).fetchall()
        return [_item_dict(row, []) for row in rows]

    def set_batch_status(self, batch_id: str, status: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE batches SET status = ?, updated_at = ? WHERE batch_id = ?",
                (status, _now(), batch_id),
            )

    def set_item_status(
        self,
        item_id: str,
        status: str,
        *,
        error_code: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE batch_items SET status = ?, error_code = ?, updated_at = ?
                WHERE item_id = ?
                """,
                (status, error_code, _now(), item_id),
            )

    def start_run(self, item_id: str, model_id: str) -> str:
        run_id = f"run_{uuid.uuid4().hex}"
        with self._connect() as connection:
            attempt = (
                int(
                    connection.execute(
                        "SELECT COUNT(*) FROM batch_runs WHERE item_id = ?", (item_id,)
                    ).fetchone()[0]
                )
                + 1
            )
            connection.execute(
                """
                INSERT INTO batch_runs(
                    run_id, item_id, attempt, requested_model_id, status, created_at
                ) VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (run_id, item_id, attempt, model_id, _now()),
            )
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        actual_model_id: str | None = None,
        trace_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE batch_runs SET status = ?, actual_model_id = ?, trace_id = ?,
                    error_code = ?, completed_at = ? WHERE run_id = ?
                """,
                (status, actual_model_id, trace_id, error_code, _now(), run_id),
            )


class BatchManager:
    def __init__(
        self,
        *,
        store: BatchStore,
        runtime: RouteTuneRuntime,
        router: LabelRouter,
        model_pool: ModelPoolStore,
        security: SecurityPipeline,
        traces: TraceStore,
        artifact_root: Path = Path("artifacts/batches"),
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.router = router
        self.model_pool = model_pool
        self.security = security
        self.traces = traces
        self.artifact_root = artifact_root

    def import_jsonl(
        self,
        *,
        name: str,
        content: str,
        mode: BatchMode,
        model_tag: str | None = None,
        model_ids: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        if len(content.encode("utf-8")) > 10_000_000:
            raise ValueError("JSONL upload exceeds the 10 MB MVP limit")
        parsed = _parse_jsonl(content)
        selected_model_ids: list[str] = []
        if mode == "compare":
            requested_model_ids = list(dict.fromkeys(model_ids or []))
            if requested_model_ids:
                for model_id in requested_model_ids:
                    try:
                        self.model_pool.get_entry(model_id, require_enabled=True)
                    except KeyError as exc:
                        raise ValueError(f"unknown candidate model: {model_id}") from exc
                selected_model_ids = requested_model_ids
            else:
                candidates = self.model_pool.list(tag=model_tag, enabled_only=True)
                selected_model_ids = [item["model_id"] for item in candidates]
            if len(selected_model_ids) < 2:
                raise ValueError("compare batches require at least two enabled candidate models")

        items: list[dict[str, Any]] = []
        for position, raw in enumerate(parsed, start=1):
            messages = [ChatMessage(**item) for item in raw["messages"]]
            redacted_messages, findings = _redact_messages(self.security, messages)
            blocked = self.security.blocked_types(RedactionResult(text="", findings=findings))
            serialized_raw = json.dumps(raw["messages"], ensure_ascii=False, separators=(",", ":"))
            serialized_redacted = json.dumps(
                [
                    {"role": message.role, "content": message.content}
                    for message in redacted_messages
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            item: dict[str, Any] = {
                "item_id": f"item_{uuid.uuid4().hex}",
                "external_id": raw.get("id") or f"item-{position:04d}",
                "position": position,
                "input_hash": hashlib.sha256(serialized_raw.encode()).hexdigest(),
                "messages_json": serialized_redacted,
                "metadata_json": json.dumps(raw.get("metadata", {}), ensure_ascii=False),
                "generation_options_json": json.dumps(
                    {
                        key: value
                        for key, value in {
                            "max_tokens": (
                                max_tokens if max_tokens is not None else raw.get("max_tokens")
                            ),
                            "temperature": (
                                temperature
                                if temperature is not None
                                else raw.get("temperature")
                            ),
                            "top_p": top_p if top_p is not None else raw.get("top_p"),
                            "top_k": top_k if top_k is not None else raw.get("top_k"),
                        }.items()
                        if value is not None
                    },
                    separators=(",", ":"),
                ),
                "status": "blocked" if blocked else "pending",
                "error_code": f"INPUT_BLOCKED:{','.join(blocked)}" if blocked else None,
            }
            if not blocked and mode == "route":
                user_text = next(
                    message.content for message in redacted_messages if message.role == "user"
                )
                decision = self.router.decide(item["external_id"], user_text)
                item["route_label"] = decision.selected_route_label
                try:
                    resolved_label, resolved_model_id = self._resolve_available_label(
                        decision.selected_route_label
                    )
                    item["route_label"] = resolved_label
                    item["assigned_model_id"] = resolved_model_id
                except ValueError:
                    item["status"] = "failed"
                    item["error_code"] = "NO_MODEL_FOR_ROUTE_LABEL"
            items.append(item)
        batch_id = self.store.create_batch(
            name=name,
            mode=mode,
            model_tag=model_tag,
            model_ids=selected_model_ids,
            items=items,
        )
        result = self.store.get_batch(batch_id)
        assert result is not None
        return result

    def run(self, batch_id: str) -> None:
        batch = self.store.get_batch(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        self.store.set_batch_status(batch_id, "running")
        for item in self.store.pending_items(batch_id):
            self.store.set_item_status(item["item_id"], "running")
            model_ids = (
                [item["assigned_model_id"]] if batch["mode"] == "route" else batch["model_ids"]
            )
            successes = 0
            for model_id in model_ids:
                if not model_id:
                    continue
                if self._execute(item, model_id, compare=batch["mode"] == "compare"):
                    successes += 1
            expected = len(model_ids)
            self.store.set_item_status(
                item["item_id"],
                "completed" if successes == expected and expected else "failed",
                error_code=None if successes == expected and expected else "MODEL_CALL_FAILED",
            )
        refreshed = self.store.get_batch(batch_id)
        assert refreshed is not None
        terminal = all(
            item["status"] in {"completed", "failed", "blocked"} for item in refreshed["items"]
        )
        self.store.set_batch_status(batch_id, "completed" if terminal else "ready")

    def retry_item(self, batch_id: str, item_id: str, model_id: str | None = None) -> None:
        batch = self.store.get_batch(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        item = next((value for value in batch["items"] if value["item_id"] == item_id), None)
        if item is None:
            raise KeyError(item_id)
        if model_id is None:
            model_id = self._next_retry_model(item)
        self.store.set_batch_status(batch_id, "running")
        self.store.set_item_status(item_id, "running")
        success = self._execute(item, model_id, compare=True)
        self.store.set_item_status(
            item_id,
            "completed" if success else "failed",
            error_code=None if success else "RETRY_MODEL_CALL_FAILED",
        )
        self.store.set_batch_status(batch_id, "completed")

    def details(self, batch_id: str) -> dict[str, Any] | None:
        batch = self.store.get_batch(batch_id)
        if batch is None:
            return None
        for item in batch["items"]:
            for run in item["runs"]:
                run["trace"] = self.traces.get(run["trace_id"]) if run["trace_id"] else None
        batch["progress"] = _batch_progress(batch)
        return batch

    def train_candidate_bundle(
        self, batch_id: str, *, strategy: str = "auto"
    ) -> dict[str, Any]:
        batch = self.details(batch_id)
        if batch is None:
            raise KeyError(batch_id)
        if batch["mode"] != "compare":
            raise ValueError("only compare batches can train a label router")
        records: list[EvaluationRecord] = []
        incomplete: list[str] = []
        model_ids = list(batch["model_ids"])
        for item in batch["items"]:
            candidates: list[CandidateResult] = []
            successful_calls = 0
            for model_id in model_ids:
                run = next(
                    (
                        value
                        for value in reversed(item["runs"])
                        if value["requested_model_id"] == model_id
                    ),
                    None,
                )
                trace = run["trace"] if run else None
                if not run or run["status"] not in {"completed", "failed"} or not trace:
                    incomplete.append(f"{item['external_id']}:{model_id}")
                    continue
                model = self.model_pool.get_entry(model_id)
                if run["status"] == "failed" or trace.get("status") == "failed":
                    failure_code = str(
                        trace.get("error_code") or run.get("error_code") or "MODEL_CALL_FAILED"
                    )
                    candidates.append(
                        CandidateResult(
                            model_id=model_id,
                            annotation=Annotation(
                                passed=False,
                                failure_reason=f"technical_failure:{failure_code}"[:200],
                            ),
                            cost_rank=model.cost_rank,
                            latency_ms=float(trace.get("latency_ms") or 0),
                            output_hash=hashlib.sha256(
                                f"technical_failure:{failure_code}".encode()
                            ).hexdigest(),
                        )
                    )
                    continue
                successful_calls += 1
                annotations = trace.get("annotations", [])
                if not annotations:
                    incomplete.append(f"{item['external_id']}:{model_id}")
                    continue
                latest = annotations[-1]
                candidates.append(
                    CandidateResult(
                        model_id=model_id,
                        annotation=Annotation(
                            passed=bool(latest["passed"]),
                            failure_reason=latest.get("failure_reason"),
                            score=latest.get("score"),
                            comment=latest.get("comment"),
                        ),
                        cost_rank=model.cost_rank,
                        latency_ms=float(trace.get("latency_ms") or 0),
                        output_hash=trace["output_hash"],
                    )
                )
            if successful_calls == 0:
                incomplete.append(f"{item['external_id']}:all_models_failed")
            if len(candidates) == len(model_ids) and successful_calls > 0:
                user_text = next(
                    message["content"] for message in item["messages"] if message["role"] == "user"
                )
                records.append(
                    EvaluationRecord(
                        request_id=item["external_id"],
                        request_text=user_text,
                        candidates=tuple(candidates),
                        metadata={str(key): str(value) for key, value in item["metadata"].items()},
                    )
                )
        if incomplete:
            preview = ", ".join(incomplete[:5])
            raise ValueError(
                f"compare outputs need annotations before training; missing {len(incomplete)}: "
                f"{preview}"
            )
        if len(records) < 10:
            raise ValueError("label training requires at least 10 fully annotated items")
        mappings = {
            model_id: self.model_pool.get_entry(model_id).route_label for model_id in model_ids
        }
        bundle = train_label_bundle(
            records,
            mappings,
            route_labels=self.router.bundle.route_labels,
            fallback_label=self.router.bundle.fallback_label,
            strategy=strategy,
        )
        output = self.artifact_root / batch_id / bundle.version
        bundle.save(output)
        return {
            "batch_id": batch_id,
            "bundle_version": bundle.version,
            "training_record_count": bundle.training_record_count,
            "requested_strategy": strategy,
            "trainer_id": bundle.trainer_id,
            "output": str(output),
            "published": False,
        }

    def _execute(self, item: dict[str, Any], model_id: str, *, compare: bool) -> bool:
        run_id = self.store.start_run(item["item_id"], model_id)
        raw_messages = item.get("messages")
        if raw_messages is None:
            raw_messages = json.loads(item["messages_json"])
        messages = [ChatMessage(**value) for value in raw_messages]
        options = item.get("generation_options") or {}
        generation_options = (
            GenerationOptions(
                max_tokens=options.get("max_tokens"),
                temperature=options.get("temperature"),
                top_p=options.get("top_p"),
                top_k=options.get("top_k"),
            )
            if options
            else None
        )
        try:
            response = self.runtime.complete(
                messages,
                request_id=f"{item['external_id']}:{run_id}",
                model_id_override=model_id,
                override_reason="BATCH_COMPARE_MODEL" if compare else "BATCH_ROUTED_MODEL",
                allow_provider_fallback=not compare,
                generation_options=generation_options,
                trace_source="batch",
            )
        except Exception as error:
            trace_id = getattr(error, "trace_id", None)
            self.store.finish_run(
                run_id,
                status="failed",
                trace_id=trace_id,
                error_code=error.__class__.__name__.upper(),
            )
            return False
        self.store.finish_run(
            run_id,
            status="completed",
            actual_model_id=response.actual_model_id,
            trace_id=response.trace_id,
        )
        return True

    def _next_retry_model(self, item: dict[str, Any]) -> str:
        attempted = {run["requested_model_id"] for run in item["runs"] if run["requested_model_id"]}
        current_label = item.get("route_label") or self.router.bundle.route_labels[0]
        labels = self.router.bundle.route_labels
        start = labels.index(current_label) if current_label in labels else 0
        for label in labels[start:]:
            candidates = self.model_pool.list(
                route_label=label, enabled_only=True, online_only=True
            )
            for candidate in candidates:
                if candidate["model_id"] not in attempted:
                    return candidate["model_id"]
        raise ValueError("no untried model is available at this or a higher route tier")

    def _resolve_available_label(self, route_label: str) -> tuple[str, str]:
        labels = self.router.bundle.route_labels
        start = labels.index(route_label)
        for label in labels[start:]:
            candidates = self.model_pool.list(
                route_label=label, enabled_only=True, online_only=True
            )
            if candidates:
                return label, str(candidates[0]["model_id"])
        raise ValueError(f"no model is available at or above route label: {route_label}")


def _parse_jsonl(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    external_ids: set[str] = set()
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
            raise ValueError(f"line {line_number} must contain a messages array")
        messages = row["messages"]
        roles = [item.get("role") for item in messages if isinstance(item, dict)]
        if len(roles) != len(messages) or roles not in (["user"], ["system", "user"]):
            raise ValueError(f"line {line_number} messages must be [user] or [system, user]")
        if any(
            not isinstance(item.get("content"), str) or not item["content"].strip()
            for item in messages
        ):
            raise ValueError(f"line {line_number} message content must be non-empty text")
        max_tokens = row.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or not 1 <= max_tokens <= 1_000_000
        ):
            raise ValueError(f"line {line_number} max_tokens must be an integer from 1 to 1000000")
        temperature = row.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError(f"line {line_number} temperature must be a number from 0 to 2")
        top_p = row.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0 <= top_p <= 1
        ):
            raise ValueError(f"line {line_number} top_p must be a number from 0 to 1")
        top_k = row.get("top_k")
        if top_k is not None and (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 1_000_000
        ):
            raise ValueError(f"line {line_number} top_k must be an integer from 1 to 1000000")
        external_id = str(row.get("id") or f"item-{line_number:04d}")
        if external_id in external_ids:
            raise ValueError(f"duplicate id at line {line_number}: {external_id}")
        external_ids.add(external_id)
        row["id"] = external_id
        rows.append(row)
    if not rows:
        raise ValueError("JSONL upload contains no records")
    if len(rows) > 1_000:
        raise ValueError("a batch can contain at most 1000 records")
    return rows


def _redact_messages(
    security: SecurityPipeline, messages: list[ChatMessage]
) -> tuple[list[ChatMessage], tuple[SensitiveFinding, ...]]:
    redacted: list[ChatMessage] = []
    findings: list[SensitiveFinding] = []
    for message in messages:
        result = security.redact(message.content)
        redacted.append(ChatMessage(role=message.role, content=result.text))
        findings.extend(result.findings)
    return redacted, tuple(findings)


def _batch_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in (
        "completed_count",
        "failed_count",
        "blocked_count",
        "expected_run_count",
        "finished_run_count",
        "failed_run_count",
        "annotated_run_count",
    ):
        result[key] = int(result.get(key) or 0)
    return result


def _batch_progress(batch: dict[str, Any]) -> dict[str, int | bool]:
    expected = (
        len(batch["items"]) * len(batch["model_ids"])
        if batch["mode"] == "compare"
        else len(batch["items"])
    )
    finished = 0
    failed = 0
    successful = 0
    annotated = 0
    items_with_success: set[str] = set()
    empty_hash = hashlib.sha256(b"").hexdigest()
    pairs = (
        [
            (item, model_id)
            for item in batch["items"]
            for model_id in batch["model_ids"]
        ]
        if batch["mode"] == "compare"
        else [(item, None) for item in batch["items"]]
    )
    for item, model_id in pairs:
        attempts = [
            run
            for run in item["runs"]
            if model_id is None or run["requested_model_id"] == model_id
        ]
        latest_run = attempts[-1] if attempts else None
        valid_run = (
            latest_run
            if latest_run
            and latest_run["status"] == "completed"
            and latest_run.get("trace")
            and latest_run["trace"].get("status") == "completed"
            and latest_run["trace"].get("output_hash") != empty_hash
            else None
        )
        if valid_run is not None:
            finished += 1
            successful += 1
            items_with_success.add(item["item_id"])
            if valid_run["trace"].get("annotations"):
                annotated += 1
        elif latest_run and latest_run["status"] == "failed":
            finished += 1
            failed += 1
    return {
        "expected": expected,
        "finished": finished,
        "failed": failed,
        "successful": successful,
        "annotated": annotated,
        "manual_expected": successful,
        "technical_failed": failed,
        "train_ready": bool(
            batch["mode"] == "compare"
            and expected > 0
            and finished == expected
            and annotated == successful
            and len(items_with_success) == len(batch["items"])
        ),
    }


def _item_dict(row: sqlite3.Row, runs: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(row)
    result["messages"] = json.loads(result.pop("messages_json"))
    result["metadata"] = json.loads(result.pop("metadata_json"))
    result["generation_options"] = json.loads(result.pop("generation_options_json", "{}"))
    result["runs"] = runs
    return result
