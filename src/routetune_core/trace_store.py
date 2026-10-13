from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from routetune_core.contracts import Annotation
from routetune_core.security import SensitiveFinding

StorageMode = Literal["metadata_only", "redacted"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TraceStore:
    def __init__(self, path: Path, storage_mode: StorageMode = "redacted") -> None:
        if storage_mode not in {"metadata_only", "redacted"}:
            raise ValueError("storage_mode must be metadata_only or redacted")
        self.path = path
        self.storage_mode = storage_mode
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
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER NOT NULL
                );
                INSERT INTO schema_version(version)
                SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_version);

                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    input_text TEXT,
                    input_findings_json TEXT NOT NULL,
                    selected_route_label TEXT,
                    selected_model_id TEXT,
                    actual_model_id TEXT,
                    bundle_version TEXT,
                    decision_json TEXT,
                    fallback_reason TEXT,
                    output_hash TEXT,
                    output_text TEXT,
                    output_findings_json TEXT,
                    provider_request_id TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    latency_ms REAL,
                    error_code TEXT,
                    deployment_id TEXT,
                    release_id TEXT,
                    request_source TEXT NOT NULL DEFAULT 'api'
                );
                CREATE INDEX IF NOT EXISTS traces_created_at_idx ON traces(created_at DESC);

                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL REFERENCES traces(trace_id),
                    output_hash TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    failure_reason TEXT,
                    score REAL,
                    comment TEXT,
                    annotator_id TEXT,
                    schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS annotations_trace_idx ON annotations(trace_id);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(traces)").fetchall()
            }
            if "selected_route_label" not in columns:
                connection.execute("ALTER TABLE traces ADD COLUMN selected_route_label TEXT")
            if "deployment_id" not in columns:
                connection.execute("ALTER TABLE traces ADD COLUMN deployment_id TEXT")
            if "release_id" not in columns:
                connection.execute("ALTER TABLE traces ADD COLUMN release_id TEXT")
            if "request_source" not in columns:
                connection.execute(
                    "ALTER TABLE traces ADD COLUMN request_source TEXT NOT NULL DEFAULT 'api'"
                )
                connection.execute(
                    "UPDATE traces SET request_source = 'batch' "
                    "WHERE decision_json LIKE '%BATCH_COMPARE_MODEL%' "
                    "OR decision_json LIKE '%BATCH_ROUTED_MODEL%'"
                )

    def start(
        self,
        *,
        request_id: str,
        input_hash: str,
        input_text: str,
        findings: tuple[SensitiveFinding, ...],
        deployment_id: str | None = None,
        release_id: str | None = None,
        request_source: str = "api",
    ) -> str:
        if request_source not in {"api", "batch"}:
            raise ValueError("request_source must be api or batch")
        trace_id = f"tr_{uuid.uuid4().hex}"
        stored_text = input_text if self.storage_mode == "redacted" else None
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO traces(
                    trace_id, request_id, created_at, status, input_hash, input_text,
                    input_findings_json, deployment_id, release_id, request_source
                ) VALUES (?, ?, ?, 'started', ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    request_id,
                    _now(),
                    input_hash,
                    stored_text,
                    _findings_json(findings),
                    deployment_id,
                    release_id,
                    request_source,
                ),
            )
        return trace_id

    def set_route(self, trace_id: str, decision: object) -> None:
        selected_model_id = getattr(decision, "selected_model_id", None)
        selected_route_label = getattr(decision, "selected_route_label", None)
        bundle_version = getattr(decision, "bundle_version", None)
        prediction = getattr(decision, "prediction", None)
        if bundle_version is None and prediction is not None:
            bundle_version = getattr(prediction, "bundle_version", None)
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE traces SET selected_route_label = ?, selected_model_id = ?,
                    bundle_version = ?, decision_json = ?
                WHERE trace_id = ?
                """,
                (
                    selected_route_label,
                    selected_model_id,
                    bundle_version,
                    json.dumps(asdict(decision), ensure_ascii=False, separators=(",", ":")),
                    trace_id,
                ),
            )

    def set_selected_model(self, trace_id: str, model_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE traces SET selected_model_id = ? WHERE trace_id = ?",
                (model_id, trace_id),
            )

    def complete(
        self,
        trace_id: str,
        *,
        actual_model_id: str,
        output_hash: str,
        output_text: str,
        output_findings: tuple[SensitiveFinding, ...],
        provider_request_id: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        fallback_reason: str | None = None,
    ) -> None:
        stored_text = output_text if self.storage_mode == "redacted" else None
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE traces SET
                    completed_at = ?, status = 'completed', actual_model_id = ?,
                    fallback_reason = ?, output_hash = ?, output_text = ?,
                    output_findings_json = ?, provider_request_id = ?, prompt_tokens = ?,
                    completion_tokens = ?, latency_ms = ?
                WHERE trace_id = ?
                """,
                (
                    _now(),
                    actual_model_id,
                    fallback_reason,
                    output_hash,
                    stored_text,
                    _findings_json(output_findings),
                    provider_request_id,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                    trace_id,
                ),
            )

    def fail(self, trace_id: str, error_code: str, *, status: str = "failed") -> None:
        if status not in {"failed", "blocked"}:
            raise ValueError("failure status must be failed or blocked")
        with self._connect() as connection:
            connection.execute(
                "UPDATE traces SET completed_at = ?, status = ?, error_code = ? WHERE trace_id = ?",
                (_now(), status, error_code, trace_id),
            )

    def annotate(
        self,
        trace_id: str,
        annotation: Annotation,
        *,
        annotator_id: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            trace = connection.execute(
                "SELECT output_hash, status FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if trace is None:
                raise KeyError(trace_id)
            if trace["status"] != "completed" or not trace["output_hash"]:
                raise ValueError("only completed traces with an output can be annotated")
            latest = connection.execute(
                "SELECT * FROM annotations WHERE trace_id = ? ORDER BY created_at DESC LIMIT 1",
                (trace_id,),
            ).fetchone()
            if (
                latest is not None
                and latest["output_hash"] == trace["output_hash"]
                and bool(latest["passed"]) == annotation.passed
                and latest["failure_reason"] == annotation.failure_reason
                and latest["score"] == annotation.score
                and latest["comment"] == annotation.comment
                and latest["annotator_id"] == annotator_id
                and latest["schema_version"] == annotation.schema_version
            ):
                return _annotation_dict(latest)
            annotation_id = f"ann_{uuid.uuid4().hex}"
            created_at = _now()
            connection.execute(
                """
                INSERT INTO annotations(
                    annotation_id, trace_id, output_hash, passed, failure_reason, score,
                    comment, annotator_id, schema_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    annotation_id,
                    trace_id,
                    trace["output_hash"],
                    int(annotation.passed),
                    annotation.failure_reason,
                    annotation.score,
                    annotation.comment,
                    annotator_id,
                    annotation.schema_version,
                    created_at,
                ),
            )
        return {
            "annotation_id": annotation_id,
            "trace_id": trace_id,
            "output_hash": trace["output_hash"],
            "passed": annotation.passed,
            "failure_reason": annotation.failure_reason,
            "score": annotation.score,
            "comment": annotation.comment,
            "annotator_id": annotator_id,
            "schema_version": annotation.schema_version,
            "created_at": created_at,
        }

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            trace = connection.execute(
                "SELECT * FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if trace is None:
                return None
            annotations = connection.execute(
                "SELECT * FROM annotations WHERE trace_id = ? ORDER BY created_at",
                (trace_id,),
            ).fetchall()
        result = _trace_dict(trace)
        result["annotations"] = [_annotation_dict(item) for item in annotations]
        return result

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = min(max(limit, 1), 200)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM traces ORDER BY created_at DESC LIMIT ?", (safe_limit,)
            ).fetchall()
        return [_trace_dict(item) for item in rows]

    def stats(self, window: str = "all", request_source: str = "api") -> dict[str, Any]:
        windows = {"24h": 24, "7d": 24 * 7, "30d": 24 * 30, "all": None}
        if window not in windows:
            raise ValueError("window must be one of 24h, 7d, 30d or all")
        if request_source not in {"api", "batch", "all"}:
            raise ValueError("request_source must be api, batch or all")
        hours = windows[window]
        cutoff = (datetime.now(UTC) - timedelta(hours=hours)).isoformat() if hours else None
        conditions: list[str] = []
        values: list[object] = []
        if cutoff:
            conditions.append("created_at >= ?")
            values.append(cutoff)
        if request_source != "all":
            conditions.append("request_source = ?")
            values.append(request_source)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        annotation_where = (
            "WHERE " + " AND ".join("t." + condition for condition in conditions)
            if conditions
            else ""
        )
        params = tuple(values)
        with self._connect() as connection:
            totals = connection.execute(
                f"""
                SELECT COUNT(*) AS total,
                       SUM(status = 'completed') AS completed,
                       SUM(status = 'failed') AS failed,
                       SUM(status = 'blocked') AS blocked,
                       AVG(CASE WHEN status = 'completed' THEN latency_ms END) AS avg_latency_ms,
                       SUM(COALESCE(prompt_tokens, 0)) AS prompt_tokens,
                       SUM(COALESCE(completion_tokens, 0)) AS completion_tokens
                FROM traces {where}
                """,
                params,
            ).fetchone()
            latency_rows = connection.execute(
                f"SELECT latency_ms FROM traces {where} "
                f"{'AND' if where else 'WHERE'} status = 'completed' AND latency_ms IS NOT NULL",
                params,
            ).fetchall()
            models = connection.execute(
                f"""
                SELECT COALESCE(actual_model_id, selected_model_id, 'unassigned') AS model_id,
                       COUNT(*) AS requests,
                       SUM(status = 'completed') AS completed,
                       SUM(status IN ('failed', 'blocked')) AS failed,
                       AVG(
                           CASE WHEN status = 'completed' THEN latency_ms END
                       ) AS average_latency_ms,
                       SUM(COALESCE(prompt_tokens, 0)) AS prompt_tokens,
                       SUM(COALESCE(completion_tokens, 0)) AS completion_tokens
                FROM traces {where}
                GROUP BY COALESCE(actual_model_id, selected_model_id, 'unassigned')
                ORDER BY requests DESC
                """,
                params,
            ).fetchall()
            route_labels = connection.execute(
                f"""
                SELECT COALESCE(selected_route_label, 'unassigned') AS route_label,
                       COUNT(*) AS requests
                FROM traces {where}
                GROUP BY COALESCE(selected_route_label, 'unassigned') ORDER BY requests DESC
                """,
                params,
            ).fetchall()
            errors = connection.execute(
                f"""
                SELECT COALESCE(error_code, status) AS error_code, COUNT(*) AS requests
                FROM traces {where}
                {'AND' if where else 'WHERE'} status IN ('failed', 'blocked')
                GROUP BY COALESCE(error_code, status) ORDER BY requests DESC
                """,
                params,
            ).fetchall()
            timeline = connection.execute(
                f"""
                SELECT substr(created_at, 1, 10) AS bucket,
                       COUNT(*) AS requests,
                       SUM(status = 'completed') AS completed,
                       SUM(status IN ('failed', 'blocked')) AS failed,
                       SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)) AS tokens
                FROM traces {where}
                GROUP BY substr(created_at, 1, 10) ORDER BY bucket
                """,
                params,
            ).fetchall()
            labels = connection.execute(
                f"""
                SELECT COUNT(*) AS total, SUM(a.passed = 1) AS passed
                FROM annotations a JOIN traces t ON t.trace_id = a.trace_id
                {annotation_where}
                """,
                params,
            ).fetchone()
        latencies = sorted(float(item["latency_ms"]) for item in latency_rows)
        annotation_total = int(labels["total"] or 0)
        passed = int(labels["passed"] or 0)
        total = int(totals["total"] or 0)
        completed = int(totals["completed"] or 0)
        prompt_tokens = int(totals["prompt_tokens"] or 0)
        completion_tokens = int(totals["completion_tokens"] or 0)
        return {
            "window": window,
            "request_source": request_source,
            "requests": {
                "total": total,
                "completed": completed,
                "failed": int(totals["failed"] or 0),
                "blocked": int(totals["blocked"] or 0),
                "success_rate": completed / total if total else None,
                "average_latency_ms": float(totals["avg_latency_ms"] or 0.0),
                "p50_latency_ms": _percentile(latencies, 0.50),
                "p95_latency_ms": _percentile(latencies, 0.95),
            },
            "tokens": {
                "prompt": prompt_tokens,
                "completion": completion_tokens,
                "total": prompt_tokens + completion_tokens,
            },
            "models": [
                {
                    **dict(item),
                    "completed": int(item["completed"] or 0),
                    "failed": int(item["failed"] or 0),
                    "average_latency_ms": float(item["average_latency_ms"] or 0.0),
                    "prompt_tokens": int(item["prompt_tokens"] or 0),
                    "completion_tokens": int(item["completion_tokens"] or 0),
                }
                for item in models
            ],
            "route_labels": [dict(item) for item in route_labels],
            "errors": [dict(item) for item in errors],
            "timeline": [dict(item) for item in timeline],
            "annotations": {
                "total": annotation_total,
                "passed": passed,
                "pass_rate": passed / annotation_total if annotation_total else None,
            },
        }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    index = max(0, min(len(values) - 1, round((len(values) - 1) * percentile)))
    return values[index]


def _findings_json(findings: tuple[SensitiveFinding, ...]) -> str:
    return json.dumps(
        [item.to_safe_dict() for item in findings], ensure_ascii=False, separators=(",", ":")
    )


def _json_or_none(value: object) -> object:
    return json.loads(value) if isinstance(value, str) else None


def _trace_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    for key in ("input_findings_json", "output_findings_json", "decision_json"):
        result[key.removesuffix("_json")] = _json_or_none(result.pop(key))
    return result


def _annotation_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["passed"] = bool(result["passed"])
    return result
