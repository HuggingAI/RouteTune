from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any

from routetune_core.label_router import LabelEngineBundle, LabelRouter
from routetune_core.model_pool import (
    ModelPoolSnapshot,
    ModelPoolStore,
    RegistryAdapter,
)
from routetune_core.runtime import RouteTuneRuntime, RuntimeResponse
from routetune_core.security import SecurityPipeline, SecurityPolicy
from routetune_core.security_store import SecurityDictionaryStore
from routetune_core.trace_store import TraceStore


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReleaseStore:
    """Persistent immutable releases and the mutable production pointer."""

    def __init__(self, path: Path, artifact_root: Path | None = None) -> None:
        self.path = path
        self.artifact_root = artifact_root or path.parent / "release-artifacts"
        self.artifact_root.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS route_releases (
                    release_id TEXT PRIMARY KEY,
                    bundle_version TEXT NOT NULL,
                    bundle_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    source_batch_id TEXT,
                    model_snapshot_json TEXT NOT NULL,
                    security_snapshot_json TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    published_at TEXT
                );
                CREATE INDEX IF NOT EXISTS route_releases_created_idx
                    ON route_releases(created_at DESC);

                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    active_release_id TEXT REFERENCES route_releases(release_id),
                    updated_at TEXT NOT NULL
                );
                INSERT OR IGNORE INTO deployments(deployment_id, active_release_id, updated_at)
                    VALUES ('production', NULL, '');
                """
            )

    def create(
        self,
        *,
        bundle: LabelEngineBundle,
        model_snapshot: list[dict[str, Any]],
        security_snapshot: dict[str, Any],
        source_batch_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        release_id = f"rel_{uuid.uuid4().hex}"
        bundle_path = self.artifact_root / release_id / "label-bundle"
        bundle.save(bundle_path)
        created_at = _now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO route_releases(
                    release_id, bundle_version, bundle_path, status, source_batch_id,
                    model_snapshot_json, security_snapshot_json, notes, created_at
                ) VALUES (?, ?, ?, 'candidate', ?, ?, ?, ?, ?)
                """,
                (
                    release_id,
                    bundle.version,
                    str(bundle_path),
                    source_batch_id,
                    json.dumps(model_snapshot, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(security_snapshot, ensure_ascii=False, separators=(",", ":")),
                    notes,
                    created_at,
                ),
            )
        release = self.get(release_id)
        assert release is not None
        return release

    def activate(self, release_id: str, *, deployment_id: str = "production") -> dict[str, Any]:
        published_at = _now()
        with self._connect() as connection:
            release = connection.execute(
                "SELECT release_id FROM route_releases WHERE release_id = ?", (release_id,)
            ).fetchone()
            if release is None:
                raise KeyError(release_id)
            current = connection.execute(
                "SELECT active_release_id FROM deployments WHERE deployment_id = ?",
                (deployment_id,),
            ).fetchone()
            if current is None:
                raise KeyError(deployment_id)
            if current["active_release_id"] and current["active_release_id"] != release_id:
                connection.execute(
                    "UPDATE route_releases SET status = 'superseded' WHERE release_id = ?",
                    (current["active_release_id"],),
                )
            connection.execute(
                "UPDATE route_releases SET status = 'published', published_at = ? "
                "WHERE release_id = ?",
                (published_at, release_id),
            )
            connection.execute(
                "UPDATE deployments SET active_release_id = ?, updated_at = ? "
                "WHERE deployment_id = ?",
                (release_id, published_at, deployment_id),
            )
        deployment = self.deployment(deployment_id)
        assert deployment is not None
        return deployment

    def get(self, release_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM route_releases WHERE release_id = ?", (release_id,)
            ).fetchone()
        return _release_dict(row, internal=False) if row is not None else None

    def load(self, release_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM route_releases WHERE release_id = ?", (release_id,)
            ).fetchone()
        if row is None:
            raise KeyError(release_id)
        return _release_dict(row, internal=True)

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM route_releases ORDER BY created_at DESC"
            ).fetchall()
        return [_release_dict(row, internal=False) for row in rows]

    def latest_for_source_batch(
        self, source_batch_id: str, trainer_id: str | None = None
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM route_releases WHERE source_batch_id = ? "
                "ORDER BY created_at DESC",
                (source_batch_id,),
            ).fetchall()
        releases = [_release_dict(row, internal=False) for row in rows]
        if trainer_id is not None:
            releases = [item for item in releases if item["trainer_id"] == trainer_id]
        return releases[0] if releases else None

    def deployment(self, deployment_id: str = "production") -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT d.deployment_id, d.active_release_id, d.updated_at,
                       r.bundle_version, r.published_at
                FROM deployments d
                LEFT JOIN route_releases r ON r.release_id = d.active_release_id
                WHERE d.deployment_id = ?
                """,
                (deployment_id,),
            ).fetchone()
        return dict(row) if row is not None else None


class DeploymentManager:
    """Owns the production runtime and swaps immutable release snapshots."""

    deployment_id = "production"

    def __init__(
        self,
        *,
        draft_runtime: RouteTuneRuntime,
        model_pool: ModelPoolStore,
        dictionaries: SecurityDictionaryStore,
        traces: TraceStore,
        releases: ReleaseStore,
    ) -> None:
        if not isinstance(draft_runtime.router, LabelRouter):
            raise TypeError("versioned deployments require label routing")
        if not isinstance(draft_runtime.adapter, RegistryAdapter):
            raise TypeError("versioned deployments require RegistryAdapter")
        self.draft_runtime = draft_runtime
        self.model_pool = model_pool
        self.dictionaries = dictionaries
        self.traces = traces
        self.releases = releases
        self._client = draft_runtime.adapter.client
        self._base_detectors = tuple(draft_runtime.security.detectors[:-1])
        self._lock = RLock()
        self._last_refresh = 0.0

        deployment = releases.deployment(self.deployment_id)
        assert deployment is not None
        active_release_id = deployment["active_release_id"]
        if active_release_id is None:
            initial = self.create_candidate(
                draft_runtime.router.bundle,
                notes="Initial release created from the startup configuration.",
            )
            releases.activate(initial["release_id"], deployment_id=self.deployment_id)
            active_release_id = initial["release_id"]
        self._active_release_id = str(active_release_id)
        self._active_runtime = self._build_runtime(releases.load(self._active_release_id))

    @property
    def active_release_id(self) -> str:
        return self._active_release_id

    @property
    def active_runtime(self) -> RouteTuneRuntime:
        self.refresh_if_needed()
        return self._active_runtime

    def create_candidate(
        self,
        bundle: LabelEngineBundle,
        *,
        source_batch_id: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        security_snapshot = {
            "dictionary_terms": self.dictionaries.combined_enabled_terms(),
            "blocked_types": sorted(self.draft_runtime.security.policy.blocked_types),
        }
        return self.releases.create(
            bundle=bundle,
            model_snapshot=self.model_pool.snapshot(),
            security_snapshot=security_snapshot,
            source_batch_id=source_batch_id,
            notes=notes,
        )

    def create_candidate_from_path(
        self,
        bundle_path: Path,
        *,
        source_batch_id: str | None = None,
    ) -> dict[str, Any]:
        return self.create_candidate(
            LabelEngineBundle.load(bundle_path), source_batch_id=source_batch_id
        )

    def publish(self, release_id: str) -> dict[str, Any]:
        release = self.releases.load(release_id)
        runtime = self._build_runtime(release)
        with self._lock:
            deployment = self.releases.activate(release_id, deployment_id=self.deployment_id)
            self._active_runtime = runtime
            self._active_release_id = release_id
            self._last_refresh = monotonic()
        return deployment

    def complete(self, *args: Any, **kwargs: Any) -> RuntimeResponse:
        runtime = self.active_runtime
        return runtime.complete(*args, **kwargs)

    def refresh_if_needed(self, *, force: bool = False) -> None:
        current_time = monotonic()
        if not force and current_time - self._last_refresh < 1.0:
            return
        self._last_refresh = current_time
        deployment = self.releases.deployment(self.deployment_id)
        if deployment is None or deployment["active_release_id"] is None:
            return
        release_id = str(deployment["active_release_id"])
        if release_id == self._active_release_id:
            return
        runtime = self._build_runtime(self.releases.load(release_id))
        with self._lock:
            self._active_runtime = runtime
            self._active_release_id = release_id

    def _build_runtime(self, release: dict[str, Any]) -> RouteTuneRuntime:
        bundle = LabelEngineBundle.load(Path(release["bundle_path"]))
        pool = ModelPoolSnapshot(release["model_snapshot"])
        if not pool.list(enabled_only=True, online_only=True):
            raise ValueError("a release requires at least one enabled online model")
        security_data = release["security_snapshot"]
        security = SecurityPipeline(
            detectors=self._base_detectors,
            policy=SecurityPolicy(blocked_types=frozenset(security_data.get("blocked_types", []))),
        )
        security.set_managed_dictionary_terms(security_data.get("dictionary_terms", {}))
        return RouteTuneRuntime(
            router=LabelRouter(bundle),
            adapter=RegistryAdapter(pool, client=self._client, status_store=self.model_pool),
            security=security,
            traces=self.traces,
            model_pool=pool,
            deployment_id=self.deployment_id,
            release_id=str(release["release_id"]),
        )


def _release_dict(row: sqlite3.Row, *, internal: bool) -> dict[str, Any]:
    models = json.loads(row["model_snapshot_json"])
    security = json.loads(row["security_snapshot_json"])
    training_record_count = 0
    trainer_id = "unknown"
    router_family = "probabilistic-classifier"
    try:
        manifest = json.loads(
            (Path(row["bundle_path"]) / "label-manifest.json").read_text(encoding="utf-8")
        )
        training_record_count = int(manifest.get("training_record_count", 0))
        trainer_id = str(manifest.get("trainer_id") or manifest.get("classifier") or "unknown")
        router_family = str(manifest.get("router_family") or router_family)
        if router_family == "system-one-classifier":
            router_family = "probabilistic-classifier"
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    result: dict[str, Any] = {
        "release_id": row["release_id"],
        "bundle_version": row["bundle_version"],
        "status": row["status"],
        "source_batch_id": row["source_batch_id"],
        "notes": row["notes"],
        "created_at": row["created_at"],
        "published_at": row["published_at"],
        "training_record_count": training_record_count,
        "trainer_id": trainer_id,
        "router_family": router_family,
        "model_count": len(models),
        "online_model_count": sum(
            bool(item.get("enabled", True)) and "evaluation-only" not in item.get("tags", [])
            for item in models
        ),
        "security_entity_types": sorted(security.get("dictionary_terms", {})),
        "blocked_types": sorted(security.get("blocked_types", [])),
    }
    if internal:
        result.update(
            {
                "bundle_path": row["bundle_path"],
                "model_snapshot": models,
                "security_snapshot": security,
            }
        )
    return result
