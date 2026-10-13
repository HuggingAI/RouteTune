from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ENTITY_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class SecurityDictionaryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS security_dictionaries (
                    dictionary_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    terms_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create(self, name: str, terms: dict[str, list[str]]) -> dict[str, Any]:
        if not name.strip():
            raise ValueError("dictionary name must not be blank")
        normalized = normalize_dictionary_terms(terms)
        dictionary_id = f"dict_{uuid.uuid4().hex}"
        now = _now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO security_dictionaries(
                        dictionary_id, name, terms_json, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, 1, ?, ?)
                    """,
                    (
                        dictionary_id,
                        name.strip(),
                        json.dumps(normalized, ensure_ascii=False, separators=(",", ":")),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("dictionary name already exists") from exc
        result = self.get(dictionary_id)
        assert result is not None
        return result

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM security_dictionaries ORDER BY created_at DESC"
            ).fetchall()
        return [_safe_dictionary(row) for row in rows]

    def get(self, dictionary_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM security_dictionaries WHERE dictionary_id = ?",
                (dictionary_id,),
            ).fetchone()
        return _safe_dictionary(row) if row is not None else None

    def set_enabled(self, dictionary_id: str, enabled: bool) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE security_dictionaries SET enabled = ?, updated_at = ?
                WHERE dictionary_id = ?
                """,
                (int(enabled), _now(), dictionary_id),
            )
        if cursor.rowcount == 0:
            raise KeyError(dictionary_id)
        result = self.get(dictionary_id)
        assert result is not None
        return result

    def combined_enabled_terms(self) -> dict[str, list[str]]:
        combined: dict[str, list[str]] = {}
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT terms_json FROM security_dictionaries WHERE enabled = 1"
            ).fetchall()
        for row in rows:
            for entity_type, values in json.loads(row["terms_json"]).items():
                combined.setdefault(entity_type, []).extend(values)
        return {key: sorted(set(values)) for key, values in combined.items()}


def normalize_dictionary_terms(terms: dict[str, list[str]]) -> dict[str, list[str]]:
    if not terms or len(terms) > 50:
        raise ValueError("dictionary must contain 1 to 50 entity types")
    normalized: dict[str, list[str]] = {}
    total_terms = 0
    total_characters = 0
    for raw_entity_type, raw_values in terms.items():
        entity_type = raw_entity_type.strip().upper()
        if not _ENTITY_TYPE.fullmatch(entity_type):
            raise ValueError("entity types must use A-Z, 0-9 and underscore")
        if not isinstance(raw_values, list):
            raise ValueError("each entity type must map to a string array")
        values: list[str] = []
        for raw_value in raw_values:
            if not isinstance(raw_value, str):
                raise ValueError("dictionary terms must be strings")
            value = raw_value.strip()
            if not value or len(value) > 200:
                raise ValueError("dictionary terms must contain 1 to 200 characters")
            values.append(value)
            total_characters += len(value)
        values = sorted(set(values))
        if not values:
            raise ValueError("each entity type must contain at least one term")
        normalized[entity_type] = values
        total_terms += len(values)
    if total_terms > 10_000 or total_characters > 500_000:
        raise ValueError("dictionary exceeds the 10000 term or 500000 character limit")
    return normalized


def _safe_dictionary(row: sqlite3.Row) -> dict[str, Any]:
    terms = json.loads(row["terms_json"])
    return {
        "dictionary_id": row["dictionary_id"],
        "name": row["name"],
        "enabled": bool(row["enabled"]),
        "entity_types": sorted(terms),
        "term_count": sum(len(values) for values in terms.values()),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
