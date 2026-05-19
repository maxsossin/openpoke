"""Agent descriptor persistence - structured metadata for execution agents."""

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from ...logging_config import logger

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
# Legacy JSON path — read on first run to migrate existing data into SQLite.
_DESCRIPTOR_PATH = _DATA_DIR / "execution_agents" / "descriptors.json"
_DB_PATH = _DATA_DIR / "execution_agents" / "descriptors.db"
_db_lock = threading.Lock()


@dataclass
class AgentDescriptor:
    agent_name: str
    summary: str
    tags: list[str]
    last_active: str
    status: str
    last_output_snippet: str


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema() -> None:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _db_lock, _connect() as conn:
        conn.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS agent_descriptors (
            agent_name          TEXT    PRIMARY KEY,
            summary             TEXT    NOT NULL DEFAULT '',
            tags                TEXT    NOT NULL DEFAULT '[]',
            last_active         TEXT    NOT NULL DEFAULT '',
            status              TEXT    NOT NULL DEFAULT 'idle',
            last_output_snippet TEXT    NOT NULL DEFAULT ''
        );
        """)
    _migrate_from_json()


def _migrate_from_json() -> None:
    """Import descriptors.json into SQLite on first run.

    If the JSON file exists and the SQLite table is empty, all entries are
    imported atomically. After this point SQLite is the sole authority.

    Fails loudly (raises RuntimeError) if the JSON file exists but cannot be
    parsed — silent data loss from an unreadable file is not acceptable.
    """
    if not _DESCRIPTOR_PATH.exists():
        return
    with _db_lock, _connect() as conn:
        count = int(conn.execute(
            "SELECT COUNT(*) FROM agent_descriptors"
        ).fetchone()[0])
        if count > 0:
            # SQLite is already populated; JSON is superseded.
            return
        try:
            with open(_DESCRIPTOR_PATH) as f:
                raw = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"descriptors.json exists but cannot be imported into SQLite: {exc}. "
                "Delete or repair the file to continue."
            ) from exc
        for name, d in raw.items():
            conn.execute(
                "INSERT OR REPLACE INTO agent_descriptors"
                " (agent_name, summary, tags, last_active, status, last_output_snippet)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    d.get("agent_name", name),
                    d.get("summary", ""),
                    json.dumps(d.get("tags", [])),
                    d.get("last_active", ""),
                    d.get("status", "idle"),
                    d.get("last_output_snippet", ""),
                ),
            )
        logger.info("Migrated %d descriptors from JSON to SQLite", len(raw))


def load_descriptors() -> dict[str, AgentDescriptor]:
    try:
        with _db_lock, _connect() as conn:
            rows = conn.execute(
                "SELECT agent_name, summary, tags, last_active, status, last_output_snippet"
                " FROM agent_descriptors"
            ).fetchall()
        result = {}
        for row in rows:
            tags = json.loads(row["tags"]) if row["tags"] else []
            result[row["agent_name"]] = AgentDescriptor(
                agent_name=row["agent_name"],
                summary=row["summary"],
                tags=tags,
                last_active=row["last_active"],
                status=row["status"],
                last_output_snippet=row["last_output_snippet"],
            )
        return result
    except Exception as exc:
        logger.warning(f"Failed to load descriptors: {exc}")
        return {}


def save_descriptor(descriptor: AgentDescriptor) -> None:
    try:
        with _db_lock, _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO agent_descriptors"
                " (agent_name, summary, tags, last_active, status, last_output_snippet)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    descriptor.agent_name,
                    descriptor.summary,
                    json.dumps(descriptor.tags),
                    descriptor.last_active,
                    descriptor.status,
                    descriptor.last_output_snippet,
                ),
            )
    except Exception as exc:
        logger.warning(f"Failed to save descriptor for {descriptor.agent_name}: {exc}")


_ensure_schema()
