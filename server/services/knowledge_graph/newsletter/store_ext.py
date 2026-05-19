"""Schema extensions and persistence for the newsletter intelligence subsystem.

All new tables live in the same SQLite database as KnowledgeGraphStore and
share its threading.Lock — the lock is borrowed by reference so every DB
operation across both stores is serialised through a single mutex.

Tables added (all additive, never replace existing):
    kg_newsletter_sources   — publication fingerprints per node_id
    kg_newsletter_meta      — per-email newsletter classification record
    kg_story_framings       — per-source story perspectives (divergence preserved)
    kg_topic_attention      — daily windowed attention for momentum tracking
    kg_contrarian_positions — dissenting positions with evidence accumulation

Migrations:
    kg_node_facts.novelty_score REAL DEFAULT 1.0
    kg_edges.novelty_score      REAL DEFAULT 1.0
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..store import KnowledgeGraphStore
from ....logging_config import logger

_SCHEMA_EXTENSIONS = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS kg_newsletter_sources (
    node_id           INTEGER PRIMARY KEY REFERENCES kg_nodes(id),
    publication_name  TEXT    NOT NULL,
    sender_patterns   TEXT    NOT NULL DEFAULT '[]',
    first_seen_at     TEXT    NOT NULL,
    email_count       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS kg_newsletter_meta (
    email_id        TEXT    PRIMARY KEY,
    source_node_id  INTEGER REFERENCES kg_nodes(id),
    is_newsletter   INTEGER NOT NULL DEFAULT 0,
    detected_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS kg_story_framings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    story_node_id   INTEGER NOT NULL REFERENCES kg_nodes(id),
    source_node_id  INTEGER NOT NULL REFERENCES kg_nodes(id),
    framing_text    TEXT    NOT NULL,
    sentiment       TEXT    NOT NULL DEFAULT 'neutral',
    source_email_id TEXT    NOT NULL,
    source_email_ts TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    is_active       INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_kg_story_framings_story
    ON kg_story_framings (story_node_id, is_active);

CREATE INDEX IF NOT EXISTS idx_kg_story_framings_source
    ON kg_story_framings (source_node_id, is_active);

CREATE TABLE IF NOT EXISTS kg_topic_attention (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_node_id   INTEGER NOT NULL REFERENCES kg_nodes(id),
    source_node_id  INTEGER NOT NULL REFERENCES kg_nodes(id),
    window_date     TEXT    NOT NULL,
    mention_count   INTEGER NOT NULL DEFAULT 0,
    intensity_score REAL    NOT NULL DEFAULT 0.0,
    updated_at      TEXT    NOT NULL,
    UNIQUE(topic_node_id, source_node_id, window_date)
);

CREATE INDEX IF NOT EXISTS idx_kg_topic_attention_topic_date
    ON kg_topic_attention (topic_node_id, window_date);

CREATE TABLE IF NOT EXISTS kg_contrarian_positions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_node_id       INTEGER NOT NULL REFERENCES kg_nodes(id),
    source_node_id      INTEGER NOT NULL REFERENCES kg_nodes(id),
    position_text       TEXT    NOT NULL,
    prevailing_view     TEXT    NOT NULL,
    evidence_email_ids  TEXT    NOT NULL DEFAULT '[]',
    evidence_count      INTEGER NOT NULL DEFAULT 1,
    status              TEXT    NOT NULL DEFAULT 'pending',
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_contrarian_topic
    ON kg_contrarian_positions (topic_node_id, status);
"""

# Migrations that add columns to existing tables. Each is tried once;
# "duplicate column name" errors are silently swallowed.
_MIGRATIONS = [
    "ALTER TABLE kg_node_facts ADD COLUMN novelty_score REAL NOT NULL DEFAULT 1.0",
    "ALTER TABLE kg_edges      ADD COLUMN novelty_score REAL NOT NULL DEFAULT 1.0",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today_iso() -> str:
    return date.today().isoformat()


class NewsletterGraphStore:
    """Newsletter-specific persistence.

    Borrows KnowledgeGraphStore's threading.Lock so all database operations
    across both stores are serialised through a single mutex, preserving the
    existing atomicity guarantees.
    """

    def __init__(self, kg_store: KnowledgeGraphStore) -> None:
        self._db_path: Path = kg_store._db_path
        self._lock: threading.Lock = kg_store._lock
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA_EXTENSIONS)
        for sql in _MIGRATIONS:
            try:
                with self._lock, self._connect() as conn:
                    conn.execute(sql)
            except Exception:
                pass  # column already exists; safe to ignore

    # ------------------------------------------------------------------
    # Newsletter sources
    # ------------------------------------------------------------------

    def get_or_create_source(
        self,
        node_id: int,
        publication_name: str,
        sender_pattern: str,
    ) -> None:
        """Ensure a newsletter source record exists for node_id."""
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT sender_patterns FROM kg_newsletter_sources WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if existing is None:
                patterns = json.dumps([sender_pattern])
                conn.execute(
                    "INSERT INTO kg_newsletter_sources"
                    " (node_id, publication_name, sender_patterns, first_seen_at, email_count)"
                    " VALUES (?, ?, ?, ?, 0)",
                    (node_id, publication_name, patterns, now),
                )
            else:
                patterns_list: List[str] = json.loads(existing["sender_patterns"])
                if sender_pattern not in patterns_list:
                    patterns_list.append(sender_pattern)
                    conn.execute(
                        "UPDATE kg_newsletter_sources SET sender_patterns = ? WHERE node_id = ?",
                        (json.dumps(patterns_list), node_id),
                    )

    def increment_source_email_count(self, node_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE kg_newsletter_sources"
                " SET email_count = email_count + 1 WHERE node_id = ?",
                (node_id,),
            )

    def get_all_sources(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ks.node_id, ks.publication_name, ks.sender_patterns,"
                "       ks.email_count, n.canonical_name"
                " FROM kg_newsletter_sources ks"
                " JOIN kg_nodes n ON ks.node_id = n.id",
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Newsletter email metadata
    # ------------------------------------------------------------------

    def record_newsletter_meta(
        self,
        email_id: str,
        source_node_id: Optional[int],
        is_newsletter: bool,
    ) -> None:
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO kg_newsletter_meta"
                " (email_id, source_node_id, is_newsletter, detected_at)"
                " VALUES (?, ?, ?, ?)",
                (email_id, source_node_id, int(is_newsletter), now),
            )

    def update_newsletter_meta_source(self, email_id: str, source_node_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE kg_newsletter_meta SET source_node_id = ? WHERE email_id = ?",
                (source_node_id, email_id),
            )

    # ------------------------------------------------------------------
    # Story framings (divergent perspectives — never collapsed)
    # ------------------------------------------------------------------

    def insert_story_framing(
        self,
        *,
        story_node_id: int,
        source_node_id: int,
        framing_text: str,
        sentiment: str,
        source_email_id: str,
        source_email_ts: str,
    ) -> int:
        """Insert a framing record and return its ID.

        Each framing is a distinct row even when the same source covers
        the same story repeatedly. Callers must not deduplicate framings —
        the timeline of evolving perspectives is the signal.
        """
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO kg_story_framings"
                " (story_node_id, source_node_id, framing_text, sentiment,"
                "  source_email_id, source_email_ts, created_at, is_active)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    story_node_id, source_node_id, framing_text, sentiment,
                    source_email_id, source_email_ts, now,
                ),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def get_story_framings(
        self,
        story_node_id: int,
        *,
        exclude_source_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return active framings, newest first, optionally excluding one source."""
        with self._lock, self._connect() as conn:
            if exclude_source_id is not None:
                rows = conn.execute(
                    "SELECT sf.framing_text, sf.sentiment, sf.source_email_ts,"
                    "       n.canonical_name AS source_name, sf.source_node_id"
                    " FROM kg_story_framings sf"
                    " JOIN kg_nodes n ON sf.source_node_id = n.id"
                    " WHERE sf.story_node_id = ? AND sf.is_active = 1"
                    "   AND sf.source_node_id != ?"
                    " ORDER BY sf.source_email_ts DESC LIMIT ?",
                    (story_node_id, exclude_source_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT sf.framing_text, sf.sentiment, sf.source_email_ts,"
                    "       n.canonical_name AS source_name, sf.source_node_id"
                    " FROM kg_story_framings sf"
                    " JOIN kg_nodes n ON sf.source_node_id = n.id"
                    " WHERE sf.story_node_id = ? AND sf.is_active = 1"
                    " ORDER BY sf.source_email_ts DESC LIMIT ?",
                    (story_node_id, limit),
                ).fetchall()
        return [dict(r) for r in rows]

    def count_distinct_story_sources(self, story_node_id: int) -> int:
        """Count distinct sources with any active framing for this story."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT source_node_id)"
                " FROM kg_story_framings"
                " WHERE story_node_id = ? AND is_active = 1",
                (story_node_id,),
            ).fetchone()
        return int(row[0])

    def count_distinct_prevailing_sources(
        self,
        story_node_id: int,
        window_days: int = 14,
    ) -> int:
        """Count distinct sources with framings in the last window_days days."""
        cutoff = (date.today() - timedelta(days=window_days)).isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT source_node_id)"
                " FROM kg_story_framings"
                " WHERE story_node_id = ? AND is_active = 1"
                "   AND source_email_ts > ?",
                (story_node_id, cutoff),
            ).fetchone()
        return int(row[0])

    # ------------------------------------------------------------------
    # Topic attention (momentum)
    # ------------------------------------------------------------------

    def record_topic_attention(
        self,
        *,
        topic_node_id: int,
        source_node_id: int,
        novelty_score: float,
        window_date: Optional[str] = None,
    ) -> None:
        """Increment the daily attention bucket for topic × source.

        novelty_score is accumulated into intensity_score so that
        high-novelty mentions outweigh low-novelty ones in acceleration
        calculations.
        """
        date_str = window_date or _today_iso()
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO kg_topic_attention"
                " (topic_node_id, source_node_id, window_date,"
                "  mention_count, intensity_score, updated_at)"
                " VALUES (?, ?, ?, 1, ?, ?)"
                " ON CONFLICT(topic_node_id, source_node_id, window_date)"
                " DO UPDATE SET mention_count   = mention_count   + 1,"
                "               intensity_score = intensity_score + ?,"
                "               updated_at      = ?",
                (
                    topic_node_id, source_node_id, date_str,
                    novelty_score, now,
                    novelty_score, now,
                ),
            )

    def get_topic_recent_mention_count(
        self,
        topic_node_id: int,
        *,
        days: int = 7,
    ) -> int:
        """Total novel mentions across all sources in the last N days."""
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(mention_count), 0)"
                " FROM kg_topic_attention"
                " WHERE topic_node_id = ? AND window_date > ?",
                (topic_node_id, cutoff),
            ).fetchone()
        return int(row[0])

    def get_topic_window_stats(
        self,
        topic_node_id: int,
        *,
        recent_days: int = 7,
        prior_days: int = 7,
    ) -> Dict[str, Any]:
        """Aggregate stats for recent and prior rolling windows."""
        today = date.today()
        recent_start = (today - timedelta(days=recent_days)).isoformat()
        prior_end = recent_start
        prior_start = (today - timedelta(days=recent_days + prior_days)).isoformat()

        with self._lock, self._connect() as conn:
            recent = conn.execute(
                "SELECT COALESCE(SUM(mention_count), 0) AS mentions,"
                "       COALESCE(SUM(intensity_score), 0.0) AS intensity,"
                "       COUNT(DISTINCT source_node_id) AS sources"
                " FROM kg_topic_attention"
                " WHERE topic_node_id = ? AND window_date > ?",
                (topic_node_id, recent_start),
            ).fetchone()
            prior = conn.execute(
                "SELECT COALESCE(SUM(mention_count), 0) AS mentions,"
                "       COALESCE(SUM(intensity_score), 0.0) AS intensity"
                " FROM kg_topic_attention"
                " WHERE topic_node_id = ? AND window_date > ? AND window_date <= ?",
                (topic_node_id, prior_start, prior_end),
            ).fetchone()

        return {
            "recent_mentions": int(recent["mentions"]),
            "recent_intensity": float(recent["intensity"]),
            "recent_sources": int(recent["sources"]),
            "prior_mentions": int(prior["mentions"]),
            "prior_intensity": float(prior["intensity"]),
        }

    def get_all_tracked_topic_ids(self, min_recent_mentions: int = 3) -> List[int]:
        """Topic node IDs with at least min_recent_mentions in the last 7 days."""
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT topic_node_id, SUM(mention_count) AS total"
                " FROM kg_topic_attention"
                " WHERE window_date > ?"
                " GROUP BY topic_node_id"
                " HAVING total >= ?",
                (cutoff, min_recent_mentions),
            ).fetchall()
        return [int(r["topic_node_id"]) for r in rows]

    def get_node_canonical_name(self, node_id: int) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT canonical_name FROM kg_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        return row["canonical_name"] if row else None

    def get_top_momentum_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return topics sorted by recent intensity for query-time surfacing."""
        cutoff = (date.today() - timedelta(days=7)).isoformat()
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ta.topic_node_id,"
                "       n.canonical_name,"
                "       SUM(ta.mention_count)   AS total_mentions,"
                "       SUM(ta.intensity_score) AS total_intensity,"
                "       COUNT(DISTINCT ta.source_node_id) AS source_count"
                " FROM kg_topic_attention ta"
                " JOIN kg_nodes n ON ta.topic_node_id = n.id"
                " WHERE ta.window_date > ?"
                " GROUP BY ta.topic_node_id"
                " ORDER BY total_intensity DESC"
                " LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Contrarian positions
    # ------------------------------------------------------------------

    def upsert_contrarian_position(
        self,
        *,
        topic_node_id: int,
        source_node_id: int,
        position_text: str,
        prevailing_view: str,
        email_id: str,
        min_evidence_for_confirmed: int = 2,
    ) -> Tuple[str, int]:
        """Create or update a contrarian position.

        Returns (status, evidence_count).
        Status transitions: 'pending' → 'confirmed' when evidence_count
        reaches min_evidence_for_confirmed (default 2).
        """
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id, evidence_email_ids, evidence_count, status"
                " FROM kg_contrarian_positions"
                " WHERE topic_node_id = ? AND source_node_id = ?"
                " ORDER BY created_at DESC LIMIT 1",
                (topic_node_id, source_node_id),
            ).fetchone()

            if existing is None:
                emails = json.dumps([email_id])
                conn.execute(
                    "INSERT INTO kg_contrarian_positions"
                    " (topic_node_id, source_node_id, position_text, prevailing_view,"
                    "  evidence_email_ids, evidence_count, status, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, 1, 'pending', ?, ?)",
                    (
                        topic_node_id, source_node_id, position_text,
                        prevailing_view, emails, now, now,
                    ),
                )
                return "pending", 1

            emails_list: List[str] = json.loads(existing["evidence_email_ids"])
            if email_id not in emails_list:
                emails_list.append(email_id)
            new_count = len(emails_list)
            new_status = (
                "confirmed"
                if new_count >= min_evidence_for_confirmed
                else existing["status"]
            )
            conn.execute(
                "UPDATE kg_contrarian_positions"
                " SET evidence_email_ids = ?, evidence_count = ?, status = ?,"
                "     position_text = ?, updated_at = ?"
                " WHERE id = ?",
                (
                    json.dumps(emails_list), new_count, new_status,
                    position_text, now, existing["id"],
                ),
            )
            return new_status, new_count

    def get_confirmed_contrarians(
        self,
        topic_node_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return confirmed contrarian positions, optionally filtered by topic."""
        with self._lock, self._connect() as conn:
            if topic_node_id is not None:
                rows = conn.execute(
                    "SELECT cp.*, n_t.canonical_name AS topic_name,"
                    "        n_s.canonical_name AS source_name"
                    " FROM kg_contrarian_positions cp"
                    " JOIN kg_nodes n_t ON cp.topic_node_id = n_t.id"
                    " JOIN kg_nodes n_s ON cp.source_node_id = n_s.id"
                    " WHERE cp.topic_node_id = ? AND cp.status = 'confirmed'"
                    " ORDER BY cp.updated_at DESC LIMIT ?",
                    (topic_node_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT cp.*, n_t.canonical_name AS topic_name,"
                    "        n_s.canonical_name AS source_name"
                    " FROM kg_contrarian_positions cp"
                    " JOIN kg_nodes n_t ON cp.topic_node_id = n_t.id"
                    " JOIN kg_nodes n_s ON cp.source_node_id = n_s.id"
                    " WHERE cp.status = 'confirmed'"
                    " ORDER BY cp.updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Narrative query helpers
    # ------------------------------------------------------------------

    def query_story_narrative(self, story_node_id: int) -> Dict[str, Any]:
        """Full narrative for a story: metadata facts + all source framings."""
        with self._lock, self._connect() as conn:
            story_row = conn.execute(
                "SELECT canonical_name FROM kg_nodes WHERE id = ? AND node_type = 'story'",
                (story_node_id,),
            ).fetchone()
            if story_row is None:
                return {}

            facts_rows = conn.execute(
                "SELECT fact_key, fact_value FROM kg_node_facts"
                " WHERE node_id = ? AND is_active = 1 AND flagged = 0"
                " ORDER BY fact_key",
                (story_node_id,),
            ).fetchall()

            framings = conn.execute(
                "SELECT sf.framing_text, sf.sentiment, sf.source_email_ts,"
                "       n.canonical_name AS source_name"
                " FROM kg_story_framings sf"
                " JOIN kg_nodes n ON sf.source_node_id = n.id"
                " WHERE sf.story_node_id = ? AND sf.is_active = 1"
                " ORDER BY sf.source_email_ts ASC",
                (story_node_id,),
            ).fetchall()

        facts: Dict[str, str] = {r["fact_key"]: r["fact_value"] for r in facts_rows}
        framing_list = [dict(f) for f in framings]
        sentiments = [f["sentiment"] for f in framing_list]
        sources = list({f["source_name"] for f in framing_list})

        return {
            "story": story_row["canonical_name"],
            "facts": facts,
            "framings": framing_list,
            "source_count": len(sources),
            "sources": sources,
            "sentiments": sentiments,
        }

    def search_stories_by_topic(self, topic_name: str) -> List[Dict[str, Any]]:
        """Story nodes linked to a topic by name (substring match)."""
        pattern = f"%{topic_name}%"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT n.id, n.canonical_name"
                " FROM kg_nodes n"
                " JOIN kg_edges e ON e.from_node_id = n.id AND e.is_active = 1"
                " JOIN kg_nodes nt ON e.to_node_id = nt.id"
                " WHERE n.node_type = 'story'"
                "   AND (nt.canonical_name LIKE ? OR n.canonical_name LIKE ?)"
                " LIMIT 10",
                (pattern, pattern),
            ).fetchall()
        return [dict(r) for r in rows]


_nl_store_instance: Optional[NewsletterGraphStore] = None


def get_newsletter_graph_store() -> NewsletterGraphStore:
    global _nl_store_instance
    if _nl_store_instance is None:
        from ..store import get_knowledge_graph_store
        _nl_store_instance = NewsletterGraphStore(get_knowledge_graph_store())
    return _nl_store_instance


__all__ = ["NewsletterGraphStore", "get_newsletter_graph_store"]
