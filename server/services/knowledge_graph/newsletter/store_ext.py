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
from datetime import datetime, timedelta, timezone
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
    story_node_id       INTEGER NOT NULL REFERENCES kg_nodes(id),
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
    ON kg_contrarian_positions (story_node_id, status);

CREATE TABLE IF NOT EXISTS kg_claims (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    story_node_id       INTEGER REFERENCES kg_nodes(id),
    source_node_id      INTEGER REFERENCES kg_nodes(id),
    claim_text          TEXT    NOT NULL,
    claim_date          TEXT    NOT NULL,
    verification_status TEXT    NOT NULL DEFAULT 'pending',
    verified_at         TEXT,
    source_email_id     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_claims_story
    ON kg_claims (story_node_id, verification_status);

CREATE TABLE IF NOT EXISTS kg_pending_framings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    story_title     TEXT    NOT NULL,
    source_node_id  INTEGER NOT NULL REFERENCES kg_nodes(id),
    framing_text    TEXT    NOT NULL,
    sentiment       TEXT    NOT NULL DEFAULT 'neutral',
    source_email_id TEXT    NOT NULL,
    source_email_ts TEXT    NOT NULL,
    key_claim       TEXT    NOT NULL DEFAULT '',
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_kg_pending_framings_title
    ON kg_pending_framings (story_title);
"""

# Migrations that add columns to existing tables. Each is tried once;
# "duplicate column name" errors are silently swallowed.
_MIGRATIONS = [
    # Momentum alert cooldown: NULL means the topic has never triggered an alert.
    "ALTER TABLE kg_topic_attention ADD COLUMN last_alerted_at TEXT",
    # Rename: the column stored story node IDs, not topic node IDs.
    # Safe on populated databases (SQLite 3.25+); indexes are updated automatically.
    # Silently ignored on fresh databases where the schema already uses story_node_id.
    "ALTER TABLE kg_contrarian_positions RENAME COLUMN topic_node_id TO story_node_id",
    # Edge property payload — NULL means no properties (all existing rows safe).
    "ALTER TABLE kg_edges ADD COLUMN properties TEXT",
    # Key claim persisted on story framings for claim-vs-claim contrarian assessment.
    "ALTER TABLE kg_story_framings ADD COLUMN key_claim TEXT NOT NULL DEFAULT ''",
    # Source credibility scoring — updated by claim verification and novelty scorer.
    "ALTER TABLE kg_newsletter_sources ADD COLUMN claim_correct_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE kg_newsletter_sources ADD COLUMN claim_total_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE kg_newsletter_sources ADD COLUMN avg_novelty_score REAL NOT NULL DEFAULT 1.0",
    # Source consistency profiling — JSON fingerprint updated by aggregation query.
    "ALTER TABLE kg_newsletter_sources ADD COLUMN sentiment_profile TEXT NOT NULL DEFAULT '{}'",
    # Novelty at first observation — 1.0 for all pre-existing rows (safe on populated databases).
    "ALTER TABLE kg_node_facts ADD COLUMN novelty_score REAL NOT NULL DEFAULT 1.0",
    "ALTER TABLE kg_edges ADD COLUMN novelty_score REAL NOT NULL DEFAULT 1.0",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


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
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                if (
                    "duplicate column name" not in msg
                    and "already exists" not in msg
                    and "no such column" not in msg  # RENAME COLUMN already applied
                ):
                    raise

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
        key_claim: str = "",
    ) -> int:
        """Insert a framing record and return its ID.

        Each framing is a distinct row even when the same source covers
        the same story repeatedly. Callers must not deduplicate framings —
        the timeline of evolving perspectives is the signal.

        key_claim: the most specific, falsifiable claim from this framing.
        Stored for claim-vs-claim contrarian assessment (Fix 6). Defaults to
        empty string for callers that do not supply a claim.
        """
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO kg_story_framings"
                " (story_node_id, source_node_id, framing_text, sentiment,"
                "  source_email_id, source_email_ts, created_at, is_active, key_claim)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)",
                (
                    story_node_id, source_node_id, framing_text, sentiment,
                    source_email_id, source_email_ts, now, key_claim,
                ),
            )
            framing_id: int = cursor.lastrowid  # type: ignore[assignment]

        # Index this framing in ChromaDB so future semantic novelty checks can
        # compare against it. Best-effort: SQLite write already succeeded.
        try:
            from .novelty import index_framing_text
            index_framing_text(framing_id, framing_text, story_node_id)
        except Exception:
            pass

        return framing_id

    def get_story_framings(
        self,
        story_node_id: int,
        *,
        exclude_source_id: Optional[int] = None,
        limit: int = 20,
        since_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return active framings, newest first, optionally excluding one source.

        since_days: when set, restricts to framings whose source_email_ts is
        within the last N days (UTC). Use this to avoid months-old framings
        being treated as the current prevailing view.
        """
        cutoff = (
            (datetime.now(timezone.utc).date() - timedelta(days=since_days)).isoformat()
            if since_days is not None
            else None
        )
        with self._lock, self._connect() as conn:
            date_clause = " AND sf.source_email_ts > ?" if cutoff else ""
            excl_clause = " AND sf.source_node_id != ?" if exclude_source_id is not None else ""
            base = (
                "SELECT sf.framing_text, sf.sentiment, sf.source_email_ts,"
                "       n.canonical_name AS source_name, sf.source_node_id,"
                "       COALESCE(sf.key_claim, '') AS key_claim"
                " FROM kg_story_framings sf"
                " JOIN kg_nodes n ON sf.source_node_id = n.id"
                f" WHERE sf.story_node_id = ? AND sf.is_active = 1{date_clause}{excl_clause}"
                " ORDER BY sf.source_email_ts DESC LIMIT ?"
            )
            params: list = [story_node_id]
            if cutoff:
                params.append(cutoff)
            if exclude_source_id is not None:
                params.append(exclude_source_id)
            params.append(limit)
            rows = conn.execute(base, params).fetchall()
        return [dict(r) for r in rows]

    def get_recent_framings_for_topic(
        self,
        topic_node_id: int,
        limit: int = 5,
    ) -> List[str]:
        """Return recent framing texts from stories involving this topic.

        Traverses kg_edges (story → topic, edge_type='involves') to find stories
        that cover this topic, then returns their most recent active framings from
        kg_story_framings. Used by compute_topic_novelty() for semantic comparison.

        Call chain: compute_topic_novelty (novelty.py) → here → SQLite JOIN.
        Blocking local I/O only; no external calls.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT sf.framing_text"
                " FROM kg_story_framings sf"
                " JOIN kg_edges e ON sf.story_node_id = e.from_node_id"
                " WHERE e.to_node_id = ? AND e.edge_type = 'involves'"
                "   AND e.is_active = 1 AND sf.is_active = 1"
                " ORDER BY sf.source_email_ts DESC LIMIT ?",
                (topic_node_id, limit),
            ).fetchall()
        return [r["framing_text"] for r in rows]

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
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=window_days)).isoformat()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT source_node_id)"
                " FROM kg_story_framings"
                " WHERE story_node_id = ? AND is_active = 1"
                "   AND source_email_ts > ?",
                (story_node_id, cutoff),
            ).fetchone()
        return int(row[0])

    def store_pending_framing(
        self,
        *,
        story_title: str,
        source_node_id: int,
        framing_text: str,
        sentiment: str,
        source_email_id: str,
        source_email_ts: str,
        key_claim: str = "",
    ) -> None:
        """Hold a framing for a story that has not yet accumulated enough source coverage.

        Called from thread_story when max_sources < STORY_MIN_SOURCES. The
        framing is linked to a real story_node once coverage threshold is met,
        preserving the full framing timeline from the first publisher onward.
        """
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO kg_pending_framings"
                " (story_title, source_node_id, framing_text, sentiment,"
                "  source_email_id, source_email_ts, key_claim, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    story_title, source_node_id, framing_text, sentiment,
                    source_email_id, source_email_ts, key_claim, now,
                ),
            )

    def pop_pending_framings(self, story_title: str) -> List[Dict[str, Any]]:
        """Return and delete all pending framings for the given story title.

        Called immediately after a story node is created so the full framing
        timeline (including pre-creation publishers) is stored in kg_story_framings.
        Rows are ordered by created_at ASC to preserve chronological sequence.
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT source_node_id, framing_text, sentiment,"
                "       source_email_id, source_email_ts, key_claim"
                " FROM kg_pending_framings WHERE story_title = ?"
                " ORDER BY created_at ASC",
                (story_title,),
            ).fetchall()
            if rows:
                conn.execute(
                    "DELETE FROM kg_pending_framings WHERE story_title = ?",
                    (story_title,),
                )
        return [dict(r) for r in rows]

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
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
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
        today = datetime.now(timezone.utc).date()
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

    def get_all_tracked_topic_ids(
        self,
        min_recent_mentions: int = 3,
        cooldown_cutoff: Optional[str] = None,
    ) -> List[int]:
        """Topic node IDs with at least min_recent_mentions in the last 7 days.

        cooldown_cutoff: when provided (ISO timestamp), topics whose MAX(last_alerted_at)
        is at or after this value are excluded. This is the momentum alert cooldown
        check — it runs as a single SQL HAVING condition, not a post-fetch filter.
        """
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        params: list = [cutoff, min_recent_mentions]
        cooldown_clause = ""
        if cooldown_cutoff is not None:
            cooldown_clause = " AND COALESCE(MAX(last_alerted_at), '') < ?"
            params.append(cooldown_cutoff)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT topic_node_id, SUM(mention_count) AS total"
                " FROM kg_topic_attention"
                " WHERE window_date > ?"
                " GROUP BY topic_node_id"
                f" HAVING total >= ?{cooldown_clause}",
                params,
            ).fetchall()
        return [int(r["topic_node_id"]) for r in rows]

    def mark_topic_alerted(self, topic_node_id: int, alerted_at: str) -> None:
        """Stamp last_alerted_at on all attention rows for this topic in the current window.

        Updating all rows in the 7-day window ensures MAX(last_alerted_at) is
        visible to the cooldown check in get_all_tracked_topic_ids() regardless
        of which specific date row is queried. Must be called only after a
        successful alert dispatch.
        """
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE kg_topic_attention SET last_alerted_at = ?"
                " WHERE topic_node_id = ? AND window_date > ?",
                (alerted_at, topic_node_id, cutoff),
            )

    def get_node_canonical_name(self, node_id: int) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT canonical_name FROM kg_nodes WHERE id = ?",
                (node_id,),
            ).fetchone()
        return row["canonical_name"] if row else None

    def get_top_momentum_topics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return topics sorted by recent intensity for query-time surfacing."""
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
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
        story_node_id: int,
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
                " WHERE story_node_id = ? AND source_node_id = ?"
                " ORDER BY created_at DESC LIMIT 1",
                (story_node_id, source_node_id),
            ).fetchone()

            if existing is None:
                emails = json.dumps([email_id])
                conn.execute(
                    "INSERT INTO kg_contrarian_positions"
                    " (story_node_id, source_node_id, position_text, prevailing_view,"
                    "  evidence_email_ids, evidence_count, status, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, 1, 'pending', ?, ?)",
                    (
                        story_node_id, source_node_id, position_text,
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
        story_node_id: Optional[int] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return confirmed contrarian positions, optionally filtered by story."""
        with self._lock, self._connect() as conn:
            if story_node_id is not None:
                rows = conn.execute(
                    "SELECT cp.*, n_t.canonical_name AS story_name,"
                    "        n_s.canonical_name AS source_name"
                    " FROM kg_contrarian_positions cp"
                    " JOIN kg_nodes n_t ON cp.story_node_id = n_t.id"
                    " JOIN kg_nodes n_s ON cp.source_node_id = n_s.id"
                    " WHERE cp.story_node_id = ? AND cp.status = 'confirmed'"
                    " ORDER BY cp.updated_at DESC LIMIT ?",
                    (story_node_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT cp.*, n_t.canonical_name AS story_name,"
                    "        n_s.canonical_name AS source_name"
                    " FROM kg_contrarian_positions cp"
                    " JOIN kg_nodes n_t ON cp.story_node_id = n_t.id"
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
        safe = topic_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{safe}%"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT n.id, n.canonical_name"
                " FROM kg_nodes n"
                " JOIN kg_edges e ON e.from_node_id = n.id"
                "   AND e.edge_type = 'involves' AND e.is_active = 1"
                " JOIN kg_nodes nt ON e.to_node_id = nt.id AND nt.node_type = 'topic'"
                " WHERE n.node_type = 'story'"
                "   AND (nt.canonical_name LIKE ? ESCAPE '\\' OR n.canonical_name LIKE ? ESCAPE '\\')"
                " LIMIT 10",
                (pattern, pattern),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Multi-hop traversal queries
    # ------------------------------------------------------------------

    def get_stories_covering_topic(
        self,
        topic_name: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Two-hop: story nodes whose involves edges point to a topic matching topic_name.

        Traversal: topic_node ← (involves) ← story_node.
        Returns story id, canonical_name, and the matched topic name.
        """
        safe = topic_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{safe}%"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT s.id, s.canonical_name, t.canonical_name AS topic_name"
                " FROM kg_nodes t"
                " JOIN kg_edges e ON t.id = e.to_node_id"
                "   AND e.edge_type = 'involves' AND e.is_active = 1"
                " JOIN kg_nodes s ON e.from_node_id = s.id"
                "   AND s.node_type = 'story' AND s.is_deprecated = 0"
                " WHERE t.node_type = 'topic' AND t.canonical_name LIKE ? ESCAPE '\\'"
                " LIMIT ?",
                (pattern, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_topics_covered_by_source(
        self,
        source_name: str,
        days: int = 30,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Two-hop: topic nodes reachable from a source via covers → involves edges.

        Traversal: source_node → (covers) → story_node → (involves) → topic_node.
        days: restrict the source → story covers edges to those whose
        source_email_timestamp is within the last N days.
        """
        safe = source_name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{safe}%"
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).isoformat(timespec="seconds")
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT t.id, t.canonical_name, t.node_type"
                " FROM kg_nodes src"
                " JOIN kg_edges e1 ON src.id = e1.from_node_id"
                "   AND e1.edge_type = 'covers' AND e1.is_active = 1"
                "   AND e1.source_email_timestamp > ?"
                " JOIN kg_nodes s ON e1.to_node_id = s.id"
                "   AND s.node_type = 'story' AND s.is_deprecated = 0"
                " JOIN kg_edges e2 ON s.id = e2.from_node_id"
                "   AND e2.edge_type = 'involves' AND e2.is_active = 1"
                " JOIN kg_nodes t ON e2.to_node_id = t.id"
                "   AND t.node_type = 'topic' AND t.is_deprecated = 0"
                " WHERE src.node_type = 'newsletter_source'"
                "   AND src.canonical_name LIKE ? ESCAPE '\\' AND src.is_deprecated = 0"
                " LIMIT ?",
                (cutoff, pattern, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Claim tracking
    # ------------------------------------------------------------------

    def insert_claim(
        self,
        *,
        story_node_id: Optional[int],
        source_node_id: int,
        claim_text: str,
        claim_date: str,
        source_email_id: str,
    ) -> int:
        """Insert a new claim record and return its ID.

        Persists the key_claim extracted by the newsletter LLM into a
        dedicated table linked to the story and source. Verification status
        starts as 'pending' and is updated by external verification passes.
        """
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO kg_claims"
                " (story_node_id, source_node_id, claim_text, claim_date,"
                "  verification_status, verified_at, source_email_id)"
                " VALUES (?, ?, ?, ?, 'pending', NULL, ?)",
                (story_node_id, source_node_id, claim_text, claim_date, source_email_id),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def increment_claim_count(self, source_node_id: int) -> None:
        """Increment claim_total_count for a newsletter source.

        Separate from mark_claim_verified so callers can count submitted
        claims independently of whether they have been verified.
        """
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE kg_newsletter_sources"
                " SET claim_total_count = claim_total_count + 1"
                " WHERE node_id = ?",
                (source_node_id,),
            )

    def mark_claim_verified(
        self,
        claim_id: int,
        *,
        is_correct: bool,
        novelty_score: Optional[float] = None,
    ) -> None:
        """Record the outcome of a claim verification and update source credibility.

        Updates verification_status and verified_at on the claim row, then
        atomically increments claim_total_count (and claim_correct_count when
        is_correct=True) on the source. When novelty_score is provided,
        avg_novelty_score is updated using the incremental running-average
        formula — (old_avg * old_count + new_score) / new_count — so precision
        is not lost through a cumulative-sum approach.
        """
        now = _utc_now_iso()
        status = "correct" if is_correct else "incorrect"

        with self._lock, self._connect() as conn:
            claim_row = conn.execute(
                "SELECT source_node_id FROM kg_claims WHERE id = ?",
                (claim_id,),
            ).fetchone()
            if claim_row is None:
                return
            source_node_id = int(claim_row["source_node_id"])

            conn.execute(
                "UPDATE kg_claims SET verification_status = ?, verified_at = ? WHERE id = ?",
                (status, now, claim_id),
            )

            if novelty_score is not None:
                # avg_novelty_score in the SET expression uses the OLD claim_total_count
                # (SQLite evaluates all SET RHS with pre-update row values) so the
                # incremental formula (old_avg * old_count + new_score) / (old_count + 1)
                # is computed correctly before claim_total_count is incremented.
                conn.execute(
                    "UPDATE kg_newsletter_sources"
                    " SET claim_total_count  = claim_total_count + 1,"
                    "     claim_correct_count = claim_correct_count + ?,"
                    "     avg_novelty_score   = (avg_novelty_score * claim_total_count + ?)"
                    "                           / (claim_total_count + 1)"
                    " WHERE node_id = ?",
                    (1 if is_correct else 0, novelty_score, source_node_id),
                )
            else:
                conn.execute(
                    "UPDATE kg_newsletter_sources"
                    " SET claim_total_count  = claim_total_count + 1,"
                    "     claim_correct_count = claim_correct_count + ?"
                    " WHERE node_id = ?",
                    (1 if is_correct else 0, source_node_id),
                )

    # ------------------------------------------------------------------
    # Source credibility
    # ------------------------------------------------------------------

    def get_source_credibility(self, node_id: int) -> Dict[str, Any]:
        """Return credibility metrics for a newsletter source node."""
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT claim_correct_count, claim_total_count, avg_novelty_score"
                " FROM kg_newsletter_sources WHERE node_id = ?",
                (node_id,),
            ).fetchone()
        if row is None:
            return {"claim_accuracy": None, "avg_novelty_score": None}
        total = int(row["claim_total_count"])
        correct = int(row["claim_correct_count"])
        return {
            "claim_accuracy": round(correct / total, 3) if total > 0 else None,
            "claim_total": total,
            "claim_correct": correct,
            "avg_novelty_score": float(row["avg_novelty_score"]),
        }

    def get_all_sources(self) -> List[Dict[str, Any]]:
        """Return all tracked newsletter sources with credibility data."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ks.node_id, ks.publication_name, ks.sender_patterns,"
                "       ks.email_count, n.canonical_name,"
                "       ks.claim_correct_count, ks.claim_total_count,"
                "       ks.avg_novelty_score,"
                "       COALESCE(ks.sentiment_profile, '{}') AS sentiment_profile"
                " FROM kg_newsletter_sources ks"
                " JOIN kg_nodes n ON ks.node_id = n.id",
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Source consistency profiling
    # ------------------------------------------------------------------

    def update_source_sentiment_profile(self, node_id: int) -> Dict[str, Any]:
        """Recompute and persist the sentiment fingerprint for a source.

        Aggregates all active framings from kg_story_framings for this source,
        counts bearish/bullish/neutral/etc. per topic, and writes a summary
        back to kg_newsletter_sources.sentiment_profile. Returns the computed
        profile so the caller can log or surface it.

        Aggregation is read-heavy and designed to be called infrequently
        (e.g. once per momentum check cycle, not per email).
        """
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT sf.sentiment, t.canonical_name AS topic"
                " FROM kg_story_framings sf"
                " JOIN kg_edges e ON sf.story_node_id = e.from_node_id"
                "   AND e.edge_type = 'involves' AND e.is_active = 1"
                " JOIN kg_nodes t ON e.to_node_id = t.id AND t.node_type = 'topic'"
                " WHERE sf.source_node_id = ? AND sf.is_active = 1",
                (node_id,),
            ).fetchall()

        sentiment_by_topic: Dict[str, Dict[str, int]] = {}
        for r in rows:
            topic = r["topic"]
            sent = r["sentiment"]
            if topic not in sentiment_by_topic:
                sentiment_by_topic[topic] = {}
            sentiment_by_topic[topic][sent] = sentiment_by_topic[topic].get(sent, 0) + 1

        bearish_topics = [
            t for t, counts in sentiment_by_topic.items()
            if counts.get("bearish", 0) > sum(counts.values()) * 0.5
        ]
        bullish_topics = [
            t for t, counts in sentiment_by_topic.items()
            if counts.get("bullish", 0) > sum(counts.values()) * 0.5
        ]
        total_framings = sum(sum(c.values()) for c in sentiment_by_topic.values())
        neutral_count = sum(
            c.get("neutral", 0) for c in sentiment_by_topic.values()
        )
        neutral_rate = round(neutral_count / total_framings, 3) if total_framings > 0 else 1.0

        profile = {
            "bearish_topics": bearish_topics,
            "bullish_topics": bullish_topics,
            "neutral_rate": neutral_rate,
        }
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE kg_newsletter_sources SET sentiment_profile = ? WHERE node_id = ?",
                (json.dumps(profile), node_id),
            )
        return profile


_nl_store_instance: Optional[NewsletterGraphStore] = None


def get_newsletter_graph_store() -> NewsletterGraphStore:
    global _nl_store_instance
    if _nl_store_instance is None:
        from ..store import get_knowledge_graph_store
        _nl_store_instance = NewsletterGraphStore(get_knowledge_graph_store())
    return _nl_store_instance


__all__ = ["NewsletterGraphStore", "get_newsletter_graph_store"]
