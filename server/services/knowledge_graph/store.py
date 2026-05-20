"""SQLite-backed persistence for the email knowledge graph."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...logging_config import logger

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DB_PATH = _DATA_DIR / "knowledge_graph.db"

# Cosine similarity threshold for merging a new node name into an existing node.
# A submitted name scoring at or above this against an existing node of the same
# type is treated as a synonym and resolved to the existing canonical name.
_NODE_SIMILARITY_THRESHOLD = 0.88

_kg_chroma_collection = None
_kg_chroma_lock = threading.Lock()

CONFIDENCE_THRESHOLD = 0.6
_STALE_DAYS = 90


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_kg_collection():
    """Get the ChromaDB collection for KG node embeddings.

    Uses double-checked locking (same pattern as vector_index._get_collection).
    Returns None if ChromaDB is unavailable; callers fall back to exact matching.
    """
    global _kg_chroma_collection
    if _kg_chroma_collection is None:
        with _kg_chroma_lock:
            if _kg_chroma_collection is None:
                try:
                    from .shared_chroma import get_chroma_client
                    client = get_chroma_client()
                    if client is not None:
                        _kg_chroma_collection = client.get_or_create_collection(
                            "kg_nodes",
                            metadata={"hnsw:space": "cosine"},
                        )
                except Exception as exc:
                    logger.debug(
                        "KG ChromaDB unavailable; entity synonym resolution disabled: %s", exc
                    )
    return _kg_chroma_collection


def _resolve_node_name(
    node_type: str,
    canonical_name: str,
    threshold: float = _NODE_SIMILARITY_THRESHOLD,
) -> str:
    """Return the canonical_name of an existing node if semantically equivalent.

    Queries ChromaDB for the closest node of the same type. Returns the existing
    node's canonical_name when similarity >= threshold, otherwise returns the
    submitted canonical_name unchanged.

    threshold defaults to _NODE_SIMILARITY_THRESHOLD (0.88) for entity resolution.
    Story title matching uses a lower threshold (0.80) because titles legitimately
    vary more than entity names.

    Falls back to returning canonical_name (exact-match behaviour) when:
      - ChromaDB is unavailable
      - The collection is empty
      - Any ChromaDB error occurs
    """
    try:
        collection = _get_kg_collection()
        if collection is None:
            return canonical_name
        count = collection.count()
        if count == 0:
            return canonical_name
        results = collection.query(
            query_texts=[canonical_name],
            n_results=1,
            where={"node_type": node_type},
            include=["documents", "distances"],
        )
        ids = (results.get("ids") or [[]])[0]
        if not ids:
            return canonical_name
        best_distance = results["distances"][0][0]
        best_doc = results["documents"][0][0]
        similarity = 1.0 - best_distance
        if similarity >= threshold:
            logger.debug(
                "Entity synonym resolved to existing node",
                extra={
                    "submitted": canonical_name,
                    "resolved": best_doc,
                    "node_type": node_type,
                    "similarity": round(similarity, 3),
                    "threshold": threshold,
                },
            )
            return best_doc
    except Exception as exc:
        logger.debug("Entity resolution ChromaDB query failed, using exact match: %s", exc)
    return canonical_name


def _index_node(node_type: str, canonical_name: str) -> None:
    """Upsert a node into ChromaDB for future entity synonym resolution.

    Idempotent — safe to call on nodes that are already indexed.
    Silently skips when ChromaDB is unavailable.
    """
    try:
        collection = _get_kg_collection()
        if collection is None:
            return
        doc_id = f"{node_type}::{canonical_name}"
        collection.upsert(
            ids=[doc_id],
            documents=[canonical_name],
            metadatas=[{"node_type": node_type}],
        )
    except Exception as exc:
        logger.debug("KG node ChromaDB indexing failed: %s", exc)


class KnowledgeGraphStore:
    """Low-level persistence for the knowledge graph backed by SQLite.

    Threading model: one threading.Lock guards all connections, matching
    TriggerStore. SQLite WAL mode allows concurrent readers from external
    tools but the lock ensures atomic read-check-write sequences within
    this process.
    """

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        # executescript commits any open transaction before running; safe with isolation_level=None
        schema = """
        PRAGMA journal_mode=WAL;

        CREATE TABLE IF NOT EXISTS kg_nodes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            node_type       TEXT    NOT NULL,
            canonical_name  TEXT    NOT NULL,
            is_deprecated   INTEGER NOT NULL DEFAULT 0,
            created_at      TEXT    NOT NULL,
            updated_at      TEXT    NOT NULL,
            UNIQUE(node_type, canonical_name)
        );

        CREATE INDEX IF NOT EXISTS idx_kg_nodes_type_name
            ON kg_nodes (node_type, canonical_name);

        CREATE TABLE IF NOT EXISTS kg_node_facts (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id                 INTEGER NOT NULL REFERENCES kg_nodes(id),
            fact_key                TEXT    NOT NULL,
            fact_value              TEXT    NOT NULL,
            version                 INTEGER NOT NULL DEFAULT 1,
            confidence              REAL    NOT NULL DEFAULT 1.0,
            is_active               INTEGER NOT NULL DEFAULT 1,
            flagged                 INTEGER NOT NULL DEFAULT 0,
            novelty_score           REAL    NOT NULL DEFAULT 1.0,
            source_email_id         TEXT    NOT NULL,
            source_email_timestamp  TEXT    NOT NULL,
            extracted_at            TEXT    NOT NULL,
            created_at              TEXT    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_kg_facts_node_key_active
            ON kg_node_facts (node_id, fact_key, is_active);

        CREATE INDEX IF NOT EXISTS idx_kg_facts_email_id
            ON kg_node_facts (source_email_id);

        CREATE TABLE IF NOT EXISTS kg_edges (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            from_node_id            INTEGER NOT NULL REFERENCES kg_nodes(id),
            to_node_id              INTEGER NOT NULL REFERENCES kg_nodes(id),
            edge_type               TEXT    NOT NULL,
            version                 INTEGER NOT NULL DEFAULT 1,
            confidence              REAL    NOT NULL DEFAULT 1.0,
            is_active               INTEGER NOT NULL DEFAULT 1,
            flagged                 INTEGER NOT NULL DEFAULT 0,
            novelty_score           REAL    NOT NULL DEFAULT 1.0,
            source_email_id         TEXT    NOT NULL,
            source_email_timestamp  TEXT    NOT NULL,
            extracted_at            TEXT    NOT NULL,
            created_at              TEXT    NOT NULL,
            properties              TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_kg_edges_from_active
            ON kg_edges (from_node_id, edge_type, is_active);

        CREATE INDEX IF NOT EXISTS idx_kg_edges_to_active
            ON kg_edges (to_node_id, edge_type, is_active);

        CREATE TABLE IF NOT EXISTS kg_processed_emails (
            email_id        TEXT PRIMARY KEY,
            processed_at    TEXT NOT NULL
        );
        """
        with self._lock, self._connect() as conn:
            conn.executescript(schema)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def is_email_processed(self, email_id: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM kg_processed_emails WHERE email_id = ?",
                (email_id,),
            ).fetchone()
        return row is not None

    def mark_email_processed(self, email_id: str) -> None:
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO kg_processed_emails (email_id, processed_at) VALUES (?, ?)",
                (email_id, now),
            )

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def get_or_create_node(self, node_type: str, canonical_name: str) -> int:
        """Return the node ID, creating the node atomically if it does not exist.

        Before creating a new node, a synonym normalization pass runs inside this
        lock: ChromaDB is queried for an existing node of the same type whose
        embedding is within _NODE_SIMILARITY_THRESHOLD of canonical_name. If one
        is found, that node's ID is returned instead of creating a duplicate. This
        prevents topic fragmentation from synonymous names (e.g. "AI regulation"
        vs "AI policy"). Falls back to exact matching when ChromaDB is unavailable.
        """
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            # Fast path: exact match
            row = conn.execute(
                "SELECT id FROM kg_nodes"
                " WHERE node_type = ? AND canonical_name = ? AND is_deprecated = 0",
                (node_type, canonical_name),
            ).fetchone()
            if row is not None:
                node_id = int(row["id"])
            else:
                # Normalization pass: resolve synonym to an existing node before creating
                resolved_name = _resolve_node_name(node_type, canonical_name)
                if resolved_name != canonical_name:
                    resolved_row = conn.execute(
                        "SELECT id FROM kg_nodes"
                        " WHERE node_type = ? AND canonical_name = ? AND is_deprecated = 0",
                        (node_type, resolved_name),
                    ).fetchone()
                    if resolved_row is not None:
                        # Synonym resolved — return existing node without creating a new one
                        return int(resolved_row["id"])
                    # resolved_name not found (stale ChromaDB index); fall through to create

                conn.execute(
                    "INSERT OR IGNORE INTO kg_nodes"
                    " (node_type, canonical_name, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?)",
                    (node_type, canonical_name, now, now),
                )
                row = conn.execute(
                    "SELECT id FROM kg_nodes WHERE node_type = ? AND canonical_name = ?",
                    (node_type, canonical_name),
                ).fetchone()
                node_id = int(row["id"])

        # Index outside lock — best-effort, non-atomic with SQLite write; idempotent
        _index_node(node_type, canonical_name)
        return node_id

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def upsert_fact(
        self,
        *,
        node_id: int,
        fact_key: str,
        fact_value: str,
        confidence: float,
        source_email_id: str,
        source_email_timestamp: str,
        extracted_at: str,
        novelty_score: float = 1.0,
    ) -> Tuple[bool, bool]:
        """Insert or update a node fact atomically.

        Returns (inserted, contradiction_detected).

        Contradiction: an active fact already exists for (node_id, fact_key)
        with a different fact_value. When detected, the old fact is set
        is_active=0 (preserved, not deleted) and a new fact with
        version=old_version+1 is inserted. The entire read-check-write
        sequence is protected by self._lock.
        """
        now = _utc_now_iso()
        flagged = 1 if confidence < CONFIDENCE_THRESHOLD else 0

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id, fact_value, version FROM kg_node_facts"
                " WHERE node_id = ? AND fact_key = ? AND is_active = 1",
                (node_id, fact_key),
            ).fetchone()

            if existing is None:
                conn.execute(
                    "INSERT INTO kg_node_facts"
                    " (node_id, fact_key, fact_value, version, confidence,"
                    "  is_active, flagged, novelty_score, source_email_id, source_email_timestamp,"
                    "  extracted_at, created_at)"
                    " VALUES (?, ?, ?, 1, ?, 1, ?, ?, ?, ?, ?, ?)",
                    (
                        node_id, fact_key, fact_value, confidence, flagged,
                        novelty_score,
                        source_email_id, source_email_timestamp, extracted_at, now,
                    ),
                )
                return True, False

            existing_value = existing["fact_value"]
            existing_version = int(existing["version"])

            if existing_value == fact_value:
                return False, False

            # Contradiction: deprecate existing, insert new version
            conn.execute(
                "UPDATE kg_node_facts SET is_active = 0 WHERE id = ?",
                (existing["id"],),
            )
            conn.execute(
                "INSERT INTO kg_node_facts"
                " (node_id, fact_key, fact_value, version, confidence,"
                "  is_active, flagged, novelty_score, source_email_id, source_email_timestamp,"
                "  extracted_at, created_at)"
                " VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
                (
                    node_id, fact_key, fact_value, existing_version + 1,
                    confidence, flagged,
                    novelty_score,
                    source_email_id, source_email_timestamp, extracted_at, now,
                ),
            )
            return True, True

    def upsert_edge(
        self,
        *,
        from_node_id: int,
        to_node_id: int,
        edge_type: str,
        confidence: float,
        source_email_id: str,
        source_email_timestamp: str,
        extracted_at: str,
        properties: Optional[Dict[str, Any]] = None,
        novelty_score: float = 1.0,
    ) -> bool:
        """Insert an edge if no active edge of the same type exists between these nodes.

        properties: optional JSON-serialisable dict stored on the edge (e.g.
            {"role": "CTO", "since": "2022"} for a works_at edge). NULL properties
            means no properties — compatible with all existing rows.

        Returns True if a new edge was inserted.
        """
        now = _utc_now_iso()
        flagged = 1 if confidence < CONFIDENCE_THRESHOLD else 0
        props_json: Optional[str] = json.dumps(properties) if properties else None

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM kg_edges"
                " WHERE from_node_id = ? AND to_node_id = ? AND edge_type = ? AND is_active = 1",
                (from_node_id, to_node_id, edge_type),
            ).fetchone()

            if existing is None:
                conn.execute(
                    "INSERT INTO kg_edges"
                    " (from_node_id, to_node_id, edge_type, version, confidence,"
                    "  is_active, flagged, novelty_score, source_email_id, source_email_timestamp,"
                    "  extracted_at, created_at, properties)"
                    " VALUES (?, ?, ?, 1, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        from_node_id, to_node_id, edge_type, confidence, flagged,
                        novelty_score,
                        source_email_id, source_email_timestamp, extracted_at, now,
                        props_json,
                    ),
                )
                return True
            # Confidence is monotonically non-decreasing: use MAX so a low-quality
            # re-observation cannot overwrite high-confidence established evidence.
            # COALESCE preserves existing properties when the new edge carries none.
            # Provenance (source_email_id, source_email_timestamp, extracted_at) is
            # immutable: the first observation establishes authorship and must not be
            # overwritten by later re-observations of the same edge.
            conn.execute(
                "UPDATE kg_edges"
                " SET confidence = MAX(confidence, ?),"
                "     flagged = CASE WHEN MAX(confidence, ?) >= ? THEN 0 ELSE flagged END,"
                "     properties = COALESCE(?, properties)"
                " WHERE id = ?",
                (
                    confidence, confidence, CONFIDENCE_THRESHOLD,
                    props_json, existing["id"],
                ),
            )
        return False

    # ------------------------------------------------------------------
    # Deprecation
    # ------------------------------------------------------------------

    def deprecate_stale_facts(self, older_than_days: int = _STALE_DAYS) -> int:
        """Set is_active=0 on facts/edges whose source email is older than N days.

        Facts are never deleted — the full history remains recoverable.
        Returns the total number of rows deprecated.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        cutoff_iso = cutoff.isoformat(timespec="seconds")

        with self._lock, self._connect() as conn:
            cursor_facts = conn.execute(
                "UPDATE kg_node_facts SET is_active = 0"
                " WHERE is_active = 1 AND source_email_timestamp < ?",
                (cutoff_iso,),
            )
            cursor_edges = conn.execute(
                "UPDATE kg_edges SET is_active = 0"
                " WHERE is_active = 1 AND source_email_timestamp < ?",
                (cutoff_iso,),
            )
            deprecated = cursor_facts.rowcount + cursor_edges.rowcount

        if deprecated:
            logger.info(
                "Knowledge graph stale deprecation complete",
                extra={
                    "deprecated_facts": cursor_facts.rowcount,
                    "deprecated_edges": cursor_edges.rowcount,
                    "cutoff_days": older_than_days,
                },
            )
        return deprecated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_node_by_name(
        self,
        canonical_name: str,
        node_type: Optional[str] = None,
        include_flagged: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Return a node and its active facts/edges, or None if not found."""
        with self._lock, self._connect() as conn:
            if node_type:
                row = conn.execute(
                    "SELECT * FROM kg_nodes"
                    " WHERE canonical_name = ? AND node_type = ? AND is_deprecated = 0",
                    (canonical_name, node_type),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM kg_nodes"
                    " WHERE canonical_name = ? AND is_deprecated = 0",
                    (canonical_name,),
                ).fetchone()

            if row is None:
                return None

            node_id = int(row["id"])
            flagged_clause = "" if include_flagged else " AND flagged = 0"

            facts = conn.execute(
                f"SELECT fact_key, fact_value, confidence, version, source_email_timestamp"
                f" FROM kg_node_facts"
                f" WHERE node_id = ? AND is_active = 1{flagged_clause}"
                f" ORDER BY fact_key, version DESC",
                (node_id,),
            ).fetchall()

            edges = conn.execute(
                f"SELECT e.edge_type, e.confidence, e.version,"
                f" n.node_type AS to_type, n.canonical_name AS to_name,"
                f" e.properties"
                f" FROM kg_edges e JOIN kg_nodes n ON e.to_node_id = n.id"
                f" WHERE e.from_node_id = ? AND e.is_active = 1{flagged_clause}",
                (node_id,),
            ).fetchall()

        def _deserialize_edge(e: sqlite3.Row) -> dict:
            row = dict(e)
            props_raw = row.get("properties")
            row["properties"] = json.loads(props_raw) if isinstance(props_raw, str) else props_raw
            return row

        return {
            "id": node_id,
            "node_type": row["node_type"],
            "canonical_name": row["canonical_name"],
            "facts": [dict(f) for f in facts],
            "edges": [_deserialize_edge(e) for e in edges],
        }

    def query_nodes_by_type(
        self,
        node_type: str,
        include_flagged: bool = False,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return all non-deprecated nodes of a given type with their active facts."""
        flagged_clause = "" if include_flagged else " AND flagged = 0"

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kg_nodes"
                " WHERE node_type = ? AND is_deprecated = 0"
                " ORDER BY canonical_name LIMIT ?",
                (node_type, limit),
            ).fetchall()

            result = []
            for row in rows:
                node_id = int(row["id"])
                facts = conn.execute(
                    f"SELECT fact_key, fact_value, confidence, version, source_email_timestamp"
                    f" FROM kg_node_facts"
                    f" WHERE node_id = ? AND is_active = 1{flagged_clause}"
                    f" ORDER BY fact_key",
                    (node_id,),
                ).fetchall()
                result.append({
                    "node_type": row["node_type"],
                    "canonical_name": row["canonical_name"],
                    "facts": [dict(f) for f in facts],
                })

        return result

    def query_fact_history(self, node_id: int, fact_key: str) -> List[Dict[str, Any]]:
        """Return all versions of a fact, including deprecated ones, in version order."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kg_node_facts"
                " WHERE node_id = ? AND fact_key = ?"
                " ORDER BY version",
                (node_id, fact_key),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_nodes(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Case-insensitive substring search over canonical_name."""
        safe_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{safe_query}%"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, node_type, canonical_name FROM kg_nodes"
                " WHERE canonical_name LIKE ? ESCAPE '\\' AND is_deprecated = 0"
                " ORDER BY canonical_name LIMIT ?",
                (pattern, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def query_nodes_by_edge_target(
        self,
        to_node_id: int,
        edge_type: str,
        include_flagged: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return nodes that have an active edge of edge_type pointing TO to_node_id.

        Enables reverse lookups such as "who works_at Acme Corp?" without
        enumerating all person nodes. Each result includes the edge confidence
        and any edge properties.
        """
        flagged_clause = "" if include_flagged else " AND e.flagged = 0"
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT n.id, n.node_type, n.canonical_name,"
                f" e.confidence AS edge_confidence, e.properties AS edge_properties"
                f" FROM kg_edges e JOIN kg_nodes n ON e.from_node_id = n.id"
                f" WHERE e.to_node_id = ? AND e.edge_type = ? AND e.is_active = 1"
                f"{flagged_clause}",
                (to_node_id, edge_type),
            ).fetchall()

        def _deserialize_row(r: sqlite3.Row) -> dict:
            row = dict(r)
            props_raw = row.get("edge_properties")
            row["edge_properties"] = (
                json.loads(props_raw) if isinstance(props_raw, str) else props_raw
            )
            return row

        return [_deserialize_row(r) for r in rows]

    def query_entities_semantically(
        self,
        text: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return up to n_results entity names whose embeddings are closest to text.

        Uses the kg_nodes ChromaDB collection. Returns a list of dicts with
        keys: id (chroma doc id), canonical_name, node_type, similarity.
        Returns an empty list when ChromaDB is unavailable or the collection is empty.
        """
        try:
            collection = _get_kg_collection()
            if collection is None or collection.count() == 0:
                return []
            results = collection.query(
                query_texts=[text],
                n_results=min(n_results, collection.count()),
                include=["documents", "distances", "metadatas"],
            )
            docs = (results.get("documents") or [[]])[0]
            distances = (results.get("distances") or [[]])[0]
            metas = (results.get("metadatas") or [[]])[0]
            out = []
            for doc, dist, meta in zip(docs, distances, metas):
                out.append({
                    "canonical_name": doc,
                    "node_type": (meta or {}).get("node_type", ""),
                    "similarity": round(1.0 - dist, 4),
                })
            return out
        except Exception as exc:
            logger.debug("Semantic entity query failed: %s", exc)
            return []

    def get_node_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM kg_nodes WHERE is_deprecated = 0"
            ).fetchone()
        return int(row[0])

    def get_fact_count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM kg_node_facts WHERE is_active = 1"
            ).fetchone()
        return int(row[0])


_store_instance: Optional[KnowledgeGraphStore] = None


def get_knowledge_graph_store() -> KnowledgeGraphStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = KnowledgeGraphStore()
    return _store_instance


__all__ = [
    "KnowledgeGraphStore",
    "get_knowledge_graph_store",
    "CONFIDENCE_THRESHOLD",
]
