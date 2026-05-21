"""Shared ChromaDB PersistentClient for the knowledge graph service.

Both store.py (kg_nodes collection) and newsletter/novelty.py (story_framings
collection) target the same on-disk path. Opening two PersistentClient
instances against the same path gives each instance its own in-memory HNSW
index; writes from one client stale the other's index and cause gradual
corruption. This module ensures a single PersistentClient is shared so all
collections are obtained from the same instance.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ...logging_config import logger

_KG_CHROMA_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_graph" / "chroma"
)

_chroma_client = None
_chroma_lock = threading.Lock()


def get_chroma_client():
    """Return the shared ChromaDB PersistentClient, creating it on first call.

    Uses double-checked locking — same pattern as store.py._get_kg_collection.
    Returns None if ChromaDB is unavailable; callers must handle None gracefully.
    """
    global _chroma_client
    if _chroma_client is None:
        with _chroma_lock:
            if _chroma_client is None:
                try:
                    import chromadb
                    from chromadb.config import Settings
                    _KG_CHROMA_PATH.mkdir(parents=True, exist_ok=True)
                    _chroma_client = chromadb.PersistentClient(
                        path=str(_KG_CHROMA_PATH),
                        settings=Settings(anonymized_telemetry=False),
                    )
                except Exception as exc:
                    logger.debug("Shared ChromaDB client unavailable: %s", exc)
    return _chroma_client


__all__ = ["get_chroma_client"]
