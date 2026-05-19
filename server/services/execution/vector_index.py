"""Semantic search over agent descriptors using chromadb."""

import threading
from pathlib import Path

from ...logging_config import logger
from .descriptor import AgentDescriptor

_CHROMA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "execution_agents" / "chroma"

_chroma_collection = None
_chroma_lock = threading.Lock()


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        with _chroma_lock:
            if _chroma_collection is None:
                import chromadb
                from chromadb.config import Settings
                client = chromadb.PersistentClient(
                    path=str(_CHROMA_PATH),
                    settings=Settings(anonymized_telemetry=False),
                )
                _chroma_collection = client.get_or_create_collection("agents")
    return _chroma_collection


def upsert_descriptor(descriptor: AgentDescriptor) -> None:
    """Index or re-index an agent descriptor."""
    try:
        collection = _get_collection()
        document = (
            f"{descriptor.agent_name}. "
            f"{descriptor.summary}. "
            f"Tags: {', '.join(descriptor.tags)}"
        )
        collection.upsert(
            ids=[descriptor.agent_name],
            documents=[document],
            metadatas=[{
                "status": descriptor.status,
                "last_active": descriptor.last_active,
                "last_output_snippet": descriptor.last_output_snippet,
            }],
        )
    except Exception as exc:
        logger.warning(f"Failed to upsert descriptor for {descriptor.agent_name}: {exc}")


def search_agents(query: str, top_k: int = 5) -> list[dict]:
    """Return the top-k agents most semantically similar to query."""
    try:
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for i, agent_name in enumerate(results["ids"][0]):
            output.append({
                "agent_name": agent_name,
                "summary": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "relevance": round(1 - results["distances"][0][i], 3),
            })
        return output
    except Exception as exc:
        logger.warning(f"Agent search failed: {exc}")
        return []
