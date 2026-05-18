"""Knowledge graph query tool — read-only access for execution agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from server.logging_config import logger
from server.services.knowledge_graph import get_knowledge_graph_store
from .schemas import TASK_TOOL_NAME


def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:  # noqa: ARG001
    """Return query tool callable. agent_name unused; present for registry protocol."""
    return {TASK_TOOL_NAME: query_knowledge_graph}


def query_knowledge_graph(
    entity_name: Optional[str] = None,
    node_type: Optional[str] = None,
    list_type: Optional[str] = None,
) -> Any:
    """Query the knowledge graph for entities, facts, and relationships.

    This function is intentionally synchronous — KnowledgeGraphStore uses
    threading.Lock (not asyncio) and the execution agent runtime calls
    tool functions synchronously before awaiting the result.

    Read-only: no writes are performed here or anywhere reachable from here.
    """
    store = get_knowledge_graph_store()

    if entity_name:
        name = entity_name.strip()
        if not name:
            return {"error": "entity_name must not be empty"}

        result = store.query_node_by_name(name, node_type=node_type)
        if result:
            logger.debug(
                "KG query hit",
                extra={"entity_name": name, "node_type": result.get("node_type")},
            )
            return result

        # Exact match failed — try substring search as fallback
        candidates = store.search_nodes(name)
        if candidates:
            logger.debug(
                "KG query: no exact match, returning candidates",
                extra={"entity_name": name, "candidates": len(candidates)},
            )
            return {
                "result": None,
                "message": f"No exact match for '{name}'. Possible matches below.",
                "candidates": candidates,
            }

        return {
            "result": None,
            "message": f"No entity found matching '{name}'.",
        }

    if list_type:
        nodes = store.query_nodes_by_type(list_type)
        logger.debug(
            "KG list query",
            extra={"list_type": list_type, "count": len(nodes)},
        )
        return {
            "node_type": list_type,
            "count": len(nodes),
            "nodes": nodes,
        }

    # Neither argument provided — return graph stats as orientation
    return {
        "node_count": store.get_node_count(),
        "fact_count": store.get_fact_count(),
        "message": (
            "Provide entity_name to look up a specific entity, "
            "or list_type to enumerate all entities of a given type."
        ),
    }


__all__ = ["build_registry", "query_knowledge_graph"]
