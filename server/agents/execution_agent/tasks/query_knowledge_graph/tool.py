"""Knowledge graph query tool — read-only access for execution agents."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from server.logging_config import logger
from server.services.knowledge_graph import get_knowledge_graph_store
from server.services.knowledge_graph.newsletter.store_ext import get_newsletter_graph_store
from .schemas import TASK_TOOL_NAME


def build_registry(agent_name: str) -> Dict[str, Callable[..., Any]]:  # noqa: ARG001
    """Return query tool callable. agent_name unused; present for registry protocol."""
    return {TASK_TOOL_NAME: query_knowledge_graph}


def query_knowledge_graph(
    entity_name: Optional[str] = None,
    node_type: Optional[str] = None,
    list_type: Optional[str] = None,
    newsletter_query: Optional[str] = None,
) -> Any:
    """Query the knowledge graph for entities, facts, and relationships.

    Synchronous — KnowledgeGraphStore uses threading.Lock (not asyncio).
    Read-only: no writes are performed here or anywhere reachable from here.

    newsletter_query modes add access to the newsletter intelligence layer:
        story_narrative      — full framing timeline for a story or topic
        momentum_topics      — topics with accelerating coverage this week
        contrarian_positions — confirmed dissenting positions
        newsletter_sources   — tracked publications
    """
    store = get_knowledge_graph_store()

    # ---- Newsletter-specific query modes --------------------------------
    if newsletter_query:
        return _handle_newsletter_query(
            newsletter_query,
            entity_name=entity_name,
        )

    # ---- Standard entity lookup ----------------------------------------
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

    return {
        "node_count": store.get_node_count(),
        "fact_count": store.get_fact_count(),
        "message": (
            "Provide entity_name to look up a specific entity, "
            "list_type to enumerate entities of a given type, "
            "or newsletter_query for newsletter intelligence queries."
        ),
    }


def _handle_newsletter_query(
    mode: str,
    entity_name: Optional[str],
) -> Any:
    """Route newsletter-specific query modes. All paths are read-only."""
    try:
        nl_store = get_newsletter_graph_store()
        kg_store = get_knowledge_graph_store()
    except Exception as exc:
        return {"error": f"Newsletter store unavailable: {exc}"}

    if mode == "story_narrative":
        return _query_story_narrative(entity_name, kg_store, nl_store)

    if mode == "momentum_topics":
        topics = nl_store.get_top_momentum_topics(limit=10)
        return {
            "mode": "momentum_topics",
            "description": "Topics with highest novel-weighted coverage in the last 7 days.",
            "topics": topics,
        }

    if mode == "contrarian_positions":
        topic_node_id: Optional[int] = None
        if entity_name:
            name = entity_name.strip()
            # Try to resolve the entity to a node_id
            node = kg_store.query_node_by_name(name)
            if node:
                topic_node_id = node["id"]
        contrarians = nl_store.get_confirmed_contrarians(topic_node_id, limit=20)
        return {
            "mode": "contrarian_positions",
            "filter_topic": entity_name,
            "count": len(contrarians),
            "positions": [
                {
                    "topic": c["topic_name"],
                    "dissenting_source": c["source_name"],
                    "position": c["position_text"],
                    "prevailing_view": c["prevailing_view"],
                    "evidence_count": c["evidence_count"],
                    "updated_at": c["updated_at"],
                }
                for c in contrarians
            ],
        }

    if mode == "newsletter_sources":
        sources = nl_store.get_all_sources()
        return {
            "mode": "newsletter_sources",
            "count": len(sources),
            "sources": [
                {
                    "publication": s["publication_name"],
                    "emails_processed": s["email_count"],
                }
                for s in sources
            ],
        }

    return {"error": f"Unknown newsletter_query mode: '{mode}'"}


def _query_story_narrative(
    entity_name: Optional[str],
    kg_store: Any,
    nl_store: Any,
) -> Any:
    """Return the full framing narrative for a story matched by name or topic."""
    if not entity_name or not entity_name.strip():
        return {
            "error": "story_narrative requires entity_name set to a story title or topic name."
        }

    name = entity_name.strip()

    # Try direct story node lookup first
    story_node = kg_store.query_node_by_name(name, node_type="story")
    if story_node:
        narrative = nl_store.query_story_narrative(story_node["id"])
        logger.debug("Story narrative query hit", extra={"story": name})
        return {"mode": "story_narrative", **narrative}

    # Fallback: search for story nodes linked to this topic
    stories = nl_store.search_stories_by_topic(name)
    if not stories:
        # Also try a plain entity lookup so the caller gets something useful
        entity = kg_store.query_node_by_name(name)
        return {
            "mode": "story_narrative",
            "message": f"No story node found for '{name}'.",
            "entity": entity,
            "hint": (
                "Story nodes are created when 2+ publications cover the same topic. "
                "The topic may not yet have a story node if coverage is from a single source."
            ),
        }

    results = []
    for s in stories[:3]:
        narrative = nl_store.query_story_narrative(s["id"])
        if narrative:
            results.append(narrative)

    return {
        "mode": "story_narrative",
        "query": name,
        "stories_found": len(results),
        "stories": results,
    }


__all__ = ["build_registry", "query_knowledge_graph"]
