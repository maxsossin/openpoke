"""Schemas for the knowledge graph query task tool."""

from __future__ import annotations

from typing import Any, Dict, List

TASK_TOOL_NAME = "query_knowledge_graph"

_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TASK_TOOL_NAME,
            "description": (
                "Query the shared knowledge graph extracted from the user's email history. "
                "Returns structured facts about people, organizations, topics, events, "
                "newsletter stories, momentum signals, and contrarian positions. "
                "Use this before task_email_search to check if relevant context is already "
                "known. This tool is read-only and fast.\n\n"
                "Newsletter-aware query modes (use newsletter_query parameter):\n"
                "  'story_narrative' — full framing timeline for a specific story across all "
                "sources. Requires entity_name set to the story title or topic name.\n"
                "  'momentum_topics' — topics with accelerating newsletter coverage this week.\n"
                "  'contrarian_positions' — confirmed dissenting positions relative to consensus. "
                "Optionally filtered by topic via entity_name.\n"
                "  'newsletter_sources' — known publication sources in the corpus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": (
                            "Name of the entity to look up. Supports exact match and "
                            "falls back to substring search. "
                            "Examples: 'Alice Smith', 'Acme Corp', 'Fed rate policy'."
                        ),
                    },
                    "node_type": {
                        "type": "string",
                        "enum": [
                            "person", "organization", "topic", "event",
                            "story", "newsletter_source",
                        ],
                        "description": (
                            "Narrow lookup to a specific entity category. "
                            "Use 'story' for cross-publication narrative nodes, "
                            "'newsletter_source' for tracked publications."
                        ),
                    },
                    "list_type": {
                        "type": "string",
                        "enum": [
                            "person", "organization", "topic", "event",
                            "story", "newsletter_source",
                        ],
                        "description": (
                            "When entity_name is omitted, enumerate all entities of this type."
                        ),
                    },
                    "newsletter_query": {
                        "type": "string",
                        "enum": [
                            "story_narrative",
                            "momentum_topics",
                            "contrarian_positions",
                            "newsletter_sources",
                        ],
                        "description": (
                            "Activate a newsletter-specific query mode. "
                            "story_narrative: requires entity_name (story title or topic). "
                            "momentum_topics: returns topics accelerating in coverage this week. "
                            "contrarian_positions: returns confirmed dissenting positions; "
                            "optionally filtered by entity_name. "
                            "newsletter_sources: lists all tracked publications."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    }
]


def get_schemas() -> List[Dict[str, Any]]:
    return _SCHEMAS


__all__ = ["TASK_TOOL_NAME", "get_schemas"]
