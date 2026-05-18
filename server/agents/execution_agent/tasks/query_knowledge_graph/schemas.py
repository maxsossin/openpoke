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
                "Returns structured facts about known people, organizations, topics, and events. "
                "Use this before task_email_search to check if relevant contact or context "
                "information is already known. This tool is read-only and fast."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": (
                            "Name of the entity to look up. Supports exact match and "
                            "falls back to substring search. "
                            "Examples: 'Alice Smith', 'Acme Corp', 'Q3 Budget Review'."
                        ),
                    },
                    "node_type": {
                        "type": "string",
                        "enum": ["person", "organization", "topic", "event"],
                        "description": (
                            "Narrow lookup to a specific entity category. "
                            "Omit to search across all types."
                        ),
                    },
                    "list_type": {
                        "type": "string",
                        "enum": ["person", "organization", "topic", "event"],
                        "description": (
                            "When entity_name is omitted, return all known entities of this type. "
                            "Useful for enumerating known contacts or organizations."
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
