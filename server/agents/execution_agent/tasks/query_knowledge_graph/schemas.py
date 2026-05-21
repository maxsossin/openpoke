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
                "  'newsletter_sources' — known publication sources with credibility metrics.\n"
                "  'stories_covering_topic' — story nodes whose involves edges point to a "
                "topic matching entity_name. Two-hop traversal: topic ← story.\n"
                "  'topics_covered_by_source' — topic nodes reachable from a source via "
                "covers → involves edges. entity_name is the publication name.\n"
                "  'source_consistency_profile' — per-topic sentiment fingerprint for a "
                "publication. entity_name is the publication name.\n"
                "  'cross_story_relationships' — follows contradicts_story and follows_from "
                "edges from a story to return related stories with their latest framings. "
                "Requires entity_name set to a story title."
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
                            "stories_covering_topic",
                            "topics_covered_by_source",
                            "source_consistency_profile",
                            "cross_story_relationships",
                        ],
                        "description": (
                            "Activate a newsletter-specific query mode. "
                            "story_narrative: requires entity_name (story title or topic). "
                            "momentum_topics: returns topics accelerating in coverage this week. "
                            "contrarian_positions: confirmed dissenting positions; "
                            "optionally filtered by entity_name. "
                            "newsletter_sources: all tracked publications with credibility data. "
                            "stories_covering_topic: stories linked to entity_name topic. "
                            "topics_covered_by_source: topics covered by entity_name publication. "
                            "source_consistency_profile: sentiment fingerprint for entity_name publication. "
                            "cross_story_relationships: follow contradicts_story and follows_from "
                            "edges from entity_name story to related stories with framings."
                        ),
                    },
                    "reverse_lookup": {
                        "type": "string",
                        "description": (
                            "When set, performs a reverse edge lookup: returns all nodes that "
                            "have an active edge of this type pointing TO the entity identified "
                            "by entity_name. Example: entity_name='Acme Corp', "
                            "reverse_lookup='works_at' returns everyone who works at Acme Corp. "
                            "Supported edge types: works_at, involved_in, mentions, knows, "
                            "covers, involves, precedes, caused_by, follows_from."
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
